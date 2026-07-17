"""Assemble a :class:`~ahn_cli.pipeline.spec.PipelineSpec` into a run.

:func:`run_spec` is the ``pipeline run`` verb's engine. It wires a parsed spec
into the concrete seams the executor drives -- a :class:`TileSource`, an
:class:`~ahn_cli.pipeline.stages.reconcile.OrthoWindows`, the stage chain, and a
sink-appropriate :class:`~ahn_cli.pipeline.tiling.TilePlanner` -- calls
:func:`ahn_cli.pipeline.executor.run_pipeline` to stream every tile end-to-end
(resumable, bounded memory), then assembles the sink's deliverable:

* a ``tiles3d`` sink streams its per-tile blobs into a staging
  :class:`~ahn_cli.pipeline.manifest.TileStore` under the workdir, then
  :func:`ahn_cli.pipeline.assemble.assemble_tiles3d` stitches them into
  ``tileset.json`` (+ ``tiles.hfp`` and sidecars for the lossy profiles) in the
  output directory -- byte-identical to the standalone ``tiles3d`` verb for a
  single-tile area of interest;
* a cloud/``write`` sink writes each tile's reconciled-grid blob straight into
  the output directory (the executor's native per-tile store), the shape the
  reconcile stage's halo-kNN byte-identity is proven against.

Only the ``read`` source is wired here (a pre-populated on-disk site: an
``ahn/`` directory of LAZ sheets plus an ``ortho.tif``). A ``fetch`` source is a
deferred seam -- pre-fetch with the ``fetch`` verb and point a ``read`` stage at
the site. RAM/CPU/pool are all injected through to the executor, never sensed
live here, so a run is deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from ahn_cli.pipeline.assemble import (
    REGION_BLOB_NAME,
    assemble_tiles3d,
    region_blob_bytes,
)
from ahn_cli.pipeline.errors import PipelineError
from ahn_cli.pipeline.executor import PipelineResult, run_pipeline
from ahn_cli.pipeline.manifest import TileStore
from ahn_cli.pipeline.model import EncodedBlob, EncodedTile, GridTile
from ahn_cli.pipeline.planners import GridTilePlanner, QuadtreePlanner
from ahn_cli.pipeline.sources import ReadSource, WindowedOrtho
from ahn_cli.pipeline.spec import (
    DedupStage as DedupSpec,
)
from ahn_cli.pipeline.spec import (
    FetchStage,
    ReadStage,
    Tiles3dStage,
    spec_hash,
)
from ahn_cli.pipeline.spec import (
    ReconcileStage as ReconcileSpec,
)
from ahn_cli.pipeline.spec import (
    ThinStage as ThinSpec,
)
from ahn_cli.pipeline.stages.dedup import DedupStage
from ahn_cli.pipeline.stages.reconcile import ReconcileStage
from ahn_cli.pipeline.stages.thin import ThinStage
from ahn_cli.pipeline.stages.tiles3d import Tiles3dSink
from ahn_cli.pipeline.stages.write import GridWriteSink
from ahn_cli.pipeline.wiring import neighbors_for, thinning_for

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ahn_cli.domain import BBox
    from ahn_cli.pipeline.executor import PoolFactory, TileSource
    from ahn_cli.pipeline.machine import SystemProbe
    from ahn_cli.pipeline.model import Stage, TileContext, TilePayload
    from ahn_cli.pipeline.spec import PipelineSpec, StageSpec
    from ahn_cli.pipeline.stages.reconcile import OrthoWindows
    from ahn_cli.pipeline.tiling import TilePlanner

__all__ = ["PipelineRunResult", "run_spec"]

DEFAULT_POINT_SPACING_M = 0.5
"""Fallback AHN native point spacing (metres) driving the halo floor.

AHN's ground sampling is roughly half a metre; the reconcile stage's kNN
correctness floor scales with it. A ``read`` source carries no generation, so
this conservative default is used unless a caller overrides it.
"""

_STORE_SUBDIR = "_tiles3d_store"
"""Workdir sub-directory holding the tiles3d sink's per-tile staging blobs."""

_AHN_SUBDIR = "ahn"
_ORTHO_NAMES = ("ortho.tif", "ortho/ortho.tif")


@dataclass(frozen=True)
class PipelineRunResult:
    """The outcome of a :func:`run_spec` run.

    Contract:
        - ``out_dir`` is the deliverable directory; ``deliverable_path`` the
          primary artifact (``tileset.json`` for a tiles3d sink, the tile
          manifest for a cloud sink).
        - ``tile_count`` / ``processed`` / ``skipped`` mirror the executor's
          :class:`~ahn_cli.pipeline.executor.PipelineResult` counts.

    Invariants:
        - Frozen value object, equal by field value.
    """

    out_dir: Path
    deliverable_path: Path
    tile_count: int
    processed: int
    skipped: int


@dataclass(frozen=True)
class _RegionRecordingSink:
    """Wrap a :class:`Tiles3dSink`, persisting each tile's own region blob.

    The generic executor persists only a stage's returned
    :class:`~ahn_cli.pipeline.model.EncodedTile` blobs and knows nothing of
    per-tile regions; the cross-tile assembler needs each tile's own bounding
    region to fold children-first. This wrapper computes the region alongside
    encoding and appends it as a ``region.json`` blob, so it is persisted and
    resume-skipped exactly like the geometry/texture blobs -- no executor or
    manifest change needed.
    """

    base: Tiles3dSink

    def halo_m(self) -> float:
        """Return the wrapped sink's halo (always ``0``)."""
        return self.base.halo_m()

    def run(self, tile: TilePayload, ctx: TileContext) -> TilePayload:
        """Encode ``tile`` and append its own region as a ``region.json`` blob."""
        encoded = self.base.run(tile, ctx)
        # ``base.run`` succeeded, so ``tile`` is a GridTile; region_of needs it.
        region = self.base.region_of(cast("GridTile", tile), ctx)
        assert isinstance(encoded, EncodedTile)  # noqa: S101 -- sink post-condition
        region_blob = EncodedBlob(
            name=REGION_BLOB_NAME, data=region_blob_bytes(region)
        )
        return EncodedTile(
            key=encoded.key, blobs=(*encoded.blobs, region_blob)
        )


def _resolve_read_inputs(site: Path) -> tuple[Path, Path]:
    """Return the ``(cloud_dir, ortho_path)`` of a read-source site directory.

    The cloud directory is ``<site>/ahn`` when present, else ``<site>`` itself;
    the ortho is the first of ``<site>/ortho.tif`` / ``<site>/ortho/ortho.tif``.

    Failure modes:
        - :class:`~ahn_cli.pipeline.errors.PipelineError` if ``site`` is not a
          directory or holds no orthophoto.
    """
    if not site.is_dir():
        msg = f"read source path {site} is not a directory."
        raise PipelineError(msg)
    ahn_dir = site / _AHN_SUBDIR
    cloud_dir = ahn_dir if ahn_dir.is_dir() else site
    for name in _ORTHO_NAMES:
        candidate = site / name
        if candidate.is_file():
            return cloud_dir, candidate
    msg = (
        f"read source site {site} has no orthophoto "
        f"(looked for {list(_ORTHO_NAMES)})."
    )
    raise PipelineError(msg)


def _ortho_aoi(windows: WindowedOrtho) -> BBox:
    """Return the orthophoto's full EPSG:28992 extent as the run's area."""
    grid = windows.grid
    a, _b, c, _d, e, f = grid.transform
    return (c, f + e * grid.height, c + a * grid.width, f)


def _build_middle_stage(
    spec: StageSpec,
    windows: OrthoWindows,
    point_spacing_m: float,
) -> Stage:
    """Build one concrete middle stage (dedup/thin/reconcile) from its spec."""
    if isinstance(spec, DedupSpec):
        return DedupStage(
            include_classes=spec.include_classes,
            exclude_classes=spec.exclude_classes,
        )
    if isinstance(spec, ThinSpec):
        return ThinStage(thinning=thinning_for(spec))
    if isinstance(spec, ReconcileSpec):
        return ReconcileStage(
            method=spec.method,
            ortho=windows,
            neighbors=neighbors_for(spec.method),
            point_spacing_m=point_spacing_m,
        )
    msg = (
        f"stage {type(spec).__name__} cannot appear between the source and "
        "sink; only dedup, thin and reconcile are middle stages."
    )
    raise PipelineError(msg)


def run_spec(
    spec: PipelineSpec,
    *,
    point_spacing_m: float = DEFAULT_POINT_SPACING_M,
    probe: SystemProbe | None = None,
    cpu_count: int | None = None,
    pool_factory: PoolFactory | None = None,
    fault: Callable[[str], None] | None = None,
) -> PipelineRunResult:
    """Run ``spec`` end-to-end over its read source, returning the outcome.

    Contract:
        - Wires the spec's ``read`` source, ortho windows, stage chain and sink
          planner, calls the executor, then assembles the deliverable.
        - The area of interest is the orthophoto's full extent (a ``read``
          source's ortho delimits the deliverable); the pixel size is the
          ortho's native resolution.
        - ``probe`` / ``cpu_count`` / ``pool_factory`` / ``fault`` pass straight
          through to :func:`ahn_cli.pipeline.executor.run_pipeline` for
          deterministic, injectable RAM/CPU/pool and fault injection.

    Failure modes:
        - :class:`~ahn_cli.pipeline.errors.PipelineError` if the source is a
          ``fetch`` stage (a deferred seam), the read site lacks an ortho, or a
          stage is misplaced.
    """
    source_spec = spec.stages[0]
    sink_spec = spec.stages[-1]
    if isinstance(source_spec, FetchStage):
        msg = (
            "the pipeline verb wires only a `read` source today; pre-fetch "
            "with `ahn_cli fetch` and point a `read` stage at the site."
        )
        raise PipelineError(msg)
    # The spec guarantees a source-first chain, so the only other source is a
    # read stage; narrow to it without a defensive (uncoverable) branch.
    read_spec = cast("ReadStage", source_spec)
    cloud_dir, ortho_path = _resolve_read_inputs(Path(read_spec.path))
    windows = WindowedOrtho(ortho_path)
    pixel_size_m = windows.grid.transform[0]
    aoi = _ortho_aoi(windows)
    source: TileSource = ReadSource.from_dir(cloud_dir)

    middle = [
        _build_middle_stage(stage, windows, point_spacing_m)
        for stage in spec.stages[1:-1]
    ]
    return _run_with_sink(
        spec=spec,
        sink_spec=sink_spec,
        middle=middle,
        source=source,
        aoi=aoi,
        pixel_size_m=pixel_size_m,
        probe=probe,
        cpu_count=cpu_count,
        pool_factory=pool_factory,
        fault=fault,
    )


def _run_with_sink(  # noqa: PLR0913 -- assembled internal call; every arg is load-bearing
    *,
    spec: PipelineSpec,
    sink_spec: StageSpec,
    middle: Sequence[Stage],
    source: TileSource,
    aoi: BBox,
    pixel_size_m: float,
    probe: SystemProbe | None,
    cpu_count: int | None,
    pool_factory: PoolFactory | None,
    fault: Callable[[str], None] | None,
) -> PipelineRunResult:
    """Dispatch on the sink type: drive the executor, then assemble."""
    out_dir = spec.output
    workdir = spec.workdir
    signature = spec_hash(spec)
    if isinstance(sink_spec, Tiles3dStage):
        planner = QuadtreePlanner(
            pixel_size_m=pixel_size_m, tile_pixels=spec.tiling.tile_pixels
        )
        tree = planner.tree_for(aoi)
        sink: Stage = _RegionRecordingSink(
            Tiles3dSink(
                profile=sink_spec.profile,
                native_pixel_size_m=pixel_size_m,
                levels=tree.levels,
            )
        )
        stages: tuple[Stage, ...] = (*middle, sink)
        store_root = workdir / _STORE_SUBDIR
        result = _drive(
            planner,
            aoi,
            stages,
            source,
            signature,
            store_root,
            workdir,
            probe,
            cpu_count,
            pool_factory,
            fault,
        )
        deliverable = assemble_tiles3d(
            TileStore(store_root),
            tree,
            pixel_size_m=pixel_size_m,
            out_dir=out_dir,
            profile=sink_spec.profile,
        )
        return _result(out_dir, deliverable, result)
    # The spec guarantees a sink-last chain, so the only other sink is a write
    # stage; the write path is the fall-through (no uncoverable else branch).
    grid_planner = GridTilePlanner(
        tile_size_m=spec.tiling.tile_pixels * pixel_size_m
    )
    stages = (*middle, GridWriteSink())
    result = _drive(
        grid_planner,
        aoi,
        stages,
        source,
        signature,
        out_dir,
        workdir,
        probe,
        cpu_count,
        pool_factory,
        fault,
    )
    return _result(out_dir, result.manifest_path, result)


def _drive(  # noqa: PLR0913 -- thin pass-through to the executor's keyword seams
    planner: TilePlanner,
    aoi: BBox,
    stages: Sequence[Stage],
    source: TileSource,
    signature: str,
    store_root: Path,
    workdir: Path,
    probe: SystemProbe | None,
    cpu_count: int | None,
    pool_factory: PoolFactory | None,
    fault: Callable[[str], None] | None,
) -> PipelineResult:
    """Call :func:`run_pipeline` with the resolved halo floor."""
    halo_floor_m = max((stage.halo_m() for stage in stages), default=0.0)
    return run_pipeline(
        planner=planner,
        aoi_bbox=aoi,
        stages=stages,
        source=source,
        signature=signature,
        out_dir=store_root,
        workdir=workdir,
        halo_floor_m=halo_floor_m,
        probe=probe,
        cpu_count=cpu_count,
        pool_factory=pool_factory,
        fault=fault,
    )


def _result(
    out_dir: Path, deliverable: Path, result: PipelineResult
) -> PipelineRunResult:
    """Build a :class:`PipelineRunResult` from the executor's result."""
    return PipelineRunResult(
        out_dir=out_dir,
        deliverable_path=deliverable,
        tile_count=result.tile_count,
        processed=result.processed,
        skipped=result.skipped,
    )
