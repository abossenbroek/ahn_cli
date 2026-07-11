"""Prep-context transform/export orchestration.

The ``prep`` bounded context turns cached source tiles into finished
deliverables. WP14 wires the pipeline end to end over the already-merged prep
transforms, preserving this module's public surface (:class:`PrepRequest`,
:func:`prepare`):

1. Read the cached AHN tiles a prior ``fetch`` wrote under ``<data_dir>/ahn/``,
   each paired with its provenance-recorded extent.
2. :func:`~ahn_cli.prep.dedup.deduplicate_tiles` -- crop-before-merge plus an
   exact XYZ+GPS-time sweep -- into ``<data_dir>/pointcloud.laz``.
3. Apply the classification ``include``/``exclude`` filter.
4. Apply the graded :data:`~ahn_cli.prep.decimate.Thinning` request (voxel-grid
   or Poisson-disk) via the CPU reference backend, when one is requested. This
   is additive to the legacy nth-point decimation, which is untouched.
5. Write the site-root ``provenance.json`` recording the prep lineage.
6. Export ``<data_dir>/pointcloud.ply`` when ``export_points`` is set.

The mirrored ordering is the documented "filter classes -> clip -> decimate"
with WP10's crop-before-merge folded into step 2 (the merge is where the crop
must happen). Every stage is deterministic, so identical inputs yield
byte-identical outputs. Expected failures (a missing site layout, no tiles, a
tile with no provenance sidecar) raise the typed :class:`PrepError` so the CLI
reports a tidy message rather than leaking a traceback.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import laspy
import numpy as np
import numpy.typing as npt

from ahn_cli.domain import Product, Provenance
from ahn_cli.domain.authenticity import degenerate_cloud
from ahn_cli.prep.decimate import (
    NumpyBackend,
    Thinning,
    VoxelThinning,
    thin,
)
from ahn_cli.prep.dedup import CanonicalTile, DedupStats, deduplicate_tiles
from ahn_cli.prep.ply import export_ply
from ahn_cli.provenance import read_provenance, write_provenance

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path

    from ahn_cli.domain import BBox

_AHN_SUBDIR = "ahn"
_POINTCLOUD_LAZ = "pointcloud.laz"
_POINTCLOUD_PLY = "pointcloud.ply"
_PROVENANCE = "provenance.json"


class PrepError(RuntimeError):
    """Raised when a prep run cannot proceed for an expected reason.

    Signals a missing site layout (no ``<data_dir>/ahn/`` directory), no fetched
    AHN tiles to prepare, or a tile whose provenance sidecar -- the source of its
    crop extent -- is absent. Subclasses :class:`RuntimeError` so the CLI catches
    it and reports a clean error rather than a traceback.
    """


@dataclass(frozen=True)
class PrepRequest:
    """A validated intent to transform a fetched site.

    Contract:
        - ``data_dir`` is the site directory a prior ``fetch`` produced.
        - ``include_classes`` / ``exclude_classes`` are the parsed, validated
          classification filters; empty tuples mean "no filter on this side".
          The caller guarantees the two do not overlap.
        - ``export_points`` requests the point-cloud export.
        - ``thinning`` is the validated graded-thinning request (voxel-grid or
          Poisson-disk), or ``None`` for no additional thinning. It is additive
          to the legacy nth-point decimation, which is unaffected.

    Invariants:
        - Frozen: an immutable, hashable value object, equal by field value.
    """

    data_dir: Path
    include_classes: tuple[int, ...] = ()
    exclude_classes: tuple[int, ...] = ()
    export_points: bool = False
    thinning: Thinning | None = None


def prepare(request: PrepRequest) -> None:
    """Transform a fetched site into its finished point-cloud deliverables.

    Contract:
        - Reads the cached AHN tiles under ``<data_dir>/ahn/`` (sorted by name
          for a stable byte-hash), deduplicates them (crop-before-merge plus an
          exact XYZ+GPS-time sweep) into ``<data_dir>/pointcloud.laz``, applies
          the classification ``include``/``exclude`` filter, then the graded
          ``thinning`` request via the CPU reference backend, and writes the
          site-root ``provenance.json`` recording the run's lineage.
        - With ``export_points`` it additionally writes
          ``<data_dir>/pointcloud.ply``.

    Invariants:
        - Deterministic: identical fetched inputs yield byte-identical
          ``pointcloud.laz``, ``pointcloud.ply``, and ``provenance.json``. Every
          field of the provenance derives from the inputs (source sidecars, file
          bytes, request), never wall-clock time.

    Failure modes:
        - :class:`PrepError` if ``<data_dir>/ahn/`` is missing, holds no LAZ
          tiles, a tile lacks the provenance sidecar its crop extent needs, or
          the finished cloud is degenerate (no points survived the class
          filter/thinning, or every point sits at one identical position) —
          a degenerate deliverable is never emitted as genuine AHN output:
          the rejected ``pointcloud.laz`` is removed before the error
          propagates, and ``pointcloud.ply``/``provenance.json`` are only
          written after the verification gate, so a rejected run leaves no
          deliverable behind.
    """
    ahn_dir = request.data_dir / _AHN_SUBDIR
    if not ahn_dir.is_dir():
        msg = (
            f"no fetched AHN data at {ahn_dir}; run 'fetch' into "
            f"{request.data_dir} first."
        )
        raise PrepError(msg)
    laz_paths = sorted(ahn_dir.glob("*.LAZ"))
    if not laz_paths:
        msg = f"no AHN tiles (*.LAZ) to prepare in {ahn_dir}."
        raise PrepError(msg)

    tiles, provenances = _load_tiles(ahn_dir, laz_paths)
    output = request.data_dir / _POINTCLOUD_LAZ
    dedup_stats = deduplicate_tiles(tiles, output)
    output_points = _apply_selection(output, request)
    _verify_output_cloud(output)
    if request.export_points:
        export_ply(output, request.data_dir / _POINTCLOUD_PLY)
    _write_prep_provenance(
        request, tiles, provenances, dedup_stats, output_points
    )


def _load_tiles(
    ahn_dir: Path, laz_paths: list[Path]
) -> tuple[list[CanonicalTile], list[Provenance]]:
    """Pair each tile with the crop extent recorded in its provenance sidecar.

    Failure modes:
        - :class:`PrepError` if a tile has no provenance sidecar (the fetch
          writes one per tile; its absence means the crop extent is unknown).
    """
    tiles: list[CanonicalTile] = []
    provenances: list[Provenance] = []
    for path in laz_paths:
        sidecar = ahn_dir / f"{path.stem}.provenance.json"
        if not sidecar.is_file():
            msg = (
                f"tile {path.name} has no provenance sidecar at "
                f"{sidecar.name}; its crop extent is unknown."
            )
            raise PrepError(msg)
        provenance = read_provenance(sidecar)
        tiles.append(CanonicalTile(path=path, extent=provenance.bbox))
        provenances.append(provenance)
    return tiles, provenances


def _apply_selection(output: Path, request: PrepRequest) -> int:
    """Filter by class and thin ``output`` in place, returning the point count.

    When neither a class filter nor a thinning request applies, the deduplicated
    file is already final and is left untouched (its header point count is
    returned). Otherwise the cloud is read once, the class mask and thinning are
    applied in the documented "filter then decimate" order, and the result is
    written back deterministically.
    """
    include = request.include_classes
    exclude = request.exclude_classes
    thinning = request.thinning
    if not include and not exclude and thinning is None:
        with laspy.open(str(output)) as reader:
            return int(reader.header.point_count)
    with laspy.open(str(output)) as reader:
        las = reader.read()
    keep = _class_mask(las, include, exclude)
    las.points = las.points[keep]
    if thinning is not None:
        indices = thin(_coords(las), thinning, backend=NumpyBackend())
        las.points = las.points[indices]
    las.write(str(output))
    return len(las.points)


def _verify_output_cloud(output: Path) -> None:
    """Hard-verify the finished cloud is genuine, non-degenerate AHN data.

    Runs after the class filter and thinning, before the PLY export and the
    provenance record, so a degenerate deliverable never leaves the prep
    stage: the written header must not describe an empty cloud or a stack of
    points all at one identical position. On rejection the degenerate
    ``pointcloud.laz`` is removed before the error propagates, so it cannot
    poison a later ``copc`` run that reads the canonical output path; the
    PLY export and provenance record are written only after this gate, so
    the LAZ is the sole artefact needing cleanup.

    Failure modes:
        - :class:`PrepError` if the finished cloud is degenerate (the
          rejected ``output`` file is removed first).
    """
    with laspy.open(str(output)) as reader:
        header = reader.header
        count = int(header.point_count)
        mins = header.mins
        maxs = header.maxs
    if degenerate_cloud(count, mins, maxs):
        detail = (
            "no points survived the class filter/thinning"
            if count == 0
            else f"all {count} points sit at one identical position"
        )
        output.unlink(missing_ok=True)
        msg = (
            f"prepared cloud at {output} is degenerate ({detail}) — "
            "refusing to emit it as genuine AHN output."
        )
        raise PrepError(msg)


def _class_mask(
    las: laspy.LasData,
    include: tuple[int, ...],
    exclude: tuple[int, ...],
) -> npt.NDArray[np.bool_]:
    """Return the boolean keep-mask for the classification filter.

    A point is kept when its class is in ``include`` (or ``include`` is empty)
    and not in ``exclude``; the two sides are guaranteed non-overlapping by the
    caller. Empty on both sides keeps every point.
    """
    classification = np.asarray(las.classification)
    keep = np.ones(classification.shape[0], dtype=np.bool_)
    if include:
        keep &= np.isin(classification, np.asarray(include))
    if exclude:
        keep &= ~np.isin(classification, np.asarray(exclude))
    return keep


def _coords(las: laspy.LasData) -> npt.NDArray[np.float64]:
    """Return the ``(n, 3)`` world coordinates of ``las`` for thinning."""
    return np.column_stack(
        [
            np.asarray(las.x, dtype=np.float64),
            np.asarray(las.y, dtype=np.float64),
            np.asarray(las.z, dtype=np.float64),
        ]
    )


def _write_prep_provenance(
    request: PrepRequest,
    tiles: list[CanonicalTile],
    provenances: list[Provenance],
    dedup_stats: DedupStats,
    output_points: int,
) -> None:
    """Write the site-root ``provenance.json`` recording the prep run's lineage.

    Licence, attribution, generation, portal, and the tool version are carried
    forward from the source tiles' sidecars; the download window is their
    aggregate; the extent is the union of the tiles' crop extents. The point
    counts of each stage are recorded as request keys, so the record is
    deterministic and reproducible from the inputs alone.
    """
    base = provenances[0]
    output = request.data_dir / _POINTCLOUD_LAZ
    provenance = Provenance(
        source_portal=base.source_portal,
        product=Product.AHN_POINT_CLOUD,
        licence=base.licence,
        attribution=base.attribution,
        bbox=_union_bbox([tile.extent for tile in tiles]),
        download_started_at=_earliest(provenances),
        download_finished_at=_latest(provenances),
        input_checksum=_concat_checksum([tile.path for tile in tiles]),
        output_checksum=hashlib.sha256(output.read_bytes()).hexdigest(),
        tool_version=base.tool_version,
        generation=base.generation,
        request_keys=(
            ("stage", "prep"),
            ("source_tiles", ",".join(tile.path.stem for tile in tiles)),
            ("input_points", str(dedup_stats.input_points)),
            ("cropped_points", str(dedup_stats.cropped_points)),
            ("duplicates_removed", str(dedup_stats.duplicates_removed)),
            ("deduplicated_points", str(dedup_stats.output_points)),
            ("output_points", str(output_points)),
            ("include_classes", _class_label(request.include_classes)),
            ("exclude_classes", _class_label(request.exclude_classes)),
            ("thinning", _thinning_label(request.thinning)),
            ("points_exported", _bool_label(export=request.export_points)),
        ),
    )
    write_provenance(provenance, request.data_dir / _PROVENANCE)


def _union_bbox(extents: list[BBox]) -> BBox:
    """Return the axis-aligned union of one or more canonical tile extents."""
    minx = min(extent[0] for extent in extents)
    miny = min(extent[1] for extent in extents)
    maxx = max(extent[2] for extent in extents)
    maxy = max(extent[3] for extent in extents)
    return (minx, miny, maxx, maxy)


def _concat_checksum(paths: list[Path]) -> str:
    """Return a SHA-256 over the source tile bytes in the given (sorted) order."""
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _earliest(provenances: list[Provenance]) -> datetime:
    """Return the earliest download start across the source sidecars."""
    return min(provenance.download_started_at for provenance in provenances)


def _latest(provenances: list[Provenance]) -> datetime:
    """Return the latest download finish across the source sidecars."""
    return max(provenance.download_finished_at for provenance in provenances)


def _class_label(classes: tuple[int, ...]) -> str:
    """Render a classification filter list as a stable provenance value."""
    return ",".join(str(code) for code in classes)


def _bool_label(*, export: bool) -> str:
    """Render a boolean flag as a stable ``"true"``/``"false"`` provenance value."""
    return "true" if export else "false"


def _thinning_label(spec: Thinning | None) -> str:
    """Render a thinning request as a stable provenance value (typed dispatch)."""
    if spec is None:
        return "none"
    if isinstance(spec, VoxelThinning):
        return f"voxel:{spec.grade}"
    return f"poisson:{spec.radius}:{spec.seed}"
