"""The ``thin`` pipeline stage: class filter, then graded thinning.

:class:`ThinStage` is the tile-scoped adapter over the existing ``prep``
bounded context's graded thinning: apply the classification filter, then a
:data:`~ahn_cli.prep.decimate.Thinning` request (voxel-grid or Poisson-disk),
in the same "filter then decimate" order
:func:`ahn_cli.prep.transform.prepare` uses. It reuses
:mod:`ahn_cli.prep.decimate` and :mod:`ahn_cli.prep.voxel_stream` unchanged,
so a tiled run's thinned output is byte-identical to standalone ``prep``
over the same points.

Voxel-grid thinning always routes through
:func:`ahn_cli.prep.voxel_stream.stream_voxel_thin` -- exactly how
:func:`ahn_cli.prep.transform._apply_selection` routes every
:class:`~ahn_cli.prep.decimate.VoxelThinning` request, regardless of cloud
size -- since that is the oracle this stage must match bit-for-bit.
``stream_voxel_thin`` is file-based, so :meth:`ThinStage.run` spills the
incoming tile to a scratch LAZ under the tile's
:class:`~ahn_cli.pipeline.model.TileContext.workdir`, delegates to
``stream_voxel_thin`` unchanged, then reads the thinned result back into a
new :class:`~ahn_cli.pipeline.model.PointTile`. The scratch LAZ is written at
a fixed centimetre scale and a zero offset (:data:`_SCRATCH_SCALE_M`) --
matching AHN's native LAS precision -- so the round trip is lossless for any
point whose coordinates already sit on that (or a coarser) grid, true of
every point a real ``fetch`` -> ``dedup`` -> ``thin`` chain produces.

Poisson-disk thinning stays in memory: it calls
:func:`ahn_cli.prep.decimate.thin` directly against the CPU reference
backend (:class:`~ahn_cli.prep.decimate.NumpyBackend`), the same backend
``prep.transform`` hardcodes, so this matches unconditionally too.

:meth:`ThinStage.halo_m` returns ``0.0``: thinning only selects a subset of
the tile's own points and never looks beyond its bounds, so no source
overlap is needed. A consequence is that the voxel grid's origin (the
per-cloud coordinate minimum both :mod:`~ahn_cli.prep.decimate` and
:mod:`~ahn_cli.prep.voxel_stream` anchor it to) is local to each tile, not to
the whole area of interest -- byte-identity holds against standalone
``prep`` for a single tile spanning the whole input cloud (the granularity
this stage and its tests operate at), not as a claim that independently
thinning several tiles of a larger AOI reproduces one whole-cloud run.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import laspy
import numpy as np

from ahn_cli.pipeline.model import PointTile
from ahn_cli.prep.decimate import (
    NumpyBackend,
    PoissonThinning,
    Thinning,
    VoxelThinning,
)
from ahn_cli.prep.decimate import thin as decimate_thin
from ahn_cli.prep.voxel_stream import stream_voxel_thin

if TYPE_CHECKING:
    import numpy.typing as npt

    from ahn_cli.pipeline.model import TileContext, TilePayload

__all__ = ["ThinStage"]

_SCRATCH_SCALE_M = 0.01
"""Fixed LAS scale (metres) for the voxel scratch round trip.

Matches AHN's native centimetre precision, so re-encoding an already-genuine
AHN point's coordinates through this scale is lossless.
"""

_SCRATCH_POINT_FORMAT_RGB = 7
_SCRATCH_POINT_FORMAT_PLAIN = 6

_SCRATCH_IN_NAME = "thin_stage_in.laz"
_SCRATCH_OUT_NAME = "thin_stage_out.laz"


def _class_keep(
    classification: npt.NDArray[np.uint8],
    include: tuple[int, ...],
    exclude: tuple[int, ...],
) -> npt.NDArray[np.bool_]:
    """Return the classification-filter keep-mask.

    Mirrors :func:`ahn_cli.prep.transform._class_mask`: a point is kept when
    its class is in ``include`` (or ``include`` is empty) and not in
    ``exclude``. Empty on both sides keeps every point.
    """
    keep = np.ones(classification.shape[0], dtype=np.bool_)
    if include:
        keep &= np.isin(classification, np.asarray(include))
    if exclude:
        keep &= ~np.isin(classification, np.asarray(exclude))
    return keep


def _select(tile: PointTile, indices: npt.NDArray[np.intp]) -> PointTile:
    """Return a new :class:`PointTile` holding only ``tile``'s rows at ``indices``."""
    rgb = (
        np.ascontiguousarray(tile.rgb[indices])
        if tile.rgb is not None
        else None
    )
    return PointTile(
        x=np.ascontiguousarray(tile.x[indices]),
        y=np.ascontiguousarray(tile.y[indices]),
        z=np.ascontiguousarray(tile.z[indices]),
        gps_time=np.ascontiguousarray(tile.gps_time[indices]),
        classification=np.ascontiguousarray(tile.classification[indices]),
        rgb=rgb,
    )


def _write_scratch_laz(path: Path, tile: PointTile) -> None:
    """Serialize ``tile`` to a scratch LAZ at the fixed scale/zero offset."""
    point_format = (
        _SCRATCH_POINT_FORMAT_RGB
        if tile.rgb is not None
        else _SCRATCH_POINT_FORMAT_PLAIN
    )
    header = laspy.LasHeader(point_format=point_format, version="1.4")
    header.offsets = np.zeros(3, dtype=np.float64)
    header.scales = np.full(3, _SCRATCH_SCALE_M, dtype=np.float64)
    las = laspy.LasData(header)
    las.x = tile.x
    las.y = tile.y
    las.z = tile.z
    las.gps_time = tile.gps_time
    las.classification = tile.classification
    if tile.rgb is not None:
        las.red = tile.rgb[:, 0]
        las.green = tile.rgb[:, 1]
        las.blue = tile.rgb[:, 2]
    las.write(str(path))


def _read_scratch_laz(path: Path) -> PointTile:
    """Read a scratch LAZ (as :func:`_write_scratch_laz` wrote it) into a :class:`PointTile`."""
    with laspy.open(str(path)) as reader:
        las = reader.read()
    rgb = None
    if "red" in las.point_format.dimension_names:
        rgb = np.ascontiguousarray(
            np.column_stack(
                [
                    np.asarray(las.red),
                    np.asarray(las.green),
                    np.asarray(las.blue),
                ]
            ).astype(np.uint16)
        )
    return PointTile(
        x=np.ascontiguousarray(np.asarray(las.x, dtype=np.float64)),
        y=np.ascontiguousarray(np.asarray(las.y, dtype=np.float64)),
        z=np.ascontiguousarray(np.asarray(las.z, dtype=np.float64)),
        gps_time=np.ascontiguousarray(
            np.asarray(las.gps_time, dtype=np.float64)
        ),
        classification=np.ascontiguousarray(
            np.asarray(las.classification, dtype=np.uint8)
        ),
        rgb=rgb,
    )


@dataclass(frozen=True)
class ThinStage:
    """The ``thin`` pipeline stage: class filter, then graded thinning.

    Contract:
        - ``thinning`` is the validated :data:`~ahn_cli.prep.decimate.Thinning`
          request (:class:`~ahn_cli.prep.decimate.VoxelThinning` or
          :class:`~ahn_cli.prep.decimate.PoissonThinning`), or ``None`` for a
          class-filter-only pass.
        - ``include_classes`` / ``exclude_classes`` are the classification
          filter, applied before thinning; empty tuples mean "no filter on
          that side" (mirrors
          :class:`~ahn_cli.prep.transform.PrepRequest`).

    Invariants:
        - Frozen value object, equal by field value.
        - :meth:`halo_m` is always ``0.0``: thinning is tile-local.
        - Byte-identical to standalone ``prep`` over the same points: voxel
          thinning always routes through
          :func:`~ahn_cli.prep.voxel_stream.stream_voxel_thin` and
          Poisson-disk thinning always routes through
          :func:`~ahn_cli.prep.decimate.thin` with the CPU reference
          backend, exactly as
          :func:`~ahn_cli.prep.transform._apply_selection` does.

    Failure modes:
        - :class:`TypeError` if :meth:`run` is given a payload other than a
          :class:`~ahn_cli.pipeline.model.PointTile`.
    """

    thinning: Thinning | None
    include_classes: tuple[int, ...] = ()
    exclude_classes: tuple[int, ...] = ()

    def halo_m(self) -> float:
        """Return ``0.0``: thinning never reads beyond the tile's own points."""
        return 0.0

    def run(self, tile: TilePayload, ctx: TileContext) -> TilePayload:
        """Class-filter, then thin, ``tile`` per ``self.thinning``.

        Failure modes:
            - :class:`TypeError` if ``tile`` is not a
              :class:`~ahn_cli.pipeline.model.PointTile`.
        """
        if not isinstance(tile, PointTile):
            msg = (
                "ThinStage requires a PointTile payload; got "
                f"{type(tile).__name__}."
            )
            raise TypeError(msg)
        thinning = self.thinning
        if isinstance(thinning, VoxelThinning):
            return self._voxel_thin(tile, ctx, thinning)
        filtered = self._filter(tile)
        if thinning is None:
            return filtered
        return self._poisson_thin(filtered, thinning)

    def _filter(self, tile: PointTile) -> PointTile:
        """Apply the classification filter only (no thinning)."""
        if not self.include_classes and not self.exclude_classes:
            return tile
        keep = _class_keep(
            tile.classification, self.include_classes, self.exclude_classes
        )
        return _select(tile, np.flatnonzero(keep).astype(np.intp))

    def _poisson_thin(
        self, tile: PointTile, thinning: PoissonThinning
    ) -> PointTile:
        """Poisson-disk thin an already class-filtered tile, in memory."""
        coords = np.column_stack([tile.x, tile.y, tile.z])
        indices = decimate_thin(coords, thinning, backend=NumpyBackend())
        return _select(tile, indices)

    def _voxel_thin(
        self, tile: PointTile, ctx: TileContext, thinning: VoxelThinning
    ) -> PointTile:
        """Voxel-grid thin ``tile`` via the out-of-core ``stream_voxel_thin`` oracle."""
        with tempfile.TemporaryDirectory(dir=ctx.workdir) as scratch:
            scratch_dir = Path(scratch)
            source = scratch_dir / _SCRATCH_IN_NAME
            output = scratch_dir / _SCRATCH_OUT_NAME
            _write_scratch_laz(source, tile)
            stream_voxel_thin(
                source,
                output,
                thinning.grade,
                self.include_classes,
                self.exclude_classes,
                workdir=scratch_dir,
            )
            return _read_scratch_laz(output)
