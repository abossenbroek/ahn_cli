"""Per-tile reconcile stage: interpolate one tile's ortho grid from its cloud.

:class:`ReconcileStage` adapts the standalone ``reconcile`` verb to the
tile-streaming pipeline. It is a thin scope-narrowing wrapper: it cleans the
tile's point cloud with the verb's own :func:`~ahn_cli.reconcile.clean.select_and_dedupe`,
builds the verb's interpolator with :func:`~ahn_cli.reconcile.interpolate.build_interpolator`
(whose kNN runs through :func:`ahn_cli.reconcile.neighbors.build_tree` unchanged,
``workers=-1``), and streams the tile's ortho grid in the same row-blocks as
:func:`ahn_cli.reconcile.reconcile.reconcile` -- so a tiled run's per-pixel
estimate is byte-identical to a global one.

The load-bearing invariant is the **halo correctness floor** (:meth:`ReconcileStage.halo_m`):
a tile-edge ortho pixel must see exactly the neighbours a whole-area run would.
The estimate at a target is a function of its ``k`` nearest source points; if the
source halo reaches past the ``k``-th neighbour's distance, the tile-local kNN
set equals the global one and the estimate does not move. The floor is derived
from the reconcile neighbour count and the local point spacing via
:func:`ahn_cli.pipeline.tiling.derive_halo_floor` -- the same floor the executor's
tiling uses -- so a sparser area widens the halo. A halo below the floor can drop
a genuine neighbour and silently corrupt an edge estimate; a halo at or above it
is byte-identical to the global run.

The tile's ortho window (the RGB pixels plus the tile's
:class:`~ahn_cli.domain.grid.PixelGrid`) is supplied by an injected
:class:`OrthoWindows` source keyed off the tile context, never read live here;
the tile's local point spacing and neighbour count are stage configuration (AHN's
native spacing and the method's ``k``), so :meth:`halo_m` needs no point data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from ahn_cli.pipeline.errors import PipelineError
from ahn_cli.pipeline.model import GridTile, PointTile
from ahn_cli.pipeline.tiling import derive_halo_floor
from ahn_cli.reconcile.clean import select_and_dedupe
from ahn_cli.reconcile.interpolate import build_interpolator
from ahn_cli.reconcile.raster import ReconcileError, block_target_coordinates

if TYPE_CHECKING:
    import numpy.typing as npt

    from ahn_cli.domain.grid import PixelGrid
    from ahn_cli.pipeline.model import TileContext, TilePayload
    from ahn_cli.reconcile.interpolate import Interpolator
    from ahn_cli.reconcile.method import InterpMethod

__all__ = [
    "OrthoWindow",
    "OrthoWindows",
    "ReconcileStage",
]

_RGB_COMPONENTS = 3
"""Colour components in an ortho window's ``rgb`` plane (``(h, w, 3)``)."""

_DEFAULT_FLOOR_MARGIN = 1.5
"""Default safety multiple on the raw kNN reach (sparse/clustered areas).

Matches :data:`ahn_cli.pipeline.tiling._DEFAULT_FLOOR_MARGIN`; kept as this
context's own constant so the stage does not depend on a private tiling symbol.
"""

_BLOCK_CELLS = 262_144
"""Target cell count per row-block, mirroring ``reconcile``'s streaming schedule.

The block schedule is a pure function of the tile width and never changes a
pixel's estimate (each estimate is per-target), so it bounds peak memory without
touching the output bytes.
"""


@dataclass(frozen=True, eq=False)
class OrthoWindow:
    """One tile's orthophoto window: its pixel grid plus the RGB pixels.

    Contract:
        - ``grid`` is the tile's :class:`~ahn_cli.domain.grid.PixelGrid`; its
          pixel centres must coincide with the corresponding whole-area grid
          centres (a pixel-aligned sub-window) so a tiled estimate lands on the
          same targets as a global one.
        - ``rgb`` is the ``(grid.height, grid.width, 3)`` ``uint8`` colour plane
          (red, green, blue) for that window, the pixels the output cloud is
          coloured from.

    Invariants:
        - Frozen; identity equality (holds a numpy plane, so not value-hashable).

    Failure modes:
        - :class:`ValueError` if ``rgb`` does not have shape
          ``(grid.height, grid.width, 3)``.
    """

    grid: PixelGrid
    rgb: npt.NDArray[np.uint8]

    def __post_init__(self) -> None:
        """Validate the RGB plane matches the grid shape."""
        expected = (self.grid.height, self.grid.width, _RGB_COMPONENTS)
        if self.rgb.shape != expected:
            msg = (
                f"ortho window rgb must have shape {expected}; "
                f"got {self.rgb.shape}."
            )
            raise ValueError(msg)


@runtime_checkable
class OrthoWindows(Protocol):
    """Supplies a tile's :class:`OrthoWindow` for a given tile context.

    In the real pipeline this reads the orthophoto windowed per tile; in tests
    it is a deterministic in-RAM fake. It is the seam that keeps the stage free
    of live raster IO -- ``run`` never opens a file.
    """

    def window(self, ctx: TileContext) -> OrthoWindow:
        """Return the ortho window for the tile ``ctx`` identifies."""
        ...


@dataclass(frozen=True)
class ReconcileStage:
    """A tile-scoped ``reconcile`` :class:`~ahn_cli.pipeline.model.Stage`.

    Contract:
        - ``method`` is the reused
          :data:`~ahn_cli.reconcile.method.InterpMethod` (linear / IDW /
          kriging), already validated by its own type.
        - ``ortho`` supplies each tile's :class:`OrthoWindow`.
        - ``neighbors`` is the reconcile neighbour count driving the halo floor
          (the method's ``k`` for IDW/kriging); ``point_spacing_m`` is the
          cloud's native local spacing in metres; ``margin`` is the floor's
          safety multiple.
        - ``include_classes`` / ``exclude_classes`` are the LAS classes to keep /
          drop before interpolation (empty tuples keep every class), matching the
          standalone verb; coincident-XY returns are always de-duplicated.
        - :meth:`halo_m` returns the correctness floor; :meth:`run` maps a
          :class:`~ahn_cli.pipeline.model.PointTile` (the tile's cloud plus its
          halo) to a :class:`~ahn_cli.pipeline.model.GridTile` (heights + colour).

    Invariants:
        - Frozen value object, equal by field value.
    """

    method: InterpMethod
    ortho: OrthoWindows
    neighbors: int
    point_spacing_m: float
    include_classes: tuple[int, ...] = field(default_factory=tuple)
    exclude_classes: tuple[int, ...] = field(default_factory=tuple)
    margin: float = _DEFAULT_FLOOR_MARGIN

    def halo_m(self) -> float:
        """Return the kNN correctness floor for the source halo, in metres.

        The floor is ``sqrt(neighbors) * point_spacing_m * margin`` (via
        :func:`ahn_cli.pipeline.tiling.derive_halo_floor`): to see the ``k``
        nearest neighbours a global run would, a tile-edge pixel must reach
        roughly ``sqrt(k)`` rings of points at the local spacing, times a safety
        multiple for sparse or clustered areas. Below this a tile-edge estimate
        can silently diverge; at or above it every estimate matches the global
        run.

        Failure modes:
            - :class:`ValueError` if ``neighbors < 1``, ``point_spacing_m`` is
              not a finite positive length, or ``margin < 1``.
        """
        return derive_halo_floor(
            neighbors=self.neighbors,
            point_spacing_m=self.point_spacing_m,
            margin=self.margin,
        )

    def run(self, tile: TilePayload, ctx: TileContext) -> GridTile:
        """Interpolate the tile's ortho grid from its cloud, returning a grid.

        Contract:
            - ``tile`` is the tile's :class:`~ahn_cli.pipeline.model.PointTile`
              (local points plus the halo the plan resolved). The cloud is
              cleaned (class filter + XY de-duplication) exactly as the standalone
              verb cleans it, the interpolator is built once, and the tile's ortho
              grid is streamed in row-blocks -- so every estimate is byte-identical
              to a whole-area ``reconcile`` when the halo covers the kNN reach.
            - Returns a :class:`~ahn_cli.pipeline.model.GridTile` whose heights
              are the per-pixel estimates and whose colour is the ortho window.

        Failure modes:
            - :class:`~ahn_cli.pipeline.errors.PipelineError` if ``tile`` is not
              a :class:`~ahn_cli.pipeline.model.PointTile` (a wrongly ordered
              chain).
            - :class:`~ahn_cli.reconcile.raster.ReconcileError` if any pixel has
              no genuine estimate (an empty source or a void cell): missing data
              is a hard error -- the stage never fabricates ("infills") a value.
        """
        point = _require_point_tile(tile)
        window = self.ortho.window(ctx)
        interpolator = self._build_interpolator(point)
        heights = self._interpolate_heights(window.grid, interpolator, ctx)
        rgb = window.rgb
        return GridTile(
            heights=heights,
            red=np.ascontiguousarray(rgb[:, :, 0]),
            green=np.ascontiguousarray(rgb[:, :, 1]),
            blue=np.ascontiguousarray(rgb[:, :, 2]),
        )

    def _build_interpolator(self, point: PointTile) -> Interpolator:
        """Clean the tile's cloud, then build the reconcile interpolator once."""
        count = point.x.shape[0]
        coords = np.empty((count, 3), dtype=np.float64)
        coords[:, 0] = point.x
        coords[:, 1] = point.y
        coords[:, 2] = point.z
        cleaned = select_and_dedupe(
            coords,
            point.classification,
            self.include_classes,
            self.exclude_classes,
        )
        return build_interpolator(self.method, cleaned)

    def _interpolate_heights(
        self,
        grid: PixelGrid,
        interpolator: Interpolator,
        ctx: TileContext,
    ) -> npt.NDArray[np.float32]:
        """Stream the tile grid in row-blocks, estimating a height per pixel."""
        width, height = grid.width, grid.height
        block_rows = max(1, _BLOCK_CELLS // width)
        heights = np.empty((height, width), dtype=np.float32)
        for row_start in range(0, height, block_rows):
            rows = min(block_rows, height - row_start)
            target_xy, _, _ = block_target_coordinates(grid, row_start, rows)
            z, valid = interpolator.estimate(target_xy)
            _verify_full_coverage(valid, ctx, row_start, rows)
            heights[row_start : row_start + rows] = z.reshape(
                rows, width
            ).astype(np.float32)
        return heights


def _require_point_tile(tile: TilePayload) -> PointTile:
    """Return ``tile`` as a :class:`PointTile`, or raise for a bad chain."""
    if not isinstance(tile, PointTile):
        msg = (
            f"reconcile stage expected a PointTile; got "
            f"{type(tile).__name__}. A reconcile stage must follow a stage "
            "that produces a point cloud."
        )
        raise PipelineError(msg)
    return tile


def _verify_full_coverage(
    valid: npt.NDArray[np.bool_],
    ctx: TileContext,
    row_start: int,
    rows: int,
) -> None:
    """Hard-verify every pixel of this block got a genuine estimate.

    A void cell means the method could not derive a height from the tile's
    cloud (an empty source, or a target outside linear's convex hull). Writing
    it would fabricate a value or leave a hole, so missing data is a hard error.

    Failure modes:
        - :class:`~ahn_cli.reconcile.raster.ReconcileError` if any cell of
          ``valid`` is False, naming the void count and the affected rows.
    """
    if bool(valid.all()):
        return
    voids = int(valid.size - int(valid.sum()))
    last_row = row_start + rows - 1
    msg = (
        f"{voids} pixel(s) in rows {row_start}..{last_row} of tile "
        f"{ctx.key} have no genuine elevation estimate from the tile's "
        "cloud; missing data is an error -- the reconcile stage never "
        'fabricates ("infills") elevations.'
    )
    raise ReconcileError(msg)
