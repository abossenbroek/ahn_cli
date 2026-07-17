"""Sink-driven concrete tile planners for the ``pipeline`` verb.

The executor drives one output tile at a time; the *planner* decides which
tiles exist and at what extent. Two concrete planners back the two sink
families of the ``pipeline`` verb:

* :class:`QuadtreePlanner` -- the ``tiles3d`` sink's output grid. It maps the
  area of interest onto the pixel grid at the ortho's native resolution and
  reuses :func:`ahn_cli.tiles3d.quadtree.plan_quadtree` unchanged, emitting one
  :class:`~ahn_cli.pipeline.model.TileContext` per quadtree node (root through
  leaves). A node's world bbox is the exact pixel-edge extent of its inclusive
  pixel span, so a leaf tile (``stride == 1``) reconstructs pixel-for-pixel
  through :class:`~ahn_cli.pipeline.stages.tiles3d.Tiles3dSink` -- byte-identical
  to the standalone ``tiles3d`` build for a single-tile (``levels == 0``) area.
* :class:`~ahn_cli.pipeline.tiling.GridTilePlanner` (re-exported) -- the
  cloud/``write`` sink's output grid: a clean, pixel-aligned partition of the
  area of interest with no shared boundaries, the shape the reconcile stage's
  halo-kNN identity is proven against.

Both planners are **pure functions of the area of interest** (never of RAM), so
the deliverable's tile set is identical whatever the machine -- the two-budget
byte-identity invariant depends on it. The resolved ``halo_m`` the executor
passes is stamped onto every tile but never changes which tiles exist.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ahn_cli.domain import ensure_valid_bbox
from ahn_cli.pipeline.errors import PipelineError
from ahn_cli.pipeline.model import TileContext, TileKey
from ahn_cli.pipeline.tiling import GridTilePlanner
from ahn_cli.tiles3d.errors import Tiles3dError
from ahn_cli.tiles3d.quadtree import TilePlan, TreePlan, plan_quadtree

if TYPE_CHECKING:
    from pathlib import Path

    from ahn_cli.domain import BBox

__all__ = ["GridTilePlanner", "QuadtreePlanner", "aoi_pixel_dims"]


def aoi_pixel_dims(aoi_bbox: BBox, pixel_size_m: float) -> tuple[int, int]:
    """Return the ``(width, height)`` pixel dimensions of ``aoi_bbox``.

    The area of interest is measured in EPSG:28992 metres; at the ortho's
    native ground sampling distance ``pixel_size_m`` it spans
    ``round((maxx - minx) / pixel_size_m)`` columns and the matching rows.
    Rounding (not floor/ceil) keeps a bbox that is an exact pixel multiple on
    grid, the case every real ``fetch``/ortho pair produces.

    Failure modes:
        - :class:`ValueError` if ``aoi_bbox`` is degenerate.
        - :class:`~ahn_cli.pipeline.errors.PipelineError` if ``pixel_size_m``
          is not finite and positive, or the area rounds to under one pixel on
          either axis.
    """
    ensure_valid_bbox(aoi_bbox)
    if not math.isfinite(pixel_size_m) or pixel_size_m <= 0.0:
        msg = f"pixel_size_m must be finite and positive; got {pixel_size_m}."
        raise PipelineError(msg)
    minx, miny, maxx, maxy = aoi_bbox
    width = round((maxx - minx) / pixel_size_m)
    height = round((maxy - miny) / pixel_size_m)
    if width < 1 or height < 1:
        msg = (
            f"area of interest {aoi_bbox} spans under one pixel at "
            f"{pixel_size_m} m ({width}x{height}); nothing to tile."
        )
        raise PipelineError(msg)
    return width, height


@dataclass(frozen=True)
class QuadtreePlanner:
    """The ``tiles3d`` sink's quadtree output grid over an area of interest.

    Contract:
        - ``pixel_size_m`` is the ortho's native ground sampling distance;
          ``tile_pixels`` the quadtree leaf edge length in pixels.
        - :meth:`plan` measures the area of interest in pixels
          (:func:`aoi_pixel_dims`), plans the quadtree
          (:func:`ahn_cli.tiles3d.quadtree.plan_quadtree`), and emits one
          :class:`~ahn_cli.pipeline.model.TileContext` per node -- the root
          first, then its children depth-first -- each carrying the resolved
          ``halo_m``.
        - A node's bbox is the pixel-edge extent of its inclusive pixel span,
          anchored at the area's north-west corner: a leaf's bbox spans exactly
          its ``(col1 - col0 + 1) x (row1 - row0 + 1)`` pixels, so the tiles3d
          sink reconstructs it pixel-for-pixel.

    Invariants:
        - Frozen value object; :meth:`plan` is a pure function of its inputs.

    Failure modes:
        - :class:`~ahn_cli.pipeline.errors.PipelineError` on a bad
          ``pixel_size_m`` or a sub-pixel area (via :func:`aoi_pixel_dims`), or
          if the quadtree planner rejects the pixel dimensions (an area under
          two pixels per axis).
    """

    pixel_size_m: float
    tile_pixels: int = 256

    def plan(
        self, *, aoi_bbox: BBox, halo_m: float, workdir: Path
    ) -> tuple[TileContext, ...]:
        """Return the quadtree's tiles covering ``aoi_bbox``, root first."""
        width, height = aoi_pixel_dims(aoi_bbox, self.pixel_size_m)
        tree: TreePlan = self._plan_tree(width, height)
        minx, _miny, _maxx, maxy = aoi_bbox
        tiles: list[TileContext] = []
        self._walk(tree.root, minx, maxy, halo_m, workdir, tiles)
        return tuple(tiles)

    def levels_for(self, aoi_bbox: BBox) -> int:
        """Return the quadtree depth for ``aoi_bbox`` (0 for a single tile)."""
        return self.tree_for(aoi_bbox).levels

    def tree_for(self, aoi_bbox: BBox) -> TreePlan:
        """Return the quadtree plan for ``aoi_bbox`` (its root, depth, count)."""
        width, height = aoi_pixel_dims(aoi_bbox, self.pixel_size_m)
        return self._plan_tree(width, height)

    def _plan_tree(self, width: int, height: int) -> TreePlan:
        """Plan the quadtree, translating the tiles3d error to a pipeline one."""
        try:
            return plan_quadtree(width, height, self.tile_pixels)
        except Tiles3dError as exc:
            raise PipelineError(str(exc)) from exc

    def _walk(
        self,
        node: TilePlan,
        minx: float,
        maxy: float,
        halo_m: float,
        workdir: Path,
        out: list[TileContext],
    ) -> None:
        """Append ``node``'s context, then recurse into its children."""
        px = self.pixel_size_m
        west = minx + node.col0 * px
        east = minx + (node.col1 + 1) * px
        north = maxy - node.row0 * px
        south = maxy - (node.row1 + 1) * px
        out.append(
            TileContext(
                key=TileKey(level=node.level, tx=node.tx, ty=node.ty),
                bbox=(west, south, east, north),
                halo_m=halo_m,
                workdir=workdir,
            )
        )
        for child in node.children:
            self._walk(child, minx, maxy, halo_m, workdir, out)
