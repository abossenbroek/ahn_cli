"""The quadtree tiling plan: pixel spans, LOD strides, geometric error.

The grid is subdivided top-down until every leaf spans at most
``tile_pixels`` pixels per axis. A tile at level ``l`` (root = 0,
leaves = ``levels``) samples every ``2**(levels - l)``-th pixel of its
span — every vertex at every level is a **genuine** source sample, no
averaging or synthesis — and adjacent tiles share their boundary pixel
column/row so edges coincide within a level. ``geometric_error`` is 0
for leaves (they carry the full-resolution data) and grows with the
sampling stride above them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ahn_cli.tiles3d.errors import Tiles3dError

__all__ = [
    "TilePlan",
    "TreePlan",
    "geometric_error",
    "plan_quadtree",
    "sample_indices",
]

_DEFAULT_TILE_PIXELS = 256
_MIN_AXIS_SAMPLES = 2
_ERROR_PER_STRIDE_PIXEL = 4.0
"""Geometric error per metre of sampling stride (a display heuristic)."""


@dataclass(frozen=True)
class TilePlan:
    """One planned tile: its pixel span, LOD stride and children.

    Contract (fields):
        - ``level``: 0 for the root, ``levels`` for leaves.
        - ``tx``/``ty``: the split-index path (unique per level).
        - ``col0``/``row0``/``col1``/``row1``: the INCLUSIVE pixel span;
          adjacent tiles share their boundary column/row.
        - ``stride``: the sampling stride (``1`` for leaves).
        - ``children``: the subdivided tiles, empty for leaves.

    Invariants:
        - Frozen value object, equal by field value.
    """

    level: int
    tx: int
    ty: int
    col0: int
    row0: int
    col1: int
    row1: int
    stride: int
    children: tuple[TilePlan, ...]


@dataclass(frozen=True)
class TreePlan:
    """The whole quadtree: its root, depth and total tile count."""

    root: TilePlan
    levels: int
    tile_count: int


def plan_quadtree(
    width: int, height: int, tile_pixels: int = _DEFAULT_TILE_PIXELS
) -> TreePlan:
    """Plan the quadtree for a ``width x height`` grid.

    Contract:
        - Leaves span at most ``tile_pixels + 1`` pixels per axis
          (boundary shared) and cover every pixel of the grid.
        - Level ``l`` tiles carry ``stride = 2**(levels - l)``.

    Failure modes:
        - :class:`Tiles3dError` if either axis has fewer than 2 pixels:
          a surface needs at least two samples per axis.
    """
    if width < _MIN_AXIS_SAMPLES or height < _MIN_AXIS_SAMPLES:
        msg = (
            f"a {width}x{height} grid cannot be tiled: at least 2 "
            "pixels per axis are required for a surface."
        )
        raise Tiles3dError(msg)
    levels = 0
    while max(width, height) > tile_pixels * 2**levels:
        levels += 1
    count = [0]
    root = _subdivide(
        level=0,
        tx=0,
        ty=0,
        span=(0, 0, width - 1, height - 1),
        levels=levels,
        count=count,
    )
    return TreePlan(root=root, levels=levels, tile_count=count[0])


def sample_indices(
    start: int, stop_inclusive: int, stride: int
) -> npt.NDArray[np.int64]:
    """Return ``start..stop_inclusive`` strided, always keeping the end."""
    indices = np.arange(start, stop_inclusive + 1, stride, dtype=np.int64)
    if int(indices[-1]) != stop_inclusive:
        indices = np.append(indices, np.int64(stop_inclusive))
    return indices


def geometric_error(stride: int, pixel_size: float) -> float:
    """Return the tile's 3D Tiles geometric error in metres.

    Leaves (stride 1) carry the source data exactly: error 0. Coarser
    levels drop ``stride - 1`` of every ``stride`` samples, so their
    error scales with the stride's ground size.
    """
    if stride == 1:
        return 0.0
    return stride * pixel_size * _ERROR_PER_STRIDE_PIXEL


def _split_axis(lo: int, hi: int) -> tuple[tuple[int, int], ...]:
    """Split ``[lo, hi]`` at its midpoint, sharing the boundary index.

    An axis whose span cannot yield two children of at least two
    samples each stays whole.
    """
    if hi - lo < _MIN_AXIS_SAMPLES:
        return ((lo, hi),)
    mid = lo + (hi - lo) // 2
    return ((lo, mid), (mid, hi))


def _subdivide(
    *,
    level: int,
    tx: int,
    ty: int,
    span: tuple[int, int, int, int],
    levels: int,
    count: list[int],
) -> TilePlan:
    """Recursively build the tile at ``(level, tx, ty)`` over ``span``.

    ``span`` is the inclusive ``(col0, row0, col1, row1)`` pixel range.
    """
    col0, row0, col1, row1 = span
    count[0] += 1
    children: tuple[TilePlan, ...] = ()
    if level < levels:
        col_spans = _split_axis(col0, col1)
        row_spans = _split_axis(row0, row1)
        children = tuple(
            _subdivide(
                level=level + 1,
                tx=2 * tx + ci,
                ty=2 * ty + ri,
                span=(cspan[0], rspan[0], cspan[1], rspan[1]),
                levels=levels,
                count=count,
            )
            for ri, rspan in enumerate(row_spans)
            for ci, cspan in enumerate(col_spans)
        )
    return TilePlan(
        level=level,
        tx=tx,
        ty=ty,
        col0=col0,
        row0=row0,
        col1=col1,
        row1=row1,
        stride=2 ** (levels - level),
        children=children,
    )
