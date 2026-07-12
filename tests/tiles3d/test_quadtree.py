"""Tests for the quadtree tiling plan."""

from __future__ import annotations

import numpy as np
import pytest

from ahn_cli.tiles3d.errors import Tiles3dError
from ahn_cli.tiles3d.quadtree import (
    TilePlan,
    geometric_error,
    plan_quadtree,
    sample_indices,
)


def _leaves(tile: TilePlan) -> list[TilePlan]:
    if not tile.children:
        return [tile]
    found: list[TilePlan] = []
    for child in tile.children:
        found.extend(_leaves(child))
    return found


def _all_tiles(tile: TilePlan) -> list[TilePlan]:
    found = [tile]
    for child in tile.children:
        found.extend(_all_tiles(child))
    return found


def test_small_grid_is_a_single_leaf() -> None:
    """A grid within one tile needs no subdivision."""
    tree = plan_quadtree(6, 6)
    assert tree.levels == 0
    assert tree.tile_count == 1
    root = tree.root
    assert (root.col0, root.row0, root.col1, root.row1) == (0, 0, 5, 5)
    assert root.stride == 1
    assert root.children == ()


def test_600_grid_builds_two_levels() -> None:
    """A 600x600 grid subdivides twice into 16 leaves."""
    tree = plan_quadtree(600, 600)
    assert tree.levels == 2
    assert tree.root.stride == 4
    leaves = _leaves(tree.root)
    assert len(leaves) == 16
    assert tree.tile_count == 21  # 1 + 4 + 16
    assert all(leaf.stride == 1 for leaf in leaves)
    assert all(leaf.level == 2 for leaf in leaves)
    for tile in _all_tiles(tree.root):
        assert tile.stride == 2 ** (tree.levels - tile.level)


@pytest.mark.parametrize(
    ("width", "height"),
    [(6, 6), (257, 257), (600, 300), (2, 300), (513, 2)],
)
def test_leaves_cover_every_pixel(width: int, height: int) -> None:
    """Leaf spans tile the full grid; only boundaries are shared."""
    tree = plan_quadtree(width, height)
    coverage = np.zeros((height, width), dtype=np.int32)
    for leaf in _leaves(tree.root):
        assert leaf.col1 - leaf.col0 + 1 <= 256 + 1
        assert leaf.row1 - leaf.row0 + 1 <= 256 + 1
        coverage[leaf.row0 : leaf.row1 + 1, leaf.col0 : leaf.col1 + 1] += 1
    assert int(coverage.min()) >= 1
    # Interior pixels are covered exactly once; only shared boundary
    # rows/columns may be covered more than once.
    multi = np.argwhere(coverage > 1)
    boundary_cols = {leaf.col0 for leaf in _leaves(tree.root)} | {
        leaf.col1 for leaf in _leaves(tree.root)
    }
    boundary_rows = {leaf.row0 for leaf in _leaves(tree.root)} | {
        leaf.row1 for leaf in _leaves(tree.root)
    }
    for row, col in multi:
        assert int(col) in boundary_cols or int(row) in boundary_rows


def test_tile_ids_are_unique() -> None:
    """Every (level, tx, ty) triple appears exactly once."""
    tree = plan_quadtree(600, 300)
    ids = [(t.level, t.tx, t.ty) for t in _all_tiles(tree.root)]
    assert len(ids) == len(set(ids))


def test_thin_axis_does_not_split_below_two_samples() -> None:
    """A two-pixel-wide grid never produces a one-column tile."""
    tree = plan_quadtree(2, 300)
    for tile in _all_tiles(tree.root):
        assert tile.col1 - tile.col0 >= 1
        assert tile.row1 - tile.row0 >= 1


def test_single_pixel_axis_is_refused() -> None:
    """A grid without two samples on each axis cannot be a surface."""
    with pytest.raises(Tiles3dError, match="at least 2"):
        plan_quadtree(1, 300)


def test_sample_indices_stride_and_last() -> None:
    """Strided sampling always includes the last index."""
    assert sample_indices(0, 10, 4).tolist() == [0, 4, 8, 10]
    assert sample_indices(0, 8, 4).tolist() == [0, 4, 8]
    assert sample_indices(3, 3, 2).tolist() == [3]
    assert sample_indices(0, 5, 1).tolist() == [0, 1, 2, 3, 4, 5]


def test_geometric_error_grades() -> None:
    """Leaves are exact; coarser levels scale with their stride."""
    assert geometric_error(1, 0.5) == 0.0
    assert geometric_error(4, 0.5) == 8.0
    assert abs(geometric_error(2, 0.08) - 0.64) < 1e-12
