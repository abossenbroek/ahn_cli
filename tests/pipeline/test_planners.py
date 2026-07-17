"""Tests for the sink-driven concrete planners."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from ahn_cli.pipeline.errors import PipelineError
from ahn_cli.pipeline.model import TileKey
from ahn_cli.pipeline.planners import (
    GridTilePlanner,
    QuadtreePlanner,
    aoi_pixel_dims,
)

if TYPE_CHECKING:
    from pathlib import Path

    from ahn_cli.domain import BBox

_AOI: BBox = (0.0, 0.0, 8.0, 6.0)


def test_grid_tile_planner_is_reexported() -> None:
    """The cloud sink's grid planner is exposed here for wiring symmetry."""
    assert GridTilePlanner(tile_size_m=4.0).tile_size_m == 4.0


def test_aoi_pixel_dims() -> None:
    """The area's pixel dimensions round its metre extent at the pixel size."""
    assert aoi_pixel_dims(_AOI, 0.5) == (16, 12)


def test_aoi_pixel_dims_rejects_bad_pixel_size() -> None:
    """A non-positive pixel size is a configuration error."""
    with pytest.raises(PipelineError, match="pixel_size_m"):
        aoi_pixel_dims(_AOI, 0.0)


def test_aoi_pixel_dims_rejects_non_finite_pixel_size() -> None:
    """A NaN pixel size is a configuration error."""
    with pytest.raises(PipelineError, match="pixel_size_m"):
        aoi_pixel_dims(_AOI, float("nan"))


def test_aoi_pixel_dims_rejects_subpixel_area() -> None:
    """An area under one pixel on an axis has nothing to tile."""
    with pytest.raises(PipelineError, match="under one pixel"):
        aoi_pixel_dims((0.0, 0.0, 0.4, 6.0), 1.0)


def test_aoi_pixel_dims_rejects_degenerate_bbox() -> None:
    """A degenerate area is rejected by the domain validator."""
    with pytest.raises(ValueError, match="bbox"):
        aoi_pixel_dims((0.0, 0.0, 0.0, 6.0), 1.0)


def test_quadtree_single_tile(tmp_path: Path) -> None:
    """A small area yields a single root tile spanning the whole area."""
    planner = QuadtreePlanner(pixel_size_m=1.0, tile_pixels=256)
    tiles = planner.plan(aoi_bbox=_AOI, halo_m=0.0, workdir=tmp_path)
    assert len(tiles) == 1
    assert tiles[0].key == TileKey(level=0, tx=0, ty=0)
    assert tiles[0].bbox == _AOI
    assert planner.levels_for(_AOI) == 0


def test_quadtree_multi_level_root_first(tmp_path: Path) -> None:
    """A larger area subdivides; the root comes first and stamps the halo."""
    planner = QuadtreePlanner(pixel_size_m=1.0, tile_pixels=4)
    tiles = planner.plan(aoi_bbox=_AOI, halo_m=2.5, workdir=tmp_path)
    assert planner.levels_for(_AOI) == 1
    assert len(tiles) > 1
    assert tiles[0].key.level == 0
    assert all(ctx.halo_m == 2.5 for ctx in tiles)
    # The root bbox spans the whole area; children are strict sub-boxes.
    assert tiles[0].bbox == _AOI
    assert {ctx.key.level for ctx in tiles} == {0, 1}


def test_quadtree_rejects_too_small_for_a_surface(tmp_path: Path) -> None:
    """An area under two pixels per axis cannot be a surface."""
    planner = QuadtreePlanner(pixel_size_m=1.0, tile_pixels=256)
    with pytest.raises(PipelineError, match="at least 2"):
        planner.plan(
            aoi_bbox=(0.0, 0.0, 1.0, 1.0), halo_m=0.0, workdir=tmp_path
        )


def test_quadtree_tree_for_reports_count(tmp_path: Path) -> None:
    """``tree_for`` exposes the underlying quadtree plan."""
    _ = tmp_path
    planner = QuadtreePlanner(pixel_size_m=1.0, tile_pixels=4)
    tree = planner.tree_for(_AOI)
    assert tree.levels == 1
    assert tree.tile_count >= 1
