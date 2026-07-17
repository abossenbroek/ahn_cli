"""Tests for the in-memory :class:`DedupStage`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from ahn_cli.pipeline.model import PointTile, TileContext, TileKey
from ahn_cli.pipeline.stages.dedup import DedupStage
from tests.pipeline.harness import make_grid_tile

if TYPE_CHECKING:
    from pathlib import Path


def _ctx(workdir: Path) -> TileContext:
    return TileContext(
        key=TileKey(level=0, tx=0, ty=0),
        bbox=(0.0, 0.0, 10.0, 10.0),
        halo_m=0.0,
        workdir=workdir,
    )


def _tile(
    xs: list[float],
    ys: list[float],
    zs: list[float],
    gps: list[float],
    cls: list[int],
) -> PointTile:
    return PointTile(
        x=np.array(xs, dtype=np.float64),
        y=np.array(ys, dtype=np.float64),
        z=np.array(zs, dtype=np.float64),
        gps_time=np.array(gps, dtype=np.float64),
        classification=np.array(cls, dtype=np.uint8),
    )


def test_halo_is_zero() -> None:
    """De-duplication is tile-local."""
    assert DedupStage().halo_m() == 0.0


def test_exact_duplicates_collapse_to_smallest_index(tmp_path: Path) -> None:
    """A repeated x,y,z,gps point keeps its first occurrence, ascending order."""
    tile = _tile(
        xs=[1.0, 2.0, 1.0, 3.0],
        ys=[1.0, 2.0, 1.0, 3.0],
        zs=[5.0, 6.0, 5.0, 7.0],
        gps=[0.1, 0.2, 0.1, 0.3],
        cls=[2, 2, 2, 2],
    )
    out = DedupStage().run(tile, _ctx(tmp_path))
    assert isinstance(out, PointTile)
    # index 2 duplicates index 0 -> dropped; survivors 0,1,3 in ascending order.
    assert out.x.tolist() == [1.0, 2.0, 3.0]
    assert out.z.tolist() == [5.0, 6.0, 7.0]


def test_same_xy_different_z_is_not_a_duplicate(tmp_path: Path) -> None:
    """Two points at one XY but different Z are both kept (not exact dups)."""
    tile = _tile(
        xs=[1.0, 1.0],
        ys=[1.0, 1.0],
        zs=[5.0, 6.0],
        gps=[0.1, 0.1],
        cls=[2, 2],
    )
    out = DedupStage().run(tile, _ctx(tmp_path))
    assert isinstance(out, PointTile)
    assert out.z.tolist() == [5.0, 6.0]


def test_class_filter_include(tmp_path: Path) -> None:
    """Only included classes survive the filter."""
    tile = _tile(
        xs=[1.0, 2.0, 3.0],
        ys=[1.0, 2.0, 3.0],
        zs=[1.0, 2.0, 3.0],
        gps=[0.1, 0.2, 0.3],
        cls=[2, 6, 9],
    )
    out = DedupStage(include_classes=(2, 6)).run(tile, _ctx(tmp_path))
    assert isinstance(out, PointTile)
    assert sorted(out.classification.tolist()) == [2, 6]


def test_class_filter_exclude(tmp_path: Path) -> None:
    """Excluded classes are dropped."""
    tile = _tile(
        xs=[1.0, 2.0, 3.0],
        ys=[1.0, 2.0, 3.0],
        zs=[1.0, 2.0, 3.0],
        gps=[0.1, 0.2, 0.3],
        cls=[2, 6, 9],
    )
    out = DedupStage(exclude_classes=(9,)).run(tile, _ctx(tmp_path))
    assert isinstance(out, PointTile)
    assert 9 not in out.classification.tolist()


def test_empty_tile_stays_empty(tmp_path: Path) -> None:
    """An empty input yields an empty output (no reduceat crash)."""
    tile = _tile(xs=[], ys=[], zs=[], gps=[], cls=[])
    out = DedupStage().run(tile, _ctx(tmp_path))
    assert isinstance(out, PointTile)
    assert out.x.shape == (0,)


def test_rgb_is_carried_through(tmp_path: Path) -> None:
    """A tile's RGB plane follows its surviving points."""
    tile = PointTile(
        x=np.array([1.0, 1.0, 2.0], dtype=np.float64),
        y=np.array([1.0, 1.0, 2.0], dtype=np.float64),
        z=np.array([5.0, 5.0, 6.0], dtype=np.float64),
        gps_time=np.array([0.1, 0.1, 0.2], dtype=np.float64),
        classification=np.array([2, 2, 2], dtype=np.uint8),
        rgb=np.array(
            [[10, 20, 30], [10, 20, 30], [40, 50, 60]], dtype=np.uint16
        ),
    )
    out = DedupStage().run(tile, _ctx(tmp_path))
    assert isinstance(out, PointTile)
    assert out.rgb is not None
    assert out.rgb.tolist() == [[10, 20, 30], [40, 50, 60]]


def test_non_point_tile_is_a_type_error(tmp_path: Path) -> None:
    """A grid payload is a wiring error."""
    with pytest.raises(TypeError, match="requires a PointTile"):
        DedupStage().run(make_grid_tile(), _ctx(tmp_path))
