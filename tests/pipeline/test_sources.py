"""Tests for the read source and the windowed ortho seam."""

from __future__ import annotations

from typing import TYPE_CHECKING

import laspy
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from ahn_cli.pipeline.errors import PipelineError
from ahn_cli.pipeline.model import PointTile, TileContext, TileKey
from ahn_cli.pipeline.sources import (
    ReadSource,
    WindowedOrtho,
    find_ahn_sheets,
)
from tests.pipeline.scenes import build_site

if TYPE_CHECKING:
    from pathlib import Path

    from ahn_cli.domain import BBox


def _ctx(bbox: BBox, halo: float, workdir: Path) -> TileContext:
    return TileContext(
        key=TileKey(level=0, tx=0, ty=0),
        bbox=bbox,
        halo_m=halo,
        workdir=workdir,
    )


def _write_rgb_laz(path: Path) -> None:
    """Write a tiny LAZ with an RGB dimension (point format 7)."""
    header = laspy.LasHeader(point_format=7, version="1.4")
    header.offsets = np.zeros(3)
    header.scales = np.full(3, 0.01)
    las = laspy.LasData(header)
    las.x = np.array([1.0, 2.0])
    las.y = np.array([1.0, 2.0])
    las.z = np.array([3.0, 4.0])
    las.gps_time = np.array([0.1, 0.2])
    las.classification = np.array([2, 2], dtype=np.uint8)
    las.red = np.array([10, 40], dtype=np.uint16)
    las.green = np.array([20, 50], dtype=np.uint16)
    las.blue = np.array([30, 60], dtype=np.uint16)
    las.write(str(path))


# --- find_ahn_sheets --------------------------------------------------------


def test_find_ahn_sheets_rejects_non_directory(tmp_path: Path) -> None:
    """A missing directory is a clear error."""
    with pytest.raises(PipelineError, match="not a directory"):
        find_ahn_sheets(tmp_path / "nope")


def test_find_ahn_sheets_rejects_empty(tmp_path: Path) -> None:
    """A directory with no LAS/LAZ file is a clear error."""
    with pytest.raises(PipelineError, match="no LAS/LAZ sheet"):
        find_ahn_sheets(tmp_path)


# --- ReadSource -------------------------------------------------------------


def test_read_source_needs_a_sheet() -> None:
    """A ReadSource with no sheet is refused at construction."""
    with pytest.raises(PipelineError, match="at least one AHN sheet"):
        ReadSource([])


def test_read_source_loads_and_crops(tmp_path: Path) -> None:
    """A tile loads only the overlapping sheet's points, cropped to bbox+halo."""
    site, cloud, _ortho = build_site(tmp_path, seed=0)
    source = ReadSource.from_dir(site / "ahn")
    tile = source.load(_ctx((2.0, 2.0, 4.0, 4.0), 0.5, tmp_path))
    assert isinstance(tile.payload, PointTile)
    xs = tile.payload.x
    assert xs.size > 0
    assert float(xs.min()) >= 1.5
    assert float(xs.max()) <= 4.5
    assert (
        tile.content_hash
        == ReadSource([cloud])
        .load(_ctx((2.0, 2.0, 4.0, 4.0), 0.5, tmp_path))
        .content_hash
    )


def test_read_source_non_overlapping_tile_is_empty(tmp_path: Path) -> None:
    """A tile far from every sheet yields an empty point tile."""
    site, _cloud, _ortho = build_site(tmp_path, seed=1)
    source = ReadSource.from_dir(site / "ahn")
    tile = source.load(_ctx((1000.0, 1000.0, 1001.0, 1001.0), 0.0, tmp_path))
    assert isinstance(tile.payload, PointTile)
    assert tile.payload.x.shape == (0,)
    assert tile.payload.rgb is None


def test_read_source_carries_rgb(tmp_path: Path) -> None:
    """An RGB sheet's colour survives the crop into the tile."""
    sheet = tmp_path / "rgb.laz"
    _write_rgb_laz(sheet)
    source = ReadSource([sheet])
    tile = source.load(_ctx((0.0, 0.0, 5.0, 5.0), 0.0, tmp_path))
    assert isinstance(tile.payload, PointTile)
    assert tile.payload.rgb is not None
    assert tile.payload.rgb.shape == (2, 3)


# --- WindowedOrtho ----------------------------------------------------------


def test_windowed_ortho_is_pixel_aligned(tmp_path: Path) -> None:
    """A sub-window's pixel centres coincide with the global grid's."""
    _site, _cloud, ortho = build_site(tmp_path, width=8, height=6, seed=2)
    windows = WindowedOrtho(ortho)
    whole = windows.grid
    window = windows.window(_ctx((4.0, 0.0, 8.0, 3.0), 0.0, tmp_path))
    assert window.grid.width == 4
    assert window.grid.height == 3
    assert window.rgb.shape == (3, 4, 3)
    # The sub-window's first pixel centre equals the global grid's at (col=4).
    global_e = whole.eastings()
    sub_e = window.grid.eastings()
    assert float(sub_e[0, 0]) == float(global_e[3, 4])


def test_windowed_ortho_rejects_too_few_bands(tmp_path: Path) -> None:
    """A single-band raster cannot supply RGB."""
    path = tmp_path / "gray.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=1,
        dtype="uint8",
        crs="EPSG:28992",
        transform=from_bounds(0, 0, 4, 4, 4, 4),
    ) as dst:
        dst.write(np.zeros((1, 4, 4), dtype=np.uint8))
    with pytest.raises(PipelineError, match="band"):
        WindowedOrtho(path)


def test_windowed_ortho_rejects_out_of_bounds_window(tmp_path: Path) -> None:
    """A tile bbox escaping the raster is a clear error."""
    _site, _cloud, ortho = build_site(tmp_path, width=8, height=6, seed=3)
    windows = WindowedOrtho(ortho)
    with pytest.raises(PipelineError, match="outside the"):
        windows.window(_ctx((0.0, 0.0, 20.0, 6.0), 0.0, tmp_path))
