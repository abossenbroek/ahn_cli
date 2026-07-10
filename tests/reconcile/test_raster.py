"""Tests for the reconcile raster/point-cloud IO."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from ahn_cli.domain.grid import PixelGrid
from ahn_cli.reconcile.raster import (
    ReconcileError,
    load_cloud,
    load_ortho,
    target_coordinates,
)

if TYPE_CHECKING:
    from pathlib import Path

# The ``ortho_path`` fixture writes a 6x6 raster (see conftest).
_FIXTURE_SIZE = 6


def test_load_ortho_returns_grid_and_rgb(ortho_path: Path) -> None:
    """A valid ortho yields its grid dimensions and (h, w, 3) uint8 RGB."""
    ortho = load_ortho(ortho_path)
    assert ortho.grid.width == _FIXTURE_SIZE
    assert ortho.grid.height == _FIXTURE_SIZE
    assert ortho.rgb.shape == (_FIXTURE_SIZE, _FIXTURE_SIZE, 3)
    assert ortho.rgb.dtype == np.uint8


def test_load_ortho_missing_raises(tmp_path: Path) -> None:
    """An unreadable ortho path raises the typed ReconcileError."""
    with pytest.raises(ReconcileError, match="not readable"):
        load_ortho(tmp_path / "absent.tif")


def test_load_ortho_too_few_bands_raises(tmp_path: Path) -> None:
    """An ortho with fewer than three bands raises ReconcileError."""
    path = tmp_path / "gray.tif"
    transform = from_bounds(0.0, 0.0, 2.0, 2.0, 2, 2)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=2,
        dtype="uint8",
        crs="EPSG:28992",
        transform=transform,
    ) as dst:
        dst.write(np.zeros((2, 2, 2), dtype=np.uint8).transpose(2, 0, 1))
    with pytest.raises(ReconcileError, match="band"):
        load_ortho(path)


def test_load_cloud_returns_xyz(cloud_path: Path) -> None:
    """A valid LAZ yields its (n, 3) world coordinates."""
    xyz = load_cloud(cloud_path)
    assert xyz.shape == (200, 3)
    assert xyz.dtype == np.float64


def test_load_cloud_missing_raises(tmp_path: Path) -> None:
    """An unreadable cloud path raises the typed ReconcileError."""
    with pytest.raises(ReconcileError, match="not readable"):
        load_cloud(tmp_path / "absent.laz")


def test_target_coordinates_shapes_and_first_centre() -> None:
    """Target XY flattens the grid; the first entry is the top-left centre."""
    # North-up: origin top-left (10, 20), 2 m pixels, 3 wide x 2 high.
    grid = PixelGrid(
        width=3, height=2, transform=(2.0, 0.0, 10.0, 0.0, -2.0, 20.0)
    )
    target_xy, eastings, northings = target_coordinates(grid)
    assert target_xy.shape == (6, 2)
    assert eastings.shape == (2, 3)
    assert northings.shape == (2, 3)
    # Pixel-centre of column 0, row 0: easting 10 + 1, northing 20 - 1.
    assert math.isclose(target_xy[0, 0], 11.0)
    assert math.isclose(target_xy[0, 1], 19.0)
