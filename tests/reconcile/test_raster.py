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
    block_target_coordinates,
    load_cloud,
    open_ortho,
)

if TYPE_CHECKING:
    from pathlib import Path

# The ``ortho_path`` fixture writes a 6x6 raster (see conftest).
_FIXTURE_SIZE = 6


def test_open_ortho_grid_and_windowed_read(ortho_path: Path) -> None:
    """A valid ortho exposes its grid and reads RGB row-blocks."""
    with open_ortho(ortho_path) as ortho:
        assert ortho.grid.width == _FIXTURE_SIZE
        assert ortho.grid.height == _FIXTURE_SIZE
        full = ortho.read_rows(0, _FIXTURE_SIZE)
        assert full.shape == (_FIXTURE_SIZE, _FIXTURE_SIZE, 3)
        assert full.dtype == np.uint8
        block = ortho.read_rows(2, 3)
        assert block.shape == (3, _FIXTURE_SIZE, 3)
        # The windowed strip equals the corresponding rows of the full read.
        assert np.array_equal(block, full[2:5])


def test_open_ortho_missing_raises(tmp_path: Path) -> None:
    """An unreadable ortho path raises the typed ReconcileError."""
    with pytest.raises(ReconcileError, match="not readable"):
        open_ortho(tmp_path / "absent.tif")


def test_open_ortho_too_few_bands_raises(tmp_path: Path) -> None:
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
        dst.write(np.zeros((2, 2, 2), dtype=np.uint8))
    with pytest.raises(ReconcileError, match="band"):
        open_ortho(path)


def test_load_cloud_returns_coords_and_classification(
    cloud_path: Path,
) -> None:
    """A valid LAZ yields its (n, 3) coordinates and (n,) classification."""
    cloud = load_cloud(cloud_path)
    assert cloud.coords.shape == (200, 3)
    assert cloud.coords.dtype == np.float64
    assert cloud.classification.shape == (200,)
    assert cloud.classification.dtype == np.uint8


def test_load_cloud_missing_raises(tmp_path: Path) -> None:
    """An unreadable cloud path raises the typed ReconcileError."""
    with pytest.raises(ReconcileError, match="not readable"):
        load_cloud(tmp_path / "absent.laz")


def test_block_target_coordinates_full_grid() -> None:
    """Whole-grid block coordinates match the pixel-centre convention."""
    # North-up: origin top-left (10, 20), 2 m pixels, 3 wide x 2 high.
    grid = PixelGrid(
        width=3, height=2, transform=(2.0, 0.0, 10.0, 0.0, -2.0, 20.0)
    )
    target_xy, eastings, northings = block_target_coordinates(grid, 0, 2)
    assert target_xy.shape == (6, 2)
    assert eastings.shape == (2, 3)
    assert northings.shape == (2, 3)
    # Pixel-centre of column 0, row 0: easting 10 + 1, northing 20 - 1.
    assert math.isclose(target_xy[0, 0], 11.0)
    assert math.isclose(target_xy[0, 1], 19.0)


def test_block_target_coordinates_partial_block() -> None:
    """A partial block starts at the requested row (centre convention held)."""
    grid = PixelGrid(
        width=3, height=4, transform=(2.0, 0.0, 10.0, 0.0, -2.0, 20.0)
    )
    _, _, northings = block_target_coordinates(grid, 2, 1)
    assert northings.shape == (1, 3)
    # Row index 2 centre: northing 20 - 2*(2.5) = 20 - 5 = 15.
    assert math.isclose(northings[0, 0], 15.0)


def test_open_ortho_uniform_placeholder_raises(tmp_path: Path) -> None:
    """A uniform-colour 'ortho' (a placeholder grid) is refused outright.

    Regression for the Moerkapelle gray-cloud incident: reconcile colours its
    output from the ortho, so a synthetic constant-colour stand-in (built
    during a Beeldmateriaal outage) must fail fast — never paint the whole
    cloud a single gray.
    """
    path = tmp_path / "placeholder.tif"
    transform = from_bounds(0.0, 0.0, 4.0, 4.0, 8, 8)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=8,
        width=8,
        count=3,
        dtype="uint8",
        crs="EPSG:28992",
        transform=transform,
    ) as dst:
        dst.write(np.full((3, 8, 8), 128, dtype=np.uint8))
    with pytest.raises(ReconcileError, match="uniform colour"):
        open_ortho(path)


def test_open_ortho_accepts_partially_constant_imagery(
    tmp_path: Path,
) -> None:
    """One constant band is fine while the imagery varies overall."""
    path = tmp_path / "dusk.tif"
    transform = from_bounds(0.0, 0.0, 4.0, 4.0, 8, 8)
    bands = np.zeros((3, 8, 8), dtype=np.uint8)
    bands[0] = 10  # constant red channel
    bands[1] = np.arange(64, dtype=np.uint8).reshape(8, 8)  # varying green
    bands[2] = 200
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=8,
        width=8,
        count=3,
        dtype="uint8",
        crs="EPSG:28992",
        transform=transform,
    ) as dst:
        dst.write(bands)
    with open_ortho(path) as reader:
        assert reader.grid.width == 8
