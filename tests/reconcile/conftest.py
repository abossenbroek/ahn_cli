"""Shared synthetic ortho/cloud fixtures for the reconcile tests.

Tiny in-process fixtures (a 6x6 EPSG:28992 RGB GeoTIFF and a covering AHN LAZ)
keep the unit tests fast and offline; the real Amsterdam data is exercised
separately by the integration test.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import laspy
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

if TYPE_CHECKING:
    from pathlib import Path

WIDTH = 6
HEIGHT = 6
RES = 0.5
MINX = 100.0
MAXY = 100.0 + HEIGHT * RES
MAXX = MINX + WIDTH * RES
MINY = MAXY - HEIGHT * RES


@pytest.fixture
def ortho_path(tmp_path: Path) -> Path:
    """Write a deterministic 6x6 3-band uint8 EPSG:28992 orthophoto."""
    transform = from_bounds(MINX, MINY, MAXX, MAXY, WIDTH, HEIGHT)
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, (3, HEIGHT, WIDTH)).astype(np.uint8)
    path = tmp_path / "ortho.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=HEIGHT,
        width=WIDTH,
        count=3,
        dtype="uint8",
        crs="EPSG:28992",
        transform=transform,
    ) as dst:
        dst.write(rgb)
    return path


@pytest.fixture
def cloud_path(tmp_path: Path) -> Path:
    """Write an AHN LAZ of 204 points covering the ortho extent (planar Z).

    The four extent corners are included so the convex hull contains
    every pixel centre: hull-based methods (linear) must cover the whole
    grid now that reconcile refuses void estimates.
    """
    rng = np.random.default_rng(1)
    corners = np.array(
        [[MINX, MINY], [MAXX, MINY], [MINX, MAXY], [MAXX, MAXY]]
    )
    xy = np.vstack([rng.uniform(MINX, MAXX, (200, 2)), corners])
    z = 0.1 * xy[:, 0] + 0.2 * xy[:, 1]
    header = laspy.LasHeader(point_format=2)
    header.offsets = [MINX, MINY, 0.0]
    header.scales = [0.001, 0.001, 0.001]
    las = laspy.LasData(header)
    las.x = xy[:, 0]
    las.y = xy[:, 1]
    las.z = z
    path = tmp_path / "cloud.laz"
    las.write(str(path))
    return path
