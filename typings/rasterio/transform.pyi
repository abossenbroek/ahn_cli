"""Minimal type stub for ``rasterio.transform`` used by the test fixtures."""

from rasterio import Affine

def from_bounds(
    west: float,
    south: float,
    east: float,
    north: float,
    width: int,
    height: int,
) -> Affine: ...
