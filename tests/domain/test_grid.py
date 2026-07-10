"""Tests for the :class:`~ahn_cli.domain.grid.PixelGrid` value object.

The grid maps integer pixel indices of a north-up (or skewed) affine raster to
EPSG:28992 world coordinates at the *pixel centre*. These tests pin the
pixel-centre convention with hand-computed expected values, the ``(height,
width)`` array shape for square and non-square grids, and the degenerate-extent
guards.
"""

from __future__ import annotations

import numpy as np
import pytest

from ahn_cli.domain.grid import GeoTransform, PixelGrid

# A north-up transform: 0.5 m pixels, origin (top-left corner) at
# (194000, 443020). Coefficients are (a, b, c, d, e, f) in rasterio order:
# x = a*col + b*row + c ; y = d*col + e*row + f.
_NORTH_UP: GeoTransform = (0.5, 0.0, 194000.0, 0.0, -0.5, 443020.0)


def _close(actual: float, expected: float) -> bool:
    """Return whether two floats agree within a tight tolerance."""
    return abs(actual - expected) <= 1e-6


# --------------------------------------------------------------------------
# Value object
# --------------------------------------------------------------------------


def test_pixel_grid_is_a_frozen_value_object() -> None:
    """PixelGrid is hashable and equal by field value."""
    grid = PixelGrid(width=2, height=3, transform=_NORTH_UP)

    assert grid == PixelGrid(width=2, height=3, transform=_NORTH_UP)
    assert len({grid, PixelGrid(width=2, height=3, transform=_NORTH_UP)}) == 1


# --------------------------------------------------------------------------
# Pixel-centre world coordinates
# --------------------------------------------------------------------------


def test_eastings_are_pixel_centre_x() -> None:
    """Each easting is the world X of its pixel centre (col + 0.5)."""
    grid = PixelGrid(width=2, height=3, transform=_NORTH_UP)

    eastings = grid.eastings()

    # col 0 centre -> 194000 + 0.5*(0+0.5) = 194000.25; col 1 -> 194000.75.
    assert eastings.shape == (3, 2)
    assert _close(float(eastings[0, 0]), 194000.25)
    assert _close(float(eastings[0, 1]), 194000.75)
    # Eastings depend only on the column for a north-up grid: rows agree.
    assert np.array_equal(eastings[0], eastings[2])


def test_northings_are_pixel_centre_y() -> None:
    """Each northing is the world Y of its pixel centre (row + 0.5)."""
    grid = PixelGrid(width=2, height=3, transform=_NORTH_UP)

    northings = grid.northings()

    # row 0 centre -> 443020 - 0.5*(0+0.5) = 443019.75; row 2 -> 443018.75.
    assert northings.shape == (3, 2)
    assert _close(float(northings[0, 0]), 443019.75)
    assert _close(float(northings[2, 0]), 443018.75)
    # Northings depend only on the row for a north-up grid: columns agree.
    assert np.array_equal(northings[:, 0], northings[:, 1])


def test_skewed_transform_uses_both_axes() -> None:
    """A rotated/skewed transform mixes row and column into both coordinates."""
    skew: GeoTransform = (0.5, 0.1, 194000.0, 0.2, -0.5, 443020.0)
    grid = PixelGrid(width=1, height=1, transform=skew)

    # Centre of the sole pixel is (0.5, 0.5):
    # x = 0.5*0.5 + 0.1*0.5 + 194000 = 194000.30
    # y = 0.2*0.5 - 0.5*0.5 + 443020 = 443019.85
    assert _close(float(grid.eastings()[0, 0]), 194000.30)
    assert _close(float(grid.northings()[0, 0]), 443019.85)


def test_non_square_grid_keeps_height_by_width_shape() -> None:
    """A non-square grid yields ``(height, width)`` planes, not transposed."""
    grid = PixelGrid(width=5, height=2, transform=_NORTH_UP)

    assert grid.eastings().shape == (2, 5)
    assert grid.northings().shape == (2, 5)


# --------------------------------------------------------------------------
# Degenerate-extent guards
# --------------------------------------------------------------------------


def test_rejects_non_positive_width() -> None:
    """A width of zero (or less) has no pixels and is refused."""
    with pytest.raises(ValueError, match="width"):
        PixelGrid(width=0, height=3, transform=_NORTH_UP)


def test_rejects_non_positive_height() -> None:
    """A height of zero (or less) has no pixels and is refused."""
    with pytest.raises(ValueError, match="height"):
        PixelGrid(width=2, height=-1, transform=_NORTH_UP)
