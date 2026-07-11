"""Unit tests for the pure data-authenticity predicates."""

from __future__ import annotations

import numpy as np

from ahn_cli.domain.authenticity import (
    degenerate_cloud,
    flat_surface,
    uniform_image,
)

# --------------------------------------------------------------------------- #
# uniform_image
# --------------------------------------------------------------------------- #


def test_uniform_image_flags_a_single_colour_grid() -> None:
    """Every pixel one identical colour is the placeholder signature."""
    assert uniform_image(np.full((3, 8, 8), 128, dtype=np.uint8))


def test_uniform_image_flags_an_empty_sample() -> None:
    """No pixels at all is uniform: there is no imagery to trust."""
    assert uniform_image(np.empty((3, 0, 0), dtype=np.uint8))


def test_uniform_image_accepts_varying_imagery() -> None:
    """Any per-pixel variation clears the gate."""
    sample = np.full((3, 8, 8), 128, dtype=np.uint8)
    sample[1, 4, 4] = 129
    assert not uniform_image(sample)


def test_uniform_image_accepts_a_saturated_band_beside_variation() -> None:
    """One constant band next to a varying one is real photography."""
    rng = np.random.default_rng(7)
    sample = np.stack(
        [
            np.full((8, 8), 255, dtype=np.uint8),
            rng.integers(0, 256, size=(8, 8), dtype=np.uint8),
            np.full((8, 8), 0, dtype=np.uint8),
        ]
    )
    assert not uniform_image(sample)


def test_uniform_image_handles_a_single_band_plane() -> None:
    """A 2-D single-band raster is judged the same way."""
    assert uniform_image(np.full((8, 8), 5.0, dtype=np.float32))
    varied = np.arange(64, dtype=np.float32).reshape(8, 8)
    assert not uniform_image(varied)


# --------------------------------------------------------------------------- #
# flat_surface
# --------------------------------------------------------------------------- #


def test_flat_surface_flags_a_constant_raster() -> None:
    """Two or more valid samples all one value is not genuine terrain."""
    assert flat_surface(np.full((4, 4), 10.0, dtype=np.float32), None)


def test_flat_surface_flags_an_all_nodata_raster() -> None:
    """A raster with no valid samples at all carries no terrain."""
    assert flat_surface(np.full((4, 4), -9999.0, dtype=np.float32), -9999.0)


def test_flat_surface_flags_an_all_nan_raster() -> None:
    """Non-finite samples are never valid, even without a declared nodata."""
    assert flat_surface(np.full((4, 4), np.nan, dtype=np.float32), None)


def test_flat_surface_accepts_variation_among_valid_samples() -> None:
    """Any relief among the valid samples clears the gate."""
    values = np.full((4, 4), 10.0, dtype=np.float32)
    values[2, 2] = 12.5
    assert not flat_surface(values, None)


def test_flat_surface_ignores_nodata_when_judging_variation() -> None:
    """Voids do not count as relief: constant-valid plus voids is flat."""
    values = np.full((4, 4), 10.0, dtype=np.float32)
    values[0, 0] = -9999.0
    assert flat_surface(values, -9999.0)


def test_flat_surface_accepts_a_single_valid_sample() -> None:
    """One measurement carries no variation to judge; it is not flat."""
    assert not flat_surface(np.array([[42.0]], dtype=np.float32), None)


# --------------------------------------------------------------------------- #
# degenerate_cloud
# --------------------------------------------------------------------------- #


def test_degenerate_cloud_flags_an_empty_cloud() -> None:
    """Zero points is degenerate regardless of the recorded extremes."""
    assert degenerate_cloud(0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))


def test_degenerate_cloud_flags_stacked_duplicates() -> None:
    """Many points all at one identical XYZ is fabricated data."""
    assert degenerate_cloud(5, (1.0, 2.0, 3.0), (1.0, 2.0, 3.0))


def test_degenerate_cloud_accepts_a_single_point() -> None:
    """One point has no duplication signature and passes."""
    assert not degenerate_cloud(1, (1.0, 2.0, 3.0), (1.0, 2.0, 3.0))


def test_degenerate_cloud_accepts_any_spatial_extent() -> None:
    """Extent on any axis clears the gate."""
    assert not degenerate_cloud(5, (1.0, 2.0, 3.0), (1.0, 2.0, 4.0))
