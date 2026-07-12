"""Tests for the copc-context 0.5 m voxel outlier-aware de-duplication."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ahn_cli.copc.dedup import dedupe_voxels

VOXEL = 500  # 0.5 m in 0.001-scale integer units


def _quantized(rows: list[tuple[int, int, int]]) -> npt.NDArray[np.int64]:
    return np.asarray(rows, dtype=np.int64)


def test_empty_input_yields_empty_indices() -> None:
    """An empty cloud yields an empty survivor index array."""
    out = dedupe_voxels(np.empty((0, 3), dtype=np.int64), VOXEL)
    assert out.shape == (0,)
    assert out.dtype == np.int64


def test_singletons_all_survive() -> None:
    """Points alone in their voxel are kept untouched (native coarseness)."""
    quantized = _quantized([(0, 0, 0), (600, 0, 0), (0, 600, 0), (0, 0, 600)])
    out = dedupe_voxels(quantized, VOXEL)
    assert np.array_equal(out, np.arange(4))


def test_voxel_boundary_is_exclusive() -> None:
    """Units 499 and 500 land in different 0.5 m voxels (floor semantics)."""
    quantized = _quantized([(499, 0, 0), (500, 0, 0)])
    out = dedupe_voxels(quantized, VOXEL)
    assert np.array_equal(out, np.arange(2))


def test_negative_coordinates_use_floor_voxels() -> None:
    """Below-sea-level (negative) units voxelize by floor, not truncation."""
    # -1 and -500 share voxel -1; -501 belongs to voxel -2.
    quantized = _quantized([(0, 0, -1), (0, 0, -500), (0, 0, -501)])
    out = dedupe_voxels(quantized, VOXEL)
    # One survivor from the {-1, -500} voxel plus the lone -501 point.
    assert out.shape == (2,)
    assert 2 in out.tolist()


def test_duplicate_pair_keeps_single_original_point() -> None:
    """Two points in one voxel collapse to exactly one original index."""
    quantized = _quantized([(10, 10, 10), (20, 20, 20), (900, 0, 0)])
    out = dedupe_voxels(quantized, VOXEL)
    assert out.shape == (2,)
    assert out.tolist()[1] == 2  # the singleton always survives
    assert out.tolist()[0] in (0, 1)  # survivor is an original point


def test_survivor_is_nearest_to_voxel_median() -> None:
    """The survivor is the point closest to the voxel's component medians."""
    # Lower medians of x/y/z over the group: (10, 10, 10) -> index 1 wins.
    quantized = _quantized([(2, 2, 2), (10, 10, 10), (11, 11, 11)])
    out = dedupe_voxels(quantized, VOXEL)
    assert np.array_equal(out, np.asarray([1]))


def test_z_outlier_cannot_survive() -> None:
    """A MAD-flagged Z outlier is excluded from survivorship."""
    # Tight cluster near z=10 with one spike at z=490 (same voxel): the spike
    # is the group's x/y-median column, but its z deviation disqualifies it.
    quantized = _quantized([(1, 1, 10), (2, 2, 12), (4, 4, 11), (3, 3, 490)])
    out = dedupe_voxels(quantized, VOXEL)
    assert out.shape == (1,)
    assert out.tolist()[0] != 3


def test_zero_mad_keeps_only_exact_median_z() -> None:
    """With MAD == 0 only points at the median Z remain candidates."""
    quantized = _quantized([(0, 0, 5), (1, 0, 5), (2, 0, 5), (3, 0, 100)])
    out = dedupe_voxels(quantized, VOXEL)
    assert out.shape == (1,)
    assert out.tolist()[0] != 3


def test_identical_points_tie_break_on_lowest_index() -> None:
    """Bit-identical duplicates keep the first occurrence (stable)."""
    quantized = _quantized([(7, 7, 7), (7, 7, 7), (7, 7, 7)])
    out = dedupe_voxels(quantized, VOXEL)
    assert np.array_equal(out, np.asarray([0]))


def test_survivor_indices_are_sorted_and_deterministic() -> None:
    """Survivor indices come back sorted; identical input, identical output."""
    rng = np.random.default_rng(42)
    quantized = rng.integers(-2000, 2000, (500, 3)).astype(np.int64)
    first = dedupe_voxels(quantized, VOXEL)
    second = dedupe_voxels(quantized, VOXEL)
    assert np.array_equal(first, second)
    assert np.array_equal(first, np.sort(first))


def test_multiple_voxels_dedupe_independently() -> None:
    """Each occupied voxel contributes exactly one survivor."""
    quantized = _quantized(
        [
            (0, 0, 0),
            (100, 100, 100),  # voxel (0,0,0) duplicate
            (700, 0, 0),
            (800, 0, 0),  # voxel (1,0,0) duplicate
            (0, 700, 0),  # singleton
        ]
    )
    out = dedupe_voxels(quantized, VOXEL)
    assert out.shape == (3,)
