"""Tests for the reconcile source-cloud cleanup (class filter + XY dedup)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ahn_cli.reconcile.clean import select_and_dedupe


def _coords(
    rows: list[tuple[float, float, float]],
) -> npt.NDArray[np.float64]:
    return np.asarray(rows, dtype=np.float64)


def _cls(values: list[int]) -> npt.NDArray[np.uint8]:
    return np.asarray(values, dtype=np.uint8)


def test_coincident_xy_collapses_to_max_z() -> None:
    """Returns sharing an XY collapse to a single point at the highest Z."""
    coords = _coords([(0.0, 0.0, 1.0), (0.0, 0.0, 5.0), (1.0, 1.0, 2.0)])
    out = select_and_dedupe(coords, _cls([1, 1, 1]), (), ())
    # Output is sorted by (x, y); the (0,0) group keeps its max Z (5).
    assert np.array_equal(out, _coords([(0.0, 0.0, 5.0), (1.0, 1.0, 2.0)]))


def test_distinct_xy_all_kept() -> None:
    """With no coincident XY every point survives (content preserved)."""
    coords = _coords([(0.0, 0.0, 1.0), (1.0, 0.0, 2.0), (0.0, 1.0, 3.0)])
    out = select_and_dedupe(coords, _cls([1, 1, 1]), (), ())
    # All three survive, returned sorted by (x, y).
    assert np.array_equal(
        out, _coords([(0.0, 0.0, 1.0), (0.0, 1.0, 3.0), (1.0, 0.0, 2.0)])
    )


def test_deterministic() -> None:
    """Identical input yields byte-identical output across calls."""
    rng = np.random.default_rng(0)
    coords = rng.integers(0, 5, (300, 3)).astype(np.float64)  # forces dupes
    cls = _cls([2] * 300)
    first = select_and_dedupe(coords, cls, (), ())
    second = select_and_dedupe(coords, cls, (), ())
    assert np.array_equal(first, second)


def test_empty_input() -> None:
    """An empty cloud yields an empty result."""
    out = select_and_dedupe(
        np.empty((0, 3)), np.empty(0, dtype=np.uint8), (), ()
    )
    assert out.shape == (0, 3)


def test_include_filter_keeps_only_listed_classes() -> None:
    """An include filter keeps only points whose class is listed."""
    coords = _coords([(0.0, 0.0, 1.0), (1.0, 0.0, 2.0), (2.0, 0.0, 3.0)])
    out = select_and_dedupe(coords, _cls([2, 6, 1]), (2, 6), ())
    assert np.array_equal(out, _coords([(0.0, 0.0, 1.0), (1.0, 0.0, 2.0)]))


def test_exclude_filter_drops_listed_classes() -> None:
    """An exclude filter drops points whose class is listed (keeps the rest)."""
    coords = _coords([(0.0, 0.0, 1.0), (1.0, 0.0, 2.0), (2.0, 0.0, 9.0)])
    out = select_and_dedupe(coords, _cls([2, 6, 7]), (), (7,))
    assert np.array_equal(out, _coords([(0.0, 0.0, 1.0), (1.0, 0.0, 2.0)]))


def test_filter_then_dedupe() -> None:
    """The class filter runs before dedup: filtered-out points can't win a tie."""
    # Two points at (0,0): a noise (class 7) return at z=9 and a ground at z=1.
    coords = _coords([(0.0, 0.0, 9.0), (0.0, 0.0, 1.0)])
    out = select_and_dedupe(coords, _cls([7, 2]), (), (7,))
    # Noise dropped first, so the kept XY point is the ground z=1 (not the z=9).
    assert out.shape == (1, 3)
    assert tuple(out[0].tolist()) == (0.0, 0.0, 1.0)


def test_everything_filtered_out() -> None:
    """A filter that removes all points yields an empty result."""
    coords = _coords([(0.0, 0.0, 1.0), (1.0, 0.0, 2.0)])
    out = select_and_dedupe(coords, _cls([2, 2]), (6,), ())
    assert out.shape == (0, 3)
