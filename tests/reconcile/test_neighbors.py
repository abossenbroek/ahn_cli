"""Tests for the deterministic k-nearest-neighbour primitive."""

from __future__ import annotations

import numpy as np

from ahn_cli.reconcile.neighbors import knn


def test_shapes_and_clamps_k_to_source_count() -> None:
    """Clamp k to the source count; output shapes are (Q, k')."""
    source = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    target = np.array([[0.0, 0.0], [2.0, 0.0]])
    sq, idx = knn(target, source, k=10)
    assert sq.shape == (2, 3)
    assert idx.shape == (2, 3)


def test_returns_nearest_ascending_by_distance() -> None:
    """Neighbours come back ordered by ascending squared distance."""
    source = np.array([[0.0, 0.0], [3.0, 0.0], [1.0, 0.0]])
    target = np.array([[0.0, 0.0]])
    sq, idx = knn(target, source, k=3)
    assert idx[0].tolist() == [0, 2, 1]
    assert sq[0].tolist() == [0.0, 1.0, 9.0]


def test_squared_distance_values() -> None:
    """The reported distances are the true squared Euclidean distances."""
    source = np.array([[0.0, 0.0], [3.0, 4.0]])
    target = np.array([[0.0, 0.0]])
    sq, _ = knn(target, source, k=2)
    assert sq[0].tolist() == [0.0, 25.0]


def test_equal_distance_tie_broken_by_source_index() -> None:
    """Neighbours at an equal distance are ordered by ascending source index."""
    # Points 1 and 2 are both at distance 1 from the target; index breaks the tie.
    source = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    target = np.array([[0.0, 0.0]])
    _, idx = knn(target, source, k=4)
    assert idx[0].tolist() == [0, 1, 2, 3]


def test_empty_source_returns_zero_width() -> None:
    """No source points yields a (Q, 0) result rather than an error."""
    target = np.array([[0.0, 0.0], [1.0, 1.0]])
    sq, idx = knn(target, np.empty((0, 2)), k=5)
    assert sq.shape == (2, 0)
    assert idx.shape == (2, 0)


def test_empty_target_returns_zero_rows() -> None:
    """No targets yields a (0, k') result."""
    source = np.array([[0.0, 0.0], [1.0, 0.0]])
    sq, idx = knn(np.empty((0, 2)), source, k=2)
    assert sq.shape == (0, 2)
    assert idx.shape == (0, 2)


def test_single_neighbour_query() -> None:
    """A k of 1 returns a (Q, 1) result (scipy's 1-D shape is normalised)."""
    source = np.array([[0.0, 0.0], [5.0, 0.0]])
    target = np.array([[4.9, 0.0], [0.1, 0.0]])
    _, idx = knn(target, source, k=1)
    assert idx.tolist() == [[1], [0]]


def test_deterministic_across_calls() -> None:
    """Identical inputs yield byte-identical neighbour indices and distances."""
    rng = np.random.default_rng(0)
    source = rng.random((200, 2))
    target = rng.random((50, 2))
    sq_a, idx_a = knn(target, source, k=7)
    sq_b, idx_b = knn(target, source, k=7)
    assert np.array_equal(idx_a, idx_b)
    assert np.array_equal(sq_a, sq_b)
