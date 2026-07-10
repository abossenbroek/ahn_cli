"""Deterministic k-nearest-neighbour search (the CPU reference primitive).

Both IDW and kriging reduce to "the ``k`` nearest source points of each target
cell". This module wraps scipy's indexed ``cKDTree`` -- a fast, C-backed search
-- as that primitive, split so the tree is built **once** and queried per
row-block (:func:`build_tree` + :func:`query_knn`), which is what lets the
reconcile pipeline stream an arbitrarily large area with flat memory.
:func:`knn` is the build-and-query convenience for callers that have all targets
at once.

Determinism is a load-bearing guardrail: within each target's neighbour row the
result is sorted by ascending squared distance, ties broken by ascending source
index, so identical inputs yield byte-identical ``(sq_dist, idx)`` on any machine
running the same scipy. ``k`` is clamped to the source count; an empty source or
target degrades to a zero-width / zero-row result rather than raising, so callers
treat "no neighbours" as "void cell" uniformly.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.spatial import cKDTree


def build_tree(source_xy: npt.NDArray[np.float64]) -> cKDTree | None:
    """Build a ``cKDTree`` over the source XY, or ``None`` for an empty source.

    Contract:
        - ``source_xy`` is ``(n, 2)`` world XY.
        - Returns the tree, or ``None`` when ``n == 0`` (so callers treat every
          query as a void cell without a special case).
    """
    if source_xy.shape[0] == 0:
        return None
    return cKDTree(source_xy)


def query_knn(
    tree: cKDTree | None,
    target_xy: npt.NDArray[np.float64],
    source_count: int,
    k: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.intp]]:
    """Query a prebuilt tree for the ``k`` nearest source points of each target.

    Contract:
        - ``tree`` is the :func:`build_tree` result (``None`` for empty source);
          ``source_count`` is the source point count; ``target_xy`` is ``(q, 2)``.
        - ``k`` is clamped to ``source_count``. Returns ``(sq_dist, idx)`` each
          ``(q, k')`` with ``k' = min(k, source_count)``: row ``i`` lists target
          ``i``'s neighbours by ascending squared distance, source-index
          tie-break; ``sq_dist`` holds the true squared distances.

    Invariants:
        - Deterministic: identical inputs yield byte-identical output.
        - Empty source yields ``(q, 0)``; empty target yields ``(0, k')``.
    """
    q = target_xy.shape[0]
    k_eff = min(k, source_count)
    if tree is None or k_eff == 0:
        return (
            np.empty((q, 0), dtype=np.float64),
            np.empty((q, 0), dtype=np.intp),
        )
    if q == 0:
        return (
            np.empty((0, k_eff), dtype=np.float64),
            np.empty((0, k_eff), dtype=np.intp),
        )

    dist, raw_idx = tree.query(target_xy, k=k_eff)
    # scipy returns 1-D arrays when k_eff == 1; normalise to (q, k_eff).
    dist = dist.reshape(q, k_eff).astype(np.float64)
    idx = raw_idx.reshape(q, k_eff).astype(np.intp)
    sq = dist * dist

    # Stable per-row ordering: primary key squared distance, secondary key the
    # source index, so equidistant neighbours are returned in index order.
    order = np.lexsort((idx, sq), axis=1)
    sq_sorted = np.take_along_axis(sq, order, axis=1)
    idx_sorted = np.take_along_axis(idx, order, axis=1)
    return sq_sorted, idx_sorted


def knn(
    target_xy: npt.NDArray[np.float64],
    source_xy: npt.NDArray[np.float64],
    k: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.intp]]:
    """Build a tree over ``source_xy`` and return each target's ``k`` neighbours.

    Convenience wrapper over :func:`build_tree` + :func:`query_knn` for callers
    that hold every target at once; see :func:`query_knn` for the contract.
    """
    tree = build_tree(source_xy)
    return query_knn(tree, target_xy, source_xy.shape[0], k)
