"""Deterministic k-nearest-neighbour search (the CPU reference primitive).

Both IDW and kriging reduce to "the ``k`` nearest source points of each target
cell". This module is the numpy/scipy reference for that query and the
correctness oracle the Metal kernel is validated against.

Determinism is a load-bearing guardrail: within each target's neighbour row the
result is sorted by ascending squared distance, ties broken by ascending source
index, so identical inputs yield byte-identical ``(sq_dist, idx)`` on any machine
running the same scipy. ``k`` is clamped to the source count; an empty source or
target degrades to a zero-width / zero-row result rather than raising, so the
callers treat "no neighbours" as "void cell" uniformly.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.spatial import cKDTree


def knn(
    target_xy: npt.NDArray[np.float64],
    source_xy: npt.NDArray[np.float64],
    k: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.intp]]:
    """Return the ``k`` nearest source points of each target, deterministically.

    Contract:
        - ``target_xy`` is ``(q, 2)`` and ``source_xy`` is ``(n, 2)`` world XY.
        - ``k`` is the desired neighbour count; it is clamped to ``n``.
        - Returns ``(sq_dist, idx)`` each shaped ``(q, k')`` with
          ``k' = min(k, n)``. Row ``i`` lists target ``i``'s neighbours ordered
          by ascending squared distance, ties broken by ascending source index.
          ``sq_dist`` holds the true squared Euclidean distances; ``idx`` holds
          the source-point indices.

    Invariants:
        - Deterministic: identical inputs yield byte-identical output.
        - An empty source yields ``(q, 0)`` arrays; an empty target yields
          ``(0, k')`` arrays.
    """
    q = target_xy.shape[0]
    n = source_xy.shape[0]
    k_eff = min(k, n)
    if k_eff == 0:
        return (
            np.empty((q, 0), dtype=np.float64),
            np.empty((q, 0), dtype=np.intp),
        )
    if q == 0:
        return (
            np.empty((0, k_eff), dtype=np.float64),
            np.empty((0, k_eff), dtype=np.intp),
        )

    tree = cKDTree(source_xy)
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
