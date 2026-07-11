"""Copc-context de-duplication: one survivor per occupied 0.5 m voxel.

AHN's native coarseness is 0.5 m, so two points inside the same 0.5 m voxel
carry no extra information — they are overlap artefacts (flight-line overlap,
tile seams, interpolation duplicates). This module collapses each occupied
voxel to a **single original point**, chosen by reasoning about outliers
rather than by arrival order:

1. Per voxel, take the component-wise lower median of the member points and
   the MAD (median absolute deviation) of their Z values.
2. Points whose Z deviates from the median by more than ``3.5 * 1.4826 * MAD``
   are outliers and may not survive (with ``MAD == 0`` only points at the
   exact median Z remain candidates — the strictest, still non-empty, case).
3. The survivor is the candidate nearest (squared Euclidean, integer units) to
   the voxel's median triple, ties broken by lowest input index.

Points are never moved or synthesised: the survivor keeps its original
coordinates and (by index) its original attributes. Voxels use floor division,
so below-sea-level (negative) coordinates bucket correctly.

Determinism: pure integer/sort arithmetic; identical input yields identical
survivor indices.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

_MAD_CONSISTENCY = 1.4826  # MAD -> sigma for normal data
_ROBUST_Z_CUTOFF = 3.5  # conventional robust-z outlier threshold


def dedupe_voxels(
    quantized: npt.NDArray[np.int64],
    voxel_units: int,
) -> npt.NDArray[np.int64]:
    """Return sorted indices of the one surviving point per occupied voxel.

    Contract:
        - ``quantized`` is ``(n, 3)`` integer point coordinates in output scale
          units (e.g. millimetres for a 0.001 scale); ``voxel_units`` is the
          voxel edge in those units (500 for a 0.5 m voxel at 0.001 scale).
        - Returns sorted original indices, one per occupied voxel; singleton
          voxels keep their point untouched.

    Invariants:
        - Deterministic: identical input yields identical output.
        - Every survivor is an original input point (no synthesis).
    """
    n = quantized.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.int64)

    voxels = quantized // voxel_units  # floor division: negatives bucket down
    order = np.lexsort((voxels[:, 2], voxels[:, 1], voxels[:, 0]))
    ordered_voxels = voxels[order]
    is_start = np.ones(n, dtype=np.bool_)
    is_start[1:] = np.any(ordered_voxels[1:] != ordered_voxels[:-1], axis=1)
    group_of = np.empty(n, dtype=np.int64)
    group_of[order] = np.cumsum(is_start) - 1

    counts = np.bincount(group_of)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    median_pos = starts + (counts - 1) // 2

    # Component-wise lower medians per group (median value is a data value).
    medians = np.empty((counts.shape[0], 3), dtype=np.int64)
    for axis in range(3):
        by_value = np.lexsort((quantized[:, axis], group_of))
        medians[:, axis] = quantized[by_value, axis][median_pos]

    deviation = np.abs(quantized[:, 2] - medians[group_of, 2])
    by_deviation = np.lexsort((deviation, group_of))
    mad = deviation[by_deviation][median_pos]
    cutoff = (
        _ROBUST_Z_CUTOFF * _MAD_CONSISTENCY * mad[group_of].astype(np.float64)
    )
    is_outlier = deviation.astype(np.float64) > cutoff

    difference = quantized - medians[group_of]
    distance_sq = np.einsum("ij,ij->i", difference, difference)
    ranking = np.lexsort((np.arange(n), distance_sq, is_outlier, group_of))
    return np.sort(ranking[starts]).astype(np.int64)
