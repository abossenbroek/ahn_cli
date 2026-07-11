"""Reconcile-context source-cloud cleanup: class filter + XY de-duplication.

Raw AHN LiDAR is not a clean sample of a surface: overlapping flight lines leave
*coincident* returns at the same XY, and the cloud mixes classes (ground,
building, vegetation, water, noise). Feeding that straight into IDW/kriging
over-weights the duplicated locations and -- with a zero nugget -- makes the
kriging neighbourhood singular. :func:`select_and_dedupe` cleans the cloud once,
before the interpolator is built:

1. **Class filter** (optional): keep only ``include`` classes and/or drop
   ``exclude`` classes. Empty on both sides keeps every class.
2. **XY de-duplication** (always): collapse every set of returns sharing an XY to
   a single point at the **highest Z** -- the top surface, consistent with the
   top-down orthophoto. This removes the coincident-neighbour cause of singular
   kriging systems at the source rather than papering over it downstream.

Determinism: the result is produced by a stable lexicographic sort, so identical
input yields byte-identical output.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def select_and_dedupe(
    coords: npt.NDArray[np.float64],
    classification: npt.NDArray[np.uint8],
    include: tuple[int, ...],
    exclude: tuple[int, ...],
) -> npt.NDArray[np.float64]:
    """Filter by class, then collapse coincident-XY returns to their top Z.

    Contract:
        - ``coords`` is ``(n, 3)`` world ``(x, y, z)``; ``classification`` is the
          matching ``(n,)`` per-point class.
        - A point is kept when its class is in ``include`` (or ``include`` is
          empty) and not in ``exclude``; the survivors are then de-duplicated so
          each distinct XY appears once, at its maximum Z.
        - Returns the cleaned ``(m, 3)`` coordinates.

    Invariants:
        - Deterministic: identical input yields byte-identical output.
    """
    keep = np.ones(coords.shape[0], dtype=np.bool_)
    if include:
        keep &= np.isin(classification, np.asarray(include))
    if exclude:
        keep &= ~np.isin(classification, np.asarray(exclude))
    return _dedupe_xy_max_z(coords[keep])


def _dedupe_xy_max_z(
    coords: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Collapse coincident-XY points to one point at the highest Z (stable)."""
    if coords.shape[0] == 0:
        return coords
    # Sort by (x, y, z): coincident-XY points become adjacent with z ascending,
    # so the last row of each XY group carries that group's maximum Z.
    order = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0]))
    ordered = coords[order]
    xy = ordered[:, :2]
    is_group_end = np.ones(ordered.shape[0], dtype=np.bool_)
    is_group_end[:-1] = np.any(xy[1:] != xy[:-1], axis=1)
    return ordered[is_group_end]
