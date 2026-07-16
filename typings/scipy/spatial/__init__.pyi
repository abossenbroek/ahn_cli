"""Stub for ``scipy.spatial.cKDTree`` -- only the kNN query the reconcile uses."""

import numpy as np
import numpy.typing as npt

class QhullError(Exception):
    """Raised by Qhull-backed routines on a degenerate input geometry."""

class cKDTree:
    """A k-d tree over ``(n, d)`` points, queried for nearest neighbours."""

    def __init__(self, data: npt.NDArray[np.float64]) -> None: ...
    def query(
        self,
        x: npt.NDArray[np.float64],
        k: int = ...,
        workers: int = ...,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.intp]]: ...
