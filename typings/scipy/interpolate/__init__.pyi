"""Stub for ``scipy.interpolate.LinearNDInterpolator`` -- Delaunay-linear interp."""

import numpy as np
import numpy.typing as npt

class LinearNDInterpolator:
    """Piecewise-linear interpolant on the Delaunay triangulation of points."""

    def __init__(
        self,
        points: npt.NDArray[np.float64],
        values: npt.NDArray[np.float64],
        fill_value: float = ...,
    ) -> None: ...
    def __call__(
        self, x: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]: ...
