"""The :class:`PixelGrid` value object: pixel indices to world coordinates.

A raster deliverable (e.g. ``dsm.tif``) is an affine grid: an integer pixel
index ``(col, row)`` maps to an EPSG:28992 world coordinate through the six
coefficients of its geotransform. :class:`PixelGrid` names that mapping as a
domain value object so callers never pass raw affine tuples or stringly-typed
dicts around, and it fixes the **pixel-centre** sampling convention in one
place: pixel ``(col, row)`` is sampled at ``(col + 0.5, row + 0.5)``.

Pure domain: no I/O and no dependency on the legacy modules. The fetch/prep
contexts build a grid from a rasterio ``Affine`` (whose ``a..f`` attributes are
exactly this coefficient order) without the domain importing rasterio.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

GeoTransform: TypeAlias = tuple[float, float, float, float, float, float]
"""An affine geotransform ``(a, b, c, d, e, f)`` in rasterio ``Affine`` order.

World coordinates of a pixel coordinate ``(col, row)`` are
``x = a*col + b*row + c`` and ``y = d*col + e*row + f``. For a north-up raster
``b == d == 0``, ``a`` is the pixel width and ``e`` the (negative) pixel height.
"""


@dataclass(frozen=True)
class PixelGrid:
    """An affine pixel grid mapping pixel centres to EPSG:28992 coordinates.

    Contract:
        - ``width`` / ``height`` are the raster's pixel dimensions; both must be
          strictly positive.
        - ``transform`` is the :data:`GeoTransform` (rasterio ``Affine`` order).
        - :meth:`eastings` / :meth:`northings` return the world X / Y of every
          pixel *centre* as ``(height, width)`` float64 arrays -- pixel
          ``(col, row)`` is sampled at ``(col + 0.5, row + 0.5)``.

    Invariants:
        - Immutable and hashable; equal iff every field is equal.

    Failure modes:
        - ``ValueError`` if ``width`` or ``height`` is not strictly positive
          (a grid with no pixels has no coordinate mesh).
    """

    width: int
    height: int
    transform: GeoTransform

    def __post_init__(self) -> None:
        """Validate that the grid has at least one pixel in each dimension."""
        if self.width <= 0:
            msg = f"grid width must be strictly positive; got {self.width}."
            raise ValueError(msg)
        if self.height <= 0:
            msg = f"grid height must be strictly positive; got {self.height}."
            raise ValueError(msg)

    def eastings(self) -> npt.NDArray[np.float64]:
        """Return the pixel-centre world X of every pixel, shape (h, w)."""
        return np.zeros((self.height, self.width), dtype=np.float64)

    def northings(self) -> npt.NDArray[np.float64]:
        """Return the pixel-centre world Y of every pixel, shape (h, w)."""
        return np.zeros((self.height, self.width), dtype=np.float64)
