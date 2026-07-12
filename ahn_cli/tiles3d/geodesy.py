"""EPSG:7415 geodesy for 3D Tiles: ECEF positions and geodetic regions.

3D Tiles places content in Earth-centred Earth-fixed coordinates
(EPSG:4978) and describes ``region`` bounding volumes in EPSG:4979
geodetic radians. The pipeline's grid is EPSG:7415 (RD New + NAP);
:class:`Geodesy` wraps the two pyproj transformers.

Determinism caveat: which PROJ pipeline pyproj selects (in particular
whether the NLGEO2018 quasi-geoid grid is available for the NAP ->
ellipsoidal height step) depends on the installed PROJ data files, so
absolute outputs are deterministic per machine, not across machines.
Self-consistency is what the strict verifier relies on: it recomputes
through this same class, so build and verify always agree bit-exact.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from pyproj import Transformer

if TYPE_CHECKING:
    import numpy.typing as npt

__all__ = ["Geodesy"]

_Triple = tuple[
    "npt.NDArray[np.float64]",
    "npt.NDArray[np.float64]",
    "npt.NDArray[np.float64]",
]


class Geodesy:
    """The EPSG:7415 -> EPSG:4978 / EPSG:4979 transformer pair.

    Contract:
        - :meth:`to_ecef` maps RD/NAP ``(x, y, z)`` arrays to ECEF
          metres (EPSG:4978).
        - :meth:`to_geodetic_radians` maps the same to EPSG:4979
          longitude/latitude in **radians** plus ellipsoidal height in
          metres — the 3D Tiles ``region`` convention.

    Invariants:
        - Both methods are pure and deterministic for a given PROJ
          installation; shapes are preserved.
    """

    def __init__(self) -> None:
        """Build the two pyproj transformers (always_xy ordering)."""
        self._ecef = Transformer.from_crs(
            "EPSG:7415", "EPSG:4978", always_xy=True
        )
        self._geodetic = Transformer.from_crs(
            "EPSG:7415", "EPSG:4979", always_xy=True
        )

    def to_ecef(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
    ) -> _Triple:
        """Transform RD/NAP coordinates to ECEF (EPSG:4978) metres."""
        return _transform(self._ecef, x, y, z)

    def to_geodetic_radians(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
    ) -> _Triple:
        """Transform RD/NAP coordinates to EPSG:4979 radians + height."""
        lon, lat, height = _transform(self._geodetic, x, y, z)
        return (np.radians(lon), np.radians(lat), height)


def _transform(
    transformer: Transformer,
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
) -> _Triple:
    """Run one pyproj transform, returning float64 arrays.

    Size-1 inputs go through pyproj's scalar fast path explicitly:
    handing it a one-element array trips numpy's deprecated
    array-to-scalar conversion inside pyproj.
    """
    if x.size == 1:
        a, b, c = cast(
            "tuple[float, float, float]",
            transformer.transform(
                float(x.ravel()[0]),
                float(y.ravel()[0]),
                float(z.ravel()[0]),
            ),
        )
        return (
            np.full(x.shape, a, dtype=np.float64),
            np.full(y.shape, b, dtype=np.float64),
            np.full(z.shape, c, dtype=np.float64),
        )
    a, b, c = cast(
        "tuple[object, object, object]",
        transformer.transform(x, y, z),
    )
    return (
        np.asarray(a, dtype=np.float64),
        np.asarray(b, dtype=np.float64),
        np.asarray(c, dtype=np.float64),
    )
