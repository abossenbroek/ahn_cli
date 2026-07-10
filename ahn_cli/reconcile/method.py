"""Reconcile-context interpolation-method value objects (typed dispatch).

The ``reconcile`` verb interpolates the AHN point cloud's elevation onto the
orthophoto's grid. *Which* interpolation to use is modelled as a small set of
frozen value objects rather than a stringly-typed switch:

* :class:`LinearInterp` -- Delaunay-barycentric linear interpolation. A true
  interpolant (passes through the data); cells outside the convex hull are void.
* :class:`IdwInterp` -- inverse-distance weighting over the ``k`` nearest points.
* :class:`KrigingInterp` -- ordinary kriging over the ``k`` nearest points using
  a *fixed* :class:`Variogram` (never auto-fitted, so runs reproduce).

:data:`InterpMethod` unions the three; downstream code dispatches on the variant
(``isinstance``), never on a method string.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import numpy.typing as npt

DEFAULT_IDW_POWER = 2.0
"""Default IDW distance exponent (inverse-square weighting)."""

DEFAULT_IDW_K = 12
"""Default neighbour count for IDW."""

DEFAULT_KRIGING_K = 16
"""Default neighbour count for ordinary kriging."""


class VariogramModel(Enum):
    """The available theoretical variogram models (the ``--kriging-model`` tokens)."""

    SPHERICAL = "spherical"
    EXPONENTIAL = "exponential"
    GAUSSIAN = "gaussian"


@dataclass(frozen=True)
class Variogram:
    """A fixed theoretical variogram: semivariance as a function of lag.

    Contract:
        - ``model`` selects the shape (:class:`VariogramModel`).
        - ``nugget`` is the semivariance at an infinitesimal lag; ``>= 0``.
        - ``sill`` is the total sill the model approaches; must be
          ``>= nugget`` (so the partial sill ``sill - nugget`` is non-negative).
        - ``vrange`` is the range in metres; must be finite and strictly
          positive.

    Invariants:
        - Frozen value object, equal by field value.
        - ``semivariance(0) == 0`` for every model (the value at exactly zero
          lag; the nugget appears only for strictly positive lags).

    Failure modes:
        - :class:`ValueError` on a negative nugget, a sill below the nugget, or
          a range that is not finite and positive.
    """

    model: VariogramModel
    nugget: float
    sill: float
    vrange: float

    def __post_init__(self) -> None:
        """Reject a negative nugget, sill below nugget, or bad range."""
        if self.nugget < 0.0:
            msg = f"variogram nugget must be >= 0; got {self.nugget}."
            raise ValueError(msg)
        if self.sill < self.nugget:
            msg = (
                f"variogram sill must be >= nugget ({self.nugget}); "
                f"got {self.sill}."
            )
            raise ValueError(msg)
        if not np.isfinite(self.vrange) or self.vrange <= 0.0:
            msg = f"variogram range must be finite and positive; got {self.vrange}."
            raise ValueError(msg)

    def semivariance(
        self, lag: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Return the semivariance ``gamma(h)`` for each lag ``h`` in ``lag``.

        Contract:
            - ``lag`` is an array of non-negative distances in metres.
            - Returns an array of the same shape. ``gamma(0) == 0``; positive
              lags carry the nugget plus the partial-sill model term, clamped
              to the total ``sill``.
        """
        partial = self.sill - self.nugget
        ratio = lag / self.vrange
        if self.model is VariogramModel.SPHERICAL:
            shape = np.where(
                ratio < 1.0,
                1.5 * ratio - 0.5 * ratio**3,
                1.0,
            )
        elif self.model is VariogramModel.EXPONENTIAL:
            shape = 1.0 - np.exp(-ratio)
        else:
            shape = 1.0 - np.exp(-(ratio**2))
        gamma = self.nugget + partial * shape
        return np.where(lag == 0.0, 0.0, gamma)


@dataclass(frozen=True)
class LinearInterp:
    """A request for Delaunay-barycentric linear interpolation.

    Invariants:
        - Frozen value object with no parameters; all instances compare equal.
    """


@dataclass(frozen=True)
class IdwInterp:
    """A validated request for inverse-distance-weighted interpolation.

    Contract:
        - ``power`` is the distance exponent; must be finite and strictly
          positive (higher concentrates weight on the nearest points).
        - ``k`` is the neighbour count; must be a positive integer.

    Invariants:
        - Frozen value object, equal by field value.

    Failure modes:
        - :class:`ValueError` on a non-finite/non-positive power or ``k < 1``.
    """

    power: float = DEFAULT_IDW_POWER
    k: int = DEFAULT_IDW_K

    def __post_init__(self) -> None:
        """Reject a non-finite/non-positive power or a neighbour count below one."""
        if not np.isfinite(self.power) or self.power <= 0.0:
            msg = f"idw power must be finite and positive; got {self.power}."
            raise ValueError(msg)
        if self.k < 1:
            msg = f"idw k (neighbour count) must be >= 1; got {self.k}."
            raise ValueError(msg)


@dataclass(frozen=True)
class KrigingInterp:
    """A validated request for ordinary kriging with a fixed variogram.

    Contract:
        - ``variogram`` is the fixed :class:`Variogram` model (never fitted at
          run time, so results reproduce).
        - ``k`` is the neighbour count; must be a positive integer.

    Invariants:
        - Frozen value object, equal by field value.

    Failure modes:
        - :class:`ValueError` on ``k < 1``.
    """

    variogram: Variogram
    k: int = DEFAULT_KRIGING_K

    def __post_init__(self) -> None:
        """Reject a neighbour count below one."""
        if self.k < 1:
            msg = f"kriging k (neighbour count) must be >= 1; got {self.k}."
            raise ValueError(msg)


InterpMethod = LinearInterp | IdwInterp | KrigingInterp
"""A validated interpolation request: linear, IDW, or ordinary kriging."""
