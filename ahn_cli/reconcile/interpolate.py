"""The reconcile interpolation methods (typed dispatch on the request variant).

Each method estimates an elevation ``Z`` at target XY from the AHN source points
``(x, y, z)``, returning ``(z, valid)`` where ``valid`` marks the cells an
estimate could be produced for:

* linear -- Delaunay-barycentric linear interpolation via scipy; cells outside
  the convex hull (or a degenerate point set) are void.
* IDW -- inverse-distance weighting over the ``k`` nearest points.
* kriging -- ordinary kriging over the ``k`` nearest points; a singular per-cell
  system falls back deterministically to the IDW estimate.

The neighbour structure (a scipy Delaunay for linear, a ``cKDTree`` for IDW and
kriging) is built **once** by :func:`build_interpolator`; the returned
:class:`Interpolator` then answers :meth:`Interpolator.estimate` for any batch of
targets. This is what lets the reconcile pipeline stream an arbitrarily large
grid in row-blocks with flat memory -- a blocked run is byte-identical to a
whole-grid one because every estimate is per-target. :func:`interpolate` is the
build-and-estimate convenience for callers that hold all targets at once.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np
import numpy.typing as npt
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import QhullError

from ahn_cli.reconcile.method import (
    IdwInterp,
    KrigingInterp,
    LinearInterp,
)
from ahn_cli.reconcile.neighbors import build_tree, query_knn

if TYPE_CHECKING:
    from ahn_cli.reconcile.method import InterpMethod

_MIN_TRIANGULATION_POINTS = 3
"""Delaunay linear interpolation needs at least a triangle."""

_IDW_FALLBACK_POWER = 2.0
"""Distance exponent for the IDW estimate kriging falls back to when singular."""

_KRIGING_RCOND = 1e-10
"""Reciprocal-condition floor below which a kriging system is treated singular.

Real LiDAR carries coincident/near-coincident returns, so with a zero nugget a
neighbourhood's kriging matrix can be singular *to working precision* even when
its determinant rounds to a tiny non-zero value. A relative smallest-singular-
value test reliably catches those, where a determinant-sign test does not."""


class Interpolator(Protocol):
    """A prebuilt interpolator: estimate ``Z`` for any batch of target XY."""

    def estimate(
        self, target_xy: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
        """Return ``(z, valid)`` for ``target_xy`` (``NaN``/``False`` where void)."""
        ...


def build_interpolator(
    method: InterpMethod, source_xyz: npt.NDArray[np.float64]
) -> Interpolator:
    """Build the interpolator for ``method`` over ``source_xyz`` (structure once).

    Typed dispatch on the request variant -- no stringly-typed method switch.
    """
    if isinstance(method, LinearInterp):
        return _LinearInterpolator(source_xyz)
    if isinstance(method, IdwInterp):
        return _IdwInterpolator(source_xyz, method)
    return _KrigingInterpolator(source_xyz, method)


def interpolate(
    method: InterpMethod,
    source_xyz: npt.NDArray[np.float64],
    target_xy: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Estimate ``Z`` at each target XY (build-and-estimate convenience).

    Contract:
        - ``source_xyz`` is ``(n, 3)`` world ``(x, y, z)``; ``target_xy`` is
          ``(q, 2)`` world XY.
        - Returns ``(z, valid)`` each length ``q``: ``z`` holds the estimate
          (``NaN`` where void) and ``valid`` is ``True`` where produced.
    """
    return build_interpolator(method, source_xyz).estimate(target_xy)


def _void(
    q: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Return an all-void ``(z, valid)`` result for ``q`` targets."""
    return np.full((q,), np.nan, dtype=np.float64), np.zeros(
        (q,), dtype=np.bool_
    )


class _LinearInterpolator:
    """Delaunay-barycentric linear interpolation; outside-hull cells are void."""

    def __init__(self, source_xyz: npt.NDArray[np.float64]) -> None:
        """Build the Delaunay interpolant, or none for a degenerate point set."""
        self._interp: LinearNDInterpolator | None = None
        if source_xyz.shape[0] >= _MIN_TRIANGULATION_POINTS:
            try:
                self._interp = LinearNDInterpolator(
                    source_xyz[:, :2], source_xyz[:, 2], fill_value=np.nan
                )
            except QhullError:
                self._interp = None

    def estimate(
        self, target_xy: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
        """Evaluate the interpolant; out-of-hull (or degenerate) cells are void."""
        q = target_xy.shape[0]
        if self._interp is None:
            return _void(q)
        z = np.asarray(self._interp(target_xy), dtype=np.float64).reshape(q)
        return z, ~np.isnan(z)


class _IdwInterpolator:
    """Inverse-distance weighting over the ``k`` nearest points."""

    def __init__(
        self, source_xyz: npt.NDArray[np.float64], method: IdwInterp
    ) -> None:
        """Build the ``cKDTree`` once and hold the source Z and parameters."""
        self._z = source_xyz[:, 2]
        self._count = source_xyz.shape[0]
        self._tree = build_tree(source_xyz[:, :2])
        self._power = method.power
        self._k = method.k

    def estimate(
        self, target_xy: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
        """Return the inverse-distance-weighted estimate for ``target_xy``."""
        sq, idx = query_knn(self._tree, target_xy, self._count, self._k)
        q, k_eff = sq.shape
        if k_eff == 0:
            return _void(q)
        z = _idw_weighted_mean(sq, self._z[idx], self._power)
        return z, np.ones((q,), dtype=np.bool_)


class _KrigingInterpolator:
    """Ordinary kriging over the ``k`` nearest points with a fixed variogram."""

    def __init__(
        self, source_xyz: npt.NDArray[np.float64], method: KrigingInterp
    ) -> None:
        """Build the ``cKDTree`` once and hold the source XY/Z and the request."""
        self._xy = source_xyz[:, :2]
        self._z = source_xyz[:, 2]
        self._count = source_xyz.shape[0]
        self._tree = build_tree(self._xy)
        self._method = method

    def estimate(
        self, target_xy: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
        """Return the ordinary-kriging estimate for ``target_xy``."""
        sq, idx = query_knn(
            self._tree, target_xy, self._count, self._method.k
        )
        q, k_eff = sq.shape
        if k_eff == 0:
            return _void(q)
        z_neighbours = self._z[idx]
        lhs, rhs = _kriging_system(self._xy[idx], sq, self._method)
        # Start every cell at the deterministic IDW estimate, then overwrite the
        # cells whose kriging system is well conditioned with the kriging
        # solution. An ill-conditioned system (e.g. coincident neighbours with a
        # zero nugget) keeps the IDW fallback -- detected up front so no per-cell
        # exception handling runs and no near-singular solve is ever attempted.
        z = _idw_weighted_mean(sq, z_neighbours, _IDW_FALLBACK_POWER)
        solvable = _well_conditioned(lhs)
        if solvable.any():
            weights = np.linalg.solve(
                lhs[solvable], rhs[solvable][:, :, np.newaxis]
            )[:, :, 0]
            z[solvable] = (weights[:, :k_eff] * z_neighbours[solvable]).sum(
                axis=1
            )
        return z, np.ones((q,), dtype=np.bool_)


def _idw_weighted_mean(
    sq: npt.NDArray[np.float64],
    z_neighbours: npt.NDArray[np.float64],
    power: float,
) -> npt.NDArray[np.float64]:
    """Return the inverse-distance-weighted mean per row (coincidence-safe).

    A neighbour at exactly zero distance would carry infinite weight; such rows
    instead return the mean of their exactly-coincident neighbours' values, so
    the estimate is finite and reproduces a datum sampled at its own location.
    """
    q = sq.shape[0]
    coincident = sq == 0.0
    with np.errstate(divide="ignore"):
        weights = np.where(sq > 0.0, sq ** (-power / 2.0), 0.0)
    weight_sum = weights.sum(axis=1)
    general = np.divide(
        (weights * z_neighbours).sum(axis=1),
        weight_sum,
        out=np.zeros((q,), dtype=np.float64),
        where=weight_sum > 0.0,
    )
    coincident_count = coincident.sum(axis=1)
    exact = np.divide(
        (coincident * z_neighbours).sum(axis=1),
        coincident_count,
        out=np.zeros((q,), dtype=np.float64),
        where=coincident_count > 0,
    )
    return np.where(coincident.any(axis=1), exact, general)


def _kriging_system(
    neighbour_xy: npt.NDArray[np.float64],
    sq_to_target: npt.NDArray[np.float64],
    method: KrigingInterp,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Build the augmented ordinary-kriging systems ``(lhs, rhs)`` per target.

    ``lhs`` is ``(q, k+1, k+1)``: the neighbour-pair semivariance block bordered
    by ones (the unbiasedness constraint) with a zero corner. ``rhs`` is
    ``(q, k+1)``: the neighbour-to-target semivariances with a trailing one.
    """
    q, k_eff, _ = neighbour_xy.shape
    diff = neighbour_xy[:, :, None, :] - neighbour_xy[:, None, :, :]
    pair_lag = np.sqrt((diff * diff).sum(axis=-1))
    gamma_pairs = method.variogram.semivariance(pair_lag)
    gamma_target = method.variogram.semivariance(np.sqrt(sq_to_target))

    lhs = np.zeros((q, k_eff + 1, k_eff + 1))
    lhs[:, :k_eff, :k_eff] = gamma_pairs
    lhs[:, :k_eff, k_eff] = 1.0
    lhs[:, k_eff, :k_eff] = 1.0
    rhs = np.zeros((q, k_eff + 1))
    rhs[:, :k_eff] = gamma_target
    rhs[:, k_eff] = 1.0
    return lhs, rhs


def _well_conditioned(lhs: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
    """Return, per system, whether it is well conditioned enough to solve.

    A system is accepted when its smallest singular value exceeds
    :data:`_KRIGING_RCOND` times its largest -- a reliable relative-conditioning
    test that catches matrices singular to working precision (which a
    determinant-sign test misses).
    """
    singular_values = np.linalg.svd(lhs, compute_uv=False)
    smallest = singular_values[:, -1]
    largest = singular_values[:, 0]
    return smallest > _KRIGING_RCOND * largest
