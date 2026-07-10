"""The reconcile interpolation methods (typed dispatch on the request variant).

Each method estimates an elevation ``Z`` at every target XY from the AHN source
points ``(x, y, z)``, returning ``(z, valid)`` where ``valid`` marks the cells
an estimate could be produced for:

* :func:`_linear` -- Delaunay-barycentric linear interpolation via scipy; cells
  outside the convex hull (or a degenerate point set) are void.
* :func:`_idw` -- inverse-distance weighting over the ``k`` nearest points.
* :func:`_kriging` -- ordinary kriging over the ``k`` nearest points; a singular
  per-cell system falls back deterministically to the IDW estimate.

IDW and kriging obtain their neighbours through the injected
:class:`~ahn_cli.reconcile.backend.InterpBackend`, so the numpy reference and the
Metal accelerator share one algorithm. Every branch here is host-side, so both
backends exercise identical control flow -- only the kNN numerics differ.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import QhullError

from ahn_cli.reconcile.method import (
    IdwInterp,
    KrigingInterp,
    LinearInterp,
)

if TYPE_CHECKING:
    from ahn_cli.reconcile.backend import InterpBackend
    from ahn_cli.reconcile.method import InterpMethod

_MIN_TRIANGULATION_POINTS = 3
"""Delaunay linear interpolation needs at least a triangle."""

_IDW_FALLBACK_POWER = 2.0
"""Distance exponent for the IDW estimate kriging falls back to when singular."""


def interpolate(
    method: InterpMethod,
    source_xyz: npt.NDArray[np.float64],
    target_xy: npt.NDArray[np.float64],
    *,
    backend: InterpBackend,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Estimate ``Z`` at each target XY, dispatching on the method variant.

    Contract:
        - ``source_xyz`` is ``(n, 3)`` world ``(x, y, z)``; ``target_xy`` is
          ``(q, 2)`` world XY at which to estimate ``Z``.
        - Returns ``(z, valid)`` each length ``q``: ``z`` holds the estimate
          (``NaN`` where void) and ``valid`` is ``True`` where an estimate was
          produced.

    Typed dispatch on the request variant -- no stringly-typed method switch.
    """
    if isinstance(method, LinearInterp):
        return _linear(source_xyz, target_xy)
    if isinstance(method, IdwInterp):
        return _idw(source_xyz, target_xy, method, backend)
    return _kriging(source_xyz, target_xy, method, backend)


def _void(
    q: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Return an all-void ``(z, valid)`` result for ``q`` targets."""
    return np.full(q, np.nan), np.zeros(q, dtype=np.bool_)


def _linear(
    source_xyz: npt.NDArray[np.float64],
    target_xy: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Delaunay-barycentric linear interpolation; outside-hull cells are void."""
    q = target_xy.shape[0]
    if source_xyz.shape[0] < _MIN_TRIANGULATION_POINTS:
        return _void(q)
    try:
        interp = LinearNDInterpolator(
            source_xyz[:, :2], source_xyz[:, 2], fill_value=np.nan
        )
        estimate = interp(target_xy)
    except QhullError:
        return _void(q)
    z = np.asarray(estimate, dtype=np.float64).reshape(q)
    return z, ~np.isnan(z)


def _idw(
    source_xyz: npt.NDArray[np.float64],
    target_xy: npt.NDArray[np.float64],
    method: IdwInterp,
    backend: InterpBackend,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Inverse-distance-weighted interpolation over the ``k`` nearest points."""
    sq, idx = backend.knn(target_xy, source_xyz[:, :2], method.k)
    q, k_eff = sq.shape
    if k_eff == 0:
        return _void(q)
    z_neighbours = source_xyz[:, 2][idx]
    z = _idw_weighted_mean(sq, z_neighbours, method.power)
    return z, np.ones(q, dtype=np.bool_)


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
        out=np.zeros(q),
        where=weight_sum > 0.0,
    )
    coincident_count = coincident.sum(axis=1)
    exact = np.divide(
        (coincident * z_neighbours).sum(axis=1),
        coincident_count,
        out=np.zeros(q),
        where=coincident_count > 0,
    )
    return np.where(coincident.any(axis=1), exact, general)


def _kriging(
    source_xyz: npt.NDArray[np.float64],
    target_xy: npt.NDArray[np.float64],
    method: KrigingInterp,
    backend: InterpBackend,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Ordinary kriging over the ``k`` nearest points with a fixed variogram."""
    source_xy = source_xyz[:, :2]
    sq, idx = backend.knn(target_xy, source_xy, method.k)
    q, k_eff = sq.shape
    if k_eff == 0:
        return _void(q)
    z_neighbours = source_xyz[:, 2][idx]
    lhs, rhs = _kriging_system(source_xy[idx], sq, method)
    # Start every cell at the deterministic IDW estimate, then overwrite the
    # cells whose kriging system is non-singular with the kriging solution. A
    # singular system (e.g. coincident neighbours with a zero nugget) keeps the
    # IDW fallback -- detected up front so no per-cell exception handling runs.
    z = _idw_weighted_mean(sq, z_neighbours, _IDW_FALLBACK_POWER)
    sign, _ = np.linalg.slogdet(lhs)
    solvable = sign != 0.0
    if solvable.any():
        weights = np.linalg.solve(
            lhs[solvable], rhs[solvable][:, :, np.newaxis]
        )[:, :, 0]
        z[solvable] = (weights[:, :k_eff] * z_neighbours[solvable]).sum(
            axis=1
        )
    return z, np.ones(q, dtype=np.bool_)


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
