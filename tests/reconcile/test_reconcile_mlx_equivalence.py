"""Real-``mlx`` vs numpy equivalence for the reconcile kNN (macOS + mlx only).

This is the epic's "CPU/GPU-equivalence" DoD check for reconcile: it runs the
numpy reference and the *real* Metal ``metal_kernel`` backend on identical input
and asserts they agree. The kernel computes distances in float32, so the
contract is ``allclose`` on the squared distances plus exact neighbour-set
equality on tie-free input -- not byte-exact distances.

It is SKIPPED wherever ``mlx`` cannot be imported (all Linux CI runners), so it
provides no coverage; the numpy-backed fake in ``test_mlx_backend.py`` covers the
host adapter lines. ``mlx.core`` is loaded via ``importlib`` (never a static
import), so this module imports cleanly, and type-checks, without ``mlx`` present.
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest

from ahn_cli.reconcile.backend import NumpyBackend
from ahn_cli.reconcile.interpolate import interpolate
from ahn_cli.reconcile.method import (
    IdwInterp,
    KrigingInterp,
    Variogram,
    VariogramModel,
)
from ahn_cli.reconcile.mlx_backend import MlxBackend, MlxModule

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("mlx") is None,
    reason="requires mlx installed on Apple silicon",
)


def _real_mlx_backend() -> MlxBackend:
    """Return an MlxBackend around the real ``mlx.core`` module."""
    module = importlib.import_module("mlx.core")
    return MlxBackend(cast("MlxModule", module))


def _points(
    n: int, seed: int
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return a deterministic (xyz source, xy target) pair in a 50 m square."""
    rng = np.random.default_rng(seed)
    source = np.column_stack(
        [rng.uniform(0.0, 50.0, (n, 2)), rng.uniform(-5.0, 60.0, n)]
    )
    target = rng.uniform(0.0, 50.0, (40, 2))
    return source, target


def test_knn_matches_numpy_within_tolerance() -> None:
    """Real Metal kNN matches the numpy reference: sets exact, distances close."""
    source, target = _points(300, seed=1)
    ref_sq, ref_idx = NumpyBackend().knn(target, source[:, :2], 8)
    gpu_sq, gpu_idx = _real_mlx_backend().knn(target, source[:, :2], 8)
    assert np.array_equal(gpu_idx, ref_idx)
    assert np.allclose(gpu_sq, ref_sq, atol=1e-2)


def test_idw_estimates_match_between_backends() -> None:
    """IDW gives allclose estimates on the numpy and Metal backends."""
    source, target = _points(300, seed=2)
    method = IdwInterp(power=2.0, k=12)
    ref, _ = interpolate(method, source, target, backend=NumpyBackend())
    gpu, _ = interpolate(method, source, target, backend=_real_mlx_backend())
    assert np.allclose(gpu, ref, atol=1e-4)


def test_kriging_estimates_match_between_backends() -> None:
    """Ordinary kriging gives allclose estimates on both backends."""
    source, target = _points(300, seed=3)
    method = KrigingInterp(
        variogram=Variogram(VariogramModel.SPHERICAL, 0.0, 1.0, 20.0), k=12
    )
    ref, _ = interpolate(method, source, target, backend=NumpyBackend())
    gpu, _ = interpolate(method, source, target, backend=_real_mlx_backend())
    assert np.allclose(gpu, ref, atol=1e-3)
