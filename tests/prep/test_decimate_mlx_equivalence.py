"""Real-``mlx`` vs CPU equivalence for graded decimation (macOS + mlx only).

This is the epic's "CPU/GPU-equivalence" DoD check. It runs the numpy reference
and the *real* MLX backend on identical input and asserts they agree:

* **Voxel** -- exact retained-index-set equality (the spike's deterministic
  ``sort by (voxel_key, point_id)`` contract).
* **Poisson** -- property equivalence within the documented tolerance: MLX
  computes squared distances in float32, so a point exactly on the radius
  boundary may flip versus the float64 CPU path. The asserted contract is
  therefore the min-distance property on the MLX result plus a count-within-
  tolerance and bbox-containment check -- not byte-exact index equality.

It is SKIPPED wherever ``mlx`` cannot be imported (all Linux CI runners), so it
provides no coverage; the numpy-backed fake in ``test_decimate.py`` covers the
adapter lines. ``mlx.core`` is loaded via ``importlib`` (never a static import),
so this module imports cleanly, and type-checks, without ``mlx`` present.
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest

from ahn_cli.prep.decimate import (
    GRADE_MAX,
    GRADE_MIN,
    MlxBackend,
    MlxModule,
    NumpyBackend,
    decimate_poisson,
    decimate_voxel,
)

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("mlx") is None,
    reason="requires mlx installed on Apple silicon",
)


def _real_mlx_backend() -> MlxBackend:
    """Return an MlxBackend around the real ``mlx.core`` module."""
    module = importlib.import_module("mlx.core")
    return MlxBackend(cast("MlxModule", module))


def _cloud(n: int = 500, seed: int = 13) -> npt.NDArray[np.float64]:
    """Return a deterministic pseudo-random cloud in a 10 m cube."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 10.0, size=(n, 3)).astype(np.float64)


def _min_pairwise_distance(points: npt.NDArray[np.float64]) -> float:
    diff = points[:, None, :] - points[None, :, :]
    dist = np.sqrt((diff * diff).sum(axis=2))
    np.fill_diagonal(dist, np.inf)
    return float(dist.min())


def test_voxel_real_mlx_matches_cpu_exactly() -> None:
    """Every grade: real MLX keeps the identical index set as the CPU backend."""
    coords = _cloud()
    cpu = NumpyBackend()
    mlx = _real_mlx_backend()
    for grade in range(GRADE_MIN, GRADE_MAX + 1):
        cpu_kept = decimate_voxel(coords, grade, backend=cpu)
        mlx_kept = decimate_voxel(coords, grade, backend=mlx)
        assert cpu_kept.tolist() == mlx_kept.tolist(), grade


def test_poisson_real_mlx_matches_cpu_within_tolerance() -> None:
    """Real MLX Poisson holds the min-distance property and count tolerance."""
    coords = _cloud()
    radius = 1.5
    cpu_kept = decimate_poisson(
        coords, radius, seed=5, backend=NumpyBackend()
    )
    mlx_kept = decimate_poisson(
        coords, radius, seed=5, backend=_real_mlx_backend()
    )

    # Hard constraint: the MLX result is a valid Poisson-disk sampling.
    assert _min_pairwise_distance(coords[mlx_kept]) >= radius
    # Containment: kept points are a subset of the input.
    assert set(mlx_kept.tolist()) <= set(range(len(coords)))
    # Count within tolerance (float32 boundary flips only, if any).
    tolerance = max(2, round(0.05 * len(cpu_kept)))
    assert abs(len(mlx_kept) - len(cpu_kept)) <= tolerance
