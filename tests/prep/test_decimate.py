"""Tests for the prep-context graded-thinning transform (WP11).

Coverage strategy for the Apple-silicon accelerator on Linux CI (no ``mlx``):

* The pure-numpy :class:`NumpyBackend` is the correctness source of truth and is
  tested directly on real point clouds.
* :class:`MlxBackend` is exercised with a numpy-backed *fake* ``mlx.core``
  handle (:class:`FakeMx`), so every adapter line executes off-device and the
  wiring (ops, shapes, exact voxel equivalence to the CPU backend) is proven
  without ``mlx`` installed.
* Backend *selection* is driven through injected loaders, so the "mlx present"
  and "mlx absent" branches are both covered without ``mlx`` installed.

The real ``mlx`` vs CPU equivalence check lives in
``tests/prep/test_decimate_mlx_equivalence.py`` and is skipped unless ``mlx``
imports on Apple silicon; it is not what provides coverage here.
"""

from __future__ import annotations

from itertools import pairwise
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest

from ahn_cli.prep.decimate import (
    GRADE_MAX,
    GRADE_MIN,
    DecimationBackend,
    MlxArray,
    MlxBackend,
    NumpyBackend,
    PoissonThinning,
    VoxelThinning,
    decimate_poisson,
    decimate_voxel,
    select_backend,
    thin,
    voxel_size_for_grade,
)

# --------------------------------------------------------------------------- #
# A numpy-backed fake ``mlx.core`` handle.
#
# It implements exactly the narrow surface ``MlxBackend`` calls, so the adapter
# runs off-device on Linux and its wiring is fully covered.
# --------------------------------------------------------------------------- #


def _np(values: MlxArray) -> npt.NDArray[np.generic]:
    """Return the numpy view of a fake mlx array."""
    return np.asarray(values)


class FakeArray:
    """A numpy-wrapping stand-in for an ``mlx`` array."""

    def __init__(self, values: npt.ArrayLike) -> None:
        """Wrap ``values`` as a numpy array."""
        self.value: npt.NDArray[np.generic] = np.asarray(values)

    def __getitem__(self, key: object) -> FakeArray:
        """Slice the wrapped array (only slices are used by the adapter)."""
        return FakeArray(self.value[cast("slice", key)])

    def __array__(
        self, dtype: npt.DTypeLike = None
    ) -> npt.NDArray[np.generic]:
        """Expose the numpy view (the fake's device-to-host bridge)."""
        return np.asarray(self.value, dtype=dtype)


class FakeMx:
    """A numpy-backed fake of the ``mlx.core`` module function surface."""

    def array(self, values: object) -> FakeArray:
        """Wrap array-like ``values``."""
        return FakeArray(cast("npt.ArrayLike", values))

    def arange(self, size: int) -> FakeArray:
        """Return ``0..size-1``."""
        return FakeArray(np.arange(size))

    def argsort(self, values: MlxArray) -> FakeArray:
        """Return a stable ascending argsort of ``values``."""
        return FakeArray(np.argsort(_np(values), kind="stable"))

    def take(self, values: MlxArray, indices: MlxArray) -> FakeArray:
        """Gather ``values`` at ``indices``."""
        return FakeArray(np.take(_np(values), _np(indices).astype(np.intp)))

    def multiply(self, left: MlxArray, right: MlxArray | int) -> FakeArray:
        """Return the elementwise product."""
        other = right if isinstance(right, int) else _np(right)
        return FakeArray(np.multiply(_np(left), other))

    def add(self, left: MlxArray, right: MlxArray) -> FakeArray:
        """Return the elementwise sum."""
        return FakeArray(np.add(_np(left), _np(right)))

    def subtract(self, left: MlxArray, right: MlxArray) -> FakeArray:
        """Return the elementwise difference."""
        return FakeArray(np.subtract(_np(left), _np(right)))

    def not_equal(self, left: MlxArray, right: MlxArray) -> FakeArray:
        """Return the elementwise inequality mask."""
        return FakeArray(np.not_equal(_np(left), _np(right)))

    def sum(self, values: MlxArray, axis: int) -> FakeArray:
        """Reduce ``values`` by summation along ``axis``."""
        return FakeArray(np.sum(_np(values), axis=axis))


def _fake_mlx_backend() -> MlxBackend:
    """Return an :class:`MlxBackend` wired to the numpy-backed fake handle."""
    return MlxBackend(FakeMx())


# --------------------------------------------------------------------------- #
# Synthetic point-cloud fixtures.
# --------------------------------------------------------------------------- #


def _grid_cloud() -> npt.NDArray[np.float64]:
    """Return a dense 3D lattice: many points per coarse voxel."""
    axis = np.arange(0.0, 4.0, 0.5)
    xx, yy, zz = np.meshgrid(axis, axis, axis, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]).astype(
        np.float64
    )


def _random_cloud(n: int = 400, seed: int = 7) -> npt.NDArray[np.float64]:
    """Return a pseudo-random cloud in a 10 m cube (fixed seed)."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 10.0, size=(n, 3)).astype(np.float64)


_BACKENDS: list[DecimationBackend] = [NumpyBackend(), _fake_mlx_backend()]
"""Both backends, for parametrized behavioural equivalence."""


# --------------------------------------------------------------------------- #
# Grade -> voxel-size mapping.
# --------------------------------------------------------------------------- #


def test_grade_zero_is_identity_size() -> None:
    """Grade 0 maps to a zero edge length (the identity placeholder)."""
    assert voxel_size_for_grade(GRADE_MIN) == 0.0


def test_voxel_size_strictly_increases_for_active_grades() -> None:
    """Grades 1-9 map to strictly increasing positive voxel sizes."""
    sizes = [voxel_size_for_grade(g) for g in range(1, GRADE_MAX + 1)]
    assert all(a < b for a, b in pairwise(sizes))
    assert sizes[0] > 0.0


@pytest.mark.parametrize("grade", [-1, GRADE_MAX + 1])
def test_voxel_size_rejects_out_of_range_grade(grade: int) -> None:
    """A grade outside [0, 9] is a ValueError."""
    with pytest.raises(ValueError, match="voxel grade"):
        voxel_size_for_grade(grade)


# --------------------------------------------------------------------------- #
# Thinning value objects.
# --------------------------------------------------------------------------- #


def test_voxel_thinning_accepts_boundary_grades() -> None:
    """Both endpoint grades construct."""
    assert VoxelThinning(grade=GRADE_MIN).grade == GRADE_MIN
    assert VoxelThinning(grade=GRADE_MAX).grade == GRADE_MAX


@pytest.mark.parametrize("grade", [-1, 10])
def test_voxel_thinning_rejects_out_of_range(grade: int) -> None:
    """An out-of-range grade is rejected at construction."""
    with pytest.raises(ValueError, match="voxel grade"):
        VoxelThinning(grade=grade)


def test_poisson_thinning_defaults_seed() -> None:
    """The seed defaults to the documented fixed value."""
    assert PoissonThinning(radius=1.0).seed == 0


@pytest.mark.parametrize("radius", [0.0, -1.0, float("inf"), float("nan")])
def test_poisson_thinning_rejects_bad_radius(radius: float) -> None:
    """A non-finite or non-positive radius is rejected at construction."""
    with pytest.raises(ValueError, match="poisson radius"):
        PoissonThinning(radius=radius)


# --------------------------------------------------------------------------- #
# NumpyBackend primitives.
# --------------------------------------------------------------------------- #


def test_numpy_backend_is_named_cpu() -> None:
    """The reference backend identifies as ``cpu``."""
    assert NumpyBackend().name == "cpu"


def test_numpy_voxel_first_indices_empty() -> None:
    """No groups in, no indices out."""
    out = NumpyBackend().voxel_first_indices(np.empty(0, dtype=np.int64))
    assert out.dtype == np.intp
    assert out.size == 0


def test_numpy_voxel_first_indices_keeps_min_index_per_group() -> None:
    """Each distinct group id keeps its smallest original index."""
    groups = np.array([2, 0, 2, 1, 0, 2], dtype=np.int64)
    # group 0 -> min idx 1, group 1 -> idx 3, group 2 -> min idx 0.
    assert NumpyBackend().voxel_first_indices(groups).tolist() == [0, 1, 3]


def test_numpy_squared_distances_matches_manual() -> None:
    """Squared distances equal the hand-computed values."""
    point = np.array([0.0, 0.0, 0.0])
    others = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 2.0]])
    out = NumpyBackend().squared_distances(point, others)
    assert out.tolist() == [25.0, 4.0]


# --------------------------------------------------------------------------- #
# MlxBackend adapter (via the numpy-backed fake).
# --------------------------------------------------------------------------- #


def test_mlx_backend_is_named_mlx() -> None:
    """The accelerator backend identifies as ``mlx``."""
    assert _fake_mlx_backend().name == "mlx"


def test_mlx_voxel_first_indices_empty() -> None:
    """The adapter short-circuits an empty input to an empty result."""
    out = _fake_mlx_backend().voxel_first_indices(np.empty(0, dtype=np.int64))
    assert out.dtype == np.intp
    assert out.size == 0


def test_mlx_voxel_first_indices_matches_numpy_backend() -> None:
    """The fake-mlx adapter reproduces the reference selection exactly."""
    groups = np.array([2, 0, 2, 1, 0, 2, 1], dtype=np.int64)
    ref = NumpyBackend().voxel_first_indices(groups)
    got = _fake_mlx_backend().voxel_first_indices(groups)
    assert got.tolist() == ref.tolist()


def test_mlx_squared_distances_matches_numpy_backend() -> None:
    """The fake-mlx distance kernel reproduces the reference values."""
    point = np.array([1.0, 1.0, 1.0])
    others = np.array([[1.0, 1.0, 4.0], [5.0, 1.0, 1.0]])
    ref = NumpyBackend().squared_distances(point, others)
    got = _fake_mlx_backend().squared_distances(point, others)
    assert got.tolist() == ref.tolist()


# --------------------------------------------------------------------------- #
# Backend selection (both branches, no real mlx needed).
# --------------------------------------------------------------------------- #


def _raise_import_error(_name: str) -> object:
    """Raise ImportError to model the "mlx absent" (Linux CI) path."""
    raise ImportError


def test_select_backend_prefers_gpu_when_import_succeeds() -> None:
    """With an importer that yields a handle, the accelerator is chosen."""
    chosen = select_backend(import_module=lambda _name: FakeMx())
    assert isinstance(chosen, MlxBackend)
    assert chosen.name == "mlx"


def test_select_backend_falls_back_when_mlx_absent() -> None:
    """With an importer that raises ImportError, the CPU reference is chosen."""
    chosen = select_backend(import_module=_raise_import_error)
    assert chosen.name == "cpu"


def test_select_backend_forces_cpu_when_gpu_not_preferred() -> None:
    """``prefer_gpu=False`` never attempts to import mlx."""
    chosen = select_backend(
        prefer_gpu=False, import_module=_raise_import_error
    )
    assert chosen.name == "cpu"


# --------------------------------------------------------------------------- #
# Voxel decimation behaviour.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("backend", _BACKENDS, ids=["cpu", "fake-mlx"])
def test_voxel_grade_zero_keeps_all_points(
    backend: DecimationBackend,
) -> None:
    """Grade 0 is the identity: every point survives."""
    coords = _grid_cloud()
    kept = decimate_voxel(coords, 0, backend=backend)
    assert kept.tolist() == list(range(len(coords)))


@pytest.mark.parametrize("backend", _BACKENDS, ids=["cpu", "fake-mlx"])
def test_voxel_empty_cloud_returns_empty(
    backend: DecimationBackend,
) -> None:
    """An empty cloud thins to nothing at an active grade."""
    coords = np.empty((0, 3), dtype=np.float64)
    assert decimate_voxel(coords, 5, backend=backend).size == 0


@pytest.mark.parametrize("backend", _BACKENDS, ids=["cpu", "fake-mlx"])
def test_voxel_density_strictly_decreases_with_grade(
    backend: DecimationBackend,
) -> None:
    """Higher grade -> coarser grid -> strictly fewer kept points."""
    coords = _random_cloud(n=3000)
    counts = [
        len(decimate_voxel(coords, g, backend=backend)) for g in range(1, 7)
    ]
    assert all(a > b for a, b in pairwise(counts))


@pytest.mark.parametrize("backend", _BACKENDS, ids=["cpu", "fake-mlx"])
def test_voxel_keeps_one_point_per_occupied_voxel(
    backend: DecimationBackend,
) -> None:
    """The kept count equals the number of occupied voxels."""
    coords = _grid_cloud()
    grade = 3
    size = voxel_size_for_grade(grade)
    kept = decimate_voxel(coords, grade, backend=backend)
    occupied = {
        tuple(np.floor((coords[i] - coords.min(axis=0)) / size).astype(int))
        for i in range(len(coords))
    }
    assert len(kept) == len(occupied)


def test_voxel_is_spatially_more_uniform_than_nth_point() -> None:
    """Voxel thinning enforces a spacing floor that nth-point selection cannot.

    On a raster-ordered lattice, nth-point keeps consecutive raster points, so
    its kept set contains immediately adjacent (0.5 m apart) points. Voxel keeps
    at most one point per >=1 m cell, so its closest kept pair is strictly
    farther apart -- a concrete spatial-uniformity gain over nth-point.
    """
    coords = _grid_cloud()
    backend = NumpyBackend()
    voxel_idx = decimate_voxel(coords, 3, backend=backend)
    voxel_kept = coords[voxel_idx]
    step = max(1, len(coords) // len(voxel_idx))
    nth_kept = coords[np.arange(0, len(coords), step)]
    assert _min_pairwise_distance(voxel_kept) > _min_pairwise_distance(
        nth_kept
    )


def test_voxel_cpu_and_fake_mlx_select_identical_sets() -> None:
    """CPU and fake-mlx voxel selections are byte-identical across grades."""
    coords = _random_cloud()
    for grade in range(GRADE_MIN, GRADE_MAX + 1):
        cpu = decimate_voxel(coords, grade, backend=NumpyBackend())
        mlx = decimate_voxel(coords, grade, backend=_fake_mlx_backend())
        assert cpu.tolist() == mlx.tolist()


def test_voxel_output_is_sorted_and_deterministic() -> None:
    """Repeated runs return the identical, ascending index array."""
    coords = _random_cloud()
    first = decimate_voxel(coords, 4, backend=NumpyBackend())
    second = decimate_voxel(coords, 4, backend=NumpyBackend())
    assert first.tolist() == second.tolist()
    assert first.tolist() == sorted(first.tolist())


def test_voxel_grade_rejects_out_of_range() -> None:
    """decimate_voxel forwards the grade-range guard."""
    with pytest.raises(ValueError, match="voxel grade"):
        decimate_voxel(_grid_cloud(), 42, backend=NumpyBackend())


# --------------------------------------------------------------------------- #
# Poisson-disk decimation behaviour.
# --------------------------------------------------------------------------- #


def _min_pairwise_distance(points: npt.NDArray[np.float64]) -> float:
    diff = points[:, None, :] - points[None, :, :]
    dist = np.sqrt((diff * diff).sum(axis=2))
    np.fill_diagonal(dist, np.inf)
    return float(dist.min())


@pytest.mark.parametrize("backend", _BACKENDS, ids=["cpu", "fake-mlx"])
def test_poisson_minimum_distance_property_holds(
    backend: DecimationBackend,
) -> None:
    """No two kept points are closer than the radius."""
    coords = _random_cloud()
    radius = 1.5
    kept = decimate_poisson(coords, radius, seed=3, backend=backend)
    assert len(kept) > 1
    assert _min_pairwise_distance(coords[kept]) >= radius


@pytest.mark.parametrize("backend", _BACKENDS, ids=["cpu", "fake-mlx"])
def test_poisson_empty_cloud_returns_empty(
    backend: DecimationBackend,
) -> None:
    """An empty cloud yields no samples."""
    coords = np.empty((0, 3), dtype=np.float64)
    assert decimate_poisson(coords, 1.0, backend=backend).size == 0


@pytest.mark.parametrize("backend", _BACKENDS, ids=["cpu", "fake-mlx"])
def test_poisson_is_deterministic_for_a_seed(
    backend: DecimationBackend,
) -> None:
    """Same cloud, radius and seed -> identical kept indices."""
    coords = _random_cloud()
    first = decimate_poisson(coords, 1.2, seed=11, backend=backend)
    second = decimate_poisson(coords, 1.2, seed=11, backend=backend)
    assert first.tolist() == second.tolist()
    assert first.tolist() == sorted(first.tolist())


def test_poisson_different_seeds_differ() -> None:
    """Different seeds generally select different point sets."""
    coords = _random_cloud()
    a = decimate_poisson(coords, 1.2, seed=1, backend=NumpyBackend())
    b = decimate_poisson(coords, 1.2, seed=2, backend=NumpyBackend())
    assert a.tolist() != b.tolist()


def test_poisson_single_point_is_always_kept() -> None:
    """A lone point has no neighbour and is accepted (isolated branch)."""
    coords = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
    kept = decimate_poisson(coords, 1.0, backend=NumpyBackend())
    assert kept.tolist() == [0]


@pytest.mark.parametrize("radius", [0.0, -2.0, float("inf"), float("nan")])
def test_poisson_rejects_bad_radius(radius: float) -> None:
    """decimate_poisson guards the radius directly (not only via the VO)."""
    with pytest.raises(ValueError, match="poisson radius"):
        decimate_poisson(_random_cloud(), radius, backend=NumpyBackend())


# --------------------------------------------------------------------------- #
# Typed dispatch.
# --------------------------------------------------------------------------- #


def test_thin_dispatches_voxel() -> None:
    """A VoxelThinning routes to the voxel algorithm."""
    coords = _grid_cloud()
    got = thin(coords, VoxelThinning(grade=3), backend=NumpyBackend())
    expected = decimate_voxel(coords, 3, backend=NumpyBackend())
    assert got.tolist() == expected.tolist()


def test_thin_dispatches_poisson() -> None:
    """A PoissonThinning routes to the Poisson algorithm."""
    coords = _random_cloud()
    got = thin(
        coords, PoissonThinning(radius=1.4, seed=5), backend=NumpyBackend()
    )
    expected = decimate_poisson(coords, 1.4, seed=5, backend=NumpyBackend())
    assert got.tolist() == expected.tolist()
