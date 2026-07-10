"""Tests for the reconcile interpolation methods against the numpy oracle."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from ahn_cli.reconcile.backend import NumpyBackend
from ahn_cli.reconcile.interpolate import interpolate
from ahn_cli.reconcile.method import (
    IdwInterp,
    KrigingInterp,
    LinearInterp,
    Variogram,
    VariogramModel,
)

_BACKEND = NumpyBackend()


def _xyz(
    rows: list[tuple[float, float, float]],
) -> npt.NDArray[np.float64]:
    return np.asarray(rows, dtype=np.float64)


class TestBackend:
    """The numpy backend's identity and kNN delegation."""

    def test_name(self) -> None:
        """The numpy backend identifies itself as the CPU reference."""
        assert NumpyBackend().name == "cpu"

    def test_knn_delegates(self) -> None:
        """Return the deterministic nearest neighbours via the backend."""
        source = np.array([[0.0, 0.0], [1.0, 0.0]])
        sq, idx = NumpyBackend().knn(np.array([[0.0, 0.0]]), source, 1)
        assert idx.tolist() == [[0]]
        assert sq.tolist() == [[0.0]]


class TestLinear:
    """Delaunay-barycentric linear interpolation."""

    def test_interpolates_inside_hull(self) -> None:
        """A point inside the triangle takes the barycentric-linear value."""
        source = _xyz([(0.0, 0.0, 0.0), (1.0, 0.0, 1.0), (0.0, 1.0, 2.0)])
        z, valid = interpolate(
            LinearInterp(), source, np.array([[0.25, 0.25]]), backend=_BACKEND
        )
        assert valid.tolist() == [True]
        assert math.isclose(z[0], 0.75, abs_tol=1e-9)

    def test_outside_hull_is_void(self) -> None:
        """A point outside the convex hull is marked void."""
        source = _xyz([(0.0, 0.0, 0.0), (1.0, 0.0, 1.0), (0.0, 1.0, 2.0)])
        _, valid = interpolate(
            LinearInterp(), source, np.array([[5.0, 5.0]]), backend=_BACKEND
        )
        assert valid.tolist() == [False]

    def test_too_few_points_all_void(self) -> None:
        """Fewer than three points cannot triangulate; all targets are void."""
        source = _xyz([(0.0, 0.0, 1.0), (1.0, 0.0, 2.0)])
        _, valid = interpolate(
            LinearInterp(), source, np.array([[0.5, 0.0]]), backend=_BACKEND
        )
        assert valid.tolist() == [False]

    def test_collinear_points_all_void(self) -> None:
        """A degenerate (collinear) point set cannot triangulate; all void."""
        source = _xyz([(0.0, 0.0, 0.0), (1.0, 0.0, 1.0), (2.0, 0.0, 2.0)])
        _, valid = interpolate(
            LinearInterp(), source, np.array([[0.5, 0.0]]), backend=_BACKEND
        )
        assert valid.tolist() == [False]


class TestIdw:
    """Inverse-distance-weighted interpolation."""

    def test_single_neighbour_returns_its_value(self) -> None:
        """With k = 1 the nearest point's value is returned."""
        source = _xyz([(0.0, 0.0, 10.0), (100.0, 0.0, 20.0)])
        z, valid = interpolate(
            IdwInterp(k=1), source, np.array([[1.0, 0.0]]), backend=_BACKEND
        )
        assert valid.tolist() == [True]
        assert math.isclose(z[0], 10.0, abs_tol=1e-9)

    def test_equal_values_return_that_value(self) -> None:
        """When neighbours share a value, IDW returns it regardless of weights."""
        source = _xyz([(0.0, 0.0, 5.0), (10.0, 0.0, 5.0), (0.0, 10.0, 5.0)])
        z, _ = interpolate(
            IdwInterp(k=3), source, np.array([[3.0, 4.0]]), backend=_BACKEND
        )
        assert math.isclose(z[0], 5.0, abs_tol=1e-9)

    def test_coincident_target_returns_exact_value(self) -> None:
        """A target exactly on a source point returns that point's value."""
        source = _xyz([(2.0, 3.0, 42.0), (10.0, 0.0, 7.0)])
        z, _ = interpolate(
            IdwInterp(k=2), source, np.array([[2.0, 3.0]]), backend=_BACKEND
        )
        assert math.isclose(z[0], 42.0, abs_tol=1e-9)

    def test_matches_closed_form_two_points(self) -> None:
        """The estimate matches the inverse-square-weighted mean of two points."""
        source = _xyz([(0.0, 0.0, 10.0), (10.0, 0.0, 20.0)])
        z, _ = interpolate(
            IdwInterp(power=2.0, k=2),
            source,
            np.array([[2.0, 0.0]]),
            backend=_BACKEND,
        )
        w0, w1 = 1.0 / 2.0**2, 1.0 / 8.0**2
        expected = (w0 * 10.0 + w1 * 20.0) / (w0 + w1)
        assert math.isclose(z[0], expected, rel_tol=1e-12)

    def test_empty_source_is_void(self) -> None:
        """No source points yields a void estimate."""
        z, valid = interpolate(
            IdwInterp(),
            np.empty((0, 3)),
            np.array([[0.0, 0.0]]),
            backend=_BACKEND,
        )
        assert valid.tolist() == [False]
        assert np.isnan(z[0])


class TestKriging:
    """Ordinary kriging with a fixed variogram."""

    def _variogram(self, nugget: float = 0.0) -> Variogram:
        return Variogram(VariogramModel.SPHERICAL, nugget, 1.0, 20.0)

    def test_exact_at_data_point(self) -> None:
        """Zero-nugget kriging reproduces a datum at its own location."""
        source = _xyz(
            [
                (0.0, 0.0, 3.0),
                (10.0, 0.0, 8.0),
                (0.0, 10.0, 5.0),
                (10.0, 10.0, 1.0),
            ]
        )
        z, valid = interpolate(
            KrigingInterp(variogram=self._variogram(), k=4),
            source,
            np.array([[10.0, 0.0]]),
            backend=_BACKEND,
        )
        assert valid.tolist() == [True]
        assert math.isclose(z[0], 8.0, abs_tol=1e-6)

    def test_constant_field_returns_constant(self) -> None:
        """Kriging a constant field returns the constant everywhere."""
        source = _xyz(
            [
                (0.0, 0.0, 4.0),
                (10.0, 0.0, 4.0),
                (0.0, 10.0, 4.0),
                (10.0, 10.0, 4.0),
            ]
        )
        z, _ = interpolate(
            KrigingInterp(variogram=self._variogram(), k=4),
            source,
            np.array([[5.0, 5.0]]),
            backend=_BACKEND,
        )
        assert math.isclose(z[0], 4.0, abs_tol=1e-6)

    def test_singular_system_falls_back_mixed_batch(self) -> None:
        """A batch with one singular and one solvable row is handled per cell.

        Target B's two nearest points are coincident duplicates with a zero
        nugget, giving two identical kriging rows (singular); target A's two
        nearest are distinct (solvable). The singular-detection mask keeps A's
        kriging estimate and gives B the deterministic IDW fallback. Both must
        be finite.
        """
        source = _xyz(
            [
                (0.0, 0.0, 10.0),
                (0.0, 0.0, 10.0),
                (100.0, 0.0, 20.0),
                (101.0, 0.0, 21.0),
            ]
        )
        z, valid = interpolate(
            KrigingInterp(variogram=self._variogram(), k=2),
            source,
            np.array([[100.5, 0.0], [0.5, 0.0]]),  # A solvable, B singular
            backend=_BACKEND,
        )
        assert valid.tolist() == [True, True]
        assert np.isfinite(z).all()

    def test_all_singular_falls_back(self) -> None:
        """When every system is singular the estimate is the IDW fallback.

        The only two source points are coincident duplicates, so every target's
        kriging system is singular and the solvable mask is empty. The estimate
        must still be the finite IDW value, not NaN.
        """
        source = _xyz([(0.0, 0.0, 10.0), (0.0, 0.0, 10.0)])
        z, valid = interpolate(
            KrigingInterp(variogram=self._variogram(), k=2),
            source,
            np.array([[1.0, 0.0]]),
            backend=_BACKEND,
        )
        assert valid.tolist() == [True]
        assert math.isclose(z[0], 10.0, abs_tol=1e-9)

    def test_empty_source_is_void(self) -> None:
        """No source points yields a void estimate."""
        z, valid = interpolate(
            KrigingInterp(variogram=self._variogram(), k=4),
            np.empty((0, 3)),
            np.array([[0.0, 0.0]]),
            backend=_BACKEND,
        )
        assert valid.tolist() == [False]
        assert np.isnan(z[0])


def test_deterministic_numpy_path() -> None:
    """The numpy path yields byte-identical estimates across calls."""
    rng = np.random.default_rng(1)
    source = np.column_stack([rng.random((80, 2)) * 10.0, rng.random(80)])
    target = rng.random((25, 2)) * 10.0
    method = KrigingInterp(
        variogram=Variogram(VariogramModel.EXPONENTIAL, 0.0, 1.0, 5.0), k=8
    )
    z_a, _ = interpolate(method, source, target, backend=_BACKEND)
    z_b, _ = interpolate(method, source, target, backend=_BACKEND)
    assert np.array_equal(z_a, z_b)
