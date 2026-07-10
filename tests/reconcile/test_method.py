"""Tests for the reconcile interpolation-method value objects."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ahn_cli.reconcile.method import (
    IdwInterp,
    KrigingInterp,
    LinearInterp,
    Variogram,
    VariogramModel,
)


class TestIdwInterp:
    """The inverse-distance-weighting request value object."""

    def test_defaults(self) -> None:
        """Default power and neighbour count are the documented values."""
        idw = IdwInterp()
        assert idw.power == 2.0
        assert idw.k == 12

    def test_frozen_equal_by_value(self) -> None:
        """Two identical requests compare equal (frozen value object)."""
        assert IdwInterp(power=1.5, k=8) == IdwInterp(power=1.5, k=8)

    @pytest.mark.parametrize("power", [0.0, -1.0, float("inf"), float("nan")])
    def test_rejects_non_positive_or_non_finite_power(
        self, power: float
    ) -> None:
        """A power that is not finite and positive is rejected."""
        with pytest.raises(ValueError, match="power"):
            IdwInterp(power=power)

    @pytest.mark.parametrize("k", [0, -3])
    def test_rejects_non_positive_k(self, k: int) -> None:
        """A neighbour count below one is rejected."""
        with pytest.raises(ValueError, match="k"):
            IdwInterp(k=k)


class TestLinearInterp:
    """The Delaunay-linear request value object."""

    def test_frozen_equal_by_value(self) -> None:
        """All linear requests compare equal (no parameters)."""
        assert LinearInterp() == LinearInterp()


class TestVariogram:
    """The parameterised variogram model backing kriging."""

    def test_frozen_equal_by_value(self) -> None:
        """Two identical variograms compare equal."""
        left = Variogram(VariogramModel.SPHERICAL, 0.0, 1.0, 10.0)
        right = Variogram(VariogramModel.SPHERICAL, 0.0, 1.0, 10.0)
        assert left == right

    def test_rejects_negative_nugget(self) -> None:
        """A negative nugget is rejected."""
        with pytest.raises(ValueError, match="nugget"):
            Variogram(VariogramModel.SPHERICAL, -0.1, 1.0, 10.0)

    def test_rejects_sill_below_nugget(self) -> None:
        """A total sill below the nugget is rejected (negative partial sill)."""
        with pytest.raises(ValueError, match="sill"):
            Variogram(VariogramModel.SPHERICAL, 0.5, 0.2, 10.0)

    @pytest.mark.parametrize("vrange", [0.0, -1.0, float("inf")])
    def test_rejects_non_positive_or_non_finite_range(
        self, vrange: float
    ) -> None:
        """A range that is not finite and positive is rejected."""
        with pytest.raises(ValueError, match="range"):
            Variogram(VariogramModel.SPHERICAL, 0.0, 1.0, vrange)

    def test_semivariance_zero_lag_is_zero(self) -> None:
        """Semivariance at zero lag is zero for every model."""
        for model in VariogramModel:
            vg = Variogram(model, 0.2, 1.0, 10.0)
            got = float(vg.semivariance(np.array([0.0]))[0])
            assert math.isclose(got, 0.0, abs_tol=1e-12)

    def test_spherical_reaches_sill_beyond_range(self) -> None:
        """Spherical semivariance equals the sill at and beyond the range."""
        vg = Variogram(VariogramModel.SPHERICAL, 0.0, 2.0, 10.0)
        beyond = vg.semivariance(np.array([10.0, 25.0]))
        assert np.allclose(beyond, [2.0, 2.0])

    def test_spherical_midpoint_value(self) -> None:
        """Spherical semivariance matches the closed form at h = range/2."""
        vg = Variogram(VariogramModel.SPHERICAL, 0.0, 1.0, 10.0)
        # 1.5*(0.5) - 0.5*(0.5**3) = 0.75 - 0.0625 = 0.6875
        got = float(vg.semivariance(np.array([5.0]))[0])
        assert math.isclose(got, 0.6875, abs_tol=1e-12)

    def test_exponential_value(self) -> None:
        """Exponential semivariance matches nugget + psill*(1-exp(-h/r))."""
        vg = Variogram(VariogramModel.EXPONENTIAL, 0.1, 1.1, 10.0)
        got = float(vg.semivariance(np.array([10.0]))[0])
        assert math.isclose(got, 0.1 + 1.0 * (1.0 - math.exp(-1.0)))

    def test_gaussian_value(self) -> None:
        """Gaussian semivariance matches nugget + psill*(1-exp(-(h/r)**2))."""
        vg = Variogram(VariogramModel.GAUSSIAN, 0.0, 1.0, 10.0)
        got = float(vg.semivariance(np.array([10.0]))[0])
        assert math.isclose(got, 1.0 - math.exp(-1.0))


class TestKrigingInterp:
    """The ordinary-kriging request value object."""

    def test_defaults(self) -> None:
        """The default neighbour count is the documented value."""
        kr = KrigingInterp(
            variogram=Variogram(VariogramModel.SPHERICAL, 0.0, 1.0, 10.0)
        )
        assert kr.k == 16

    def test_frozen_equal_by_value(self) -> None:
        """Two identical kriging requests compare equal."""
        vg = Variogram(VariogramModel.SPHERICAL, 0.0, 1.0, 10.0)
        assert KrigingInterp(variogram=vg, k=8) == KrigingInterp(
            variogram=vg, k=8
        )

    @pytest.mark.parametrize("k", [0, -2])
    def test_rejects_non_positive_k(self, k: int) -> None:
        """A neighbour count below one is rejected."""
        vg = Variogram(VariogramModel.SPHERICAL, 0.0, 1.0, 10.0)
        with pytest.raises(ValueError, match="k"):
            KrigingInterp(variogram=vg, k=k)
