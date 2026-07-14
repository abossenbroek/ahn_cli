"""Tests for the EPSG:7415 geodesy transforms."""

from __future__ import annotations

import numpy as np

from ahn_cli.tiles3d.geodesy import Geodesy

# The OLV chapel dome in Amersfoort: the RD origin's defining point.
_RD_X = 155_000.0
_RD_Y = 463_000.0
_AMERSFOORT_LON_DEG = 5.387
_AMERSFOORT_LAT_DEG = 52.155


def test_to_ecef_shapes_and_magnitude() -> None:
    """ECEF coordinates keep the shape and sit on the Earth's surface."""
    geodesy = Geodesy()
    x = np.array([_RD_X, _RD_X + 100.0])
    y = np.array([_RD_Y, _RD_Y + 100.0])
    z = np.array([0.0, 10.0])
    ex, ey, ez = geodesy.to_ecef(x, y, z)
    for axis in (ex, ey, ez):
        assert axis.shape == (2,)
        assert axis.dtype == np.float64
    radius = np.sqrt(ex**2 + ey**2 + ez**2)
    assert np.all(radius > 6.3e6)
    assert np.all(radius < 6.5e6)


def test_to_geodetic_radians_matches_amersfoort() -> None:
    """The RD origin lands on Amersfoort in EPSG:4979 radians."""
    geodesy = Geodesy()
    lon, lat, h = geodesy.to_geodetic_radians(
        np.array([_RD_X]), np.array([_RD_Y]), np.array([0.0])
    )
    assert abs(np.degrees(lon[0]) - _AMERSFOORT_LON_DEG) < 0.01
    assert abs(np.degrees(lat[0]) - _AMERSFOORT_LAT_DEG) < 0.01
    assert -np.pi <= lon[0] <= np.pi
    assert -np.pi / 2 <= lat[0] <= np.pi / 2
    assert np.isfinite(h[0])


def test_transforms_are_deterministic() -> None:
    """Two calls (and two instances) agree bit for bit."""
    x = np.array([_RD_X, _RD_X + 50.0, _RD_X - 50.0])
    y = np.array([_RD_Y, _RD_Y - 25.0, _RD_Y + 25.0])
    z = np.array([-3.0, 0.0, 40.0])
    first = Geodesy().to_ecef(x, y, z)
    second = Geodesy().to_ecef(x, y, z)
    for a, b in zip(first, second, strict=True):
        assert np.array_equal(a, b)


def test_to_geodetic_from_ecef_inverts_to_ecef() -> None:
    """ECEF -> geodetic reproduces the direct RD/NAP -> geodetic result.

    The dequantized-position containment check reconstructs a game
    tile's vertices in ECEF and needs their region coordinates; going
    ECEF -> EPSG:4979 must agree with the direct EPSG:7415 -> EPSG:4979
    pipeline to within pyproj's round-trip epsilon.
    """
    geodesy = Geodesy()
    x = np.array([_RD_X, _RD_X + 40.0])
    y = np.array([_RD_Y, _RD_Y - 30.0])
    z = np.array([-2.0, 25.0])
    direct = geodesy.to_geodetic_radians(x, y, z)
    ex, ey, ez = geodesy.to_ecef(x, y, z)
    via_ecef = geodesy.to_geodetic_from_ecef(ex, ey, ez)
    for a, b in zip(direct, via_ecef, strict=True):
        assert a.shape == (2,)
        assert a.dtype == np.float64
        assert np.allclose(a, b, rtol=0.0, atol=1e-6)


def test_ecef_and_geodetic_paths_agree() -> None:
    """The two pyproj pipelines describe the same place.

    Converting the geodetic result back to ECEF by hand (spherical
    formula on the WGS84 ellipsoid) must land within metres of the
    direct ECEF result — a coarse but pipeline-independent consistency
    check.
    """
    geodesy = Geodesy()
    x = np.array([_RD_X])
    y = np.array([_RD_Y])
    z = np.array([0.0])
    ex, ey, ez = geodesy.to_ecef(x, y, z)
    lon, lat, h = geodesy.to_geodetic_radians(x, y, z)
    a = 6_378_137.0
    e2 = 6.694379990141e-3
    n = a / np.sqrt(1.0 - e2 * np.sin(lat) ** 2)
    px = (n + h) * np.cos(lat) * np.cos(lon)
    py = (n + h) * np.cos(lat) * np.sin(lon)
    pz = (n * (1.0 - e2) + h) * np.sin(lat)
    assert abs(px[0] - ex[0]) < 5.0
    assert abs(py[0] - ey[0]) < 5.0
    assert abs(pz[0] - ez[0]) < 5.0
