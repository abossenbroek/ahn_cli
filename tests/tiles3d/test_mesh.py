"""Tests for the stride-sampled RTC tile meshes."""

from __future__ import annotations

import numpy as np

from ahn_cli.tiles3d.geodesy import Geodesy
from ahn_cli.tiles3d.mesh import build_tile_mesh
from ahn_cli.tiles3d.quadtree import plan_quadtree
from tests.tiles3d.conftest import make_terrain


def test_leaf_mesh_counts_and_uvs() -> None:
    """A 6x6 leaf yields a full 36-vertex, 50-triangle grid mesh."""
    terrain = make_terrain(6, 6)
    tile = plan_quadtree(6, 6).root
    mesh = build_tile_mesh(terrain, tile, Geodesy())
    assert mesh.positions.shape == (36, 3)
    assert mesh.positions.dtype == np.float32
    assert mesh.uvs.shape == (36, 2)
    assert mesh.indices.shape == (2 * 5 * 5 * 3,)
    assert mesh.indices.dtype == np.uint32
    assert int(mesh.indices.max()) == 35
    assert mesh.uvs[0, 0] == np.float32(0.5 / 6)
    assert mesh.uvs[0, 1] == np.float32(0.5 / 6)
    assert np.all(mesh.uvs > 0.0)
    assert np.all(mesh.uvs < 1.0)
    assert mesh.cols.tolist() == [0, 1, 2, 3, 4, 5]
    assert mesh.rows.tolist() == [0, 1, 2, 3, 4, 5]


def test_strided_root_samples_genuine_pixels() -> None:
    """A stride-4 root samples every 4th pixel plus the last one."""
    terrain = make_terrain(600, 600)
    root = plan_quadtree(600, 600).root
    mesh = build_tile_mesh(terrain, root, Geodesy())
    assert root.stride == 4
    assert mesh.cols.tolist() == [*range(0, 600, 4), 599]
    assert mesh.positions.shape == (151 * 151, 3)


def test_positions_reproduce_ecef_after_runtime_rotation() -> None:
    """Rebuild ECEF exactly from y-up positions + node translation.

    The runtime rotates the glTF scene by pi/2 about x:
    ``(x, y, z) -> (x, -z, y)``. Applying that to ``center + position``
    must land on the independently computed ECEF coordinates.
    """
    terrain = make_terrain(8, 5)
    tile = plan_quadtree(8, 5).root
    geodesy = Geodesy()
    mesh = build_tile_mesh(terrain, tile, geodesy)
    world = mesh.positions.astype(np.float64) + np.array(mesh.center)
    ecef_x = world[:, 0]
    ecef_y = -world[:, 2]
    ecef_z = world[:, 1]
    ex, ey, ez = geodesy.to_ecef(
        terrain.x.astype(np.float64).ravel(),
        terrain.y.astype(np.float64).ravel(),
        terrain.z.astype(np.float64).ravel(),
    )
    assert np.allclose(ecef_x, ex, atol=1e-3)
    assert np.allclose(ecef_y, ey, atol=1e-3)
    assert np.allclose(ecef_z, ez, atol=1e-3)


def test_region_is_the_exact_geodetic_envelope() -> None:
    """The region equals the min/max of the tile's geodetic vertices."""
    terrain = make_terrain(7, 6)
    tile = plan_quadtree(7, 6).root
    geodesy = Geodesy()
    mesh = build_tile_mesh(terrain, tile, geodesy)
    lon, lat, height = geodesy.to_geodetic_radians(
        terrain.x.astype(np.float64).ravel(),
        terrain.y.astype(np.float64).ravel(),
        terrain.z.astype(np.float64).ravel(),
    )
    assert mesh.region == (
        float(lon.min()),
        float(lat.min()),
        float(lon.max()),
        float(lat.max()),
        float(height.min()),
        float(height.max()),
    )
    west, south, east, north, low, high = mesh.region
    assert west < east
    assert south < north
    assert low < high


def test_mesh_is_deterministic() -> None:
    """Two builds agree bit for bit."""
    terrain = make_terrain(9, 4)
    tile = plan_quadtree(9, 4).root
    first = build_tile_mesh(terrain, tile, Geodesy())
    second = build_tile_mesh(terrain, tile, Geodesy())
    assert np.array_equal(first.positions, second.positions)
    assert np.array_equal(first.uvs, second.uvs)
    assert np.array_equal(first.indices, second.indices)
    assert first.center == second.center
    assert first.region == second.region


def test_triangles_are_non_degenerate() -> None:
    """Every triangle references three distinct vertices."""
    terrain = make_terrain(6, 6)
    mesh = build_tile_mesh(terrain, plan_quadtree(6, 6).root, Geodesy())
    triangles = mesh.indices.reshape(-1, 3)
    for tri in triangles:
        assert len(set(tri.tolist())) == 3
