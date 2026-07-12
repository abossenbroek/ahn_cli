"""Stride-sampled RTC tile meshes in glTF's y-up frame.

:func:`build_tile_mesh` samples a tile's pixel span at its LOD stride
(every vertex is a genuine source pixel), transforms the samples to
ECEF, and stores them **relative-to-centre** (RTC): the tile's centre
becomes the glTF node ``translation`` (kept float64) and the vertices
become float32 offsets, preserving millimetre precision on an
Earth-sized frame. Both are swizzled into glTF's y-up axes
(``(x, y, z)_ecef -> (x, z, -y)_gltf``) so the runtime's mandated
y-up -> z-up rotation (about x by pi/2) reproduces ECEF exactly. The
``region`` is the exact EPSG:4979 envelope of the tile's own vertices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ahn_cli.tiles3d.quadtree import sample_indices

if TYPE_CHECKING:
    import numpy.typing as npt

    from ahn_cli.tiles3d.geodesy import Geodesy
    from ahn_cli.tiles3d.quadtree import TilePlan
    from ahn_cli.tiles3d.sources import TerrainGrid

__all__ = ["TileMesh", "build_tile_mesh"]

Region = tuple[float, float, float, float, float, float]
"""A 3D Tiles region: (west, south, east, north, minH, maxH); radians."""


@dataclass(frozen=True, eq=False)
class TileMesh:
    """One tile's mesh, ready for glb packing.

    Contract (fields):
        - ``positions``: ``(n, 3)`` float32 glTF y-up RTC vertices.
        - ``uvs``: ``(n, 2)`` float32 texel-centre texture coordinates.
        - ``indices``: ``(t * 3,)`` uint32 triangle list.
        - ``center``: the RTC node translation in the glTF frame
          (float64 precision).
        - ``region``: the exact EPSG:4979 envelope of these vertices.
        - ``cols``/``rows``: the sampled source pixel indices.

    ``eq=False``: wraps large arrays, so instances compare by identity.
    """

    positions: npt.NDArray[np.float32]
    uvs: npt.NDArray[np.float32]
    indices: npt.NDArray[np.uint32]
    center: tuple[float, float, float]
    region: Region
    cols: npt.NDArray[np.int64]
    rows: npt.NDArray[np.int64]


def build_tile_mesh(
    terrain: TerrainGrid, tile: TilePlan, geodesy: Geodesy
) -> TileMesh:
    """Build the mesh for one planned tile from the verified terrain.

    Contract:
        - Samples ``tile``'s span at its stride (genuine pixels only),
          builds the RTC float32 vertex grid, texel-centre UVs and a
          two-triangles-per-cell index list, and computes the exact
          geodetic region of the sampled vertices.

    Invariants:
        - Deterministic; pure function of its inputs.
    """
    cols = sample_indices(tile.col0, tile.col1, tile.stride)
    rows = sample_indices(tile.row0, tile.row1, tile.stride)
    grid = np.ix_(rows, cols)
    x = terrain.x[grid].astype(np.float64).ravel()
    y = terrain.y[grid].astype(np.float64).ravel()
    z = terrain.z[grid].astype(np.float64).ravel()

    ex, ey, ez = geodesy.to_ecef(x, y, z)
    center_ecef = tuple(
        (float(axis.min()) + float(axis.max())) / 2.0 for axis in (ex, ey, ez)
    )
    rel_x = ex - center_ecef[0]
    rel_y = ey - center_ecef[1]
    rel_z = ez - center_ecef[2]
    positions = np.column_stack([rel_x, rel_z, -rel_y]).astype(np.float32)
    center = (center_ecef[0], center_ecef[2], -center_ecef[1])

    uvs = _texel_centre_uvs(len(cols), len(rows))
    indices = _grid_triangles(len(cols), len(rows))

    lon, lat, height = geodesy.to_geodetic_radians(x, y, z)
    region: Region = (
        float(lon.min()),
        float(lat.min()),
        float(lon.max()),
        float(lat.max()),
        float(height.min()),
        float(height.max()),
    )
    return TileMesh(
        positions=positions,
        uvs=uvs,
        indices=indices,
        center=center,
        region=region,
        cols=cols,
        rows=rows,
    )


def _texel_centre_uvs(n_cols: int, n_rows: int) -> npt.NDArray[np.float32]:
    """Map vertex ``(j, i)`` onto the centre of texel ``(j, i)``."""
    u = (np.arange(n_cols, dtype=np.float64) + 0.5) / n_cols
    v = (np.arange(n_rows, dtype=np.float64) + 0.5) / n_rows
    uu, vv = np.meshgrid(u, v)
    return np.column_stack([uu.ravel(), vv.ravel()]).astype(np.float32)


def _grid_triangles(n_cols: int, n_rows: int) -> npt.NDArray[np.uint32]:
    """Build the two-triangles-per-cell index list, row-major vertices."""
    cell_col = np.arange(n_cols - 1, dtype=np.uint32)
    cell_row = np.arange(n_rows - 1, dtype=np.uint32)
    cc, rr = np.meshgrid(cell_col, cell_row)
    a = (rr * n_cols + cc).ravel()
    b = a + 1
    c = a + n_cols
    d = c + 1
    triangles = np.column_stack([a, c, d, a, d, b])
    return triangles.reshape(-1).astype(np.uint32)
