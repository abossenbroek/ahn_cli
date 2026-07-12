"""Pure tileset emission: verified terrain + plan -> bytes in memory.

:func:`compute_build` deterministically derives every artifact of a
3D Tiles build — one glb per quadtree tile plus the ``tileset.json``
bytes — without touching disk. ``build`` writes exactly these bytes;
``verify`` recomputes them from fresh source reads and demands the
on-disk files match byte for byte. Parents are emitted after their
children so every parent's region is the union of its own vertices and
all descendant regions (content containment by construction).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ahn_cli.tiles3d.geodesy import Geodesy
from ahn_cli.tiles3d.mesh import build_tile_mesh
from ahn_cli.tiles3d.payload import TilePayload
from ahn_cli.tiles3d.quadtree import geometric_error
from ahn_cli.tiles3d.tileset import (
    tile_entry,
    tileset_document,
    union_region,
)

if TYPE_CHECKING:
    from ahn_cli.tiles3d.mesh import Region
    from ahn_cli.tiles3d.payload import TileEncoder
    from ahn_cli.tiles3d.profile import Profile
    from ahn_cli.tiles3d.quadtree import TilePlan, TreePlan
    from ahn_cli.tiles3d.sources import TerrainGrid

__all__ = [
    "ComputedBuild",
    "compute_build",
    "pixel_size",
    "texture_uri",
    "tile_uri",
]

ProgressCallback = Callable[[int, int], None]

TILES_SUBDIR = "tiles"
TILESET_NAME = "tileset.json"
_LEAF_ERROR_FACTOR = 4.0


@dataclass(frozen=True, eq=False)
class ComputedBuild:
    """Every artifact of one build, keyed by output-relative path.

    Contract (fields):
        - ``glbs``: ``{relative uri: content bytes}`` for every tile
          (``.glb`` for the glTF profiles, ``.hf`` for heightfield).
        - ``textures``: ``{relative uri: texture bytes}`` for the profiles
          that write a sibling texture file (empty for the embedded-texture
          glTF profiles).
        - ``document``: the tileset.json document (pre-serialisation).
        - ``vertices`` / ``triangles``: totals across every tile.
    """

    glbs: dict[str, bytes]
    textures: dict[str, bytes]
    document: dict[str, object]
    vertices: int
    triangles: int


def tile_uri(tile: TilePlan, profile: Profile) -> str:
    """Return the tile's output-relative content uri for ``profile``."""
    base = f"{tile.level}-{tile.tx}-{tile.ty}"
    return f"{TILES_SUBDIR}/{base}{profile.content_suffix()}"


def texture_uri(tile: TilePlan, profile: Profile) -> str | None:
    """Return the tile's sibling texture uri, or ``None`` when embedded."""
    suffix = profile.texture_suffix()
    if suffix is None:
        return None
    base = f"{tile.level}-{tile.tx}-{tile.ty}"
    return f"{TILES_SUBDIR}/{base}{suffix}"


def pixel_size(terrain: TerrainGrid) -> float:
    """Return the grid's ground pixel size in metres (the larger axis)."""
    return max(abs(terrain.transform[0]), abs(terrain.transform[4]))


def compute_build(
    terrain: TerrainGrid,
    tree: TreePlan,
    *,
    encoder: TileEncoder,
    progress: ProgressCallback | None = None,
) -> ComputedBuild:
    """Derive every glb and the tileset document, children first.

    ``encoder`` is the profile's :class:`TileEncoder`; emission stays
    agnostic to the on-disk representation and only drives the protocol.
    """
    emitter = _Emitter(terrain, tree, encoder, progress)
    root_entry, _, root_error = emitter.emit(tree.root)
    tileset_error = (
        2.0 * root_error
        if root_error > 0.0
        else pixel_size(terrain) * _LEAF_ERROR_FACTOR
    )
    return ComputedBuild(
        glbs=emitter.glbs,
        textures=emitter.textures,
        document=tileset_document(root_entry, tileset_error),
        vertices=emitter.vertices,
        triangles=emitter.triangles,
    )


class _Emitter:
    """Depth-first, children-first in-memory tile emitter."""

    def __init__(
        self,
        terrain: TerrainGrid,
        tree: TreePlan,
        encoder: TileEncoder,
        progress: ProgressCallback | None,
    ) -> None:
        self._terrain = terrain
        self._tree = tree
        self._progress = progress
        self._geodesy = Geodesy()
        self._encoder = encoder
        self._done = 0
        self.glbs: dict[str, bytes] = {}
        self.textures: dict[str, bytes] = {}
        self.vertices = 0
        self.triangles = 0

    def emit(self, tile: TilePlan) -> tuple[dict[str, object], Region, float]:
        """Emit ``tile``'s subtree; return its entry, region and error."""
        child_entries: list[dict[str, object]] = []
        region: Region | None = None
        for child in tile.children:
            entry, child_region, _ = self.emit(child)
            child_entries.append(entry)
            region = (
                child_region
                if region is None
                else union_region(region, child_region)
            )
        mesh = build_tile_mesh(self._terrain, tile, self._geodesy)
        grid = np.ix_(mesh.rows, mesh.cols)
        error = geometric_error(tile.stride, pixel_size(self._terrain))
        payload = TilePayload(
            level=tile.level,
            tx=tile.tx,
            ty=tile.ty,
            stride=tile.stride,
            geometric_error=error,
            mesh=mesh,
            x=self._terrain.x[grid],
            y=self._terrain.y[grid],
            z=self._terrain.z[grid],
            rgb=self._terrain.rgb[grid],
        )
        encoded = self._encoder.encode(payload)
        uri = f"{TILES_SUBDIR}/{encoded.content_name}"
        self.glbs[uri] = encoded.content
        if encoded.texture is not None:
            self.textures[f"{TILES_SUBDIR}/{encoded.texture_name}"] = (
                encoded.texture
            )
        self.vertices += int(mesh.positions.shape[0])
        self.triangles += int(mesh.indices.shape[0]) // 3
        self._done += 1
        if self._progress is not None:
            self._progress(self._done, self._tree.tile_count)
        region = (
            mesh.region
            if region is None
            else union_region(region, mesh.region)
        )
        entry = tile_entry(
            region,
            error,
            uri,
            child_entries,
            root=tile.level == 0,
        )
        return entry, region, error
