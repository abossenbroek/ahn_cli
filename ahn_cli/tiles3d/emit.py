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
from ahn_cli.tiles3d.gltf import build_glb
from ahn_cli.tiles3d.mesh import build_tile_mesh
from ahn_cli.tiles3d.png import encode_png
from ahn_cli.tiles3d.quadtree import geometric_error
from ahn_cli.tiles3d.tileset import (
    tile_entry,
    tileset_document,
    union_region,
)

if TYPE_CHECKING:
    from ahn_cli.tiles3d.mesh import Region
    from ahn_cli.tiles3d.quadtree import TilePlan, TreePlan
    from ahn_cli.tiles3d.sources import TerrainGrid

__all__ = ["ComputedBuild", "compute_build", "pixel_size", "tile_uri"]

ProgressCallback = Callable[[int, int], None]

TILES_SUBDIR = "tiles"
TILESET_NAME = "tileset.json"
_LEAF_ERROR_FACTOR = 4.0


@dataclass(frozen=True, eq=False)
class ComputedBuild:
    """Every artifact of one build, keyed by output-relative path.

    Contract (fields):
        - ``glbs``: ``{relative uri: glb bytes}`` for every tile.
        - ``document``: the tileset.json document (pre-serialisation).
        - ``vertices`` / ``triangles``: totals across every tile.
    """

    glbs: dict[str, bytes]
    document: dict[str, object]
    vertices: int
    triangles: int


def tile_uri(tile: TilePlan) -> str:
    """Return the tile's output-relative content uri."""
    return f"{TILES_SUBDIR}/{tile.level}-{tile.tx}-{tile.ty}.glb"


def pixel_size(terrain: TerrainGrid) -> float:
    """Return the grid's ground pixel size in metres (the larger axis)."""
    return max(abs(terrain.transform[0]), abs(terrain.transform[4]))


def compute_build(
    terrain: TerrainGrid,
    tree: TreePlan,
    *,
    progress: ProgressCallback | None = None,
) -> ComputedBuild:
    """Derive every glb and the tileset document, children first."""
    emitter = _Emitter(terrain, tree, progress)
    root_entry, _, root_error = emitter.emit(tree.root)
    tileset_error = (
        2.0 * root_error
        if root_error > 0.0
        else pixel_size(terrain) * _LEAF_ERROR_FACTOR
    )
    return ComputedBuild(
        glbs=emitter.glbs,
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
        progress: ProgressCallback | None,
    ) -> None:
        self._terrain = terrain
        self._tree = tree
        self._progress = progress
        self._geodesy = Geodesy()
        self._done = 0
        self.glbs: dict[str, bytes] = {}
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
        sampled = self._terrain.rgb[np.ix_(mesh.rows, mesh.cols)]
        uri = tile_uri(tile)
        self.glbs[uri] = build_glb(mesh, encode_png(sampled))
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
        error = geometric_error(tile.stride, pixel_size(self._terrain))
        entry = tile_entry(
            region,
            error,
            uri,
            child_entries,
            root=tile.level == 0,
        )
        return entry, region, error
