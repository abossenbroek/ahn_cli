"""The tiles3d build orchestrator: verified terrain -> 3D Tiles 1.1.

:func:`build_tiles3d` loads the perfectly matched ortho + heights pair,
plans the quadtree, then walks it depth-first (children before parent,
so every parent's region is the union of its own vertices and all
descendants — content containment by construction) writing one glb per
tile and finally ``tileset.json``. The whole grid is held in memory:
the inputs are one site's ortho, not a nationwide mosaic.

A failed build leaves nothing behind: every path written so far is
removed before the error propagates.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ahn_cli.tiles3d.errors import Tiles3dError
from ahn_cli.tiles3d.geodesy import Geodesy
from ahn_cli.tiles3d.gltf import build_glb
from ahn_cli.tiles3d.mesh import build_tile_mesh
from ahn_cli.tiles3d.png import encode_png
from ahn_cli.tiles3d.quadtree import geometric_error, plan_quadtree
from ahn_cli.tiles3d.sources import load_terrain
from ahn_cli.tiles3d.tileset import (
    tile_entry,
    tileset_document,
    union_region,
    write_tileset,
)

if TYPE_CHECKING:
    from pathlib import Path

    from ahn_cli.tiles3d.mesh import Region
    from ahn_cli.tiles3d.quadtree import TilePlan, TreePlan
    from ahn_cli.tiles3d.sources import TerrainGrid

__all__ = ["ProgressCallback", "Tiles3dBuildResult", "build_tiles3d"]

ProgressCallback = Callable[[int, int], None]
"""An injected progress reporter: called ``(tiles_done, tile_total)``
once per written tile."""

_TILES_SUBDIR = "tiles"
_TILESET_NAME = "tileset.json"
_LEAF_ERROR_FACTOR = 4.0


def _no_op_progress(_done: int, _total: int) -> None:
    """Report nothing; the default when the caller supplies no callback."""


@dataclass(frozen=True)
class Tiles3dBuildResult:
    """The ledger of one tiles3d build.

    Contract (fields):
        - ``tileset_path``: the written ``tileset.json``.
        - ``tile_count`` / ``levels``: the quadtree's shape.
        - ``vertices`` / ``triangles``: totals across every tile.

    Invariants:
        - Frozen value object, equal by field value.
    """

    tileset_path: Path
    tile_count: int
    levels: int
    vertices: int
    triangles: int


def build_tiles3d(
    ortho: Path,
    heights: Path,
    out: Path,
    *,
    tile_pixels: int = 256,
    progress: ProgressCallback | None = None,
) -> Tiles3dBuildResult:
    """Convert the ortho map + reconciled heights into 3D Tiles 1.1.

    Contract:
        - Writes ``<out>/tileset.json`` and one
          ``<out>/tiles/<level>-<tx>-<ty>.glb`` per quadtree tile.
        - Calls ``progress(tiles_done, tile_total)`` after each tile.
        - Returns a :class:`Tiles3dBuildResult`.

    Invariants:
        - Deterministic per machine (see geodesy caveat); a failed
          build removes every partially written output before raising.

    Failure modes:
        - :class:`Tiles3dError` for every input gate
          (:func:`load_terrain`, :func:`plan_quadtree`) and for an
          unwritable output location.
    """
    report = progress if progress is not None else _no_op_progress
    terrain = load_terrain(ortho, heights)
    tree = plan_quadtree(terrain.width, terrain.height, tile_pixels)
    written: list[Path] = []
    tiles_dir = out / _TILES_SUBDIR
    try:
        tiles_dir.mkdir(parents=True, exist_ok=True)
        emitter = _TileEmitter(
            terrain=terrain,
            tree=tree,
            tiles_dir=tiles_dir,
            written=written,
            report=report,
        )
        root_entry, _, root_error = emitter.emit(tree.root)
        pixel_size = _pixel_size(terrain)
        tileset_error = (
            2.0 * root_error
            if root_error > 0.0
            else pixel_size * _LEAF_ERROR_FACTOR
        )
        tileset_path = out / _TILESET_NAME
        write_tileset(
            tileset_document(root_entry, tileset_error), tileset_path
        )
        written.append(tileset_path)
    except (Tiles3dError, OSError) as exc:
        _discard(written, tiles_dir)
        if isinstance(exc, Tiles3dError):
            raise
        msg = f"3D Tiles output at {out} is not writable: {exc}"
        raise Tiles3dError(msg) from exc
    return Tiles3dBuildResult(
        tileset_path=tileset_path,
        tile_count=tree.tile_count,
        levels=tree.levels,
        vertices=emitter.vertices,
        triangles=emitter.triangles,
    )


def _pixel_size(terrain: TerrainGrid) -> float:
    """Return the grid's ground pixel size in metres (the larger axis)."""
    return max(abs(terrain.transform[0]), abs(terrain.transform[4]))


class _TileEmitter:
    """Depth-first, children-first tile writer over one build."""

    def __init__(
        self,
        *,
        terrain: TerrainGrid,
        tree: TreePlan,
        tiles_dir: Path,
        written: list[Path],
        report: ProgressCallback,
    ) -> None:
        self._terrain = terrain
        self._tree = tree
        self._tiles_dir = tiles_dir
        self._written = written
        self._report = report
        self._geodesy = Geodesy()
        self._done = 0
        self.vertices = 0
        self.triangles = 0

    def emit(self, tile: TilePlan) -> tuple[dict[str, object], Region, float]:
        """Write ``tile``'s subtree; return its entry, region and error."""
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
        glb = build_glb(mesh, encode_png(sampled))
        name = f"{tile.level}-{tile.tx}-{tile.ty}.glb"
        path = self._tiles_dir / name
        path.write_bytes(glb)
        self._written.append(path)
        self.vertices += int(mesh.positions.shape[0])
        self.triangles += int(mesh.indices.shape[0]) // 3
        self._done += 1
        self._report(self._done, self._tree.tile_count)
        region = (
            mesh.region
            if region is None
            else union_region(region, mesh.region)
        )
        error = geometric_error(tile.stride, _pixel_size(self._terrain))
        entry = tile_entry(
            region,
            error,
            f"{_TILES_SUBDIR}/{name}",
            child_entries,
            root=tile.level == 0,
        )
        return entry, region, error


def _discard(written: list[Path], tiles_dir: Path) -> None:
    """Remove everything a rejected build wrote (never leave stale)."""
    for path in written:
        path.unlink(missing_ok=True)
    if tiles_dir.is_dir() and not any(tiles_dir.iterdir()):
        tiles_dir.rmdir()
