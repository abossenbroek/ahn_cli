"""The tiles3d build orchestrator: verified terrain -> 3D Tiles 1.1.

:func:`build_tiles3d` loads the perfectly matched ortho + heights pair,
plans the quadtree, computes every artifact in memory
(:mod:`ahn_cli.tiles3d.emit`), writes them, and finally runs the strict
post-write verifier (:mod:`ahn_cli.tiles3d.verify`) against fresh disk
reads — a build is only accepted once everything on disk survives it.
The whole grid is held in memory: the inputs are one site's ortho, not
a nationwide mosaic.

A failed build leaves nothing behind: every path written so far is
removed before the error propagates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ahn_cli.tiles3d.emit import (
    TILES_SUBDIR,
    TILESET_NAME,
    ProgressCallback,
    compute_build,
)
from ahn_cli.tiles3d.errors import Tiles3dError
from ahn_cli.tiles3d.quadtree import plan_quadtree
from ahn_cli.tiles3d.sources import load_terrain
from ahn_cli.tiles3d.tileset import write_tileset
from ahn_cli.tiles3d.verify import verify_tiles3d

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["ProgressCallback", "Tiles3dBuildResult", "build_tiles3d"]


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
          ``<out>/tiles/<level>-<tx>-<ty>.glb`` per quadtree tile, then
          hard-verifies everything written (strict re-read +
          independent recomputation) before returning.
        - Calls ``progress(tiles_done, tile_total)`` per computed tile.
        - Returns a :class:`Tiles3dBuildResult`.

    Invariants:
        - Deterministic per machine (see geodesy caveat); a failed or
          verification-rejected build removes every written output
          before raising.

    Failure modes:
        - :class:`Tiles3dError` for every input gate
          (:func:`load_terrain`, :func:`plan_quadtree`), an unwritable
          output location, and any post-write verification failure.
    """
    terrain = load_terrain(ortho, heights)
    tree = plan_quadtree(terrain.width, terrain.height, tile_pixels)
    computed = compute_build(terrain, tree, progress=progress)
    written: list[Path] = []
    tiles_dir = out / TILES_SUBDIR
    tileset_path = out / TILESET_NAME
    try:
        tiles_dir.mkdir(parents=True, exist_ok=True)
        for uri, data in computed.glbs.items():
            path = out / uri
            path.write_bytes(data)
            written.append(path)
        write_tileset(computed.document, tileset_path)
        written.append(tileset_path)
        verify_tiles3d(out, ortho, heights, tile_pixels=tile_pixels)
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
        vertices=computed.vertices,
        triangles=computed.triangles,
    )


def _discard(written: list[Path], tiles_dir: Path) -> None:
    """Remove everything a rejected build wrote (never leave stale)."""
    for path in written:
        path.unlink(missing_ok=True)
    if tiles_dir.is_dir() and not any(tiles_dir.iterdir()):
        tiles_dir.rmdir()
