"""The tiles3d build orchestrator: verified terrain -> 3D Tiles 1.1.

:func:`build_tiles3d` loads the perfectly matched ortho + heights pair,
plans the quadtree, computes every artifact in memory
(:mod:`ahn_cli.tiles3d.emit`), writes them, and finally runs the strict
post-write verifier (:mod:`ahn_cli.tiles3d.verify`) against fresh disk
reads — a build is only accepted once everything on disk survives it.
The whole grid is held in memory: the inputs are one site's ortho, not
a nationwide mosaic.

A failed build leaves nothing behind: every path written so far is
removed before the error propagates. Re-runs into the same output
directory are safe in both directions: a previous build's artifacts
(``tileset.json`` and the ``tiles/`` subtree, both tool-owned) are
held aside in a scratch directory while the new build is written and
verified, dropped only once the new build is accepted, and moved back
into place when the new build fails for any reason — a previously
verified deliverable is never destroyed by a failed rebuild.
"""

from __future__ import annotations

import shutil
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

BACKUP_SUBDIR = ".tiles3d-backup"
"""Tool-owned scratch holding the previous deliverable during a rebuild."""


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
        - A previous build's artifacts in ``out`` (the tool-owned
          ``tileset.json`` and ``tiles/`` subtree) are replaced only
          once the new build has passed verification: they are held
          aside during the rebuild and moved back into place if the
          rebuild fails for any reason. An input-gate failure never
          touches them at all.

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
    backup_dir = out / BACKUP_SUBDIR
    accepted = False
    try:
        _hold_stale(tiles_dir, tileset_path, backup_dir)
        tiles_dir.mkdir(parents=True, exist_ok=True)
        for uri, data in computed.glbs.items():
            path = out / uri
            path.write_bytes(data)
            written.append(path)
        write_tileset(computed.document, tileset_path)
        written.append(tileset_path)
        verify_tiles3d(out, ortho, heights, tile_pixels=tile_pixels)
        accepted = True
    except OSError as exc:
        msg = f"3D Tiles output at {out} is not writable: {exc}"
        raise Tiles3dError(msg) from exc
    finally:
        if accepted:
            shutil.rmtree(backup_dir, ignore_errors=True)
        else:
            _discard(written, tiles_dir)
            _restore_stale(out, backup_dir)
    return Tiles3dBuildResult(
        tileset_path=tileset_path,
        tile_count=tree.tile_count,
        levels=tree.levels,
        vertices=computed.vertices,
        triangles=computed.triangles,
    )


def _hold_stale(
    tiles_dir: Path, tileset_path: Path, backup_dir: Path
) -> None:
    """Move a previous build's artifacts aside instead of deleting them.

    ``backup_dir`` is tool-owned scratch: leftovers from a crashed
    earlier rebuild are stale by definition and removed first.
    """
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    stale = [p for p in (tiles_dir, tileset_path) if p.exists()]
    if not stale:
        return
    backup_dir.mkdir(parents=True)
    for path in stale:
        path.rename(backup_dir / path.name)


def _restore_stale(out: Path, backup_dir: Path) -> None:
    """Move a held-aside previous deliverable back into place."""
    if not backup_dir.is_dir():
        return
    for held in backup_dir.iterdir():
        held.rename(out / held.name)
    backup_dir.rmdir()


def _discard(written: list[Path], tiles_dir: Path) -> None:
    """Remove everything a rejected build wrote (never leave stale)."""
    for path in written:
        path.unlink(missing_ok=True)
    if tiles_dir.is_dir() and not any(tiles_dir.iterdir()):
        tiles_dir.rmdir()
