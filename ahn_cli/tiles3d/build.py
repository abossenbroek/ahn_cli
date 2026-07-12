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
verified deliverable is never destroyed by a failed rebuild. The swap
is two-phase with an accept-marker file as its commit point, so even
a hard kill (SIGKILL, power loss) at any moment leaves a state the
next run recovers: marker present means the artifacts in place are
verified; marker absent means whatever the backup holds is the last
verified deliverable and is put back first.
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

ACCEPT_MARKER = ".tiles3d-accepted"
"""Commit point of the swap: present only between verify and drop."""


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
    marker = out / ACCEPT_MARKER
    accepted = False
    try:
        _recover(tiles_dir, tileset_path, backup_dir, marker)
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
            try:
                _drop_backup(backup_dir, marker)
            except OSError as exc:
                msg = (
                    f"3D Tiles output at {out} could not drop the held "
                    f"previous deliverable: {exc}"
                )
                raise Tiles3dError(msg) from exc
        else:
            try:
                _discard(written, tiles_dir)
                _restore_stale(tiles_dir, tileset_path, backup_dir)
            except OSError as exc:
                msg = (
                    f"3D Tiles output at {out} could not restore the "
                    f"held previous deliverable: {exc}"
                )
                raise Tiles3dError(msg) from exc
    return Tiles3dBuildResult(
        tileset_path=tileset_path,
        tile_count=tree.tile_count,
        levels=tree.levels,
        vertices=computed.vertices,
        triangles=computed.triangles,
    )


def _recover(
    tiles_dir: Path, tileset_path: Path, backup_dir: Path, marker: Path
) -> None:
    """Reconcile the leftovers of a hard-killed earlier rebuild.

    The accept marker is the commit point of the two-phase swap. With
    it present, the artifacts in ``out`` are a verified build and the
    held copy (if any survives) is disposable. Without it, whatever
    the backup holds is the last verified deliverable and the matching
    artifacts in ``out`` are the killed run's unverified partial
    write: per artifact, the held copy wins.
    """
    if marker.exists():
        if backup_dir.is_dir():
            shutil.rmtree(backup_dir)
        marker.unlink()
        return
    _restore_stale(tiles_dir, tileset_path, backup_dir)


def _hold_stale(
    tiles_dir: Path, tileset_path: Path, backup_dir: Path
) -> None:
    """Move a previous build's artifacts aside instead of deleting them."""
    stale = [p for p in (tiles_dir, tileset_path) if p.exists()]
    if not stale:
        return
    backup_dir.mkdir(parents=True)
    for path in stale:
        path.rename(backup_dir / path.name)


def _drop_backup(backup_dir: Path, marker: Path) -> None:
    """Two-phase drop: mark the build accepted, drop the held copy.

    A hard kill anywhere in here is recovered by :func:`_recover`: the
    marker tells the next run the artifacts in ``out`` are verified.
    """
    marker.touch()
    if backup_dir.is_dir():
        shutil.rmtree(backup_dir)
    marker.unlink()


def _restore_stale(
    tiles_dir: Path, tileset_path: Path, backup_dir: Path
) -> None:
    """Move a held-aside previous deliverable back into place.

    Per artifact, the held copy wins: any ``tiles/`` or
    ``tileset.json`` still in ``out`` is an unverified leftover of the
    failed or killed run holding the backup, and is removed before the
    verified copy is renamed back. An artifact absent from the backup
    keeps the copy in ``out`` (a kill between the two hold renames
    leaves the not-yet-held, still-verified artifact in place).
    """
    if not backup_dir.is_dir():
        return
    held_tiles = backup_dir / TILES_SUBDIR
    if held_tiles.is_dir():
        if tiles_dir.is_dir():
            shutil.rmtree(tiles_dir)
        held_tiles.rename(tiles_dir)
    held_tileset = backup_dir / TILESET_NAME
    if held_tileset.is_file():
        tileset_path.unlink(missing_ok=True)
        held_tileset.rename(tileset_path)
    backup_dir.rmdir()


def _discard(written: list[Path], tiles_dir: Path) -> None:
    """Remove everything a rejected build wrote (never leave stale).

    ``tiles_dir`` is removed wholesale: when this runs, it holds only
    the rejected run's output — including any partial file a failed
    write created without ever being tracked in ``written``.
    """
    for path in written:
        path.unlink(missing_ok=True)
    if tiles_dir.is_dir():
        shutil.rmtree(tiles_dir)
