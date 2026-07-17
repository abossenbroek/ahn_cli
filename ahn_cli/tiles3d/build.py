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
directory are safe in both directions: a previous build's tool-owned
artifacts — under strict the ``tiles/`` subtree + ``tileset.json``; under
the packed lossy profiles ``tiles.hfp`` + ``tileset.json`` +
``provenance.json`` + ``manifest.json`` — are cleaned by a new build and
restored by a failed one, across profiles; whichever exist are held aside
in a scratch directory while the new build is written and verified,
dropped only once the new build is accepted, and moved back into place
when the new build fails for any reason — a previously verified
deliverable is never destroyed by a failed rebuild. The swap is two-phase
with an accept-marker file as its commit point, so even a hard kill
(SIGKILL, power loss) at any moment leaves a state the next run recovers:
marker present means the artifacts in place are verified; marker absent
means whatever the backup holds is the last verified deliverable and is
put back first.
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from ahn_cli.tiles3d.emit import (
    MANIFEST_NAME,
    TILES_HFP_NAME,
    TILES_SUBDIR,
    TILESET_NAME,
    ProgressCallback,
    compute_build,
    compute_packed_build,
)
from ahn_cli.tiles3d.errors import Tiles3dError
from ahn_cli.tiles3d.manifest import FileDigest, render_manifest
from ahn_cli.tiles3d.pack import require_free_disk, write_pack
from ahn_cli.tiles3d.parallel import (
    default_window,
    default_workers,
    ordered_encode,
)
from ahn_cli.tiles3d.profile import Profile
from ahn_cli.tiles3d.provenance import PROVENANCE_NAME, render_provenance
from ahn_cli.tiles3d.quadtree import plan_quadtree
from ahn_cli.tiles3d.sources import load_terrain
from ahn_cli.tiles3d.tileset import write_tileset
from ahn_cli.tiles3d.verify import verify_tiles3d

if TYPE_CHECKING:
    from pathlib import Path

    from ahn_cli.tiles3d.emit import ComputedBuild, PackedBuild
    from ahn_cli.tiles3d.parallel import PoolFactory

__all__ = ["ProgressCallback", "Tiles3dBuildResult", "build_tiles3d"]

BACKUP_SUBDIR = ".tiles3d-backup"
"""Tool-owned scratch holding the previous deliverable during a rebuild."""

ACCEPT_MARKER = ".tiles3d-accepted"
"""Commit point of the swap: present only between verify and drop."""

_HASH_CHUNK = 1 << 20
"""Read size for streaming a file's SHA-256 (bounded-memory manifest hash)."""


def _file_artifacts(out: Path) -> list[Path]:
    """Return the tool-owned flat-file artifacts, in a stable hold order.

    Spans every profile so a cross-profile rebuild holds/restores the right
    set: ``tileset.json`` (all profiles), ``provenance.json`` and — the packed
    lossy profiles only — ``tiles.hfp`` / ``manifest.json``. Whichever are
    absent are simply skipped by the swap.
    """
    return [
        out / TILESET_NAME,
        out / PROVENANCE_NAME,
        out / TILES_HFP_NAME,
        out / MANIFEST_NAME,
    ]


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
    profile: Profile = Profile.STRICT,
    progress: ProgressCallback | None = None,
    workers: int | None = None,
    pool_factory: PoolFactory | None = None,
) -> Tiles3dBuildResult:
    """Convert the ortho map + reconciled heights into 3D Tiles 1.1.

    Contract:
        - The **strict** profile writes ``<out>/tileset.json`` and one loose
          ``<out>/tiles/<level>-<tx>-<ty>.glb`` per tile (float32 glTF with an
          embedded PNG) and no sidecar (its output is byte-frozen).
        - The lossy **game** and **heightfield** profiles write a single
          ``<out>/tiles.hfp`` pack bundling every content blob plus the binary
          scene index (``.glb`` for game; a ``.hf`` chunk + a baseline ``.jpg``
          per tile for heightfield), a demoted ``<out>/tileset.json`` sidecar
          whose ``content.uri`` values are logical pack keys, a deterministic
          ``<out>/provenance.json`` (with the ``pack`` + ``producer`` blocks),
          and a ``<out>/manifest.json`` integrity sidecar over the three loose
          files plus the pack. No ``tiles/`` directory is written.
        - Everything written is hard-verified (strict re-read + independent
          recomputation) before returning.
        - Calls ``progress(tiles_done, tile_total)`` per computed tile.
        - ``workers`` (default the machine CPU count) fans the CPU-bound
          per-tile encode across a thread pool while the writer keeps
          streaming to disk in canonical order with a bounded window resident;
          ``workers=1`` is the serial reference and every worker count yields
          byte-identical output. ``pool_factory`` injects the pool (tests).
        - Returns a :class:`Tiles3dBuildResult`.

    Invariants:
        - Deterministic per machine (see geodesy caveat); a failed or
          verification-rejected build removes every written output
          before raising.
        - A previous build's tool-owned artifacts in ``out`` — under strict
          the ``tiles/`` subtree + ``tileset.json``; under the lossy profiles
          ``tiles.hfp`` + ``tileset.json`` + ``provenance.json`` +
          ``manifest.json`` — are replaced only once the new build has passed
          verification: whichever exist are held aside during the rebuild
          (spanning every profile, so cross-profile rebuilds neither strand
          nor misdescribe the old deliverable) and moved back into place if
          the rebuild fails for any reason. An input-gate failure never
          touches them at all.

    Failure modes:
        - :class:`Tiles3dError` for every input gate
          (:func:`load_terrain`, :func:`plan_quadtree`), an unwritable
          output location, and any post-write verification failure.
    """
    resolved_workers = default_workers() if workers is None else workers
    window = default_window(resolved_workers)
    terrain = load_terrain(ortho, heights)
    tree = plan_quadtree(terrain.width, terrain.height, tile_pixels)
    if profile is Profile.STRICT:
        computed = compute_build(
            terrain, tree, encoder=profile.encoder(), progress=progress
        )
        vertices, triangles = computed.vertices, computed.triangles

        def write_deliverable() -> None:
            _write_strict(
                out,
                computed,
                workers=resolved_workers,
                window=window,
                pool_factory=pool_factory,
            )
    else:
        packed = compute_packed_build(
            terrain, tree, profile=profile, progress=progress
        )
        vertices, triangles = packed.vertices, packed.triangles

        def write_deliverable() -> None:
            _write_packed(
                out,
                packed,
                profile,
                workers=resolved_workers,
                window=window,
                pool_factory=pool_factory,
            )

    tileset_path = out / TILESET_NAME
    backup_dir = out / BACKUP_SUBDIR
    marker = out / ACCEPT_MARKER
    accepted = False
    held = False
    try:
        _recover(out, backup_dir, marker)
        _hold_stale(out, backup_dir)
        held = True
        write_deliverable()
        verify_tiles3d(
            out, ortho, heights, tile_pixels=tile_pixels, profile=profile
        )
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
                if held:
                    _discard(out)
                _restore_stale(out, backup_dir, marker)
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
        vertices=vertices,
        triangles=triangles,
    )


def _write_strict(
    out: Path,
    computed: ComputedBuild,
    *,
    workers: int,
    window: int,
    pool_factory: PoolFactory | None,
) -> None:
    """Write the strict profile's loose ``tiles/`` glbs + ``tileset.json``.

    The per-tile encodes fan out through :func:`ordered_encode` (bounded
    window, byte-identical to the serial reference), while this loop drains
    them in emit order and streams each glb to disk under the free-disk floor.
    """
    (out / TILES_SUBDIR).mkdir(parents=True, exist_ok=True)
    blobs = ordered_encode(
        computed.order,
        computed.blob_source,
        workers=workers,
        window=window,
        pool_factory=pool_factory,
    )
    for key, (content, _texture) in zip(computed.order, blobs, strict=True):
        require_free_disk(out, len(content))
        (out / computed.uri_of[key]).write_bytes(content)
    write_tileset(computed.document, out / TILESET_NAME)


def _write_packed(
    out: Path,
    packed: PackedBuild,
    profile: Profile,
    *,
    workers: int,
    window: int,
    pool_factory: PoolFactory | None,
) -> None:
    """Write the packed lossy deliverable: pack, sidecars, manifest.

    The pack is streamed straight to ``tiles.hfp``: the per-tile encodes fan
    out through a bounded window while the writer keeps only that window
    resident, then the ``dataset_id`` it returns is embedded in
    ``provenance.json`` and the manifest ties the three loose files plus the
    pack to it. Every text sidecar is written as LF bytes on every platform.
    """
    out.mkdir(parents=True, exist_ok=True)
    hfp_path = out / TILES_HFP_NAME
    tileset_path = out / TILESET_NAME
    provenance_path = out / PROVENANCE_NAME
    manifest_path = out / MANIFEST_NAME
    dataset_id = write_pack(
        hfp_path,
        packed.entries,
        packed.blob_source,
        root_geometric_error=packed.root_geometric_error,
        content_kind=packed.content_kind,
        workers=workers,
        window=window,
        pool_factory=pool_factory,
    )
    write_tileset(packed.document, tileset_path)
    tileset_bytes = tileset_path.read_bytes()
    # The lossy profiles always render a sidecar (strict never reaches here).
    provenance = cast(
        "str", render_provenance(profile, dataset_id=dataset_id.hex())
    )
    provenance_bytes = _write_lf(provenance_path, provenance)
    files = {
        TILESET_NAME: _digest(tileset_bytes),
        PROVENANCE_NAME: _digest(provenance_bytes),
        TILES_HFP_NAME: FileDigest(
            sha256=_sha256_file(hfp_path),
            size=hfp_path.stat().st_size,
        ),
    }
    _write_lf(manifest_path, render_manifest(files, dataset_id.hex()))


def _write_lf(path: Path, text: str) -> bytes:
    """Write ``text`` as UTF-8 LF bytes on every platform; return the bytes."""
    data = text.encode("utf-8")
    path.write_bytes(data)
    return data


def _digest(data: bytes) -> FileDigest:
    """Return the in-memory ``FileDigest`` (sha256 hex + size) of ``data``."""
    return FileDigest(sha256=hashlib.sha256(data).hexdigest(), size=len(data))


def _sha256_file(path: Path) -> str:
    """Return a file's SHA-256 hex, hashed in bounded-memory chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_HASH_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _recover(out: Path, backup_dir: Path, marker: Path) -> None:
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
    _restore_stale(out, backup_dir, marker)


def _hold_stale(out: Path, backup_dir: Path) -> None:
    """Move a previous build's artifacts aside instead of deleting them.

    The held set spans every profile: the strict ``tiles/`` subtree, the
    shared ``tileset.json``, and — the packed lossy profiles only —
    ``tiles.hfp`` / ``provenance.json`` / ``manifest.json`` (whichever are
    absent are simply skipped), so a cross-profile rebuild neither strands
    nor misdescribes the previous deliverable.
    """
    stale = [
        p for p in (out / TILES_SUBDIR, *_file_artifacts(out)) if p.exists()
    ]
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


def _restore_stale(out: Path, backup_dir: Path, marker: Path) -> None:
    """Move a held-aside previous deliverable back into place.

    Per artifact, the held copy wins: any ``tiles/`` subtree or tool-owned
    flat file still in ``out`` is an unverified leftover of the failed or
    killed run holding the backup, and is removed before the verified copy
    is renamed back. An artifact absent from the backup keeps the copy in
    ``out`` (a kill between the hold renames leaves the not-yet-held, still
    verified artifact in place) — and a profile that never wrote a given
    artifact simply has none held to restore.

    A surviving accept marker means the opposite: the artifacts in
    place are a verified build and the backup is disposable (a failed
    recovery leaves both for the next run) — never restore over them.
    """
    if marker.exists() or not backup_dir.is_dir():
        return
    tiles_dir = out / TILES_SUBDIR
    held_tiles = backup_dir / TILES_SUBDIR
    if held_tiles.is_dir():
        if tiles_dir.is_dir():
            shutil.rmtree(tiles_dir)
        held_tiles.rename(tiles_dir)
    for path in _file_artifacts(out):
        held = backup_dir / path.name
        if held.is_file():
            path.unlink(missing_ok=True)
            held.rename(path)
    backup_dir.rmdir()


def _discard(out: Path) -> None:
    """Remove everything a rejected build wrote (never leave stale).

    Runs only once the hold step has completed: from then on, anything
    at these tool-owned paths is the rejected run's output — including
    partial files a failed write created, and the packed profiles' fresh
    ``tiles.hfp`` / ``provenance.json`` / ``manifest.json`` — and is
    removed wholesale. Before the hold completes, artifacts still in place
    are the untouched previous deliverable, and the caller must not call
    this at all.
    """
    for path in _file_artifacts(out):
        if path.is_file():
            path.unlink()
    tiles_dir = out / TILES_SUBDIR
    if tiles_dir.is_dir():
        shutil.rmtree(tiles_dir)
