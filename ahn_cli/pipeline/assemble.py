"""Cross-tile assembly of the ``tiles3d`` deliverable from per-tile blobs.

The executor streams one tile at a time, persisting each tile's encoded blobs
(``geometry``, an optional ``texture``, and a small ``region.json`` metadata
record) into the run's :class:`~ahn_cli.pipeline.manifest.TileStore`. Nothing in
that per-tile stream knows the tileset's tree shape, and
:class:`~ahn_cli.pipeline.manifest.ManifestEntry` deliberately carries no
region/geometric-error field. This module is the assembly step that stitches the
persisted per-tile blobs into the standalone ``tiles3d`` verb's exact on-disk
shape -- a loose ``tileset.json`` + ``tiles/`` directory for the strict profile,
or a packed ``tiles.hfp`` + ``tileset.json`` + ``provenance.json`` +
``manifest.json`` for a lossy profile -- reproducing
:func:`ahn_cli.tiles3d.build._write_strict` /
:func:`ahn_cli.tiles3d.build._write_packed` byte for byte.

The one subtlety is the **children-first region union**: a tileset entry's (and
a pack entry's) region is the union of the tile's own region with every
descendant's, so that a parent's bounding volume contains all its children
(:func:`ahn_cli.tiles3d.emit._Emitter.emit`). The assembler walks the quadtree
bottom-up, unioning each node's own region (recorded per tile) with its already
unioned children, exactly as the standalone emitter does.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from ahn_cli.pipeline.errors import PipelineError
from ahn_cli.pipeline.model import TileKey
from ahn_cli.tiles3d.emit import (
    MANIFEST_NAME,
    TILES_HFP_NAME,
    TILES_SUBDIR,
    TILESET_NAME,
    tile_uri,
)
from ahn_cli.tiles3d.manifest import FileDigest, render_manifest
from ahn_cli.tiles3d.pack import (
    CONTENT_KIND_GAME,
    CONTENT_KIND_HEIGHTFIELD,
    CONTENT_KIND_SPLAT,
    PackEntry,
    write_pack,
)
from ahn_cli.tiles3d.pack import TileKey as PackTileKey
from ahn_cli.tiles3d.profile import Profile
from ahn_cli.tiles3d.provenance import PROVENANCE_NAME, render_provenance
from ahn_cli.tiles3d.quadtree import geometric_error
from ahn_cli.tiles3d.tileset import (
    tile_entry,
    tileset_document,
    union_region,
    write_tileset,
)

if TYPE_CHECKING:
    from pathlib import Path

    from ahn_cli.pipeline.manifest import TileStore
    from ahn_cli.tiles3d.mesh import Region
    from ahn_cli.tiles3d.quadtree import TilePlan, TreePlan

__all__ = ["REGION_BLOB_NAME", "assemble_tiles3d", "region_blob_bytes"]

REGION_BLOB_NAME = "region.json"
"""The per-tile region metadata blob the region-recording sink persists."""

_GEOMETRY_BLOB = "geometry"
_TEXTURE_BLOB = "texture"

_CONTENT_KIND = {
    Profile.GAME: CONTENT_KIND_GAME,
    Profile.HEIGHTFIELD: CONTENT_KIND_HEIGHTFIELD,
    Profile.SPLAT: CONTENT_KIND_SPLAT,
}
"""Pack ``content_kind`` per lossy profile (mirrors ``emit._CONTENT_KIND``)."""

_HASH_CHUNK = 1 << 20
"""Streamed-hash read block (1 MiB), so hashing ``tiles.hfp`` stays bounded."""


def region_blob_bytes(region: Region) -> bytes:
    """Return the deterministic ``region.json`` bytes for a tile's own region."""
    return json.dumps({"region": list(region)}, sort_keys=True).encode(
        "utf-8"
    )


def _read_region(store: TileStore, key: TileKey) -> Region:
    """Read a tile's own region from its persisted ``region.json`` blob."""
    path = store.tile_dir(key) / REGION_BLOB_NAME
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        msg = f"tile {key} is missing its region blob at {path}."
        raise PipelineError(msg) from exc
    data = cast("dict[str, object]", json.loads(raw))
    region = cast("list[float]", data["region"])
    return cast("Region", tuple(region))


def _geometry_bytes(store: TileStore, key: TileKey) -> bytes:
    """Return a tile's primary (geometry) blob bytes from the store."""
    return (store.tile_dir(key) / _GEOMETRY_BLOB).read_bytes()


def _texture_bytes(store: TileStore, key: TileKey) -> bytes | None:
    """Return a tile's texture blob bytes, or ``None`` if it has none."""
    path = store.tile_dir(key) / _TEXTURE_BLOB
    return path.read_bytes() if path.is_file() else None


@dataclass(frozen=True)
class _NodeResult:
    """One assembled node: its tileset entry and its unioned region."""

    entry: dict[str, object]
    region: Region


def _assemble_node(
    node: TilePlan,
    store: TileStore,
    pixel_size_m: float,
    profile: Profile,
    entries: list[PackEntry],
) -> _NodeResult:
    """Assemble ``node`` bottom-up: union child regions, build its entry."""
    key = TileKey(level=node.level, tx=node.tx, ty=node.ty)
    child_results = [
        _assemble_node(child, store, pixel_size_m, profile, entries)
        for child in node.children
    ]
    region = _read_region(store, key)
    for child in child_results:
        region = union_region(region, child.region)
    error = geometric_error(node.stride, pixel_size_m)
    uri = tile_uri(node, profile)
    entry = tile_entry(
        region,
        error,
        uri,
        [child.entry for child in child_results],
        root=node.level == 0,
    )
    entries.append(
        PackEntry(
            key=PackTileKey(node.level, node.tx, node.ty, 0),
            region=region,
            geometric_error=error,
        )
    )
    return _NodeResult(entry=entry, region=region)


def _tileset_error(root_stride: int, pixel_size_m: float) -> float:
    """Return the tileset document's top-level geometric error.

    Mirrors :func:`ahn_cli.tiles3d.emit.compute_build`: twice the root tile's
    own error, or (when the root itself carries the source exactly, a
    single-level tree) four pixels of ground.
    """
    root_error = geometric_error(root_stride, pixel_size_m)
    if root_error > 0.0:
        return 2.0 * root_error
    return pixel_size_m * 4.0


def assemble_tiles3d(
    store: TileStore,
    tree: TreePlan,
    *,
    pixel_size_m: float,
    out_dir: Path,
    profile: Profile,
) -> Path:
    """Assemble the ``tiles3d`` deliverable in ``out_dir``; return the tileset path.

    Contract:
        - Reads each tile's persisted ``geometry`` / ``texture`` / ``region.json``
          blobs from ``store``, folds regions children-first, and writes the
          standalone verb's exact deliverable: a loose ``tileset.json`` +
          ``tiles/`` directory (strict) or a packed ``tiles.hfp`` plus
          ``tileset.json`` / ``provenance.json`` / ``manifest.json`` (lossy).

    Failure modes:
        - :class:`~ahn_cli.pipeline.errors.PipelineError` if a tile's persisted
          blobs are missing.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    entries: list[PackEntry] = []
    root = _assemble_node(tree.root, store, pixel_size_m, profile, entries)
    tileset_error = _tileset_error(tree.root.stride, pixel_size_m)
    document = tileset_document(root.entry, tileset_error)
    if profile is Profile.STRICT:
        _write_strict(store, document, out_dir)
    else:
        _write_packed(
            store, document, entries, tileset_error, out_dir, profile
        )
    return out_dir / TILESET_NAME


def _write_strict(
    store: TileStore, document: dict[str, object], out_dir: Path
) -> None:
    """Write the loose strict deliverable: ``tiles/*.glb`` + ``tileset.json``."""
    tiles_dir = out_dir / TILES_SUBDIR
    tiles_dir.mkdir(parents=True, exist_ok=True)
    root = cast("dict[str, object]", document["root"])
    _write_strict_glbs(store, root, out_dir)
    write_tileset(document, out_dir / TILESET_NAME)


def _write_strict_glbs(
    store: TileStore, node: dict[str, object], out_dir: Path
) -> None:
    """Write every content glb the tileset references, recursively.

    Every tileset node this assembler emits carries a ``content`` (via
    :func:`ahn_cli.tiles3d.tileset.tile_entry`), so the uri is always present.
    """
    uri = cast("dict[str, str]", node["content"])["uri"]
    (out_dir / uri).write_bytes(_geometry_bytes(store, _key_from_uri(uri)))
    for child in cast("list[dict[str, object]]", node.get("children", [])):
        _write_strict_glbs(store, child, out_dir)


def _key_from_uri(uri: str) -> TileKey:
    """Recover a :class:`TileKey` from a ``tiles/<level>-<tx>-<ty>.<ext>`` uri."""
    stem = uri.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    level, tx, ty = (int(part) for part in stem.split("-"))
    return TileKey(level=level, tx=tx, ty=ty)


def _write_packed(
    store: TileStore,
    document: dict[str, object],
    entries: list[PackEntry],
    tileset_error: float,
    out_dir: Path,
    profile: Profile,
) -> None:
    """Write the packed deliverable, reproducing ``build._write_packed``."""
    hfp_path = out_dir / TILES_HFP_NAME
    tileset_path = out_dir / TILESET_NAME
    provenance_path = out_dir / PROVENANCE_NAME
    manifest_path = out_dir / MANIFEST_NAME

    def blob_source(key: PackTileKey) -> tuple[bytes, bytes | None]:
        tile_key = TileKey(level=key.level, tx=key.tx, ty=key.ty)
        return _geometry_bytes(store, tile_key), _texture_bytes(
            store, tile_key
        )

    dataset_id = write_pack(
        hfp_path,
        entries,
        blob_source,
        root_geometric_error=tileset_error,
        content_kind=_CONTENT_KIND[profile],
    )
    write_tileset(document, tileset_path)
    tileset_bytes = tileset_path.read_bytes()
    provenance = cast(
        "str", render_provenance(profile, dataset_id=dataset_id.hex())
    )
    provenance_bytes = _write_lf(provenance_path, provenance)
    files = {
        TILESET_NAME: _digest(tileset_bytes),
        PROVENANCE_NAME: _digest(provenance_bytes),
        TILES_HFP_NAME: FileDigest(
            sha256=_sha256_file(hfp_path), size=hfp_path.stat().st_size
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
