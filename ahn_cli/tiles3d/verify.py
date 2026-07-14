"""The strictest post-write verifier for a 3D Tiles build.

:func:`verify_tiles3d` re-reads **everything from disk** and validates
it against both the OGC 3D Tiles 1.1 / glTF 2.0 rules and an
**independent recomputation from the source rasters**:

1.  tileset.json: exact key sets (we authored it — unknown keys are
    errors), ``asset.version == "1.1"``, root ``refine: "REPLACE"``,
    finite non-negative geometric errors, child error <= parent error,
    tileset error >= root error.
2.  regions: valid ranges and ordering; every child region contained
    in its parent's.
3.  content links: every uri stays inside the output directory, maps
    to a planned quadtree tile, is referenced exactly once, and exists;
    every on-disk tile file is referenced (no orphans).
4.  glb containers: magic, version, declared length == file size,
    exact two-chunk JSON+BIN framing.
5.  glTF internals: buffer/view/accessor bounds, POSITION min/max
    recomputed from the payload bit for bit, index count divisible by
    3, all indices in range, no degenerate triangles, UVs in [0, 1].
6.  textures: strict PNG decode (every chunk CRC verified), dimensions
    equal to the tile's sample counts, pixels bit-equal to the freshly
    sampled orthophoto.
7.  containment: every sampled vertex's EPSG:4979 coordinate inside
    the tile's stored region and every ancestor's stored region.
8.  coverage: the recomputed leaf spans cover every grid pixel.
9.  byte identity: for the strict profile every glb and tileset.json;
    for the packed lossy profiles the whole ``tiles.hfp`` pack plus
    ``tileset.json``, ``provenance.json`` and ``manifest.json`` — each
    byte-equal to a full independent rebuild from the (re-verified) sources.

For the packed profiles the per-tile checks read each tile's content from
the pack (via :func:`~ahn_cli.tiles3d.pack.read_pack`, which fully
validates the container) materialised to a scratch ``tiles/`` directory, so
the strict/game/heightfield per-tile verifiers are identical either way.
On top of the container-level rejects ``read_pack`` already enforces, the
packed profiles add three **deep pack checks**, all run before the
byte-identity backstop:

- the **two-encodings witness** — the pack index and the ``tileset.json``
  sidecar are two encodings of one scene, and must agree bit-for-bit where
  they overlap: a one-to-one, onto URI(canonical parse)↔key mapping, each
  tile's six-double ``region`` and ``geometricError`` (f64 bit patterns,
  after JSON round-trip) bit-equal, and the pack header's
  ``root_geometric_error`` bit-equal to the tileset's top-level
  ``geometricError``;
- the **chunk↔entry semantic cross-check** (heightfield only) — each ``.hf``
  chunk header's ``region`` is horizontally bit-equal to its pack index
  entry region and height-contained within it (leaves bit-equal in all six),
  the wrong-tile-under-right-key guard the pack spec assigns the verifier;
- **manifest recompute** — ``manifest.json`` byte-equals a recomputation of
  the on-disk artifacts' SHA-256 digests + sizes tied to the pack's
  ``dataset_id``.

Any violation raises :class:`Tiles3dError`; the build orchestrator
then removes everything it wrote.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
import struct
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from ahn_cli.tiles3d.emit import (
    MANIFEST_NAME,
    TILES_HFP_NAME,
    TILES_SUBDIR,
    TILESET_NAME,
    compute_build,
    compute_packed_build,
    texture_uri,
    tile_uri,
)
from ahn_cli.tiles3d.errors import Tiles3dError
from ahn_cli.tiles3d.geodesy import Geodesy
from ahn_cli.tiles3d.heightfield import decode_heightfield
from ahn_cli.tiles3d.manifest import FileDigest, render_manifest
from ahn_cli.tiles3d.pack import Pack, read_pack, write_pack
from ahn_cli.tiles3d.png import decode_png
from ahn_cli.tiles3d.profile import Profile
from ahn_cli.tiles3d.provenance import PROVENANCE_NAME, render_provenance
from ahn_cli.tiles3d.quadtree import plan_quadtree, sample_indices
from ahn_cli.tiles3d.sources import load_terrain
from ahn_cli.tiles3d.tileset import render_tileset
from ahn_cli.tiles3d.verify_game import verify_game_tile
from ahn_cli.tiles3d.verify_heightfield import verify_heightfield_tile
from ahn_cli.tiles3d.verify_splat import verify_splat_tile

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from ahn_cli.tiles3d.emit import ComputedBuild
    from ahn_cli.tiles3d.quadtree import TilePlan, TreePlan
    from ahn_cli.tiles3d.sources import TerrainGrid

__all__ = ["verify_tiles3d"]

_GLB_MAGIC = 0x46546C67
_CHUNK_JSON = 0x4E4F534A
_CHUNK_BIN = 0x004E4942
_REGION_LENGTH = 6
_TRIANGLE = 3
_HORIZONTAL = 4

_TileKey = tuple[int, int, int]
_URI_PATTERN = re.compile(
    r"tiles/(0|[1-9][0-9]*)-(0|[1-9][0-9]*)-(0|[1-9][0-9]*)\.(.+)"
)
"""Strict canonical ``tiles/<level>-<tx>-<ty>.<ext>`` parse: base-10 with no
leading zeros (a bare ``0`` where zero), used by the two-encodings witness."""

_TOP_KEYS = {"asset", "geometricError", "root"}
_ASSET = {"generator": "ahn_cli tiles3d", "version": "1.1"}
_TILE_KEYS = {"boundingVolume", "content", "geometricError"}


def _require(condition: bool, message: str) -> None:  # noqa: FBT001
    """Raise the typed verification error unless ``condition`` holds."""
    if not condition:
        raise Tiles3dError(message)


def verify_tiles3d(
    out_dir: Path,
    ortho: Path,
    heights: Path,
    *,
    tile_pixels: int = 256,
    profile: Profile = Profile.STRICT,
) -> None:
    """Hard-verify a written 3D Tiles build against its sources.

    Contract:
        - Returns only when every check in the module docstring holds,
          all from fresh disk reads.
        - ``profile`` selects the encoder for the independent rebuild so
          the byte-identity backstop reproduces the same on-disk bytes:
          for strict every ``.glb`` + ``tileset.json``; for the packed lossy
          profiles the whole ``tiles.hfp`` pack plus ``tileset.json``,
          ``provenance.json`` and ``manifest.json``. Each profile runs its
          own per-tile checks before the backstop: the strict profile's
          float32/PNG glTF/texture/containment checks; the game profile's
          four quantized/meshopt/JPEG families plus dequantized-vertex
          containment (:func:`~ahn_cli.tiles3d.verify_game.verify_game_tile`);
          the heightfield profile's ``.hf``/JPEG checks plus containment
          (:func:`~ahn_cli.tiles3d.verify_heightfield.verify_heightfield_tile`);
          or the splat profile's decode/position/colour/opacity/scale/
          rotation checks plus containment
          (:func:`~ahn_cli.tiles3d.verify_splat.verify_splat_tile`).

    Failure modes:
        - :class:`Tiles3dError` naming the first failed check.
    """
    terrain = load_terrain(ortho, heights)
    tree = plan_quadtree(terrain.width, terrain.height, tile_pixels)
    _verify_leaf_coverage(tree, terrain)
    document = _read_tileset(out_dir / TILESET_NAME)
    flat = _verify_document(document)
    tiles = _walk(tree.root)
    tiles_by_uri = {tile_uri(t, profile): t for t in tiles}
    texture_by_uri = {
        tile_uri(t, profile): texture_uri(t, profile) for t in tiles
    }
    expected_textures = {u for u in texture_by_uri.values() if u is not None}
    geodesy = Geodesy()
    if profile is Profile.STRICT:
        _verify_tile_content(
            out_dir,
            flat,
            tiles_by_uri,
            texture_by_uri,
            expected_textures,
            terrain,
            geodesy,
            profile,
        )
        computed = compute_build(terrain, tree, encoder=profile.encoder())
        _verify_byte_identity(out_dir, computed)
        return
    # Packed lossy profiles: read the content from the AHNP pack (fully
    # validating the container), cross-check the two encodings and the .hf
    # chunks against the pack index, materialise the blobs to a scratch tiles/
    # layout so the per-tile checks re-read from disk exactly as the strict
    # path does, verify the manifest recompute, then byte-compare the whole
    # packed deliverable against a rebuild.
    pack = read_pack(out_dir / TILES_HFP_NAME)
    _verify_two_encodings(pack, flat, document, profile)
    if profile is Profile.HEIGHTFIELD:
        # Runs before the per-tile source checks so a corrupted chunk region
        # attributes to this cross-check, not the chunk<->mesh comparison.
        _verify_chunk_entry(pack, tiles_by_uri)
    with TemporaryDirectory() as scratch:
        content_root = Path(scratch)
        _materialise_pack(pack, content_root, profile)
        _verify_tile_content(
            content_root,
            flat,
            tiles_by_uri,
            texture_by_uri,
            expected_textures,
            terrain,
            geodesy,
            profile,
        )
    _verify_manifest(out_dir, pack)
    _verify_packed_byte_identity(out_dir, terrain, tree, profile)


def _verify_tile_content(
    content_root: Path,
    flat: _Flat,
    tiles_by_uri: dict[str, TilePlan],
    texture_by_uri: dict[str, str | None],
    expected_textures: set[str],
    terrain: TerrainGrid,
    geodesy: Geodesy,
    profile: Profile,
) -> None:
    """Run the link + per-tile checks against ``content_root``'s tiles/.

    ``content_root`` is ``out_dir`` for the strict profile's loose tiles and
    a scratch directory holding the pack's materialised blobs for the lossy
    profiles; the per-tile verifiers are identical either way.
    """
    _verify_links(content_root, flat, tiles_by_uri, expected_textures)
    for entry, enclosing_regions in flat:
        uri = _entry_uri(entry)
        tile = tiles_by_uri[uri]
        if profile is Profile.HEIGHTFIELD:
            # This profile always writes a sibling texture (never None).
            texture = cast("str", texture_by_uri[uri])
            verify_heightfield_tile(
                content_root, uri, texture, terrain, tile, geodesy
            )
            # NAP-native profile: regions are NAP, so containment must compare
            # NAP vertex heights (not geodesy-converted ellipsoidal ones).
            _verify_containment(
                terrain,
                tile,
                enclosing_regions,
                geodesy,
                uri,
                nap_heights=True,
            )
        elif profile is Profile.SPLAT:
            verify_splat_tile(content_root, uri, terrain, tile, geodesy)
            _verify_containment(
                terrain, tile, enclosing_regions, geodesy, uri
            )
        else:
            gltf, binary = _parse_glb((content_root / uri).read_bytes(), uri)
            if profile is Profile.STRICT:
                _verify_gltf(gltf, binary, uri)
                _verify_texture(gltf, binary, terrain, tile, uri)
                _verify_containment(
                    terrain, tile, enclosing_regions, geodesy, uri
                )
            else:
                verify_game_tile(
                    gltf,
                    binary,
                    terrain,
                    tile,
                    enclosing_regions,
                    geodesy,
                    uri,
                )


def _materialise_pack(
    pack: Pack, content_root: Path, profile: Profile
) -> None:
    """Write every pack blob to ``content_root/tiles/`` under its uri name.

    Restores the loose ``tiles/<level>-<tx>-<ty>.<ext>`` layout (plus the
    heightfield ``.jpg`` siblings) from the validated pack so the untouched
    per-tile verifiers can re-read each tile from disk.
    """
    tiles_dir = content_root / TILES_SUBDIR
    tiles_dir.mkdir()
    content_suffix = profile.content_suffix()
    texture_suffix = profile.texture_suffix()
    for index, entry in enumerate(pack.entries):
        base = f"{entry.level}-{entry.tx}-{entry.ty}"
        (tiles_dir / f"{base}{content_suffix}").write_bytes(
            pack.primary_blob(index)
        )
        if texture_suffix is not None:
            texture = pack.texture_blob(index)
            # content_kind = 0 (heightfield) always carries a texture blob.
            (tiles_dir / f"{base}{texture_suffix}").write_bytes(
                cast("bytes", texture)
            )


def _f64_bits(value: float) -> bytes:
    """Return a float's little-endian IEEE 754 binary64 bit pattern."""
    return struct.pack("<d", float(value))


def _region_bits(region: Sequence[float]) -> bytes:
    """Return the concatenated bit patterns of a region's doubles."""
    return b"".join(_f64_bits(v) for v in region)


def _parse_tile_uri(uri: str, expected_ext: str) -> _TileKey:
    """Parse ``tiles/<level>-<tx>-<ty>.<ext>`` strictly into ``(level, tx, ty)``.

    The extension must be exactly ``expected_ext`` (``hf`` / ``glb``), the
    three integers base-10 with no leading zeros; any deviation is rejected.
    """
    match = _URI_PATTERN.fullmatch(uri)
    if match is None or match.group(4) != expected_ext:
        msg = (
            f"content uri {uri!r} is not the canonical "
            f"tiles/<level>-<tx>-<ty>.{expected_ext} form."
        )
        raise Tiles3dError(msg)
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


def _verify_two_encodings(
    pack: Pack,
    flat: _Flat,
    document: dict[str, Any],
    profile: Profile,
) -> None:
    """Cross-check the pack index against tileset.json, bit-for-bit.

    The two-encodings witness. Maps every ``tileset.json`` ``content.uri`` to
    a pack key by the strict
    canonical parse, demands a one-to-one onto correspondence with the pack
    index (no orphan either way), then bit-compares each matched tile's
    ``region`` and ``geometricError`` and the pack header's
    ``root_geometric_error`` against the tileset's top-level ``geometricError``.
    """
    expected_ext = profile.content_suffix()[1:]
    tileset_by_key: dict[_TileKey, dict[str, Any]] = {}
    for entry, _ in flat:
        key = _parse_tile_uri(_entry_uri(entry), expected_ext)
        tileset_by_key[key] = entry
    pack_by_key = {(e.level, e.tx, e.ty): e for e in pack.entries}
    for key in tileset_by_key:
        _require(
            key in pack_by_key,
            f"tileset entry tiles/{key[0]}-{key[1]}-{key[2]}.{expected_ext} "
            "has no matching pack index entry.",
        )
    for key in pack_by_key:
        _require(
            key in tileset_by_key,
            f"pack index entry tiles/{key[0]}-{key[1]}-{key[2]}."
            f"{expected_ext} has no matching tileset.json entry.",
        )
    _require(
        _f64_bits(pack.header.root_geometric_error)
        == _f64_bits(cast("float", document["geometricError"])),
        "the pack header root_geometric_error does not bit-equal the "
        "tileset.json top-level geometricError.",
    )
    for key, entry in tileset_by_key.items():
        pack_entry = pack_by_key[key]
        uri = _entry_uri(entry)
        region = cast("list[float]", entry["boundingVolume"]["region"])
        _require(
            _region_bits(region) == _region_bits(pack_entry.region),
            f"{uri}: the tileset region does not bit-equal the pack index "
            "entry region.",
        )
        _require(
            _f64_bits(cast("float", entry["geometricError"]))
            == _f64_bits(pack_entry.geometric_error),
            f"{uri}: the tileset geometricError does not bit-equal the pack "
            "index entry geometric_error.",
        )


def _verify_chunk_entry(
    pack: Pack, tiles_by_uri: dict[str, TilePlan]
) -> None:
    """Cross-check each ``.hf`` chunk-header region against its pack entry.

    The chunk header carries the tile's *own* mesh region; the pack index
    entry carries the *enclosing* region. Per the chunk spec's region
    semantics the four horizontal doubles are bit-equal for every tile and
    the chunk height range is contained in the entry's — bit-equal in all six
    for a leaf. ``rtc_centre`` / quantizer consistency is left to
    :func:`~ahn_cli.tiles3d.verify_heightfield.verify_heightfield_tile`
    (chunk vs the reloaded sources); this only relates the two on-disk
    encodings.
    """
    plan_by_key = {
        (tile.level, tile.tx, tile.ty): tile for tile in tiles_by_uri.values()
    }
    for index, entry in enumerate(pack.entries):
        uri = f"{TILES_SUBDIR}/{entry.level}-{entry.tx}-{entry.ty}.hf"
        chunk_region = decode_heightfield(pack.primary_blob(index)).region
        _require(
            _region_bits(chunk_region[:_HORIZONTAL])
            == _region_bits(entry.region[:_HORIZONTAL]),
            f"{uri}: the heightfield chunk horizontal region does not "
            "bit-equal the pack index entry region.",
        )
        _require(
            entry.region[4] <= chunk_region[4]
            and chunk_region[5] <= entry.region[5],
            f"{uri}: the heightfield chunk height range is not contained in "
            "the pack index entry height range.",
        )
        if not plan_by_key[(entry.level, entry.tx, entry.ty)].children:
            _require(
                _region_bits(chunk_region) == _region_bits(entry.region),
                f"{uri}: the leaf heightfield chunk region does not bit-equal "
                "the pack index entry region.",
            )


def _verify_manifest(out_dir: Path, pack: Pack) -> None:
    """Byte-compare manifest.json against a recompute of the on-disk files.

    Recomputes each loose file's + the pack's SHA-256 and size straight from
    disk, ties them to the pack's ``dataset_id``, renders the manifest and
    demands it byte-equal the written ``manifest.json`` — so a manifest whose
    digests, sizes or ``dataset_id`` drift from the artifacts they describe is
    refused before the byte-identity backstop.
    """
    files = {
        name: FileDigest(
            sha256=hashlib.sha256(data).hexdigest(), size=len(data)
        )
        for name in (TILESET_NAME, PROVENANCE_NAME, TILES_HFP_NAME)
        for data in ((out_dir / name).read_bytes(),)
    }
    expected = render_manifest(files, pack.header.dataset_id.hex()).encode(
        "utf-8"
    )
    _require(
        (out_dir / MANIFEST_NAME).read_bytes() == expected,
        "manifest.json does not match a recomputation of the on-disk "
        "artifacts' digests, sizes and dataset_id.",
    )


def _verify_packed_byte_identity(
    out_dir: Path, terrain: TerrainGrid, tree: TreePlan, profile: Profile
) -> None:
    """Byte-compare the packed deliverable against an independent rebuild.

    Rebuilds ``tiles.hfp`` (in memory), the ``tileset.json`` sidecar, the
    ``provenance.json`` (with the rebuilt pack's ``dataset_id``) and the
    ``manifest.json`` from the re-verified sources, and demands each of the
    four on-disk files match byte for byte.
    """
    packed = compute_packed_build(terrain, tree, profile=profile)
    buffer = io.BytesIO()
    dataset_id = write_pack(
        buffer,
        packed.entries,
        packed.blob_source,
        root_geometric_error=packed.root_geometric_error,
        content_kind=packed.content_kind,
    )
    pack_bytes = buffer.getvalue()
    tileset_bytes = render_tileset(packed.document).encode("utf-8")
    provenance = render_provenance(profile, dataset_id=dataset_id.hex())
    provenance_bytes = cast("str", provenance).encode("utf-8")
    members = {
        TILESET_NAME: tileset_bytes,
        PROVENANCE_NAME: provenance_bytes,
        TILES_HFP_NAME: pack_bytes,
    }
    files = {
        name: FileDigest(
            sha256=hashlib.sha256(data).hexdigest(), size=len(data)
        )
        for name, data in members.items()
    }
    manifest_bytes = render_manifest(files, dataset_id.hex()).encode("utf-8")
    for name, expected in (
        (TILES_HFP_NAME, pack_bytes),
        (TILESET_NAME, tileset_bytes),
        (PROVENANCE_NAME, provenance_bytes),
        (MANIFEST_NAME, manifest_bytes),
    ):
        _require(
            (out_dir / name).read_bytes() == expected,
            f"{name} does not byte-equal its independent rebuild.",
        )


def _walk(tile: TilePlan) -> list[TilePlan]:
    """Flatten the plan tree, parents before children."""
    tiles = [tile]
    for child in tile.children:
        tiles.extend(_walk(child))
    return tiles


def _verify_leaf_coverage(tree: TreePlan, terrain: TerrainGrid) -> None:
    """Verify the recomputed leaf spans cover every grid pixel."""
    coverage = np.zeros((terrain.height, terrain.width), dtype=np.int32)
    for tile in _walk(tree.root):
        if not tile.children:
            coverage[
                tile.row0 : tile.row1 + 1, tile.col0 : tile.col1 + 1
            ] += 1
    _require(
        bool(coverage.min() >= 1),
        "the quadtree's leaf spans do not cover every pixel of the "
        "grid; a pixel is missing from the output.",
    )


def _read_tileset(path: Path) -> dict[str, Any]:
    """Read and parse tileset.json from disk."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        msg = f"tileset.json at {path} is not readable: {exc}"
        raise Tiles3dError(msg) from exc
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        msg = f"tileset.json at {path} is not valid JSON: {exc}"
        raise Tiles3dError(msg) from exc
    _require(
        isinstance(document, dict),
        f"tileset.json at {path} is not a JSON object.",
    )
    return cast("dict[str, Any]", document)


_Flat = list[tuple[dict[str, Any], list[tuple[float, ...]]]]


def _verify_document(document: dict[str, Any]) -> _Flat:
    """Structurally verify the tileset document; flatten its entries.

    Returns every tile entry paired with the list of its enclosing
    stored regions (ancestors first, its own last), for the
    containment checks.
    """
    _require(
        set(document) == _TOP_KEYS,
        "tileset.json carries unexpected or missing top-level keys.",
    )
    _require(
        document["asset"] == _ASSET,
        "tileset.json's asset block is not the ahn_cli tiles3d "
        "version 1.1 asset.",
    )
    tileset_error = _finite_error(document["geometricError"], "tileset")
    root = document["root"]
    _require(
        isinstance(root, dict),
        "the root tile entry is not a JSON object.",
    )
    root_entry = cast("dict[str, Any]", root)
    _require(
        root_entry.get("refine") == "REPLACE",
        'the root tile must declare refine "REPLACE".',
    )
    flat: _Flat = []
    root_error = _verify_entry(
        root_entry,
        is_root=True,
        parent_error=math.inf,
        parent_region=None,
        ancestor_regions=[],
        flat=flat,
    )
    _require(
        tileset_error >= root_error,
        "the tileset geometricError is below the root tile's.",
    )
    return flat


def _verify_entry(
    entry: dict[str, Any],
    *,
    is_root: bool,
    parent_error: float,
    parent_region: tuple[float, ...] | None,
    ancestor_regions: list[tuple[float, ...]],
    flat: _Flat,
) -> float:
    """Verify one tile entry (recursively) and record it in ``flat``."""
    allowed = (
        _TILE_KEYS | {"children"} | ({"refine"} if is_root else set[str]())
    )
    _require(
        _TILE_KEYS <= set(entry) <= allowed,
        f"a tile entry carries unexpected or missing keys: {sorted(entry)}.",
    )
    error = _finite_error(entry["geometricError"], "a tile")
    _require(
        error <= parent_error,
        "a tile's geometricError exceeds its parent's.",
    )
    region = _verify_region(entry["boundingVolume"])
    if parent_region is not None:
        _require(
            _region_contains(parent_region, region),
            "a child tile's region is not contained in its parent's.",
        )
    content = entry["content"]
    _require(
        isinstance(content, dict)
        and set(cast("dict[str, Any]", content)) == {"uri"}
        and isinstance(cast("dict[str, Any]", content)["uri"], str),
        "a tile's content block is not exactly a string uri.",
    )
    flat.append((entry, [*ancestor_regions, region]))
    children = cast("list[Any]", entry.get("children", []))
    for child in children:
        _require(
            isinstance(child, dict),
            "a children list holds a non-object entry.",
        )
        _verify_entry(
            cast("dict[str, Any]", child),
            is_root=False,
            parent_error=error,
            parent_region=region,
            ancestor_regions=[*ancestor_regions, region],
            flat=flat,
        )
    return error


def _finite_error(value: object, owner: str) -> float:
    """Verify a geometricError is a finite non-negative number."""
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0,
        f"{owner} geometricError is not a finite non-negative number.",
    )
    return float(cast("float", value))


def _verify_region(volume: object) -> tuple[float, ...]:
    """Verify a boundingVolume is a valid 3D Tiles region."""
    _require(
        isinstance(volume, dict)
        and set(cast("dict[str, Any]", volume)) == {"region"},
        "a boundingVolume is not exactly a region volume.",
    )
    region_value = cast("dict[str, Any]", volume)["region"]
    _require(
        isinstance(region_value, list)
        and len(cast("list[Any]", region_value)) == _REGION_LENGTH
        and all(
            isinstance(v, (int, float)) and math.isfinite(float(v))
            for v in cast("list[Any]", region_value)
        ),
        "a region is not a list of six finite numbers.",
    )
    west, south, east, north, low, high = (
        float(v) for v in cast("list[Any]", region_value)
    )
    _require(
        -math.pi <= west < east <= math.pi
        and -math.pi / 2 <= south < north <= math.pi / 2
        and low <= high,
        "a region is degenerate or out of range.",
    )
    return (west, south, east, north, low, high)


def _region_contains(
    outer: tuple[float, ...], inner: tuple[float, ...]
) -> bool:
    """Return whether ``inner`` lies within ``outer``."""
    return (
        outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and inner[2] <= outer[2]
        and inner[3] <= outer[3]
        and outer[4] <= inner[4]
        and inner[5] <= outer[5]
    )


def _entry_uri(entry: dict[str, Any]) -> str:
    """Return the entry's content uri."""
    return cast("str", cast("dict[str, Any]", entry["content"])["uri"])


def _verify_links(
    out_dir: Path,
    flat: _Flat,
    tiles_by_uri: dict[str, TilePlan],
    expected_textures: set[str],
) -> None:
    """Verify uris are safe, planned, unique, present — no orphans.

    ``expected_textures`` are the sibling texture files a profile writes
    but does not reference from ``tileset.json`` (empty for the
    embedded-texture glTF profiles); they are excluded from the orphan sweep
    so a genuine texture is not mistaken for one. Their *presence* is not
    re-checked here: the heightfield profile is the only one with sibling
    textures, and its ``_verify_links`` runs against the scratch dir
    :func:`_materialise_pack` just filled — which writes a ``.jpg`` for every
    pack entry, and ``pack.read_pack`` guarantees every ``content_kind=0``
    entry carries a texture blob — so a texture can never be absent while its
    ``.hf`` (checked first, below) is present.
    """
    seen: set[str] = set()
    for entry, _ in flat:
        uri = _entry_uri(entry)
        resolved = (out_dir / uri).resolve()
        _require(
            not uri.startswith("/")
            and resolved.is_relative_to(out_dir.resolve()),
            f"content uri {uri} escapes the output directory.",
        )
        _require(
            uri in tiles_by_uri,
            f"content uri {uri} does not correspond to any planned "
            "quadtree tile.",
        )
        _require(
            uri not in seen,
            f"content uri {uri} is referenced more than once.",
        )
        seen.add(uri)
        _require(
            (out_dir / uri).is_file(),
            f"missing content file: {uri}.",
        )
    on_disk = {
        f"{TILES_SUBDIR}/{path.name}"
        for path in (out_dir / TILES_SUBDIR).iterdir()
    }
    orphans = on_disk - seen - expected_textures
    _require(
        not orphans,
        f"tile files on disk are not referenced by tileset.json: "
        f"{sorted(orphans)}.",
    )


def _parse_glb(data: bytes, uri: str) -> tuple[dict[str, Any], bytes]:
    """Verify the glb container framing; return (glTF JSON, BIN)."""
    _require(
        len(data) >= 28,  # noqa: PLR2004 -- header + two chunk headers
        f"{uri}: glb is too short to be a container.",
    )
    magic, version, length = struct.unpack("<III", data[:12])
    _require(magic == _GLB_MAGIC, f"{uri}: bad glb magic.")
    _require(version == 2, f"{uri}: glb version is not 2.")  # noqa: PLR2004
    _require(
        length == len(data),
        f"{uri}: glb declared length does not equal the file size.",
    )
    json_length, json_kind = struct.unpack("<II", data[12:20])
    bin_header = 20 + json_length
    _require(
        json_kind == _CHUNK_JSON
        and json_length % 4 == 0
        and bin_header + 8 <= len(data),
        f"{uri}: glb chunks are not JSON-then-BIN.",
    )
    bin_length, bin_kind = struct.unpack(
        "<II", data[bin_header : bin_header + 8]
    )
    _require(
        bin_kind == _CHUNK_BIN and bin_header + 8 + bin_length == len(data),
        f"{uri}: glb BIN chunk does not exactly fill the container.",
    )
    try:
        gltf = json.loads(data[20:bin_header])
    except json.JSONDecodeError as exc:
        msg = f"{uri}: glb JSON chunk is not valid JSON: {exc}"
        raise Tiles3dError(msg) from exc
    return cast("dict[str, Any]", gltf), data[bin_header + 8 :]


def _view_bytes(
    gltf: dict[str, Any], binary: bytes, index: int, uri: str
) -> bytes:
    """Return a bufferView's bytes after bounds-checking it."""
    view = cast("dict[str, Any]", gltf["bufferViews"][index])
    offset = int(view["byteOffset"])
    length = int(view["byteLength"])
    _require(
        offset >= 0 and offset + length <= len(binary),
        f"{uri}: bufferView {index} overruns the BIN chunk.",
    )
    return binary[offset : offset + length]


def _verify_gltf(gltf: dict[str, Any], binary: bytes, uri: str) -> None:
    """Verify the glTF document's internal consistency and payloads."""
    buffers = cast("list[dict[str, Any]]", gltf["buffers"])
    _require(
        len(buffers) == 1 and int(buffers[0]["byteLength"]) == len(binary),
        f"{uri}: buffer byteLength does not equal the BIN chunk size.",
    )
    accessors = cast("list[dict[str, Any]]", gltf["accessors"])
    positions = np.frombuffer(
        _view_bytes(gltf, binary, 0, uri), dtype="<f4"
    ).reshape(-1, 3)
    uvs = np.frombuffer(
        _view_bytes(gltf, binary, 1, uri), dtype="<f4"
    ).reshape(-1, 2)
    indices = np.frombuffer(_view_bytes(gltf, binary, 2, uri), dtype="<u4")
    _require(
        int(accessors[0]["count"]) == positions.shape[0]
        and int(accessors[1]["count"]) == uvs.shape[0]
        and int(accessors[2]["count"]) == indices.shape[0],
        f"{uri}: accessor counts do not match their bufferViews.",
    )
    stored_min = [float(v) for v in accessors[0]["min"]]
    stored_max = [float(v) for v in accessors[0]["max"]]
    actual_min = [float(v) for v in positions.min(axis=0)]
    actual_max = [float(v) for v in positions.max(axis=0)]
    _require(
        stored_min == actual_min and stored_max == actual_max,
        f"{uri}: POSITION min/max do not equal the payload extremes.",
    )
    _require(
        indices.shape[0] % _TRIANGLE == 0,
        f"{uri}: index count is not a multiple of 3.",
    )
    _require(
        bool(indices.max() < positions.shape[0]),
        f"{uri}: an index is out of range of the vertex count.",
    )
    triangles = indices.reshape(-1, _TRIANGLE)
    distinct = (
        (triangles[:, 0] != triangles[:, 1])
        & (triangles[:, 1] != triangles[:, 2])
        & (triangles[:, 0] != triangles[:, 2])
    )
    _require(
        bool(distinct.all()),
        f"{uri}: a triangle references a vertex twice (degenerate).",
    )
    _require(
        bool((uvs >= 0.0).all()) and bool((uvs <= 1.0).all()),
        f"{uri}: a texture coordinate is outside [0, 1].",
    )


def _verify_texture(
    gltf: dict[str, Any],
    binary: bytes,
    terrain: TerrainGrid,
    tile: TilePlan,
    uri: str,
) -> None:
    """Verify the embedded PNG decodes and equals the sampled ortho."""
    image_view = int(
        cast("list[dict[str, Any]]", gltf["images"])[0]["bufferView"]
    )
    pixels = decode_png(_view_bytes(gltf, binary, image_view, uri))
    cols = sample_indices(tile.col0, tile.col1, tile.stride)
    rows = sample_indices(tile.row0, tile.row1, tile.stride)
    _require(
        pixels.shape == (len(rows), len(cols), 3),
        f"{uri}: texture dimensions do not equal the tile's sample counts.",
    )
    expected = terrain.rgb[np.ix_(rows, cols)]
    _require(
        bool(np.array_equal(pixels, expected)),
        f"{uri}: texture pixels do not equal the sampled orthophoto.",
    )


def _verify_containment(
    terrain: TerrainGrid,
    tile: TilePlan,
    enclosing_regions: list[tuple[float, ...]],
    geodesy: Geodesy,
    uri: str,
    *,
    nap_heights: bool = False,
) -> None:
    """Verify the tile's vertices lie inside every enclosing region.

    ``enclosing_regions`` holds the stored regions of every ancestor
    plus the tile's own; the vertex geodetics are recomputed from the
    sources, so a shrunken or drifted stored region is caught here.

    ``nap_heights`` selects the vertical datum of the height comparison to
    match the profile's region datum. The default (``False``) compares the
    geodesy NAP→ellipsoidal vertex height against the ellipsoidal regions the
    ``strict``/``game``/``splat`` profiles emit. The **NAP-native heightfield**
    profile emits NAP regions (see
    :func:`~ahn_cli.tiles3d.heightfield.nap_region`), so it passes
    ``nap_heights=True`` and the vertex height axis is the raw source NAP
    ``terrain.z`` — never the geodesy-converted ellipsoidal height. Mixing the
    two would offset the comparison by the geoid undulation (~43 m in NL where
    the NLGEO2018 grid is installed) and falsely fail containment; keeping both
    sides NAP makes the check self-consistent regardless of geoid-grid
    availability. Horizontal lon/lat are datum-independent and always come from
    geodesy.
    """
    cols = sample_indices(tile.col0, tile.col1, tile.stride)
    rows = sample_indices(tile.row0, tile.row1, tile.stride)
    grid = np.ix_(rows, cols)
    nap_z = terrain.z[grid].astype(np.float64).ravel()
    lon, lat, ellipsoidal_height = geodesy.to_geodetic_radians(
        terrain.x[grid].astype(np.float64).ravel(),
        terrain.y[grid].astype(np.float64).ravel(),
        nap_z,
    )
    height = nap_z if nap_heights else ellipsoidal_height
    for region in enclosing_regions:
        _require(
            _points_within(lon, lat, height, region),
            f"{uri}: a vertex lies outside an enclosing region.",
        )


def _points_within(
    lon: npt.NDArray[np.float64],
    lat: npt.NDArray[np.float64],
    height: npt.NDArray[np.float64],
    region: tuple[float, ...],
) -> bool:
    """Return whether every point lies inside the region."""
    return (
        bool((lon >= region[0]).all())
        and bool((lat >= region[1]).all())
        and bool((lon <= region[2]).all())
        and bool((lat <= region[3]).all())
        and bool((height >= region[4]).all())
        and bool((height <= region[5]).all())
    )


def _verify_byte_identity(out_dir: Path, computed: ComputedBuild) -> None:
    """Verify the strict deliverable byte-equals its independent rebuild.

    The strict profile embeds its texture in each glb and writes no sidecar,
    so this covers every ``.glb`` plus ``tileset.json``; the packed lossy
    profiles have their own backstop (:func:`_verify_packed_byte_identity`).
    """
    for uri, expected in computed.glbs.items():
        _require(
            (out_dir / uri).read_bytes() == expected,
            f"{uri} does not byte-equal its independent rebuild.",
        )
    _require(
        (out_dir / TILESET_NAME).read_bytes()
        == render_tileset(computed.document).encode("utf-8"),
        "tileset.json does not byte-equal its independent rebuild.",
    )
