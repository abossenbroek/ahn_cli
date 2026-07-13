"""The ``AHNP`` pack container: single-pass writer + validating reader.

The lossy ``game`` and ``heightfield`` profiles bundle every content blob
plus a self-describing binary scene index into a single ``tiles.hfp``
pack; the pack *is* the runtime's scene (it never parses JSON on the play
path). This module is the only place that knows the container byte layout;
the normative specification the Rust runtime codes against lives in
``docs/superpowers/specs/2026-07-12-hfp-pack-format.md`` and this
producer/reader mirrors it exactly.

**Layout.** Five contiguous regions: a fixed 128-byte header, the index
region (a ``level_count x 16 B`` level directory followed by
``tile_count x 96 B`` entries sorted by ``(level, tz, ty, tx)``), the
``tile_count x 64 B`` hash section, then every content blob concatenated in
index order, each 16-byte aligned with zero inter-blob padding. All
multi-byte fields are little-endian; hashes are SHA-256, CRCs are
CRC-32/ISO-HDLC (:func:`zlib.crc32`).

**Writer.** :func:`write_pack` is single-pass and bounded-memory: it
streams the blobs (one tile resident at a time, fetched lazily by key)
into the blob region, recording each blob's offset/size/SHA-256, then
seeks back to patch the level directory, index, hash section,
``dataset_id`` and the two CRCs. Only the small metadata regions are held
in memory; the blobs never are.

**Reader.** :func:`read_pack` is the reference *validating* reader — every
reject in the spec's conforming-reader checklist lives here, once, so the
verifier (task 3d) calls one place. It verifies ``header_crc32`` before
trusting any count, ``index_crc32`` before parsing entries, recomputes
``dataset_id`` from the hash section, and verifies every blob's SHA-256.
"""

from __future__ import annotations

import hashlib
import math
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO

from ahn_cli.tiles3d.errors import Tiles3dError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ahn_cli.tiles3d.mesh import Region

__all__ = [
    "BLOB_ALIGNMENT",
    "CONTENT_KIND_GAME",
    "CONTENT_KIND_HEIGHTFIELD",
    "FORMAT_VERSION",
    "HASH_RECORD_SIZE",
    "HEADER_SIZE",
    "INDEX_ENTRY_SIZE",
    "INDEX_OFFSET",
    "LEVEL_RECORD_SIZE",
    "MAGIC",
    "NO_TEXTURE_SHA256",
    "LevelRecord",
    "Pack",
    "PackEntry",
    "PackHeader",
    "PackIndexEntry",
    "TileKey",
    "read_pack",
    "write_pack",
]

MAGIC = b"AHNP"
"""The 4-byte pack magic; any other leading bytes are a decode error."""

FORMAT_VERSION = 1
"""The pack format version this module reads and writes."""

CONTENT_KIND_HEIGHTFIELD = 0
"""``content_kind`` for a heightfield pack: ``.hf`` primary + ``.jpg`` texture."""

CONTENT_KIND_GAME = 1
"""``content_kind`` for a game pack: ``.glb`` primary, no texture blob."""

HEADER_SIZE = 128
LEVEL_RECORD_SIZE = 16
INDEX_ENTRY_SIZE = 96
HASH_RECORD_SIZE = 64
INDEX_OFFSET = 128
BLOB_ALIGNMENT = 16

NO_TEXTURE_SHA256 = b"\x00" * 32
"""The ``texture_sha256`` sentinel for a game tile: 32 zero bytes, matching
the zeroed ``texture_offset`` / ``texture_size`` slots (no empty-string hash)."""

# The header up to but excluding header_crc32 (bytes [0, 124)): magic,
# format_version, tile_count, level_count, index_offset, index_size,
# hash_offset, hash_size, file_size, root_geometric_error, dataset_id,
# index_crc32, reserved, content_kind, pad. This is the span header_crc32
# covers.
_HEADER_PREFIX_FMT = "<4sIIIQQQQQd32sIII16s"
_HEADER_FMT = _HEADER_PREFIX_FMT + "I"
_HEADER_CRC_SPAN = struct.calcsize(_HEADER_PREFIX_FMT)

_LEVEL_FMT = "<IIII"
_ENTRY_FMT = "<4I7d2Q2I"
_HASH_FMT = "<32s32s"

_U32_MAX = 0xFFFFFFFF
_U32_RANGE = 1 << 32
_U64_RANGE = 1 << 64

_REGION_FIELD_NAMES = (
    "region[0]",
    "region[1]",
    "region[2]",
    "region[3]",
    "region[4]",
    "region[5]",
    "geometric_error",
)
"""Names of the seven per-entry ``float64`` fields, for reject messages."""


@dataclass(frozen=True)
class TileKey:
    """A tile's quadtree key ``(level, tx, ty, tz)`` (``tz`` is ``0`` in v1).

    The pack's index sort order is ``sort_key`` — ``(level, tz, ty, tx)``,
    level-major then row-major — not the field order; sort by ``sort_key``.
    """

    level: int
    tx: int
    ty: int
    tz: int = 0

    @property
    def sort_key(self) -> tuple[int, int, int, int]:
        """The ``(level, tz, ty, tx)`` tuple the index sorts by."""
        return (self.level, self.tz, self.ty, self.tx)


@dataclass(frozen=True)
class PackEntry:
    """One tile's up-front pack metadata: its key, region and error.

    Contract (fields):
        - ``key``: the tile's :class:`TileKey`.
        - ``region``: the tile's enclosing EPSG:4979 region (six doubles,
          west/south/east/north/min-height/max-height), bit-equal to the
          ``tileset.json`` bounding volume.
        - ``geometric_error``: the tile's 3D Tiles geometric error (leaves
          ``0``), bit-equal to the tileset ``geometricError``.

    The blob bytes are supplied separately (streamed by key), so an entry
    is small and known before any blob is encoded.
    """

    key: TileKey
    region: Region
    geometric_error: float


@dataclass(frozen=True)
class LevelRecord:
    """One level-directory record (16 bytes).

    Contract (fields):
        - ``first_entry``: index of this level's first entry.
        - ``entry_count``: number of entries at this level.
        - ``tx_count`` / ``ty_count``: distinct column / row counts.
    """

    first_entry: int
    entry_count: int
    tx_count: int
    ty_count: int


@dataclass(frozen=True)
class PackHeader:
    """The parsed, integrity-checked 128-byte pack header.

    Every field is exactly as stored (and validated) in the header; see the
    format spec for meanings.
    """

    format_version: int
    tile_count: int
    level_count: int
    index_offset: int
    index_size: int
    hash_offset: int
    hash_size: int
    file_size: int
    root_geometric_error: float
    dataset_id: bytes
    index_crc32: int
    content_kind: int
    header_crc32: int


@dataclass(frozen=True)
class PackIndexEntry:
    """One decoded index entry: its key, region, error and blob extents."""

    level: int
    tx: int
    ty: int
    tz: int
    region: Region
    geometric_error: float
    primary_offset: int
    texture_offset: int
    primary_size: int
    texture_size: int
    primary_sha256: bytes
    texture_sha256: bytes


@dataclass(frozen=True, eq=False)
class Pack:
    """A fully validated pack: header, level directory, entries and blobs.

    Contract (fields):
        - ``header``: the parsed :class:`PackHeader`.
        - ``levels``: the level directory in ascending level order.
        - ``entries``: the index entries in ``(level, tz, ty, tx)`` order.

    ``primary_blob`` / ``texture_blob`` slice the tile's blob bytes out of
    the pack. ``eq=False``: wraps the raw file bytes, so instances compare
    by identity.
    """

    header: PackHeader
    levels: tuple[LevelRecord, ...]
    entries: tuple[PackIndexEntry, ...]
    _data: bytes = field(repr=False)

    def primary_blob(self, index: int) -> bytes:
        """Return entry ``index``'s primary blob bytes."""
        entry = self.entries[index]
        return self._data[
            entry.primary_offset : entry.primary_offset + entry.primary_size
        ]

    def texture_blob(self, index: int) -> bytes | None:
        """Return entry ``index``'s texture blob, or ``None`` if absent."""
        entry = self.entries[index]
        if entry.texture_size == 0:
            return None
        return self._data[
            entry.texture_offset : entry.texture_offset + entry.texture_size
        ]


# --------------------------------------------------------------------------
# Writer
# --------------------------------------------------------------------------


def write_pack(
    dest: Path | BinaryIO,
    entries: Sequence[PackEntry],
    blob_source: Callable[[TileKey], tuple[bytes, bytes | None]],
    *,
    root_geometric_error: float,
    content_kind: int,
) -> bytes:
    """Stream a spec-valid ``AHNP`` pack to ``dest`` in one bounded pass.

    Contract:
        - ``entries`` may arrive in any order; they are sorted internally by
          ``(level, tz, ty, tx)`` and define the pack's ``tile_count`` and
          ``level_count`` (the distinct levels, which must be a contiguous
          ``0..N-1``).
        - ``blob_source(key)`` returns ``(primary_bytes, texture_bytes)``
          for one tile, fetched lazily in index order so at most one tile's
          blobs are resident. ``texture_bytes`` is ``None`` for a game pack
          (``content_kind = 1``) and non-empty for a heightfield pack
          (``content_kind = 0``).
        - Writes the header (with ``dataset_id`` and both CRCs), the level
          directory, the sorted index, the hash section, then every blob in
          index order 16-aligned with zero inter-blob padding, and returns
          the pack's ``dataset_id`` (32 bytes). ``dest`` may be a path or a
          seekable binary file; a path is opened, written and closed.
        - Deterministic: identical bytes for identical inputs.

    Failure modes (each a :class:`~ahn_cli.tiles3d.errors.Tiles3dError`):
        - ``content_kind`` not in ``{0, 1}``; ``root_geometric_error``
          non-finite; ``entries`` empty; a duplicate tile key; a non-zero
          ``tz``; a non-finite or non-well-ordered region / geometric error;
          a level set that is not a contiguous ``0..N-1``; an empty primary
          blob; a heightfield tile with a missing / empty texture, or a game
          tile whose ``blob_source`` returns a texture.
    """
    if content_kind not in (CONTENT_KIND_HEIGHTFIELD, CONTENT_KIND_GAME):
        msg = f"pack content_kind {content_kind} is not 0 (heightfield) or 1 (game)."
        raise Tiles3dError(msg)
    if not math.isfinite(root_geometric_error):
        msg = (
            f"pack root_geometric_error {root_geometric_error} is non-finite."
        )
        raise Tiles3dError(msg)
    if not entries:
        msg = "pack must have at least one tile; entries is empty."
        raise Tiles3dError(msg)
    ordered = _order_entries(entries)
    levels = _level_directory(ordered)
    if isinstance(dest, Path):
        with dest.open("wb") as handle:
            return _stream(
                handle,
                ordered,
                levels,
                blob_source,
                root_geometric_error,
                content_kind,
            )
    return _stream(
        dest, ordered, levels, blob_source, root_geometric_error, content_kind
    )


def _order_entries(entries: Sequence[PackEntry]) -> list[PackEntry]:
    """Sort entries by key and validate keys / regions before writing."""
    ordered = sorted(entries, key=lambda entry: entry.key.sort_key)
    for index, entry in enumerate(ordered):
        key = entry.key
        if key.tz != 0:
            msg = f"pack entry {key} has tz={key.tz}; must be 0 in v1."
            raise Tiles3dError(msg)
        if index > 0 and ordered[index - 1].key.sort_key == key.sort_key:
            msg = f"pack has a duplicate tile key {key}."
            raise Tiles3dError(msg)
        _require_u32(key.level, f"entry {key} level")
        _require_u32(key.tx, f"entry {key} tx")
        _require_u32(key.ty, f"entry {key} ty")
        _require_u32(key.tz, f"entry {key} tz")
        _check_region(entry.region, entry.geometric_error, key)
    return ordered


def _require_u32(value: int, field: str) -> int:
    """Return ``value`` if it fits an unsigned 32-bit field, else reject.

    A struct-level guard so an out-of-range key or blob size surfaces as a
    :class:`~ahn_cli.tiles3d.errors.Tiles3dError` (naming the field and
    value) rather than a raw ``struct.error``.
    """
    if not 0 <= value < _U32_RANGE:
        msg = f"pack {field} {value} is out of range for a 32-bit field."
        raise Tiles3dError(msg)
    return value


def _require_u64(value: int, field: str) -> int:
    """Return ``value`` if it fits an unsigned 64-bit field, else reject.

    Guards the computed absolute offsets/sizes the header and index pack as
    ``uint64`` so an overflow is a typed error, not a ``struct.error``.
    """
    if not 0 <= value < _U64_RANGE:
        msg = f"pack {field} {value} is out of range for a 64-bit field."
        raise Tiles3dError(msg)
    return value


def _check_region(
    region: Region, geometric_error: float, key: TileKey
) -> None:
    """Reject a non-finite or non-well-ordered region / geometric error."""
    values = (*region, geometric_error)
    for name, value in zip(_REGION_FIELD_NAMES, values, strict=True):
        if not math.isfinite(value):
            msg = f"pack entry {key} field {name} is non-finite ({value})."
            raise Tiles3dError(msg)
    if (
        region[0] > region[2]
        or region[1] > region[3]
        or region[4] > region[5]
    ):
        msg = f"pack entry {key} region {region} is not well-ordered."
        raise Tiles3dError(msg)


def _level_directory(ordered: list[PackEntry]) -> list[LevelRecord]:
    """Build the level directory, rejecting a non-contiguous level set."""
    max_level = ordered[-1].key.level
    records: list[LevelRecord] = []
    cursor = 0
    for level in range(max_level + 1):
        run = [entry for entry in ordered if entry.key.level == level]
        if not run:
            msg = (
                f"pack levels must be a contiguous 0..N range; level "
                f"{level} has no tiles."
            )
            raise Tiles3dError(msg)
        records.append(
            LevelRecord(
                first_entry=cursor,
                entry_count=len(run),
                tx_count=len({entry.key.tx for entry in run}),
                ty_count=len({entry.key.ty for entry in run}),
            )
        )
        cursor += len(run)
    return records


@dataclass(frozen=True)
class _Layout:
    """The derived region sizes/offsets of a pack, from its two counts."""

    tile_count: int
    level_count: int
    index_size: int
    hash_offset: int
    hash_size: int
    blob_region_start: int


def _layout(tile_count: int, level_count: int) -> _Layout:
    """Derive the pack's region offsets/sizes from its tile / level counts."""
    index_size = (
        level_count * LEVEL_RECORD_SIZE + tile_count * INDEX_ENTRY_SIZE
    )
    hash_offset = INDEX_OFFSET + index_size
    hash_size = tile_count * HASH_RECORD_SIZE
    return _Layout(
        tile_count=tile_count,
        level_count=level_count,
        index_size=index_size,
        hash_offset=hash_offset,
        hash_size=hash_size,
        blob_region_start=hash_offset + hash_size,
    )


def _stream(
    handle: BinaryIO,
    ordered: list[PackEntry],
    levels: list[LevelRecord],
    blob_source: Callable[[TileKey], tuple[bytes, bytes | None]],
    root_geometric_error: float,
    content_kind: int,
) -> bytes:
    """Write blobs from ``blob_region_start`` then seek back to patch meta."""
    layout = _layout(len(ordered), len(levels))
    handle.seek(layout.blob_region_start)
    cursor = layout.blob_region_start
    index_bytes = bytearray()
    hash_bytes = bytearray()
    for entry in ordered:
        primary, texture = blob_source(entry.key)
        if not primary:
            msg = f"pack entry {entry.key} has an empty primary blob."
            raise Tiles3dError(msg)
        _require_u32(len(primary), f"entry {entry.key} primary_size")
        cursor = _align(handle, cursor)
        primary_offset = cursor
        handle.write(primary)
        cursor += len(primary)
        primary_sha = hashlib.sha256(primary).digest()
        texture_offset, texture_size, texture_sha, cursor = _write_texture(
            handle, texture, cursor, content_kind, entry.key
        )
        index_bytes += _pack_entry(
            entry,
            primary_offset,
            texture_offset,
            len(primary),
            texture_size,
        )
        hash_bytes += struct.pack(_HASH_FMT, primary_sha, texture_sha)
    file_size = cursor

    directory_bytes = b"".join(
        struct.pack(
            _LEVEL_FMT,
            record.first_entry,
            record.entry_count,
            record.tx_count,
            record.ty_count,
        )
        for record in levels
    )
    dataset_id = hashlib.sha256(hash_bytes).digest()
    index_crc32 = zlib.crc32(directory_bytes + bytes(index_bytes)) & _U32_MAX
    header = _pack_header(
        layout,
        file_size,
        root_geometric_error,
        dataset_id,
        index_crc32,
        content_kind,
    )
    handle.seek(0)
    handle.write(
        header + directory_bytes + bytes(index_bytes) + bytes(hash_bytes)
    )
    return dataset_id


def _write_texture(
    handle: BinaryIO,
    texture: bytes | None,
    cursor: int,
    content_kind: int,
    key: TileKey,
) -> tuple[int, int, bytes, int]:
    """Write (or refuse) the texture blob; return its offset/size/sha/cursor."""
    if content_kind == CONTENT_KIND_GAME:
        if texture is not None:
            msg = f"pack entry {key}: game tile must have no texture blob."
            raise Tiles3dError(msg)
        return 0, 0, NO_TEXTURE_SHA256, cursor
    if not texture:
        msg = f"pack entry {key}: heightfield tile needs a non-empty texture."
        raise Tiles3dError(msg)
    _require_u32(len(texture), f"entry {key} texture_size")
    cursor = _align(handle, cursor)
    texture_offset = cursor
    handle.write(texture)
    cursor += len(texture)
    return (
        texture_offset,
        len(texture),
        hashlib.sha256(texture).digest(),
        cursor,
    )


def _align(handle: BinaryIO, cursor: int) -> int:
    """Write zero padding up to the next 16-byte boundary; return the offset."""
    pad = (-cursor) % BLOB_ALIGNMENT
    if pad:
        handle.write(b"\x00" * pad)
    return cursor + pad


def _pack_entry(
    entry: PackEntry,
    primary_offset: int,
    texture_offset: int,
    primary_size: int,
    texture_size: int,
) -> bytes:
    """Pack one 96-byte index entry."""
    key = entry.key
    return struct.pack(
        _ENTRY_FMT,
        key.level,
        key.tx,
        key.ty,
        key.tz,
        *entry.region,
        entry.geometric_error,
        _require_u64(primary_offset, f"entry {key} primary_offset"),
        _require_u64(texture_offset, f"entry {key} texture_offset"),
        primary_size,
        texture_size,
    )


def _pack_header(
    layout: _Layout,
    file_size: int,
    root_geometric_error: float,
    dataset_id: bytes,
    index_crc32: int,
    content_kind: int,
) -> bytes:
    """Pack the 128-byte header, computing ``header_crc32`` over ``[0, 124)``."""
    prefix = struct.pack(
        _HEADER_PREFIX_FMT,
        MAGIC,
        FORMAT_VERSION,
        _require_u32(layout.tile_count, "tile_count"),
        _require_u32(layout.level_count, "level_count"),
        INDEX_OFFSET,
        _require_u64(layout.index_size, "index_size"),
        _require_u64(layout.hash_offset, "hash_offset"),
        _require_u64(layout.hash_size, "hash_size"),
        _require_u64(file_size, "file_size"),
        root_geometric_error,
        dataset_id,
        index_crc32,
        0,
        content_kind,
        b"\x00" * 16,
    )
    header_crc32 = zlib.crc32(prefix) & _U32_MAX
    return prefix + struct.pack("<I", header_crc32)


# --------------------------------------------------------------------------
# Reader
# --------------------------------------------------------------------------


def read_pack(source: Path | bytes) -> Pack:
    """Read and fully validate a pack, returning a :class:`Pack`.

    Contract:
        - Accepts a path or the raw pack bytes, and runs every reject in the
          spec's conforming-reader checklist in an order that never trusts
          an unverified count: ``header_crc32`` before any count,
          ``index_crc32`` before the entries, then the level directory, the
          index entries, the blob extents/alignment/padding, and finally the
          cold checks — ``dataset_id`` recomputed from the hash section and
          every blob's SHA-256 against its hash record.
        - Returns the header, level directory, entries and blob accessors.

    Failure modes:
        - :class:`~ahn_cli.tiles3d.errors.Tiles3dError` for any violation
          (bad magic/version/CRCs, truncation/trailing bytes, unsorted or
          duplicate keys, ``tz != 0``, unknown ``content_kind``, non-zero
          ``reserved``/``pad``, misaligned/overlapping/out-of-bounds or
          non-ascending blobs, non-zero inter-blob padding, a
          non-well-ordered region, a ``dataset_id`` or blob-hash mismatch).
    """
    data = source.read_bytes() if isinstance(source, Path) else bytes(source)
    header = _read_header(data)
    levels = _read_directory(data, header)
    entries = _read_entries(data, header, levels)
    _verify_blobs(data, header, entries)
    _verify_hashes(data, header, entries)
    return Pack(
        header=header,
        levels=tuple(levels),
        entries=tuple(entries),
        _data=data,
    )


def _read_header(data: bytes) -> PackHeader:
    """Parse and validate the 128-byte header, deriving nothing untrusted."""
    if len(data) < HEADER_SIZE:
        msg = f"pack is {len(data)} bytes, shorter than the {HEADER_SIZE}-byte header."
        raise Tiles3dError(msg)
    fields = struct.unpack(_HEADER_FMT, data[:HEADER_SIZE])
    if fields[0] != MAGIC:
        msg = f"pack has bad magic {fields[0]!r}; expected {MAGIC!r}."
        raise Tiles3dError(msg)
    if int(fields[1]) != FORMAT_VERSION:
        msg = f"pack format_version {int(fields[1])} is not {FORMAT_VERSION}."
        raise Tiles3dError(msg)
    stored_crc = int(fields[15])
    actual_crc = zlib.crc32(data[:_HEADER_CRC_SPAN]) & _U32_MAX
    if actual_crc != stored_crc:
        msg = (
            f"pack header_crc32 {stored_crc:#010x} does not match the "
            f"computed {actual_crc:#010x}; the header is corrupt."
        )
        raise Tiles3dError(msg)
    _validate_header_fields(data)
    header = PackHeader(
        format_version=FORMAT_VERSION,
        tile_count=int(fields[2]),
        level_count=int(fields[3]),
        index_offset=int(fields[4]),
        index_size=int(fields[5]),
        hash_offset=int(fields[6]),
        hash_size=int(fields[7]),
        file_size=int(fields[8]),
        root_geometric_error=float(fields[9]),
        dataset_id=bytes(fields[10]),
        index_crc32=int(fields[11]),
        content_kind=int(fields[13]),
        header_crc32=stored_crc,
    )
    _check_layout(data, header)
    return header


def _validate_header_fields(data: bytes) -> None:
    """Reject bad counts, a non-finite root error, or reserved/pad/kind."""
    fields = struct.unpack(_HEADER_FMT, data[:HEADER_SIZE])
    tile_count = int(fields[2])
    level_count = int(fields[3])
    if tile_count == 0 or level_count == 0:
        msg = (
            f"pack must have >= 1 tile and >= 1 level; got tile_count="
            f"{tile_count}, level_count={level_count}."
        )
        raise Tiles3dError(msg)
    if level_count > tile_count:
        msg = (
            f"pack level_count {level_count} exceeds tile_count {tile_count} "
            f"(each level holds >= 1 tile)."
        )
        raise Tiles3dError(msg)
    root_geometric_error = float(fields[9])
    if not math.isfinite(root_geometric_error):
        msg = (
            f"pack root_geometric_error {root_geometric_error} is non-finite."
        )
        raise Tiles3dError(msg)
    reserved = int(fields[12])
    if reserved != 0:
        msg = f"pack reserved field is {reserved}, must be 0."
        raise Tiles3dError(msg)
    if fields[14] != b"\x00" * 16:
        msg = "pack header pad contains a non-zero byte, must be all zero."
        raise Tiles3dError(msg)
    content_kind = int(fields[13])
    if content_kind not in (CONTENT_KIND_HEIGHTFIELD, CONTENT_KIND_GAME):
        msg = f"pack content_kind {content_kind} is not 0 or 1."
        raise Tiles3dError(msg)


def _check_layout(data: bytes, header: PackHeader) -> None:
    """Reject any stored offset/size that disagrees with the counts."""
    layout = _layout(header.tile_count, header.level_count)
    checks = (
        ("index_offset", header.index_offset, INDEX_OFFSET),
        ("index_size", header.index_size, layout.index_size),
        ("hash_offset", header.hash_offset, layout.hash_offset),
        ("hash_size", header.hash_size, layout.hash_size),
        ("file_size", header.file_size, len(data)),
    )
    for name, stored, expected in checks:
        if stored != expected:
            msg = f"pack {name} is {stored}, expected {expected}."
            raise Tiles3dError(msg)
    if layout.blob_region_start > len(data):
        msg = "pack index/hash region extends past end of file (truncated)."
        raise Tiles3dError(msg)
    index_region = data[INDEX_OFFSET : INDEX_OFFSET + layout.index_size]
    actual_crc = zlib.crc32(index_region) & _U32_MAX
    if actual_crc != header.index_crc32:
        msg = (
            f"pack index_crc32 {header.index_crc32:#010x} does not match the "
            f"computed {actual_crc:#010x}; the index is corrupt."
        )
        raise Tiles3dError(msg)


def _read_directory(data: bytes, header: PackHeader) -> list[LevelRecord]:
    """Parse the level directory and validate its run continuity."""
    records: list[LevelRecord] = []
    cursor = 0
    for level in range(header.level_count):
        offset = INDEX_OFFSET + level * LEVEL_RECORD_SIZE
        first_entry, entry_count, tx_count, ty_count = struct.unpack_from(
            _LEVEL_FMT, data, offset
        )
        if first_entry != cursor:
            msg = (
                f"pack level {level} first_entry {first_entry} breaks run "
                f"continuity (expected {cursor})."
            )
            raise Tiles3dError(msg)
        records.append(
            LevelRecord(
                first_entry=first_entry,
                entry_count=entry_count,
                tx_count=tx_count,
                ty_count=ty_count,
            )
        )
        cursor += entry_count
    if cursor != header.tile_count:
        msg = (
            f"pack level directory covers {cursor} entries, not tile_count "
            f"{header.tile_count}."
        )
        raise Tiles3dError(msg)
    return records


def _read_entries(
    data: bytes, header: PackHeader, levels: list[LevelRecord]
) -> list[PackIndexEntry]:
    """Parse and validate the sorted index entries."""
    base = INDEX_OFFSET + header.level_count * LEVEL_RECORD_SIZE
    level_of = _level_of_entry(levels)
    hash_base = header.hash_offset
    entries: list[PackIndexEntry] = []
    previous_order: tuple[int, int, int, int] | None = None
    for index in range(header.tile_count):
        fields = struct.unpack_from(
            _ENTRY_FMT, data, base + index * INDEX_ENTRY_SIZE
        )
        level, tx, ty, tz = (
            int(fields[0]),
            int(fields[1]),
            int(fields[2]),
            int(fields[3]),
        )
        if tz != 0:
            msg = f"pack entry {index} has tz={tz}; must be 0 in v1."
            raise Tiles3dError(msg)
        region: Region = (
            float(fields[4]),
            float(fields[5]),
            float(fields[6]),
            float(fields[7]),
            float(fields[8]),
            float(fields[9]),
        )
        geometric_error = float(fields[10])
        _check_region(region, geometric_error, TileKey(level, tx, ty, tz))
        if level != level_of[index]:
            msg = (
                f"pack entry {index} declares level {level} but sits in the "
                f"level {level_of[index]} run."
            )
            raise Tiles3dError(msg)
        order = (level, tz, ty, tx)
        if previous_order is not None and order <= previous_order:
            msg = (
                f"pack entry {index} key {order} is not strictly after the "
                f"previous {previous_order} (unsorted or duplicate)."
            )
            raise Tiles3dError(msg)
        previous_order = order
        primary_sha, texture_sha = struct.unpack_from(
            _HASH_FMT, data, hash_base + index * HASH_RECORD_SIZE
        )
        entries.append(
            PackIndexEntry(
                level=level,
                tx=tx,
                ty=ty,
                tz=tz,
                region=region,
                geometric_error=geometric_error,
                primary_offset=int(fields[11]),
                texture_offset=int(fields[12]),
                primary_size=int(fields[13]),
                texture_size=int(fields[14]),
                primary_sha256=bytes(primary_sha),
                texture_sha256=bytes(texture_sha),
            )
        )
    return entries


def _level_of_entry(levels: list[LevelRecord]) -> list[int]:
    """Return the level owning each entry index, from the directory runs."""
    owner: list[int] = []
    for level, record in enumerate(levels):
        owner.extend([level] * record.entry_count)
    return owner


def _verify_blobs(
    data: bytes, header: PackHeader, entries: list[PackIndexEntry]
) -> None:
    """Validate every blob's alignment, ordering, bounds and padding."""
    blob_region_start = header.hash_offset + header.hash_size
    cursor = blob_region_start
    previous_offset = -1
    for index, entry in enumerate(entries):
        cursor, previous_offset = _check_blob(
            data,
            entry.primary_offset,
            entry.primary_size,
            cursor,
            previous_offset,
            f"entry {index} primary",
        )
        if header.content_kind == CONTENT_KIND_GAME:
            if entry.texture_offset != 0 or entry.texture_size != 0:
                msg = (
                    f"pack entry {index}: game tile must have texture_offset "
                    f"and texture_size 0."
                )
                raise Tiles3dError(msg)
            continue
        if entry.texture_size == 0:
            msg = f"pack entry {index}: heightfield tile has no texture blob."
            raise Tiles3dError(msg)
        cursor, previous_offset = _check_blob(
            data,
            entry.texture_offset,
            entry.texture_size,
            cursor,
            previous_offset,
            f"entry {index} texture",
        )
    if cursor != header.file_size:
        msg = (
            f"pack has {header.file_size - cursor} trailing byte(s) after the "
            f"final blob."
        )
        raise Tiles3dError(msg)


def _check_blob(
    data: bytes,
    offset: int,
    size: int,
    cursor: int,
    previous_offset: int,
    label: str,
) -> tuple[int, int]:
    """Validate one blob's alignment/order/padding/bounds; advance the cursor."""
    if offset % BLOB_ALIGNMENT != 0:
        msg = f"pack {label} offset {offset} is not 16-byte aligned."
        raise Tiles3dError(msg)
    if offset <= previous_offset:
        msg = f"pack {label} offset {offset} is not strictly ascending."
        raise Tiles3dError(msg)
    if offset < cursor:
        msg = f"pack {label} offset {offset} overlaps the previous blob."
        raise Tiles3dError(msg)
    if data[cursor:offset] != b"\x00" * (offset - cursor):
        msg = f"pack {label}: inter-blob padding before offset {offset} is non-zero."
        raise Tiles3dError(msg)
    if offset + size > len(data):
        msg = f"pack {label} range ends past end of file."
        raise Tiles3dError(msg)
    return offset + size, offset


def _verify_hashes(
    data: bytes, header: PackHeader, entries: list[PackIndexEntry]
) -> None:
    """Recompute ``dataset_id`` then every blob's SHA-256 (cold checks)."""
    hash_section = data[
        header.hash_offset : header.hash_offset + header.hash_size
    ]
    dataset_id = hashlib.sha256(hash_section).digest()
    if dataset_id != header.dataset_id:
        msg = (
            "pack dataset_id does not match the SHA-256 of the hash section "
            "(hash section corrupt)."
        )
        raise Tiles3dError(msg)
    for index, entry in enumerate(entries):
        primary = data[
            entry.primary_offset : entry.primary_offset + entry.primary_size
        ]
        if hashlib.sha256(primary).digest() != entry.primary_sha256:
            msg = f"pack entry {index} primary blob fails its SHA-256."
            raise Tiles3dError(msg)
        if header.content_kind == CONTENT_KIND_GAME:
            if entry.texture_sha256 != NO_TEXTURE_SHA256:
                msg = (
                    f"pack entry {index}: game tile texture_sha256 must be 32 "
                    f"zero bytes."
                )
                raise Tiles3dError(msg)
            continue
        texture = data[
            entry.texture_offset : entry.texture_offset + entry.texture_size
        ]
        if hashlib.sha256(texture).digest() != entry.texture_sha256:
            msg = f"pack entry {index} texture blob fails its SHA-256."
            raise Tiles3dError(msg)
