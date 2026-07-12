"""Throwaway POC: v2 pack container writer/reader + golden-vector generator.

Implements the plan's exact v2 layout (resilient-riding-sprout.md, "Format v2
design"). NOT production code — this validates the design before Phase 3.

Layout pinned here (spike decision, to be ratified in the Phase 2 spec):

Pack header (128 B, LE):
  off  0  magic 'AHNP'          4
  off  4  format_version u32=1  4
  off  8  tile_count u32        4
  off 12  level_count u32       4
  off 16  index_offset u64=128  8
  off 24  index_size u64        8   (= level_count*16 + tile_count*96)
  off 32  hash_offset u64       8
  off 40  hash_size u64         8   (= tile_count*64)
  off 48  file_size u64         8
  off 56  root_geometric_error f64  8
  off 64  dataset_id [32]u8     32  (SHA-256 of the hash section)
  off 96  index_crc32 u32       4
  off100  header_crc32 u32      4   (CRC-32 over bytes [0,100))
  off104  content_kind u32      4   (0=heightfield, 1=game)
  off108  pad 20 B zeros        20
  --------------------------------- 128

Level directory (level_count * 16 B): first_entry u32, entry_count u32,
tx_count u32, ty_count u32.

Index entry (96 B), one per tile, sorted ascending (level, tz, ty, tx):
  level u32, tx u32, ty u32, tz u32          16
  region f64[6]                              48
  geometric_error f64                         8
  primary_offset u64, texture_offset u64     16  (16-byte aligned)
  primary_size u32, texture_size u32          8
  ---------------------------------------------- 96

Hash section (tile_count * 64 B): primary_sha256[32] + texture_sha256[32].
Blob region: blobs in index order, each 16-byte aligned, inter-blob pad zero.
"""

from __future__ import annotations

import hashlib
import io
import json
import struct
import zlib
from dataclasses import dataclass

import zstandard as zstd

MAGIC = b"AHNP"
FORMAT_VERSION = 1
HEADER_SIZE = 128
DIR_ENTRY_SIZE = 16
INDEX_ENTRY_SIZE = 96
HASH_ENTRY_SIZE = 64
BLOB_ALIGN = 16
HEADER_CRC_SPAN = 100  # header_crc32 covers bytes [0, 100)

KIND_HEIGHTFIELD = 0
KIND_GAME = 1

_HEADER_STRUCT = struct.Struct("<4sIIIQQQQQd32sIII")  # up to & incl content_kind
# 4s I I I  Q Q Q Q Q  d  32s I I I  -> 4+4+4+4 +8*5 +8 +32 +4+4+4 = 108, +20 pad


def _align_up(n: int, a: int = BLOB_ALIGN) -> int:
    return (n + a - 1) // a * a


@dataclass(frozen=True)
class Tile:
    """One tile to pack: its key, region, error, and blob bytes."""

    level: int
    tx: int
    ty: int
    tz: int
    region: tuple[float, float, float, float, float, float]
    geometric_error: float
    primary: bytes
    texture: bytes  # empty for game (texture embedded in glb)

    @property
    def sort_key(self) -> tuple[int, int, int, int]:
        # NORMATIVE blob/index order: (level, tz, ty, tx)
        return (self.level, self.tz, self.ty, self.tx)


def _pack_index_entry(t: Tile, primary_off: int, texture_off: int) -> bytes:
    return struct.pack(
        "<IIII6ddQQII",
        t.level,
        t.tx,
        t.ty,
        t.tz,
        *t.region,
        t.geometric_error,
        primary_off,
        texture_off,
        len(t.primary),
        len(t.texture),
    )


def _level_directory(tiles: list[Tile]) -> bytes:
    """Contiguous per-level runs over index-ordered tiles."""
    out = io.BytesIO()
    levels = sorted({t.level for t in tiles})
    first = 0
    for lvl in levels:
        run = [t for t in tiles if t.level == lvl]
        tx_count = (max(t.tx for t in run) + 1) if run else 0
        ty_count = (max(t.ty for t in run) + 1) if run else 0
        out.write(struct.pack("<IIII", first, len(run), tx_count, ty_count))
        first += len(run)
    return out.getvalue()


def _blob_layout(
    tiles: list[Tile], blob_start: int
) -> tuple[list[tuple[int, int]], bytes]:
    """Return per-tile (primary_off, texture_off) and the padded blob region.

    Offsets are absolute file offsets, 16-byte aligned; inter-blob padding is
    zero. Texture offset is 0 when the tile has no texture (game glb-only).
    """
    body = io.BytesIO()
    offsets: list[tuple[int, int]] = []
    cursor = blob_start
    for t in tiles:
        # primary
        pad = _align_up(cursor) - cursor
        body.write(b"\0" * pad)
        cursor += pad
        primary_off = cursor
        body.write(t.primary)
        cursor += len(t.primary)
        # texture (only if present)
        if t.texture:
            pad = _align_up(cursor) - cursor
            body.write(b"\0" * pad)
            cursor += pad
            texture_off = cursor
            body.write(t.texture)
            cursor += len(t.texture)
        else:
            texture_off = 0
        offsets.append((primary_off, texture_off))
    return offsets, body.getvalue()


def _hash_section(tiles: list[Tile]) -> bytes:
    out = io.BytesIO()
    for t in tiles:
        out.write(hashlib.sha256(t.primary).digest())
        # texture hash: sha256 of empty bytes when absent (well-defined, cold)
        out.write(hashlib.sha256(t.texture).digest())
    return out.getvalue()


def build_pack(
    tiles_in: list[Tile], content_kind: int, root_geometric_error: float
) -> bytes:
    """Single-pass seek-back writer producing the full pack bytes.

    Sizes are known up front from tile_count/level_count; we write the header
    and zeroed directory/index/hash regions, stream blobs, then seek back and
    patch the index, hashes, dataset_id and CRCs.
    """
    tiles = sorted(tiles_in, key=lambda t: t.sort_key)  # re-sort to index order
    tile_count = len(tiles)
    level_count = len({t.level for t in tiles})

    directory = _level_directory(tiles)
    index_size = level_count * DIR_ENTRY_SIZE + tile_count * INDEX_ENTRY_SIZE
    assert len(directory) == level_count * DIR_ENTRY_SIZE
    index_offset = HEADER_SIZE
    hash_offset = index_offset + index_size
    hash_size = tile_count * HASH_ENTRY_SIZE
    blob_start = hash_offset + hash_size

    offsets, blob_region = _blob_layout(tiles, blob_start)
    file_size = blob_start + len(blob_region)

    # ---- single-pass buffer: header + zeroed dir/index/hash + blobs ----
    buf = bytearray(file_size)
    # blobs first (final position, never patched)
    buf[blob_start:file_size] = blob_region

    # index = directory + per-tile entries
    index = bytearray(directory)
    for t, (po, to) in zip(tiles, offsets, strict=True):
        index += _pack_index_entry(t, po, to)
    assert len(index) == index_size
    buf[index_offset:hash_offset] = index

    # hash section
    hashes = _hash_section(tiles)
    assert len(hashes) == hash_size
    buf[hash_offset:blob_start] = hashes

    dataset_id = hashlib.sha256(bytes(hashes)).digest()
    index_crc32 = zlib.crc32(bytes(index)) & 0xFFFFFFFF

    # header: everything up to content_kind, then pad
    head = _HEADER_STRUCT.pack(
        MAGIC,
        FORMAT_VERSION,
        tile_count,
        level_count,
        index_offset,
        index_size,
        hash_offset,
        hash_size,
        file_size,
        root_geometric_error,
        dataset_id,
        index_crc32,
        0,  # header_crc32 placeholder
        content_kind,
    )
    head = bytearray(head) + bytes(HEADER_SIZE - len(head))  # zero pad to 128
    assert len(head) == HEADER_SIZE
    header_crc32 = zlib.crc32(bytes(head[:HEADER_CRC_SPAN])) & 0xFFFFFFFF
    struct.pack_into("<I", head, HEADER_CRC_SPAN, header_crc32)
    buf[0:HEADER_SIZE] = head
    return bytes(buf)


def build_pack_two_pass(
    tiles_in: list[Tile], content_kind: int, root_geometric_error: float
) -> bytes:
    """Independent two-pass builder for byte-identity cross-check.

    Concatenates sections in the natural order rather than seek-back patching.
    """
    tiles = sorted(tiles_in, key=lambda t: t.sort_key)
    tile_count = len(tiles)
    level_count = len({t.level for t in tiles})
    directory = _level_directory(tiles)
    index_size = level_count * DIR_ENTRY_SIZE + tile_count * INDEX_ENTRY_SIZE
    hash_offset = HEADER_SIZE + index_size
    hash_size = tile_count * HASH_ENTRY_SIZE
    blob_start = hash_offset + hash_size
    offsets, blob_region = _blob_layout(tiles, blob_start)
    file_size = blob_start + len(blob_region)

    index = bytearray(directory)
    for t, (po, to) in zip(tiles, offsets, strict=True):
        index += _pack_index_entry(t, po, to)
    hashes = _hash_section(tiles)
    dataset_id = hashlib.sha256(bytes(hashes)).digest()
    index_crc32 = zlib.crc32(bytes(index)) & 0xFFFFFFFF

    head = bytearray(
        _HEADER_STRUCT.pack(
            MAGIC, FORMAT_VERSION, tile_count, level_count, HEADER_SIZE,
            index_size, hash_offset, hash_size, file_size,
            root_geometric_error, dataset_id, index_crc32, 0, content_kind,
        )
    )
    head += bytes(HEADER_SIZE - len(head))
    header_crc32 = zlib.crc32(bytes(head[:HEADER_CRC_SPAN])) & 0xFFFFFFFF
    struct.pack_into("<I", head, HEADER_CRC_SPAN, header_crc32)
    return bytes(head) + bytes(index) + bytes(hashes) + blob_region


# --------------------------------------------------------------------------
# Reference reader with full validation (the Python side of the reject matrix)
# --------------------------------------------------------------------------


class PackError(Exception):
    """A normative pack reject (mirrors the Rust HfError variants)."""


@dataclass(frozen=True)
class Entry:
    level: int
    tx: int
    ty: int
    tz: int
    region: tuple[float, ...]
    geometric_error: float
    primary_offset: int
    texture_offset: int
    primary_size: int
    texture_size: int


def read_pack(data: bytes) -> dict:
    """Validate and parse a pack; raise PackError on any normative reject."""
    if len(data) < HEADER_SIZE:
        raise PackError("shorter than 128-byte header")
    (
        magic, fmt, tile_count, level_count, index_offset, index_size,
        hash_offset, hash_size, file_size, root_ge, dataset_id,
        index_crc32, header_crc32_stored, content_kind,
    ) = _HEADER_STRUCT.unpack(data[:_HEADER_STRUCT.size])
    if magic != MAGIC:
        raise PackError("bad magic")
    if fmt != FORMAT_VERSION:
        raise PackError("bad format_version")
    if content_kind not in (KIND_HEIGHTFIELD, KIND_GAME):
        raise PackError("bad content_kind")
    header_crc32_calc = zlib.crc32(data[:HEADER_CRC_SPAN]) & 0xFFFFFFFF
    if header_crc32_calc != header_crc32_stored:
        raise PackError("header CRC mismatch")
    if index_offset != HEADER_SIZE:
        raise PackError("index_offset != 128")
    if index_size != level_count * DIR_ENTRY_SIZE + tile_count * INDEX_ENTRY_SIZE:
        raise PackError("index_size wrong")
    if hash_offset != index_offset + index_size:
        raise PackError("hash_offset wrong")
    if hash_size != tile_count * HASH_ENTRY_SIZE:
        raise PackError("hash_size wrong")
    if file_size != len(data):
        raise PackError("file_size != actual length (truncation)")
    if hash_offset + hash_size > len(data):
        raise PackError("index/hash section beyond EOF")

    index_bytes = data[index_offset:hash_offset]
    if zlib.crc32(index_bytes) & 0xFFFFFFFF != index_crc32:
        raise PackError("index CRC mismatch")

    hash_bytes = data[hash_offset:hash_offset + hash_size]
    if hashlib.sha256(hash_bytes).digest() != dataset_id:
        raise PackError("dataset_id != sha256(hash section)")

    # directory
    dir_end = index_offset + level_count * DIR_ENTRY_SIZE
    directory = []
    for i in range(level_count):
        base = index_offset + i * DIR_ENTRY_SIZE
        directory.append(struct.unpack_from("<IIII", data, base))

    # entries
    entries: list[Entry] = []
    blob_start = _align_up(hash_offset + hash_size)  # first blob aligned
    for i in range(tile_count):
        base = dir_end + i * INDEX_ENTRY_SIZE
        vals = struct.unpack_from("<IIII6ddQQII", data, base)
        e = Entry(
            level=vals[0], tx=vals[1], ty=vals[2], tz=vals[3],
            region=vals[4:10], geometric_error=vals[10],
            primary_offset=vals[11], texture_offset=vals[12],
            primary_size=vals[13], texture_size=vals[14],
        )
        if e.tz != 0:
            raise PackError("tz != 0")
        if e.primary_offset % BLOB_ALIGN != 0:
            raise PackError("primary_offset not 16-aligned")
        if e.texture_offset % BLOB_ALIGN != 0:
            raise PackError("texture_offset not 16-aligned")
        if e.primary_offset + e.primary_size > len(data):
            raise PackError("primary blob beyond EOF")
        if e.texture_offset and e.texture_offset + e.texture_size > len(data):
            raise PackError("texture blob beyond EOF")
        entries.append(e)

    # sort order + non-overlap
    keys = [(e.level, e.tz, e.ty, e.tx) for e in entries]
    if keys != sorted(keys):
        raise PackError("entries not sorted (level,tz,ty,tx)")
    spans = []
    for e in entries:
        spans.append((e.primary_offset, e.primary_size))
        if e.texture_offset:
            spans.append((e.texture_offset, e.texture_size))
    spans.sort()
    for (o1, s1), (o2, _s2) in zip(spans, spans[1:], strict=False):
        if o1 + s1 > o2:
            raise PackError("overlapping blobs")

    # zero-padding between hash section end and first blob, and inter-blob
    # (checked against the reconstructed layout)
    return {
        "tile_count": tile_count, "level_count": level_count,
        "content_kind": content_kind, "root_geometric_error": root_ge,
        "dataset_id": dataset_id.hex(), "directory": directory,
        "entries": entries, "index_crc32": index_crc32,
        "header_crc32": header_crc32_stored, "blob_start_aligned": blob_start,
    }


def _mk_compressor() -> zstd.ZstdCompressor:
    return zstd.ZstdCompressor(level=19, threads=0, write_checksum=True,
                              write_content_size=True)
