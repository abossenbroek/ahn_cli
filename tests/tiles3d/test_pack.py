"""Tests for the ``AHNP`` pack container writer + validating reader.

The container is checked against the normative byte layout restated here
(not against its own source): a golden micro-pack for both content kinds
with struct-level assertions at every documented offset, writer
determinism against an independent in-memory two-pass reference, the sort
over sparse / non-square grids, and one negative test per reject in the
spec's conforming-reader checklist (with the full truncation matrix and a
hash-section bit-flip caught by the ``dataset_id`` recompute).
"""

from __future__ import annotations

import hashlib
import io
import struct
import zlib
from typing import TYPE_CHECKING

import pytest

from ahn_cli.tiles3d.errors import Tiles3dError
from ahn_cli.tiles3d.pack import (
    BLOB_ALIGNMENT,
    CONTENT_KIND_GAME,
    CONTENT_KIND_HEIGHTFIELD,
    FORMAT_VERSION,
    HASH_RECORD_SIZE,
    HEADER_SIZE,
    INDEX_ENTRY_SIZE,
    INDEX_OFFSET,
    LEVEL_RECORD_SIZE,
    MAGIC,
    NO_TEXTURE_SHA256,
    PackEntry,
    TileKey,
    read_pack,
    write_pack,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    BlobSource = Callable[[TileKey], tuple[bytes, bytes | None]]

# The layout restated from the normative spec, so the container is checked
# against the document rather than against its own constants.
_HEADER_PREFIX_FMT = "<4sIIIQQQQQd32sIII16s"
_HEADER_FMT = _HEADER_PREFIX_FMT + "I"
_LEVEL_FMT = "<IIII"
_ENTRY_FMT = "<4I7d2Q2I"
_HASH_FMT = "<32s32s"

# Absolute header field offsets.
_O_MAGIC, _O_VERSION, _O_TILE_COUNT, _O_LEVEL_COUNT = 0, 4, 8, 12
_O_INDEX_OFFSET, _O_INDEX_SIZE, _O_HASH_OFFSET, _O_HASH_SIZE = 16, 24, 32, 40
_O_FILE_SIZE, _O_ROOT_GE, _O_DATASET_ID = 48, 56, 64
_O_INDEX_CRC, _O_RESERVED, _O_CONTENT_KIND, _O_PAD, _O_HEADER_CRC = (
    96,
    100,
    104,
    108,
    124,
)

# Within an index entry (bytes from the entry's start).
_E_TX, _E_TY, _E_TZ = 4, 8, 12
_E_REGION0 = 16
_E_PRIMARY_OFFSET, _E_TEXTURE_OFFSET = 72, 80
_E_PRIMARY_SIZE, _E_TEXTURE_SIZE = 88, 92

_U32 = 0xFFFFFFFF


def _hf_entries() -> list[PackEntry]:
    """Three heightfield entries across two levels, deliberately unsorted."""
    return [
        PackEntry(TileKey(1, 1, 0), (0.10, 0.20, 0.30, 0.40, 1.0, 2.0), 0.0),
        PackEntry(TileKey(0, 0, 0), (0.00, 0.05, 0.50, 0.60, 0.0, 3.0), 4.0),
        PackEntry(TileKey(1, 0, 0), (0.00, 0.05, 0.20, 0.40, 0.5, 2.5), 0.0),
    ]


# Primary sizes deliberately mix a 16-multiple (no padding) and non-multiples
# (padding), so both the pad==0 and pad>0 writer branches are exercised.
_HF_BLOBS = {
    (0, 0, 0): (b"PRIMARY-ROOT-XYZ", b"jpeg-root-bytes"),
    (1, 0, 0): (b"prim-10", b"tex-10-longer-data"),
    (1, 1, 0): (b"prim-11-data!", b"t11"),
}


def _hf_source(key: TileKey) -> tuple[bytes, bytes | None]:
    return _HF_BLOBS[(key.level, key.tx, key.ty)]


def _game_entries() -> list[PackEntry]:
    return [
        PackEntry(TileKey(1, 0, 0), (0.0, 0.0, 0.2, 0.4, 0.5, 2.5), 0.0),
        PackEntry(TileKey(0, 0, 0), (0.0, 0.0, 0.5, 0.6, 0.0, 3.0), 4.0),
    ]


_GAME_BLOBS = {
    (0, 0, 0): b"glb-root-bytes--",
    (1, 0, 0): b"glb-child",
}


def _game_source(key: TileKey) -> tuple[bytes, bytes | None]:
    return _GAME_BLOBS[(key.level, key.tx, key.ty)], None


def _write_hf(
    entries: list[PackEntry] | None = None, root: float = 8.0
) -> bytes:
    buf = io.BytesIO()
    write_pack(
        buf,
        entries if entries is not None else _hf_entries(),
        _hf_source,
        root_geometric_error=root,
        content_kind=CONTENT_KIND_HEIGHTFIELD,
    )
    return buf.getvalue()


def _write_game() -> bytes:
    buf = io.BytesIO()
    write_pack(
        buf,
        _game_entries(),
        _game_source,
        root_geometric_error=8.0,
        content_kind=CONTENT_KIND_GAME,
    )
    return buf.getvalue()


def _entry_offset(data: bytes | bytearray, index: int) -> int:
    level_count = struct.unpack_from("<I", data, _O_LEVEL_COUNT)[0]
    return (
        INDEX_OFFSET
        + level_count * LEVEL_RECORD_SIZE
        + index * INDEX_ENTRY_SIZE
    )


def _fix_header_crc(data: bytearray) -> None:
    crc = zlib.crc32(bytes(data[:124])) & _U32
    struct.pack_into("<I", data, _O_HEADER_CRC, crc)


def _fix_index_crc(data: bytearray) -> None:
    index_size = struct.unpack_from("<Q", data, _O_INDEX_SIZE)[0]
    crc = (
        zlib.crc32(bytes(data[INDEX_OFFSET : INDEX_OFFSET + index_size]))
        & _U32
    )
    struct.pack_into("<I", data, _O_INDEX_CRC, crc)
    _fix_header_crc(data)


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------


def test_layout_constants() -> None:
    """The exported sizes match the normative layout."""
    assert MAGIC == b"AHNP"
    assert FORMAT_VERSION == 1
    assert HEADER_SIZE == 128
    assert LEVEL_RECORD_SIZE == 16
    assert INDEX_ENTRY_SIZE == 96
    assert HASH_RECORD_SIZE == 64
    assert INDEX_OFFSET == 128
    assert BLOB_ALIGNMENT == 16
    assert NO_TEXTURE_SHA256 == b"\x00" * 32
    assert struct.calcsize(_HEADER_FMT) == HEADER_SIZE
    assert struct.calcsize(_ENTRY_FMT) == INDEX_ENTRY_SIZE


# --------------------------------------------------------------------------
# Golden micro-packs — struct-level assertions at every documented offset
# --------------------------------------------------------------------------


def test_golden_heightfield_header() -> None:
    """A 3-tile heightfield pack's header matches the spec byte-for-byte."""
    data = _write_hf()
    tile_count, level_count = 3, 2
    index_size = level_count * 16 + tile_count * 96
    hash_offset = INDEX_OFFSET + index_size
    hash_size = tile_count * 64

    assert data[_O_MAGIC : _O_MAGIC + 4] == b"AHNP"
    assert struct.unpack_from("<I", data, _O_VERSION)[0] == 1
    assert struct.unpack_from("<I", data, _O_TILE_COUNT)[0] == tile_count
    assert struct.unpack_from("<I", data, _O_LEVEL_COUNT)[0] == level_count
    assert struct.unpack_from("<Q", data, _O_INDEX_OFFSET)[0] == 128
    assert struct.unpack_from("<Q", data, _O_INDEX_SIZE)[0] == index_size
    assert struct.unpack_from("<Q", data, _O_HASH_OFFSET)[0] == hash_offset
    assert struct.unpack_from("<Q", data, _O_HASH_SIZE)[0] == hash_size
    assert struct.unpack_from("<Q", data, _O_FILE_SIZE)[0] == len(data)
    assert struct.unpack_from("<d", data, _O_ROOT_GE)[0] == 8.0
    assert struct.unpack_from("<I", data, _O_RESERVED)[0] == 0
    assert struct.unpack_from("<I", data, _O_CONTENT_KIND)[0] == 0
    assert data[_O_PAD : _O_PAD + 16] == b"\x00" * 16

    # dataset_id = SHA-256 of the hash section; the two CRCs cover their spans.
    hash_section = data[hash_offset : hash_offset + hash_size]
    assert (
        data[_O_DATASET_ID : _O_DATASET_ID + 32]
        == hashlib.sha256(hash_section).digest()
    )
    index_region = data[INDEX_OFFSET:hash_offset]
    assert struct.unpack_from("<I", data, _O_INDEX_CRC)[0] == (
        zlib.crc32(index_region) & _U32
    )
    assert struct.unpack_from("<I", data, _O_HEADER_CRC)[0] == (
        zlib.crc32(bytes(data[:124])) & _U32
    )

    # Level directory: contiguous runs covering the entry array once.
    assert struct.unpack_from(_LEVEL_FMT, data, INDEX_OFFSET) == (0, 1, 1, 1)
    assert struct.unpack_from(_LEVEL_FMT, data, INDEX_OFFSET + 16) == (
        1,
        2,
        2,
        1,
    )


def test_golden_heightfield_entries_and_blobs() -> None:
    """Every index entry, blob and hash record matches the spec layout."""
    data = _write_hf()
    ordered = sorted(_hf_entries(), key=lambda e: e.key.sort_key)
    level_count = 2
    index_size = level_count * 16 + len(ordered) * 96
    hash_offset = INDEX_OFFSET + index_size
    blob_region_start = hash_offset + len(ordered) * 64
    entry_base = INDEX_OFFSET + level_count * 16
    cursor = blob_region_start
    for index, entry in enumerate(ordered):
        fields = struct.unpack_from(_ENTRY_FMT, data, entry_base + index * 96)
        assert (fields[0], fields[1], fields[2], fields[3]) == (
            entry.key.level,
            entry.key.tx,
            entry.key.ty,
            0,
        )
        assert tuple(fields[4:10]) == entry.region
        assert fields[10] == entry.geometric_error
        primary_offset, texture_offset = fields[11], fields[12]
        primary_size, texture_size = fields[13], fields[14]
        primary, texture = _hf_source(entry.key)
        assert texture is not None
        # Alignment + zero inter-blob padding.
        assert primary_offset % 16 == 0
        assert data[cursor:primary_offset] == b"\x00" * (
            primary_offset - cursor
        )
        assert data[primary_offset : primary_offset + primary_size] == primary
        cursor = primary_offset + primary_size
        assert texture_offset % 16 == 0
        assert data[cursor:texture_offset] == b"\x00" * (
            texture_offset - cursor
        )
        assert data[texture_offset : texture_offset + texture_size] == texture
        cursor = texture_offset + texture_size
        # Hash record.
        record = struct.unpack_from(_HASH_FMT, data, hash_offset + index * 64)
        assert record[0] == hashlib.sha256(primary).digest()
        assert record[1] == hashlib.sha256(texture).digest()
    assert cursor == len(data)


def test_golden_game_byte_layout() -> None:
    """A 2-tile game pack zeroes the texture slots and sha, kind == 1."""
    data = _write_game()
    assert struct.unpack_from("<I", data, _O_CONTENT_KIND)[0] == 1
    hash_offset = struct.unpack_from("<Q", data, _O_HASH_OFFSET)[0]
    entry_base = INDEX_OFFSET + 2 * 16
    for index in range(2):
        fields = struct.unpack_from(_ENTRY_FMT, data, entry_base + index * 96)
        assert fields[12] == 0  # texture_offset
        assert fields[14] == 0  # texture_size
        record = struct.unpack_from(_HASH_FMT, data, hash_offset + index * 64)
        assert record[1] == NO_TEXTURE_SHA256


# --------------------------------------------------------------------------
# Reader round-trip
# --------------------------------------------------------------------------


def test_read_round_trips_heightfield() -> None:
    """read_pack recovers entries, levels and blobs in index order."""
    data = _write_hf()
    pack = read_pack(data)
    assert pack.header.tile_count == 3
    assert pack.header.level_count == 2
    assert pack.header.content_kind == CONTENT_KIND_HEIGHTFIELD
    assert pack.header.root_geometric_error == 8.0
    assert [(e.level, e.tx, e.ty) for e in pack.entries] == [
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
    ]
    for index, entry in enumerate(pack.entries):
        primary, texture = _hf_source(
            TileKey(entry.level, entry.tx, entry.ty)
        )
        assert pack.primary_blob(index) == primary
        assert pack.texture_blob(index) == texture
    assert pack.levels[0].entry_count == 1
    assert pack.levels[1].entry_count == 2


def test_read_round_trips_game() -> None:
    """A game pack reads back with no texture blobs."""
    pack = read_pack(_write_game())
    assert pack.header.content_kind == CONTENT_KIND_GAME
    for index, entry in enumerate(pack.entries):
        assert pack.texture_blob(index) is None
        assert (
            pack.primary_blob(index)
            == _GAME_BLOBS[(entry.level, entry.tx, entry.ty)]
        )


def test_read_accepts_a_path(tmp_path: Path) -> None:
    """write_pack to a path and read_pack from a path round-trips."""
    path = tmp_path / "tiles.hfp"
    dataset_id = write_pack(
        path,
        _hf_entries(),
        _hf_source,
        root_geometric_error=8.0,
        content_kind=CONTENT_KIND_HEIGHTFIELD,
    )
    pack = read_pack(path)
    assert pack.header.dataset_id == dataset_id
    assert pack.header.tile_count == 3


# --------------------------------------------------------------------------
# Determinism, sort, two-pass reference
# --------------------------------------------------------------------------


def test_write_is_deterministic() -> None:
    """Encoding identical inputs twice yields identical bytes."""
    assert _write_hf() == _write_hf()
    assert _write_game() == _write_game()


def test_write_sorts_regardless_of_input_order() -> None:
    """The output is invariant to the order entries are supplied in."""
    forward = _write_hf(_hf_entries())
    reversed_entries = list(reversed(_hf_entries()))
    assert _write_hf(reversed_entries) == forward


def _reference_pack(
    entries: list[PackEntry],
    source: BlobSource,
    root: float,
    content_kind: int,
) -> bytes:
    """Build a pack independently, in two in-memory passes (the doubt-P1 oracle)."""
    ordered = sorted(entries, key=lambda e: e.key.sort_key)
    tile_count = len(ordered)
    level_count = ordered[-1].key.level + 1
    index_size = level_count * 16 + tile_count * 96
    hash_offset = 128 + index_size
    hash_size = tile_count * 64
    cursor = hash_offset + hash_size
    blob = bytearray()
    records: list[tuple[PackEntry, int, int, int, int]] = []
    hashes = bytearray()
    for entry in ordered:
        primary, texture = source(entry.key)
        pad = (-cursor) % 16
        blob += b"\x00" * pad
        cursor += pad
        primary_offset = cursor
        blob += primary
        cursor += len(primary)
        primary_sha = hashlib.sha256(primary).digest()
        if content_kind == CONTENT_KIND_GAME:
            texture_offset, texture_size, texture_sha = (
                0,
                0,
                NO_TEXTURE_SHA256,
            )
        else:
            assert texture is not None
            pad = (-cursor) % 16
            blob += b"\x00" * pad
            cursor += pad
            texture_offset = cursor
            blob += texture
            cursor += len(texture)
            texture_size = len(texture)
            texture_sha = hashlib.sha256(texture).digest()
        records.append(
            (
                entry,
                primary_offset,
                texture_offset,
                len(primary),
                texture_size,
            )
        )
        hashes += struct.pack(_HASH_FMT, primary_sha, texture_sha)
    file_size = cursor
    directory = bytearray()
    first = 0
    for level in range(level_count):
        run = [e for e in ordered if e.key.level == level]
        directory += struct.pack(
            _LEVEL_FMT,
            first,
            len(run),
            len({e.key.tx for e in run}),
            len({e.key.ty for e in run}),
        )
        first += len(run)
    index = bytearray()
    for entry, po, to, ps, ts in records:
        index += struct.pack(
            _ENTRY_FMT,
            entry.key.level,
            entry.key.tx,
            entry.key.ty,
            entry.key.tz,
            *entry.region,
            entry.geometric_error,
            po,
            to,
            ps,
            ts,
        )
    dataset_id = hashlib.sha256(bytes(hashes)).digest()
    index_crc = zlib.crc32(bytes(directory) + bytes(index)) & _U32
    prefix = struct.pack(
        _HEADER_PREFIX_FMT,
        b"AHNP",
        1,
        tile_count,
        level_count,
        128,
        index_size,
        hash_offset,
        hash_size,
        file_size,
        root,
        dataset_id,
        index_crc,
        0,
        content_kind,
        b"\x00" * 16,
    )
    header = prefix + struct.pack("<I", zlib.crc32(prefix) & _U32)
    return (
        header + bytes(directory) + bytes(index) + bytes(hashes) + bytes(blob)
    )


def test_stream_matches_two_pass_reference_heightfield() -> None:
    """The single-pass stream is byte-identical to the two-pass oracle."""
    assert _write_hf() == _reference_pack(
        _hf_entries(), _hf_source, 8.0, CONTENT_KIND_HEIGHTFIELD
    )


def test_stream_matches_two_pass_reference_game() -> None:
    """The game single-pass stream matches the two-pass oracle too."""
    assert _write_game() == _reference_pack(
        _game_entries(), _game_source, 8.0, CONTENT_KIND_GAME
    )


def test_sort_handles_sparse_non_square_grids() -> None:
    """Adversarial keys sort by (level, tz, ty, tx); grids need not be dense."""
    keys = [
        TileKey(0, 0, 0),
        TileKey(1, 0, 0),
        TileKey(1, 1, 0),
        TileKey(1, 0, 1),
        TileKey(2, 3, 0),
        TileKey(2, 0, 2),
    ]
    entries = [
        PackEntry(
            k, (0.0, 0.0, 1.0, 1.0, 0.0, 1.0), 4.0 if k.level == 0 else 0.0
        )
        for k in keys
    ]

    def source(key: TileKey) -> tuple[bytes, bytes | None]:
        tag = f"{key.level}{key.tx}{key.ty}".encode()
        return b"p" + tag, b"t" + tag

    buf = io.BytesIO()
    write_pack(
        buf,
        list(reversed(entries)),
        source,
        root_geometric_error=8.0,
        content_kind=CONTENT_KIND_HEIGHTFIELD,
    )
    pack = read_pack(buf.getvalue())
    assert [(e.level, e.tx, e.ty) for e in pack.entries] == [
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (1, 0, 1),
        (2, 3, 0),
        (2, 0, 2),
    ]
    # Level 2 is sparse and non-square: two entries, two columns, two rows.
    assert pack.levels[2].entry_count == 2
    assert pack.levels[2].tx_count == 2
    assert pack.levels[2].ty_count == 2


# --------------------------------------------------------------------------
# Writer rejects
# --------------------------------------------------------------------------


def _write_one(
    entries: list[PackEntry],
    source: BlobSource,
    content_kind: int,
    root: float = 1.0,
) -> None:
    write_pack(
        io.BytesIO(),
        entries,
        source,
        root_geometric_error=root,
        content_kind=content_kind,
    )


def test_write_rejects_unknown_content_kind() -> None:
    """The writer refuses a content_kind outside {0, 1}."""
    with pytest.raises(Tiles3dError, match="content_kind"):
        _write_one(_hf_entries(), _hf_source, 9)


def test_write_rejects_non_finite_root() -> None:
    """The writer refuses a non-finite root geometric error."""
    with pytest.raises(Tiles3dError, match="root_geometric_error"):
        _write_one(
            _hf_entries(), _hf_source, CONTENT_KIND_HEIGHTFIELD, float("nan")
        )


def test_write_rejects_empty_entries() -> None:
    """The writer refuses an empty entry list (no tiles)."""
    with pytest.raises(Tiles3dError, match="at least one tile"):
        _write_one([], _hf_source, CONTENT_KIND_HEIGHTFIELD)


def test_write_rejects_duplicate_key() -> None:
    """The writer refuses two entries sharing a tile key."""
    entries = [
        *_hf_entries(),
        PackEntry(TileKey(1, 1, 0), (0.0, 0.0, 1.0, 1.0, 0.0, 1.0), 0.0),
    ]
    with pytest.raises(Tiles3dError, match="duplicate"):
        _write_one(entries, _hf_source, CONTENT_KIND_HEIGHTFIELD)


def test_write_rejects_non_zero_tz() -> None:
    """The writer refuses a non-zero tz in a key."""
    entries = [
        PackEntry(TileKey(0, 0, 0, 1), (0.0, 0.0, 1.0, 1.0, 0.0, 1.0), 0.0)
    ]
    with pytest.raises(Tiles3dError, match="tz="):
        _write_one(entries, lambda _k: (b"x", b"y"), CONTENT_KIND_HEIGHTFIELD)


def test_write_rejects_non_finite_region() -> None:
    """The writer refuses a non-finite region double."""
    entries = [
        PackEntry(
            TileKey(0, 0, 0), (float("inf"), 0.0, 1.0, 1.0, 0.0, 1.0), 0.0
        )
    ]
    with pytest.raises(Tiles3dError, match="non-finite"):
        _write_one(entries, lambda _k: (b"x", b"y"), CONTENT_KIND_HEIGHTFIELD)


def test_write_rejects_non_well_ordered_region() -> None:
    """The writer refuses a region whose west exceeds its east."""
    entries = [
        PackEntry(TileKey(0, 0, 0), (1.0, 0.0, 0.0, 1.0, 0.0, 1.0), 0.0)
    ]
    with pytest.raises(Tiles3dError, match="well-ordered"):
        _write_one(entries, lambda _k: (b"x", b"y"), CONTENT_KIND_HEIGHTFIELD)


def test_write_rejects_non_contiguous_levels() -> None:
    """The writer refuses a level set with a gap."""
    entries = [
        PackEntry(TileKey(0, 0, 0), (0.0, 0.0, 1.0, 1.0, 0.0, 1.0), 4.0),
        PackEntry(TileKey(2, 0, 0), (0.0, 0.0, 1.0, 1.0, 0.0, 1.0), 0.0),
    ]
    with pytest.raises(Tiles3dError, match="contiguous"):
        _write_one(entries, lambda _k: (b"x", b"y"), CONTENT_KIND_HEIGHTFIELD)


def test_write_rejects_empty_primary() -> None:
    """The writer refuses an empty primary blob."""
    entries = [
        PackEntry(TileKey(0, 0, 0), (0.0, 0.0, 1.0, 1.0, 0.0, 1.0), 0.0)
    ]
    with pytest.raises(Tiles3dError, match="empty primary"):
        _write_one(entries, lambda _k: (b"", b"y"), CONTENT_KIND_HEIGHTFIELD)


def test_write_rejects_game_tile_with_texture() -> None:
    """The writer refuses a game tile carrying a texture."""
    entries = [
        PackEntry(TileKey(0, 0, 0), (0.0, 0.0, 1.0, 1.0, 0.0, 1.0), 0.0)
    ]
    with pytest.raises(Tiles3dError, match="no texture blob"):
        _write_one(entries, lambda _k: (b"glb", b"tex"), CONTENT_KIND_GAME)


def test_write_rejects_heightfield_without_texture() -> None:
    """The writer refuses a heightfield tile without a texture."""
    entries = [
        PackEntry(TileKey(0, 0, 0), (0.0, 0.0, 1.0, 1.0, 0.0, 1.0), 0.0)
    ]
    with pytest.raises(Tiles3dError, match="non-empty texture"):
        _write_one(
            entries, lambda _k: (b"hf", None), CONTENT_KIND_HEIGHTFIELD
        )


# --------------------------------------------------------------------------
# Reader rejects — header
# --------------------------------------------------------------------------


def test_reject_short_input() -> None:
    """Input shorter than the 128-byte header is refused."""
    with pytest.raises(Tiles3dError, match="shorter than"):
        read_pack(b"\x00" * 100)


def test_reject_bad_magic() -> None:
    """A wrong magic is refused."""
    data = bytearray(_write_hf())
    data[0] = ord("X")
    with pytest.raises(Tiles3dError, match="magic"):
        read_pack(bytes(data))


def test_reject_bad_version() -> None:
    """A wrong format_version is refused."""
    data = bytearray(_write_hf())
    struct.pack_into("<I", data, _O_VERSION, 2)
    with pytest.raises(Tiles3dError, match="format_version"):
        read_pack(bytes(data))


def test_reject_header_crc_mismatch() -> None:
    """A corrupted header with a stale header_crc32 is refused."""
    data = bytearray(_write_hf())
    struct.pack_into("<I", data, _O_TILE_COUNT, 999)  # left uncorrected
    with pytest.raises(Tiles3dError, match="header_crc32"):
        read_pack(bytes(data))


def test_reject_zero_tile_count() -> None:
    """A zero tile_count is refused."""
    data = bytearray(_write_hf())
    struct.pack_into("<I", data, _O_TILE_COUNT, 0)
    _fix_header_crc(data)
    with pytest.raises(Tiles3dError, match=">= 1 tile"):
        read_pack(bytes(data))


def test_reject_zero_level_count() -> None:
    """A zero level_count is refused."""
    data = bytearray(_write_hf())
    struct.pack_into("<I", data, _O_LEVEL_COUNT, 0)
    _fix_header_crc(data)
    with pytest.raises(Tiles3dError, match=">= 1 tile"):
        read_pack(bytes(data))


def test_reject_level_count_exceeds_tile_count() -> None:
    """A level_count above tile_count is refused."""
    data = bytearray(_write_hf())
    struct.pack_into("<I", data, _O_LEVEL_COUNT, 5)
    _fix_header_crc(data)
    with pytest.raises(Tiles3dError, match="exceeds tile_count"):
        read_pack(bytes(data))


def test_reject_non_finite_root_geometric_error() -> None:
    """A non-finite root geometric error is refused."""
    data = bytearray(_write_hf())
    struct.pack_into("<d", data, _O_ROOT_GE, float("inf"))
    _fix_header_crc(data)
    with pytest.raises(Tiles3dError, match="root_geometric_error"):
        read_pack(bytes(data))


def test_reject_non_zero_reserved() -> None:
    """A non-zero reserved field is refused."""
    data = bytearray(_write_hf())
    struct.pack_into("<I", data, _O_RESERVED, 7)
    _fix_header_crc(data)
    with pytest.raises(Tiles3dError, match="reserved"):
        read_pack(bytes(data))


def test_reject_non_zero_pad() -> None:
    """A non-zero pad byte is refused."""
    data = bytearray(_write_hf())
    data[_O_PAD] = 1
    _fix_header_crc(data)
    with pytest.raises(Tiles3dError, match="pad"):
        read_pack(bytes(data))


def test_reject_unknown_content_kind() -> None:
    """A content_kind outside {0, 1} is refused."""
    data = bytearray(_write_hf())
    struct.pack_into("<I", data, _O_CONTENT_KIND, 2)
    _fix_header_crc(data)
    with pytest.raises(Tiles3dError, match="content_kind"):
        read_pack(bytes(data))


@pytest.mark.parametrize(
    ("offset", "value", "match"),
    [
        (_O_INDEX_OFFSET, 64, "index_offset"),
        (_O_INDEX_SIZE, 999, "index_size"),
        (_O_HASH_OFFSET, 999, "hash_offset"),
        (_O_HASH_SIZE, 999, "hash_size"),
        (_O_FILE_SIZE, 999, "file_size"),
    ],
)
def test_reject_layout_field_mismatch(
    offset: int, value: int, match: str
) -> None:
    """Each stored offset/size must agree with the counts and file length."""
    data = bytearray(_write_hf())
    struct.pack_into("<Q", data, offset, value)
    _fix_header_crc(data)
    with pytest.raises(Tiles3dError, match=match):
        read_pack(bytes(data))


def test_reject_truncated_index_region() -> None:
    """A file whose index/hash region runs past EOF is refused as truncated."""
    data = bytearray(_write_hf()[:140])
    struct.pack_into("<Q", data, _O_FILE_SIZE, len(data))
    _fix_header_crc(data)
    with pytest.raises(Tiles3dError, match="truncated"):
        read_pack(bytes(data))


def test_reject_index_crc_mismatch() -> None:
    """A mutated index region with a stale index_crc32 is refused."""
    data = bytearray(_write_hf())
    data[_entry_offset(data, 0) + _E_REGION0] ^= (
        0xFF  # index byte, crc left stale
    )
    with pytest.raises(Tiles3dError, match="index_crc32"):
        read_pack(bytes(data))


# --------------------------------------------------------------------------
# Reader rejects — level directory
# --------------------------------------------------------------------------


def test_reject_directory_first_entry_nonzero() -> None:
    """A level-0 first_entry other than 0 is refused."""
    data = bytearray(_write_hf())
    struct.pack_into("<I", data, INDEX_OFFSET, 1)  # level 0 first_entry
    _fix_index_crc(data)
    with pytest.raises(Tiles3dError, match="run continuity"):
        read_pack(bytes(data))


def test_reject_directory_run_discontinuity() -> None:
    """A directory run discontinuity is refused."""
    data = bytearray(_write_hf())
    struct.pack_into("<I", data, INDEX_OFFSET + 16, 2)  # level 1 first_entry
    _fix_index_crc(data)
    with pytest.raises(Tiles3dError, match="run continuity"):
        read_pack(bytes(data))


def test_reject_directory_entry_count_sum() -> None:
    """A directory whose entry counts do not sum to tile_count is refused."""
    data = bytearray(_write_hf())
    struct.pack_into(
        "<I", data, INDEX_OFFSET + 16 + 4, 1
    )  # level 1 entry_count
    _fix_index_crc(data)
    with pytest.raises(Tiles3dError, match="covers"):
        read_pack(bytes(data))


# --------------------------------------------------------------------------
# Reader rejects — index entries
# --------------------------------------------------------------------------


def test_reject_entry_non_zero_tz() -> None:
    """An entry with a non-zero tz is refused."""
    data = bytearray(_write_hf())
    struct.pack_into("<I", data, _entry_offset(data, 0) + _E_TZ, 1)
    _fix_index_crc(data)
    with pytest.raises(Tiles3dError, match="tz="):
        read_pack(bytes(data))


def test_reject_entry_non_finite_region() -> None:
    """An entry with a non-finite region double is refused."""
    data = bytearray(_write_hf())
    struct.pack_into(
        "<d", data, _entry_offset(data, 0) + _E_REGION0, float("nan")
    )
    _fix_index_crc(data)
    with pytest.raises(Tiles3dError, match="non-finite"):
        read_pack(bytes(data))


def test_reject_entry_region_not_well_ordered() -> None:
    """An entry whose region is not well-ordered is refused."""
    data = bytearray(_write_hf())
    struct.pack_into("<d", data, _entry_offset(data, 0) + _E_REGION0, 9.0)
    _fix_index_crc(data)
    with pytest.raises(Tiles3dError, match="well-ordered"):
        read_pack(bytes(data))


def test_reject_entry_level_out_of_run() -> None:
    """An entry whose level disagrees with its directory run is refused."""
    data = bytearray(_write_hf())
    struct.pack_into("<I", data, _entry_offset(data, 2), 0)  # level field
    _fix_index_crc(data)
    with pytest.raises(Tiles3dError, match="sits in the level"):
        read_pack(bytes(data))


def test_reject_entry_unsorted_or_duplicate() -> None:
    """An entry key not strictly after the previous is refused."""
    data = bytearray(_write_hf())
    struct.pack_into(
        "<I", data, _entry_offset(data, 2) + _E_TX, 0
    )  # dup of entry 1
    _fix_index_crc(data)
    with pytest.raises(Tiles3dError, match="unsorted or duplicate"):
        read_pack(bytes(data))


# --------------------------------------------------------------------------
# Reader rejects — blobs
# --------------------------------------------------------------------------


def test_reject_blob_misaligned() -> None:
    """A primary blob offset that is not 16-aligned is refused."""
    data = bytearray(_write_hf())
    entry0 = read_pack(bytes(data)).entries[0]
    struct.pack_into(
        "<Q",
        data,
        _entry_offset(data, 0) + _E_PRIMARY_OFFSET,
        entry0.primary_offset + 1,
    )
    _fix_index_crc(data)
    with pytest.raises(Tiles3dError, match="16-byte aligned"):
        read_pack(bytes(data))


def test_reject_blob_not_ascending() -> None:
    """A blob offset that is not strictly ascending is refused."""
    data = bytearray(_write_hf())
    struct.pack_into(
        "<Q", data, _entry_offset(data, 1) + _E_PRIMARY_OFFSET, 16
    )
    _fix_index_crc(data)
    with pytest.raises(Tiles3dError, match="strictly ascending"):
        read_pack(bytes(data))


def test_reject_blob_overlap() -> None:
    """A blob starting inside the previous blob's extent is refused."""
    entries = [
        PackEntry(TileKey(0, 0, 0), (0.0, 0.0, 1.0, 1.0, 0.0, 1.0), 4.0),
        PackEntry(TileKey(1, 0, 0), (0.0, 0.0, 0.5, 0.5, 0.0, 0.5), 0.0),
    ]
    blobs = {(0, 0, 0): b"X" * 40, (1, 0, 0): b"Y" * 8}
    buf = io.BytesIO()
    write_pack(
        buf,
        entries,
        lambda k: (blobs[(k.level, k.tx, k.ty)], None),
        root_geometric_error=8.0,
        content_kind=CONTENT_KIND_GAME,
    )
    data = bytearray(buf.getvalue())
    entry0 = read_pack(bytes(data)).entries[0]
    overlap = (
        entry0.primary_offset + 16
    )  # aligned, inside [offset, offset+40)
    struct.pack_into(
        "<Q", data, _entry_offset(data, 1) + _E_PRIMARY_OFFSET, overlap
    )
    _fix_index_crc(data)
    with pytest.raises(Tiles3dError, match="overlaps"):
        read_pack(bytes(data))


def test_reject_non_zero_inter_blob_padding() -> None:
    """Non-zero inter-blob padding is refused."""
    data = bytearray(_write_hf())
    entry = read_pack(bytes(data)).entries[
        1
    ]  # 'prim-10', padded before texture
    pad_position = entry.primary_offset + entry.primary_size
    assert entry.texture_offset > pad_position
    data[pad_position] = 0xFF
    with pytest.raises(Tiles3dError, match="padding"):
        read_pack(bytes(data))


def test_reject_blob_past_end_of_file() -> None:
    """A blob range extending past EOF is refused."""
    data = bytearray(_write_hf())
    struct.pack_into(
        "<I", data, _entry_offset(data, 2) + _E_PRIMARY_SIZE, 10_000_000
    )
    _fix_index_crc(data)
    with pytest.raises(Tiles3dError, match="past end of file"):
        read_pack(bytes(data))


def test_reject_trailing_bytes() -> None:
    """Trailing bytes after the final blob are refused."""
    data = bytearray(_write_hf())
    data += b"\x00" * 16
    struct.pack_into("<Q", data, _O_FILE_SIZE, len(data))
    _fix_header_crc(data)
    with pytest.raises(Tiles3dError, match="trailing"):
        read_pack(bytes(data))


def test_reject_game_entry_with_texture_fields() -> None:
    """A game entry with non-zero texture slots is refused."""
    data = bytearray(_write_game())
    struct.pack_into("<I", data, _entry_offset(data, 0) + _E_TEXTURE_SIZE, 5)
    _fix_index_crc(data)
    with pytest.raises(
        Tiles3dError, match="texture_offset and texture_size 0"
    ):
        read_pack(bytes(data))


def test_reject_heightfield_entry_without_texture() -> None:
    """A heightfield entry with a zero-size texture is refused."""
    data = bytearray(_write_hf())
    struct.pack_into("<I", data, _entry_offset(data, 0) + _E_TEXTURE_SIZE, 0)
    _fix_index_crc(data)
    with pytest.raises(Tiles3dError, match="no texture blob"):
        read_pack(bytes(data))


# --------------------------------------------------------------------------
# Reader rejects — hash section / dataset_id
# --------------------------------------------------------------------------


def test_reject_hash_section_bitflip_via_dataset_id() -> None:
    """A single flipped hash-section byte is caught by the dataset_id recompute."""
    data = bytearray(_write_hf())
    hash_offset = struct.unpack_from("<Q", data, _O_HASH_OFFSET)[0]
    data[hash_offset] ^= 0xFF
    with pytest.raises(Tiles3dError, match="dataset_id"):
        read_pack(bytes(data))


def test_reject_primary_blob_hash_mismatch() -> None:
    """A flipped primary-blob byte is caught by its SHA-256."""
    data = bytearray(_write_hf())
    entry0 = read_pack(bytes(data)).entries[0]
    data[entry0.primary_offset] ^= 0xFF
    with pytest.raises(Tiles3dError, match="primary blob fails"):
        read_pack(bytes(data))


def test_reject_texture_blob_hash_mismatch() -> None:
    """A flipped texture-blob byte is caught by its SHA-256."""
    data = bytearray(_write_hf())
    entry0 = read_pack(bytes(data)).entries[0]
    data[entry0.texture_offset] ^= 0xFF
    with pytest.raises(Tiles3dError, match="texture blob fails"):
        read_pack(bytes(data))


def test_reject_game_non_zero_texture_sha() -> None:
    """A game tile whose texture_sha256 is not all-zero is refused."""
    data = bytearray(_write_game())
    hash_offset = struct.unpack_from("<Q", data, _O_HASH_OFFSET)[0]
    hash_size = struct.unpack_from("<Q", data, _O_HASH_SIZE)[0]
    data[hash_offset + 32] ^= 0xFF  # entry 0 texture_sha256
    # Recompute dataset_id so the sentinel check (not dataset_id) fires.
    dataset_id = hashlib.sha256(
        bytes(data[hash_offset : hash_offset + hash_size])
    ).digest()
    struct.pack_into("<32s", data, _O_DATASET_ID, dataset_id)
    _fix_header_crc(data)
    with pytest.raises(Tiles3dError, match="32 zero bytes"):
        read_pack(bytes(data))


# --------------------------------------------------------------------------
# Truncation matrix
# --------------------------------------------------------------------------


@pytest.mark.parametrize("keep", [0, 64, 127, 128, 150, 300, -1])
def test_reject_truncation_matrix(keep: int) -> None:
    """Truncation at any point (short header through last byte) is refused."""
    data = _write_hf()
    truncated = data[:keep] if keep >= 0 else data[:-1]
    with pytest.raises(Tiles3dError):
        read_pack(truncated)
