"""Tests for the deterministic PNG codec."""

from __future__ import annotations

import struct
import zlib

import numpy as np
import pytest

from ahn_cli.tiles3d.errors import Tiles3dError
from ahn_cli.tiles3d.png import decode_png, encode_png
from tests.tiles3d.conftest import synth_rgb

_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def test_round_trip_is_bit_exact() -> None:
    """decode(encode(x)) == x for arbitrary uint8 RGB."""
    rgb = synth_rgb(7, 5, seed=9)
    assert np.array_equal(decode_png(encode_png(rgb)), rgb)


def test_encoding_is_deterministic() -> None:
    """Two encodings of the same image are byte-identical."""
    rgb = synth_rgb(16, 16, seed=10)
    assert encode_png(rgb) == encode_png(rgb)


def test_chunk_layout_and_crcs_are_valid() -> None:
    """The stream is signature + IHDR + IDAT + IEND with true CRCs."""
    data = encode_png(synth_rgb(4, 3))
    assert data.startswith(_SIGNATURE)
    pos = len(_SIGNATURE)
    seen: list[bytes] = []
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        kind = data[pos + 4 : pos + 8]
        payload = data[pos + 8 : pos + 8 + length]
        (crc,) = struct.unpack(
            ">I", data[pos + 8 + length : pos + 12 + length]
        )
        assert crc == zlib.crc32(kind + payload)
        seen.append(kind)
        pos += 12 + length
    assert seen == [b"IHDR", b"IDAT", b"IEND"]


def test_bad_signature_is_refused() -> None:
    """A non-PNG buffer is refused."""
    with pytest.raises(Tiles3dError, match="signature"):
        decode_png(b"not a png at all")


def test_flipped_crc_is_refused() -> None:
    """A corrupted chunk CRC is always caught."""
    data = bytearray(encode_png(synth_rgb(4, 4)))
    data[-5] ^= 0xFF  # inside IEND's CRC
    with pytest.raises(Tiles3dError, match="CRC"):
        decode_png(bytes(data))


def test_wrong_colour_type_is_refused() -> None:
    """Only 8-bit RGB (colour type 2) is our format."""
    data = bytearray(encode_png(synth_rgb(4, 4)))
    # IHDR payload starts at 16; colour type is its 10th byte.
    ihdr_payload = len(_SIGNATURE) + 8
    data[ihdr_payload + 9] = 6  # RGBA
    payload = bytes(data[ihdr_payload : ihdr_payload + 13])
    crc = zlib.crc32(b"IHDR" + payload)
    data[ihdr_payload + 13 : ihdr_payload + 17] = struct.pack(">I", crc)
    with pytest.raises(Tiles3dError, match="8-bit RGB"):
        decode_png(bytes(data))


def test_wrong_chunk_order_is_refused() -> None:
    """A stream without IHDR first is refused."""
    data = encode_png(synth_rgb(4, 4))
    body = data[len(_SIGNATURE) :]
    idat_start = body.index(b"IDAT") - 4
    iend_start = body.index(b"IEND") - 4
    ihdr = body[:idat_start]
    idat = body[idat_start:iend_start]
    iend = body[iend_start:]
    reordered = _SIGNATURE + idat + ihdr + iend
    with pytest.raises(Tiles3dError, match="IHDR"):
        decode_png(reordered)


def test_nonzero_filter_byte_is_refused() -> None:
    """Scanlines must use filter 0 (None)."""
    rgb = synth_rgb(3, 2)
    raw = bytearray()
    for row in range(2):
        raw.append(1)  # Sub filter -- not ours
        raw.extend(rgb[row].tobytes())
    compressed = zlib.compress(bytes(raw), 6)
    ihdr = struct.pack(">IIBBBBB", 3, 2, 8, 2, 0, 0, 0)
    data = (
        _SIGNATURE
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", compressed)
        + _chunk(b"IEND", b"")
    )
    with pytest.raises(Tiles3dError, match="filter"):
        decode_png(data)


def test_wrong_decoded_length_is_refused() -> None:
    """An inflated payload of the wrong size is refused."""
    ihdr = struct.pack(">IIBBBBB", 3, 2, 8, 2, 0, 0, 0)
    compressed = zlib.compress(b"\x00" * 5, 6)  # far too short
    data = (
        _SIGNATURE
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", compressed)
        + _chunk(b"IEND", b"")
    )
    with pytest.raises(Tiles3dError, match="length"):
        decode_png(data)


def test_corrupt_zlib_stream_is_refused() -> None:
    """An IDAT that does not inflate is refused."""
    ihdr = struct.pack(">IIBBBBB", 3, 2, 8, 2, 0, 0, 0)
    data = (
        _SIGNATURE
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", b"\x00garbage")
        + _chunk(b"IEND", b"")
    )
    with pytest.raises(Tiles3dError, match="inflate"):
        decode_png(data)


def test_trailing_chunk_after_iend_is_refused() -> None:
    """Nothing may follow IEND."""
    data = encode_png(synth_rgb(4, 4)) + _chunk(b"tEXt", b"x")
    with pytest.raises(Tiles3dError, match="IEND"):
        decode_png(data)


def test_truncated_stream_is_refused() -> None:
    """A stream cut inside a chunk header is refused."""
    data = encode_png(synth_rgb(4, 4))
    with pytest.raises(Tiles3dError, match="truncated"):
        decode_png(data[:-6])


def test_stream_cut_inside_a_chunk_body_is_refused() -> None:
    """A stream cut inside a chunk's CRC is refused."""
    data = encode_png(synth_rgb(4, 4))
    with pytest.raises(Tiles3dError, match="truncated"):
        decode_png(data[:-2])


def _chunk(kind: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(kind + payload)
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", crc)
    )
