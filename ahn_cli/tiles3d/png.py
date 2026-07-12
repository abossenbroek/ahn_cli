"""Deterministic PNG codec for the tile textures (stdlib zlib only).

The encoder writes the minimal, byte-deterministic PNG the tiles need:
8-bit RGB (colour type 2), filter 0 on every scanline, one IDAT with a
pinned zlib level, no ancillary chunks (and so no timestamps). The
decoder is the verifier's: it accepts **only** that shape, verifies the
CRC-32 of every chunk, and refuses anything else with a typed
:class:`Tiles3dError` — it is a checker for our own files, not a
general PNG reader.
"""

from __future__ import annotations

import struct
import zlib
from typing import TYPE_CHECKING

import numpy as np

from ahn_cli.tiles3d.errors import Tiles3dError

if TYPE_CHECKING:
    import numpy.typing as npt

__all__ = ["decode_png", "encode_png"]

_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_ZLIB_LEVEL = 6
"""Pinned compression level: zlib output is deterministic per level."""

_BIT_DEPTH = 8
_COLOUR_TYPE_RGB = 2
_CHANNELS = 3


def encode_png(rgb: npt.NDArray[np.uint8]) -> bytes:
    """Encode an ``(h, w, 3)`` uint8 image as a deterministic PNG."""
    height, width = rgb.shape[:2]
    scanlines = bytearray()
    for row in range(height):
        scanlines.append(0)  # filter type None
        scanlines.extend(rgb[row].tobytes())
    ihdr = struct.pack(
        ">IIBBBBB", width, height, _BIT_DEPTH, _COLOUR_TYPE_RGB, 0, 0, 0
    )
    idat = zlib.compress(bytes(scanlines), _ZLIB_LEVEL)
    return (
        _SIGNATURE
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", idat)
        + _chunk(b"IEND", b"")
    )


def decode_png(data: bytes) -> npt.NDArray[np.uint8]:
    """Decode (and strictly validate) a PNG written by :func:`encode_png`.

    Contract:
        - Returns the ``(h, w, 3)`` uint8 image.

    Failure modes:
        - :class:`Tiles3dError` for a bad signature, truncated stream,
          wrong chunk order, any CRC mismatch, a non-8-bit-RGB IHDR, a
          non-inflating IDAT, a non-zero scanline filter, a decoded
          length mismatch, or bytes after IEND.
    """
    if not data.startswith(_SIGNATURE):
        msg = "texture is not a PNG: bad signature."
        raise Tiles3dError(msg)
    chunks = _split_chunks(data[len(_SIGNATURE) :])
    kinds = [kind for kind, _ in chunks]
    if (
        len(kinds) < 3  # noqa: PLR2004 -- IHDR + >=1 IDAT + IEND
        or kinds[0] != b"IHDR"
        or kinds[-1] != b"IEND"
        or any(kind != b"IDAT" for kind in kinds[1:-1])
    ):
        msg = "texture PNG must be exactly IHDR, IDAT(s), IEND in order."
        raise Tiles3dError(msg)
    width, height = _parse_ihdr(chunks[0][1])
    compressed = b"".join(payload for _, payload in chunks[1:-1])
    try:
        raw = zlib.decompress(compressed)
    except zlib.error as exc:
        msg = f"texture PNG's pixel data does not inflate: {exc}"
        raise Tiles3dError(msg) from exc
    stride = 1 + width * _CHANNELS
    if len(raw) != height * stride:
        msg = (
            f"texture PNG decoded length {len(raw)} does not match "
            f"{height} rows of {stride} bytes."
        )
        raise Tiles3dError(msg)
    rows = np.frombuffer(raw, dtype=np.uint8).reshape(height, stride)
    if not bool(np.all(rows[:, 0] == 0)):
        msg = "texture PNG uses a scanline filter other than None."
        raise Tiles3dError(msg)
    return rows[:, 1:].reshape(height, width, _CHANNELS).copy()


def _chunk(kind: bytes, payload: bytes) -> bytes:
    """Frame one PNG chunk with its length and CRC-32."""
    crc = zlib.crc32(kind + payload)
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", crc)
    )


def _split_chunks(body: bytes) -> list[tuple[bytes, bytes]]:
    """Split the post-signature bytes into CRC-verified chunks."""
    chunks: list[tuple[bytes, bytes]] = []
    pos = 0
    while pos < len(body):
        header_end = pos + 8
        if header_end > len(body):
            msg = "texture PNG is truncated inside a chunk header."
            raise Tiles3dError(msg)
        (length,) = struct.unpack(">I", body[pos : pos + 4])
        kind = body[pos + 4 : header_end]
        payload_end = header_end + length
        crc_end = payload_end + 4
        if crc_end > len(body):
            msg = "texture PNG is truncated inside a chunk."
            raise Tiles3dError(msg)
        payload = body[header_end:payload_end]
        (crc,) = struct.unpack(">I", body[payload_end:crc_end])
        if crc != zlib.crc32(kind + payload):
            msg = (
                "texture PNG chunk "
                f"{kind.decode('latin-1')} fails its CRC-32 check."
            )
            raise Tiles3dError(msg)
        chunks.append((kind, payload))
        pos = crc_end
        if kind == b"IEND" and pos != len(body):
            msg = "texture PNG carries bytes after IEND."
            raise Tiles3dError(msg)
    return chunks


def _parse_ihdr(payload: bytes) -> tuple[int, int]:
    """Validate the IHDR as 8-bit RGB and return (width, height)."""
    width, height, depth, colour, comp, filt, interlace = struct.unpack(
        ">IIBBBBB", payload
    )
    if (depth, colour, comp, filt, interlace) != (
        _BIT_DEPTH,
        _COLOUR_TYPE_RGB,
        0,
        0,
        0,
    ):
        msg = (
            "texture PNG is not plain 8-bit RGB "
            "(depth 8, colour type 2, no interlace)."
        )
        raise Tiles3dError(msg)
    return int(width), int(height)
