"""Strict reader for the reconciled EXR (reconcile's dense-grid output).

The reconcile context writes a deterministic, uncompressed, scanline
OpenEXR with exactly the six FLOAT channels ``B, G, R, X, Y, Z`` (see
``ahn_cli/reconcile/writers.py``). This reader accepts **only** that
layout, byte for byte: magic, version, the exact attribute set with the
writer's pinned values, a self-consistent scanline offset table, and
nothing before, between, or after the expected bytes. Any deviation is a
:class:`Tiles3dError` naming the failed check — a mangled heights file
must never flow silently into a tileset.

The whole file is loaded into memory (one site's grid, not nationwide).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ahn_cli.tiles3d.errors import Tiles3dError

if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt

__all__ = ["ReconciledExr", "read_reconciled_exr"]

_MAGIC = 0x01312F76
_VERSION = 2
_PIXEL_TYPE_FLOAT = 2
_CHANNELS: tuple[bytes, ...] = (b"B", b"G", b"R", b"X", b"Y", b"Z")

_EXPECTED_ATTR_TYPES: dict[bytes, bytes] = {
    b"channels": b"chlist",
    b"compression": b"compression",
    b"dataWindow": b"box2i",
    b"displayWindow": b"box2i",
    b"lineOrder": b"lineOrder",
    b"pixelAspectRatio": b"float",
    b"screenWindowCenter": b"v2f",
    b"screenWindowWidth": b"float",
}
"""The exact attribute set (name -> type) the reconcile writer emits."""

_PINNED_ATTR_VALUES: dict[bytes, bytes] = {
    b"pixelAspectRatio": struct.pack("<f", 1.0),
    b"screenWindowCenter": struct.pack("<2f", 0.0, 0.0),
    b"screenWindowWidth": struct.pack("<f", 1.0),
}
"""Display attributes the writer pins; any other value is not our file."""


@dataclass(frozen=True)
class ReconciledExr:
    """The six planes of a reconciled EXR, each ``(height, width)``.

    Contract:
        - ``x``/``y`` are the pixel-centre EPSG:28992 coordinates, ``z``
          the NAP elevation, ``r``/``g``/``b`` the ortho colour in
          ``0..1`` — all ``float32``, exactly as stored.

    Invariants:
        - Frozen value object; arrays are the parsed file content.
    """

    width: int
    height: int
    x: npt.NDArray[np.float32]
    y: npt.NDArray[np.float32]
    z: npt.NDArray[np.float32]
    r: npt.NDArray[np.float32]
    g: npt.NDArray[np.float32]
    b: npt.NDArray[np.float32]


class _Cursor:
    """A bounds-checked sequential reader over the file bytes."""

    def __init__(self, data: bytes, path: Path) -> None:
        self.data = data
        self.pos = 0
        self._path = path

    def take(self, count: int) -> bytes:
        """Return the next ``count`` bytes or fail as truncated."""
        end = self.pos + count
        if end > len(self.data):
            msg = f"reconciled EXR at {self._path} is truncated."
            raise Tiles3dError(msg)
        chunk = self.data[self.pos : end]
        self.pos = end
        return chunk

    def take_string(self) -> bytes:
        """Return the next NUL-terminated string (without the NUL)."""
        end = self.data.find(b"\x00", self.pos)
        if end < 0:
            msg = f"reconciled EXR at {self._path} is truncated."
            raise Tiles3dError(msg)
        value = self.data[self.pos : end]
        self.pos = end + 1
        return value


def read_reconciled_exr(path: Path) -> ReconciledExr:
    """Parse a reconciled EXR, refusing anything but the exact layout.

    Contract:
        - Returns the six float32 planes exactly as stored on disk.

    Failure modes:
        - :class:`Tiles3dError` for an unreadable file or any deviation
          from the reconcile writer's layout: magic, version, attribute
          set, channel list, compression, windows, line order, pinned
          display attributes, offset table, scanline framing, truncation
          or trailing bytes.
    """
    try:
        data = path.read_bytes()
    except OSError as exc:
        msg = f"reconciled EXR at {path} is not readable: {exc}"
        raise Tiles3dError(msg) from exc
    cursor = _Cursor(data, path)
    magic, version = struct.unpack("<II", cursor.take(8))
    if magic != _MAGIC:
        msg = f"file at {path} has no EXR magic number."
        raise Tiles3dError(msg)
    if version != _VERSION:
        msg = f"reconciled EXR at {path} has EXR version {version}, not 2."
        raise Tiles3dError(msg)
    attributes = _parse_attributes(cursor, path)
    _verify_pinned_attributes(attributes, path)
    _verify_channel_list(attributes[b"channels"], path)
    width, height = _parse_windows(attributes, path)
    _verify_offset_table(cursor, width, height, path)
    planes = _parse_scanlines(cursor, width, height, path)
    if cursor.pos != len(data):
        msg = (
            f"reconciled EXR at {path} has "
            f"{len(data) - cursor.pos} trailing byte(s)."
        )
        raise Tiles3dError(msg)
    return ReconciledExr(
        width=width,
        height=height,
        x=planes[b"X"],
        y=planes[b"Y"],
        z=planes[b"Z"],
        r=planes[b"R"],
        g=planes[b"G"],
        b=planes[b"B"],
    )


def _parse_attributes(cursor: _Cursor, path: Path) -> dict[bytes, bytes]:
    """Read the header attributes and verify the exact name/type set."""
    attributes: dict[bytes, bytes] = {}
    types: dict[bytes, bytes] = {}
    while True:
        name = cursor.take_string()
        if name == b"":
            break
        type_name = cursor.take_string()
        (size,) = struct.unpack("<I", cursor.take(4))
        types[name] = type_name
        attributes[name] = cursor.take(size)
    if types != _EXPECTED_ATTR_TYPES:
        msg = (
            f"reconciled EXR at {path} does not carry exactly the "
            "expected header attribute set of the reconcile writer."
        )
        raise Tiles3dError(msg)
    return attributes


def _verify_pinned_attributes(
    attributes: dict[bytes, bytes], path: Path
) -> None:
    """Verify the compression/order/display attributes are the pinned ones."""
    if attributes[b"compression"] != struct.pack("<B", 0):
        msg = f"reconciled EXR at {path} is not uncompressed (compression)."
        raise Tiles3dError(msg)
    if attributes[b"lineOrder"] != struct.pack("<B", 0):
        msg = f"reconciled EXR at {path} has a non-increasing-Y line order."
        raise Tiles3dError(msg)
    for name, expected in _PINNED_ATTR_VALUES.items():
        if attributes[name] != expected:
            msg = (
                f"reconciled EXR at {path} has an unexpected "
                f"{name.decode('ascii')} value."
            )
            raise Tiles3dError(msg)


def _verify_channel_list(value: bytes, path: Path) -> None:
    """Verify the chlist is exactly B, G, R, X, Y, Z FLOAT at 1:1."""
    expected = b""
    for name in _CHANNELS:
        expected += (
            name
            + b"\x00"
            + struct.pack("<i", _PIXEL_TYPE_FLOAT)
            + struct.pack("<B", 0)
            + b"\x00\x00\x00"
            + struct.pack("<i", 1)
            + struct.pack("<i", 1)
        )
    expected += b"\x00"
    if value != expected:
        msg = (
            f"reconciled EXR at {path} does not carry exactly the six "
            "FLOAT channels B, G, R, X, Y, Z at 1:1 sampling."
        )
        raise Tiles3dError(msg)


def _parse_windows(
    attributes: dict[bytes, bytes], path: Path
) -> tuple[int, int]:
    """Verify the data/display windows and return (width, height)."""
    x0, y0, x1, y1 = struct.unpack("<4i", attributes[b"dataWindow"])
    if (x0, y0) != (0, 0) or x1 < 0 or y1 < 0:
        msg = (
            f"reconciled EXR at {path} has a data window not anchored "
            "at (0, 0) with positive extent."
        )
        raise Tiles3dError(msg)
    if attributes[b"displayWindow"] != attributes[b"dataWindow"]:
        msg = (
            f"reconciled EXR at {path} has a display window differing "
            "from its data window."
        )
        raise Tiles3dError(msg)
    return x1 + 1, y1 + 1


def _verify_offset_table(
    cursor: _Cursor, width: int, height: int, path: Path
) -> None:
    """Verify every scanline offset matches the fixed writer layout."""
    table = np.frombuffer(cursor.take(height * 8), dtype="<u8")
    block = 8 + width * 4 * len(_CHANNELS)
    first = cursor.pos
    expected = first + np.arange(height, dtype=np.uint64) * np.uint64(block)
    if not np.array_equal(table, expected):
        msg = (
            f"reconciled EXR at {path} has a scanline offset table "
            "inconsistent with an uncompressed increasing-Y layout."
        )
        raise Tiles3dError(msg)


def _parse_scanlines(
    cursor: _Cursor, width: int, height: int, path: Path
) -> dict[bytes, npt.NDArray[np.float32]]:
    """Read every scanline into per-channel ``(height, width)`` planes."""
    planes = {
        name: np.empty((height, width), dtype=np.float32)
        for name in _CHANNELS
    }
    row_bytes = width * 4
    for row in range(height):
        y, size = struct.unpack("<ii", cursor.take(8))
        if y != row or size != row_bytes * len(_CHANNELS):
            msg = (
                f"reconciled EXR at {path} has a malformed scanline "
                f"at row {row}."
            )
            raise Tiles3dError(msg)
        for name in _CHANNELS:
            planes[name][row] = np.frombuffer(
                cursor.take(row_bytes), dtype="<f4"
            )
    return planes
