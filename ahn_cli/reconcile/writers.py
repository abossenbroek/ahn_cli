"""Deterministic writers for the reconciled cloud (laz / ply / pt / exr).

The reconcile result is a dense ``(h, w, 6)`` grid -- ``X, Y, Z, R, G, B`` per
ortho pixel (RGB as ``0..255`` floats) -- plus a ``(h, w)`` validity mask. The
four writers, chosen by typed dispatch on :class:`OutputFormat`, differ by how
each format treats the grid:

* **laz / ply / pt** are point lists: they flatten and emit only the *valid*
  cells (``mask`` true). RGB is scaled to each format's native colour type.
* **exr** is a dense image: it emits the *full* grid, collapsing a void cell's
  ``Z`` to a ``0.0`` sentinel (its X/Y/RGB are kept), mirroring ``positions.exr``.

Every writer is byte-deterministic: no timestamps or host metadata are written.
The LAZ header's date/software fields are pinned to constants; the PLY, PT, and
EXR payloads are hand-packed little-endian, so identical input yields identical
bytes across runs and machines.
"""

from __future__ import annotations

import datetime
import struct
from enum import Enum
from typing import TYPE_CHECKING

import laspy
import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from pathlib import Path

_RGB_TO_UINT16 = 257
"""Scale a ``0..255`` colour to the full ``0..65535`` LAS ``uint16`` range."""

_PINNED_DATE = datetime.date(2020, 1, 1)
"""Fixed LAS header creation date, so the output carries no wall-clock time."""

_GENERATING_SOFTWARE = "ahn_cli reconcile"
"""Fixed LAS ``generating_software`` header field (deterministic)."""

_SYSTEM_IDENTIFIER = "ahn_cli"
"""Fixed LAS ``system_identifier`` header field (deterministic)."""

_LAS_SCALE = 0.001
"""LAS coordinate scale (1 mm), sub-centimetre-faithful for RD coordinates."""

_PLY_HEADER_TEMPLATE = (
    "ply\n"
    "format binary_little_endian 1.0\n"
    "element vertex {count}\n"
    "property double x\n"
    "property double y\n"
    "property double z\n"
    "property uchar red\n"
    "property uchar green\n"
    "property uchar blue\n"
    "end_header\n"
)
"""Static, deterministic coloured-PLY header; only the vertex ``count`` varies."""

_PLY_DTYPE = np.dtype(
    [
        ("x", "<f8"),
        ("y", "<f8"),
        ("z", "<f8"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
)
"""Packed little-endian PLY vertex record (24-byte xyz double + 3-byte rgb)."""

# OpenEXR container constants (see the OpenEXR file-layout specification), shared
# with ``positions.py``'s discipline: uncompressed scanline, no date attribute.
_EXR_MAGIC = 0x01312F76
_EXR_VERSION = 2
_PIXEL_TYPE_FLOAT = 2
_COMPRESSION_NONE = 0
_LINE_ORDER_INCREASING_Y = 0
_EXR_CHANNELS: tuple[bytes, ...] = (b"B", b"G", b"R", b"X", b"Y", b"Z")
"""The reconciled-cloud EXR channels in the alphabetical order EXR requires."""


class OutputFormat(Enum):
    """The reconciled-cloud output formats (the ``--format`` tokens)."""

    LAZ = "laz"
    PLY = "ply"
    PT = "pt"
    EXR = "exr"


def write_reconciled(
    output_format: OutputFormat,
    grid: npt.NDArray[np.float64],
    mask: npt.NDArray[np.bool_],
    path: Path,
) -> int:
    """Write the reconciled grid in ``output_format``, returning the count.

    Contract:
        - ``grid`` is ``(h, w, 6)`` ``float64`` (``X, Y, Z, R, G, B``; RGB in
          ``0..255``); ``mask`` is ``(h, w)`` ``bool``.
        - For ``laz``/``ply``/``pt`` the return is the number of valid points
          written; for ``exr`` it is the pixel count ``h * w``.

    Typed dispatch on the format variant -- no stringly-typed format switch.
    """
    if output_format is OutputFormat.EXR:
        return _write_exr(grid, mask, path)
    points = grid[mask]
    if output_format is OutputFormat.LAZ:
        return _write_laz(points, path)
    if output_format is OutputFormat.PLY:
        return _write_ply(points, path)
    return _write_pt(points, path)


def _write_laz(points: npt.NDArray[np.float64], path: Path) -> int:
    """Write valid points to a deterministic LAZ (point format 2, RGB)."""
    count = points.shape[0]
    header = laspy.LasHeader(point_format=2)
    header.scales = np.array([_LAS_SCALE, _LAS_SCALE, _LAS_SCALE])
    if count == 0:
        header.offsets = np.zeros(3)
    else:
        header.offsets = np.floor(points[:, :3].min(axis=0))
    header.creation_date = _PINNED_DATE
    header.generating_software = _GENERATING_SOFTWARE
    header.system_identifier = _SYSTEM_IDENTIFIER

    record = laspy.ScaleAwarePointRecord.zeros(count, header=header)
    record.x = points[:, 0]
    record.y = points[:, 1]
    record.z = points[:, 2]
    rgb16 = points[:, 3:6].astype(np.uint16) * _RGB_TO_UINT16
    record.red = rgb16[:, 0]
    record.green = rgb16[:, 1]
    record.blue = rgb16[:, 2]

    las = laspy.LasData(header)
    las.points = record
    las.write(str(path))
    return count


def _write_ply(points: npt.NDArray[np.float64], path: Path) -> int:
    """Write valid points to a deterministic binary coloured PLY."""
    count = points.shape[0]
    record = np.empty(count, dtype=_PLY_DTYPE)
    record["x"] = points[:, 0]
    record["y"] = points[:, 1]
    record["z"] = points[:, 2]
    record["red"] = points[:, 3].astype(np.uint8)
    record["green"] = points[:, 4].astype(np.uint8)
    record["blue"] = points[:, 5].astype(np.uint8)
    header = _PLY_HEADER_TEMPLATE.format(count=count)
    with path.open("wb") as sink:
        sink.write(header.encode("ascii"))
        sink.write(record.tobytes())
    return count


def _write_pt(points: npt.NDArray[np.float64], path: Path) -> int:
    """Write valid points as a raw little-endian ``float32`` ``[N, 6]`` blob.

    The layout is ``(x, y, z, r, g, b)`` per point with RGB in ``0..255``,
    loadable via ``torch.frombuffer``/``numpy.fromfile`` -- no torch dependency.
    """
    count = points.shape[0]
    path.write_bytes(points.astype("<f4", copy=False).tobytes())
    return count


def _write_exr(
    grid: npt.NDArray[np.float64],
    mask: npt.NDArray[np.bool_],
    path: Path,
) -> int:
    """Write the full grid as a 6-channel uncompressed OpenEXR (void Z -> 0)."""
    height, width = mask.shape
    easting = grid[:, :, 0].astype(np.float32)
    northing = grid[:, :, 1].astype(np.float32)
    elevation = np.where(mask, grid[:, :, 2], 0.0).astype(np.float32)
    red = (grid[:, :, 3] / 255.0).astype(np.float32)
    green = (grid[:, :, 4] / 255.0).astype(np.float32)
    blue = (grid[:, :, 5] / 255.0).astype(np.float32)
    planes = {
        b"B": blue,
        b"G": green,
        b"R": red,
        b"X": easting,
        b"Y": northing,
        b"Z": elevation,
    }
    path.write_bytes(_encode_exr(planes, width, height))
    return width * height


def _exr_attribute(name: bytes, type_name: bytes, value: bytes) -> bytes:
    """Encode one EXR header attribute ``name : type = value``."""
    size = struct.pack("<I", len(value))
    return name + b"\x00" + type_name + b"\x00" + size + value


def _exr_channel_list() -> bytes:
    """Encode the ``chlist`` value: FLOAT channels at 1:1 sampling, in order."""
    body = b""
    for name in _EXR_CHANNELS:
        body += (
            name
            + b"\x00"
            + struct.pack("<i", _PIXEL_TYPE_FLOAT)
            + struct.pack("<B", 0)  # pLinear
            + b"\x00\x00\x00"  # reserved
            + struct.pack("<i", 1)  # xSampling
            + struct.pack("<i", 1)  # ySampling
        )
    return body + b"\x00"


def _exr_header(width: int, height: int) -> bytes:
    """Encode the magic, version, and fixed header attribute block."""
    box = struct.pack("<4i", 0, 0, width - 1, height - 1)
    attributes = (
        _exr_attribute(b"channels", b"chlist", _exr_channel_list())
        + _exr_attribute(
            b"compression",
            b"compression",
            struct.pack("<B", _COMPRESSION_NONE),
        )
        + _exr_attribute(b"dataWindow", b"box2i", box)
        + _exr_attribute(b"displayWindow", b"box2i", box)
        + _exr_attribute(
            b"lineOrder",
            b"lineOrder",
            struct.pack("<B", _LINE_ORDER_INCREASING_Y),
        )
        + _exr_attribute(
            b"pixelAspectRatio", b"float", struct.pack("<f", 1.0)
        )
        + _exr_attribute(
            b"screenWindowCenter", b"v2f", struct.pack("<2f", 0.0, 0.0)
        )
        + _exr_attribute(
            b"screenWindowWidth", b"float", struct.pack("<f", 1.0)
        )
    )
    prefix = struct.pack("<I", _EXR_MAGIC) + struct.pack("<I", _EXR_VERSION)
    return prefix + attributes + b"\x00"


def _encode_exr(
    planes: dict[bytes, npt.NDArray[np.float32]], width: int, height: int
) -> bytes:
    """Encode the named ``(h, w)`` float planes to uncompressed EXR bytes."""
    header = _exr_header(width, height)
    row_bytes = width * 4
    channel_count = len(_EXR_CHANNELS)
    block_size = 8 + row_bytes * channel_count
    data_size = struct.pack("<i", row_bytes * channel_count)
    first_block = len(header) + height * 8
    offset_table = b"".join(
        struct.pack("<Q", first_block + row * block_size)
        for row in range(height)
    )
    blocks = bytearray()
    for row in range(height):
        blocks += struct.pack("<i", row)
        blocks += data_size
        for name in _EXR_CHANNELS:
            blocks += planes[name][row].astype("<f4", copy=False).tobytes()
    return header + offset_table + bytes(blocks)
