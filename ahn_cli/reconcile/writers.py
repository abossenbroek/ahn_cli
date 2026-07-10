"""Streaming, deterministic writers for the reconciled cloud (laz/ply/pt/exr).

The reconcile pipeline feeds the grid to a writer **one row-block at a time**
(``(rows, w, 6)`` -- ``X, Y, Z, R, G, B``; RGB as ``0..255`` floats -- plus a
``(rows, w)`` validity mask), so an arbitrarily large area streams to disk with
flat memory. :func:`open_writer` returns a :class:`ReconciledWriter` chosen by
typed dispatch on :class:`OutputFormat`; the caller pushes blocks with
:meth:`ReconciledWriter.write_block` and finalises with
:meth:`ReconciledWriter.close`. The formats differ by how each treats the grid:

* **laz / ply / pt** are point lists: each block emits only its *valid* cells
  (``mask`` true), in row-major order, so the concatenation is the whole cloud.
  RGB is scaled to each format's native colour type.
* **exr** is a dense image: it writes the header and scanline offset table up
  front (the height is known) and appends every row's scanline, collapsing a
  void cell's ``Z`` to a ``0.0`` sentinel (X/Y/RGB kept), mirroring
  ``positions.exr``.

Every writer is byte-deterministic for a fixed block schedule: no timestamps or
host metadata are written. The PLY, PT and EXR payloads are hand-packed
little-endian; the LAZ header's date/software/offset fields are pinned to
constants. (LAZ compression chunks on the write schedule, so its *bytes* depend
on the fixed block size while its *points* do not; the uncompressed formats are
block-size-independent.) :func:`write_reconciled` writes a whole grid as a single
block for callers that have it in memory.
"""

from __future__ import annotations

import datetime
import shutil
import struct
from enum import Enum
from typing import TYPE_CHECKING, Protocol

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


class ReconciledWriter(Protocol):
    """A streaming writer: push row-blocks, then close to finalise."""

    def write_block(
        self,
        grid_block: npt.NDArray[np.float64],
        mask_block: npt.NDArray[np.bool_],
    ) -> None:
        """Append one ``(rows, w, 6)`` grid block with its ``(rows, w)`` mask."""
        ...

    def close(self) -> int:
        """Finalise the file and return the count written (points, or pixels)."""
        ...


def open_writer(
    output_format: OutputFormat,
    path: Path,
    width: int,
    height: int,
    x_offset: float,
    y_offset: float,
) -> ReconciledWriter:
    """Open a streaming writer for ``output_format`` (typed dispatch).

    ``width``/``height`` are the full grid dimensions (the EXR offset table and
    LAS header need them up front); ``x_offset``/``y_offset`` are the LAS
    coordinate offsets (the grid's south-west corner), known before any block.
    """
    if output_format is OutputFormat.LAZ:
        return _LazWriter(path, x_offset, y_offset)
    if output_format is OutputFormat.PLY:
        return _PlyWriter(path)
    if output_format is OutputFormat.PT:
        return _PtWriter(path)
    return _ExrWriter(path, width, height)


def write_reconciled(
    output_format: OutputFormat,
    grid: npt.NDArray[np.float64],
    mask: npt.NDArray[np.bool_],
    path: Path,
) -> int:
    """Write a whole grid as a single block (in-memory convenience).

    Contract:
        - ``grid`` is ``(h, w, 6)`` ``float64`` (``X, Y, Z, R, G, B``; RGB in
          ``0..255``); ``mask`` is ``(h, w)`` ``bool``.
        - Returns the point count (laz/ply/pt) or pixel count (exr) written.
    """
    height, width = mask.shape
    x_offset = float(np.floor(grid[:, :, 0].min()))
    y_offset = float(np.floor(grid[:, :, 1].min()))
    writer = open_writer(
        output_format, path, width, height, x_offset, y_offset
    )
    writer.write_block(grid, mask)
    return writer.close()


class _PtWriter:
    """Streams valid points as a raw little-endian ``float32`` ``[N, 6]`` blob.

    The layout is ``(x, y, z, r, g, b)`` per point with RGB in ``0..255``,
    loadable via ``torch.frombuffer``/``numpy.fromfile`` -- no torch dependency.
    """

    def __init__(self, path: Path) -> None:
        """Open the output blob for streaming appends."""
        self._sink = path.open("wb")
        self._count = 0

    def write_block(
        self,
        grid_block: npt.NDArray[np.float64],
        mask_block: npt.NDArray[np.bool_],
    ) -> None:
        """Append this block's valid points as float32 rows."""
        points = grid_block[mask_block]
        self._sink.write(points.astype("<f4", copy=False).tobytes())
        self._count += points.shape[0]

    def close(self) -> int:
        """Close the blob and return the point count."""
        self._sink.close()
        return self._count


class _PlyWriter:
    """Streams valid points to a binary coloured PLY (header written on close).

    The vertex count is unknown until every block is seen, so the payload is
    streamed to a temp file and the final PLY (header + payload) is assembled on
    :meth:`close` -- bounded extra disk, never memory.
    """

    def __init__(self, path: Path) -> None:
        """Open a temp payload file for streaming vertex records."""
        self._path = path
        self._tmp = path.with_name(path.name + ".payload")
        self._sink = self._tmp.open("wb")
        self._count = 0

    def write_block(
        self,
        grid_block: npt.NDArray[np.float64],
        mask_block: npt.NDArray[np.bool_],
    ) -> None:
        """Append this block's valid points as packed PLY vertex records."""
        points = grid_block[mask_block]
        record = np.empty(points.shape[0], dtype=_PLY_DTYPE)
        record["x"] = points[:, 0]
        record["y"] = points[:, 1]
        record["z"] = points[:, 2]
        record["red"] = points[:, 3].astype(np.uint8)
        record["green"] = points[:, 4].astype(np.uint8)
        record["blue"] = points[:, 5].astype(np.uint8)
        self._sink.write(record.tobytes())
        self._count += points.shape[0]

    def close(self) -> int:
        """Assemble ``header + payload`` and return the vertex count."""
        self._sink.close()
        header = _PLY_HEADER_TEMPLATE.format(count=self._count)
        with self._path.open("wb") as out:
            out.write(header.encode("ascii"))
            with self._tmp.open("rb") as payload:
                shutil.copyfileobj(payload, out)
        self._tmp.unlink()
        return self._count


class _LazWriter:
    """Streams valid points to a deterministic LAZ (point format 2, RGB)."""

    def __init__(self, path: Path, x_offset: float, y_offset: float) -> None:
        """Open a LAZ writer with a pinned header and fixed coordinate offsets."""
        header = laspy.LasHeader(point_format=2)
        header.scales = np.array([_LAS_SCALE, _LAS_SCALE, _LAS_SCALE])
        header.offsets = np.array([x_offset, y_offset, 0.0])
        header.creation_date = _PINNED_DATE
        header.generating_software = _GENERATING_SOFTWARE
        header.system_identifier = _SYSTEM_IDENTIFIER
        self._header = header
        self._writer = laspy.open(str(path), mode="w", header=header)
        self._count = 0

    def write_block(
        self,
        grid_block: npt.NDArray[np.float64],
        mask_block: npt.NDArray[np.bool_],
    ) -> None:
        """Append this block's valid points to the LAZ stream."""
        points = grid_block[mask_block]
        if points.shape[0] == 0:
            return
        record = laspy.ScaleAwarePointRecord.zeros(
            points.shape[0], header=self._header
        )
        record.x = points[:, 0]
        record.y = points[:, 1]
        record.z = points[:, 2]
        rgb16 = points[:, 3:6].astype(np.uint16) * _RGB_TO_UINT16
        record.red = rgb16[:, 0]
        record.green = rgb16[:, 1]
        record.blue = rgb16[:, 2]
        self._writer.write_points(record)
        self._count += points.shape[0]

    def close(self) -> int:
        """Close the LAZ stream and return the point count."""
        self._writer.close()
        return self._count


class _ExrWriter:
    """Streams the full grid as a 6-channel uncompressed OpenEXR (void Z -> 0).

    The header and the scanline offset table are written on open (the height is
    known), then each row's scanline is appended, so the image streams with flat
    memory.
    """

    def __init__(self, path: Path, width: int, height: int) -> None:
        """Write the header and precomputed scanline offset table."""
        self._sink = path.open("wb")
        self._width = width
        self._height = height
        self._row = 0
        header = _exr_header(width, height)
        row_bytes = width * 4
        block_size = 8 + row_bytes * len(_EXR_CHANNELS)
        self._data_size = struct.pack("<i", row_bytes * len(_EXR_CHANNELS))
        first_block = len(header) + height * 8
        offset_table = b"".join(
            struct.pack("<Q", first_block + row * block_size)
            for row in range(height)
        )
        self._sink.write(header + offset_table)

    def write_block(
        self,
        grid_block: npt.NDArray[np.float64],
        mask_block: npt.NDArray[np.bool_],
    ) -> None:
        """Append every row of this block as an EXR scanline."""
        planes = {
            b"X": grid_block[:, :, 0].astype(np.float32),
            b"Y": grid_block[:, :, 1].astype(np.float32),
            b"Z": np.where(mask_block, grid_block[:, :, 2], 0.0).astype(
                np.float32
            ),
            b"R": (grid_block[:, :, 3] / 255.0).astype(np.float32),
            b"G": (grid_block[:, :, 4] / 255.0).astype(np.float32),
            b"B": (grid_block[:, :, 5] / 255.0).astype(np.float32),
        }
        for row in range(grid_block.shape[0]):
            self._sink.write(struct.pack("<i", self._row))
            self._sink.write(self._data_size)
            for name in _EXR_CHANNELS:
                self._sink.write(
                    planes[name][row].astype("<f4", copy=False).tobytes()
                )
            self._row += 1

    def close(self) -> int:
        """Close the EXR and return the pixel count ``width * height``."""
        self._sink.close()
        return self._width * self._height


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
