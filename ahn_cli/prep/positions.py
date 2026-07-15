"""Prep-context raster export: a DSM raster to a ``positions.exr`` position map.

TouchDesigner ingests a *position map*: a floating-point image where each pixel
carries the world coordinate of a surface sample rather than a colour. This
transform turns ``data/<site>/dsm.tif`` -- the WP7 single-band float32 elevation
raster (EPSG:28992, north-up geotransform, nodata declared) -- into a 3-channel
float32 ``positions.exr`` where, per pixel:

* **R** = world X (easting) of the pixel *centre*, from the geotransform;
* **G** = world Y (northing) of the pixel centre;
* **B** = Z, the DSM elevation at that pixel.

**Nodata policy.** A void (nodata) elevation pixel collapses its Z to the
sentinel ``0.0`` while keeping its X/Y set to the true pixel-centre world
coordinate. This keeps the position grid geometrically intact (a void reads as
"ground plane at this easting/northing" rather than a NaN that many downstream
tools mishandle), and the count of collapsed pixels is returned for provenance.

**Determinism.** The container is a hand-written *uncompressed scanline*
OpenEXR, mirroring WP13's byte-deterministic PLY discipline: no library is used,
so no timestamp/owner/capDate/comments attribute is ever emitted. Every header
attribute, the scanline offset table, and the little-endian FLOAT payload are
written byte-for-byte, so identical input yields byte-identical output across
runs and machines. Channels are stored in the alphabetical order OpenEXR
mandates (B, G, R), each an uncompressed float32 scanline.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import numpy.typing as npt
import rasterio
from rasterio.errors import RasterioIOError

from ahn_cli.domain.authenticity import flat_surface
from ahn_cli.domain.grid import GeoTransform, PixelGrid

if TYPE_CHECKING:
    from pathlib import Path

    from ahn_cli.domain import ProgressCallback

# OpenEXR container constants (see the OpenEXR file-layout specification).
_EXR_MAGIC = 0x01312F76
_EXR_VERSION = 2  # single-part scanline, short names, no flags set
_PIXEL_TYPE_FLOAT = 2  # 0=UINT, 1=HALF, 2=FLOAT
_COMPRESSION_NONE = 0
_LINE_ORDER_INCREASING_Y = 0
_CHANNEL_NAMES = (b"B", b"G", b"R")
"""Channel names in the alphabetical order the EXR chlist requires."""

_NODATA_Z = np.float32(0.0)
"""Sentinel Z for a void (nodata) pixel; its easting/northing stay set."""


class PositionsExportError(Exception):
    """Raised when the source DSM raster cannot be read for the export.

    Signals an absent or unreadable ``dsm.tif``; raised in place of the raw
    :class:`rasterio.errors.RasterioIOError` so a caller (e.g. the CLI) can
    report a tidy message rather than a traceback.
    """


@dataclass(frozen=True)
class PositionsExportStats:
    """The ledger of one ``positions.exr`` export.

    Contract (fields):
        - ``width`` / ``height``: the exported image's pixel dimensions, equal
          to the source DSM raster's dimensions.
        - ``nodata_pixels``: the number of void pixels whose Z was collapsed to
          the ``0.0`` sentinel (``0`` when the raster declares no nodata).

    Invariants:
        - Frozen value object, equal by field value; safe to record verbatim in
          a provenance sidecar.
    """

    width: int
    height: int
    nodata_pixels: int


def _attribute(name: bytes, type_name: bytes, value: bytes) -> bytes:
    """Encode one EXR header attribute ``name : type = value``."""
    size = struct.pack("<I", len(value))
    return name + b"\x00" + type_name + b"\x00" + size + value


def _channel_list() -> bytes:
    """Encode the ``chlist`` value: FLOAT B, G, R at 1:1 sampling."""
    body = b""
    for name in _CHANNEL_NAMES:
        body += (
            name
            + b"\x00"
            + struct.pack("<i", _PIXEL_TYPE_FLOAT)
            + struct.pack("<B", 0)  # pLinear
            + b"\x00\x00\x00"  # reserved
            + struct.pack("<i", 1)  # xSampling
            + struct.pack("<i", 1)  # ySampling
        )
    return body + b"\x00"  # NUL terminates the channel list


def _header(width: int, height: int) -> bytes:
    """Encode the magic, version, and the fixed header attribute block.

    Only the data/display window varies with the image size; every other
    attribute is a controlled constant. No timestamp/owner attribute is written.
    """
    box = struct.pack("<4i", 0, 0, width - 1, height - 1)
    attributes = (
        _attribute(b"channels", b"chlist", _channel_list())
        + _attribute(
            b"compression",
            b"compression",
            struct.pack("<B", _COMPRESSION_NONE),
        )
        + _attribute(b"dataWindow", b"box2i", box)
        + _attribute(b"displayWindow", b"box2i", box)
        + _attribute(
            b"lineOrder",
            b"lineOrder",
            struct.pack("<B", _LINE_ORDER_INCREASING_Y),
        )
        + _attribute(b"pixelAspectRatio", b"float", struct.pack("<f", 1.0))
        + _attribute(
            b"screenWindowCenter", b"v2f", struct.pack("<2f", 0.0, 0.0)
        )
        + _attribute(b"screenWindowWidth", b"float", struct.pack("<f", 1.0))
    )
    prefix = struct.pack("<I", _EXR_MAGIC) + struct.pack("<I", _EXR_VERSION)
    return prefix + attributes + b"\x00"  # NUL terminates the header


def _encode_exr(
    easting: npt.NDArray[np.float32],
    northing: npt.NDArray[np.float32],
    elevation: npt.NDArray[np.float32],
) -> bytes:
    """Encode the three ``(h, w)`` float32 planes to uncompressed EXR bytes.

    The scanline offset table is computed exactly (every scanline block is the
    same size for an uncompressed FLOAT image), then each scanline is written as
    its ``y`` coordinate, its pixel-data size, and the alphabetical B/G/R rows.
    """
    height, width = elevation.shape
    header = _header(width, height)
    row_bytes = width * 4  # one float32 channel row
    block_size = 8 + row_bytes * 3  # y(4) + data-size(4) + B/G/R rows
    data_size = struct.pack("<i", row_bytes * 3)
    first_block = len(header) + height * 8  # header, then the offset table
    offset_table = b"".join(
        struct.pack("<Q", first_block + row * block_size)
        for row in range(height)
    )
    planes = {b"B": elevation, b"G": northing, b"R": easting}
    blocks = bytearray()
    for row in range(height):
        blocks += struct.pack("<i", row)  # y coordinate (dataWindow yMin = 0)
        blocks += data_size
        for name in _CHANNEL_NAMES:
            blocks += planes[name][row].astype("<f4", copy=False).tobytes()
    return header + offset_table + bytes(blocks)


def _no_op_progress(_done: int, _total: int) -> None:
    """Report nothing; the default when the caller supplies no callback."""


def export_positions(
    dsm_path: Path,
    output_path: Path,
    *,
    progress: ProgressCallback | None = None,
) -> PositionsExportStats:
    """Export ``dsm_path`` to a byte-deterministic ``positions.exr`` at ``output_path``.

    Contract:
        - ``dsm_path`` is a readable single-band float32 DSM GeoTIFF in
          EPSG:28992 with a north-up geotransform; nodata may be declared or not.
        - ``output_path`` receives a 3-channel float32 uncompressed OpenEXR whose
          pixels are ``R = pixel-centre easting``, ``G = pixel-centre northing``,
          ``B = elevation``. Void (nodata) pixels keep X/Y and take ``Z = 0.0``.
        - Returns a :class:`PositionsExportStats` with the image dimensions and
          the count of nodata pixels collapsed to the sentinel.
        - Calls ``progress(0, 1)`` before the export and ``progress(1, 1)``
          after it completes (a single-raster export); defaults to a no-op
          so callers that don't care about progress are unaffected.

    Invariants:
        - Deterministic: identical input yields byte-identical output, with no
          timestamp/owner/capDate attribute ever written.
        - The output image dimensions equal the source raster's dimensions.

    Failure modes:
        - :class:`PositionsExportError` if the DSM raster is absent or
          unreadable, or if it carries no genuine relief (no valid pixels at
          all, or one constant elevation across every valid pixel — a
          placeholder surface, not measured AHN DSM data).
    """
    report = progress if progress is not None else _no_op_progress
    report(0, 1)
    try:
        with rasterio.open(str(dsm_path)) as dataset:
            elevation = np.asarray(dataset.read(1), dtype=np.float32)
            # A rasterio ``Affine`` is a 9-tuple (a, b, c, d, e, f, 0, 0, 1);
            # its first six entries are the geotransform coefficients. The
            # (unstubbed) Affine exposes neither typed members nor ``__iter__``,
            # so it is cast to the tuple it is at runtime before slicing.
            affine = cast("tuple[float, ...]", dataset.transform)
            transform = cast("GeoTransform", affine[:6])
            width = int(dataset.width)
            height = int(dataset.height)
            nodata = dataset.nodata
    except RasterioIOError as exc:
        msg = f"DSM raster at {dsm_path} is not readable: {exc}"
        raise PositionsExportError(msg) from exc

    if flat_surface(elevation, nodata):
        msg = (
            f"DSM raster at {dsm_path} carries no genuine relief (no valid "
            "pixels, or one constant elevation everywhere) — that is a "
            "placeholder surface, not measured AHN DSM data; refusing to "
            "export a position map from it."
        )
        raise PositionsExportError(msg)

    grid = PixelGrid(width=width, height=height, transform=transform)
    easting = grid.eastings().astype(np.float32)
    northing = grid.northings().astype(np.float32)
    z = elevation.copy()
    if nodata is None:
        nodata_pixels = 0
    else:
        void_mask = elevation == np.float32(nodata)
        nodata_pixels = int(np.count_nonzero(void_mask))
        z[void_mask] = _NODATA_Z

    output_path.write_bytes(_encode_exr(easting, northing, z))
    report(1, 1)
    return PositionsExportStats(
        width=width, height=height, nodata_pixels=nodata_pixels
    )
