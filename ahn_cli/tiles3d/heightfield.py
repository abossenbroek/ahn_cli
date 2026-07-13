"""The ``.hf`` heightfield chunk codec: header + zstd-framed height plane.

The heightfield profile (Approach C) stores each tile as a compact vendor
chunk instead of a glTF: a fixed 120-byte little-endian header followed by
one zstandard frame of the tile's quantized elevation plane. This module
is the only place that knows the ``.hf`` byte layout; the normative
specification (version 2) the Rust runtime decoder codes against lives in
``docs/specs/2026-07-12-heightfield-chunk-format.md`` and this
codec mirrors it exactly.

**Coordinate contract (load-bearing).** The stored plane is the *quantized
NAP height plane* — the genuine sampled source heights (``payload.z``),
quantized to ``uint16`` along that single axis via
:func:`~ahn_cli.tiles3d.quantize.quantize_axis` at the 12-bit
:data:`MAX_LEVEL`. It is **not** an ECEF-swizzled mesh axis: a heightfield
tile is a geodetic-grid product whose footprint travels in the header
``region`` — the tile's *own* mesh region, contained within (not always
equal to) the enclosing ``tileset.json`` bounding volume — and whose
implicit vertex X/Y, UVs and connectivity a runtime reconstructs from that
region. ``rtc_centre`` rides in the header only as an A-profile alignment
anchor. Every stored value is a real source sample requantized — never
averaged or infilled.

**Integrity.** The header carries a :data:`~zlib.crc32` (CRC-32/ISO-HDLC)
over its first 112 bytes so a corrupt ``width``/``height``/``payload_len``
is rejected before it can size an allocation, and the zstd frame is written
with its RFC 8878 content checksum so a bit-flipped payload fails to decode
rather than decoding to wrong bytes.

**Determinism.** zstandard is pinned (:data:`ZSTD_LEVEL` = 3,
single-threaded, one-shot compress with the content size and checksum
embedded), so the same plane yields the same bytes; :func:`zstandard_version`
exposes the library version for provenance. Pure module, no I/O beyond
in-memory (de)compression.
"""

from __future__ import annotations

import math
import struct
import zlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import zstandard as zstd

from ahn_cli.tiles3d.errors import Tiles3dError
from ahn_cli.tiles3d.quantize import axis_error_bound, quantize_axis

if TYPE_CHECKING:
    import numpy.typing as npt

    from ahn_cli.tiles3d.mesh import Region
    from ahn_cli.tiles3d.payload import TilePayload

__all__ = [
    "HEADER_SIZE",
    "MAGIC",
    "MAX_AXIS_ERROR_M",
    "MAX_LEVEL",
    "VERSION",
    "ZSTD_LEVEL",
    "DecodedHeightfield",
    "decode_heightfield",
    "encode_heightfield",
    "zstandard_version",
]

MAGIC = b"AHNH"
"""The 4-byte chunk magic; any other leading bytes are a decode error."""

VERSION = 2
"""The format version this codec reads and writes."""

ZSTD_LEVEL = 3
"""Pinned zstandard level: the stakeholder-pinned middle ground from the
codec bake-off. The per-tile-quantized height payload is near-incompressible,
so level 3 matches level 19's ratio at these ≤132 KB granules while encoding
~5x faster. Fixed here as a module constant."""

MAX_LEVEL = 4095
"""The 12-bit maximum quantization level for the height axis. Levels are
stored in the same 2-byte ``uint16`` container as the A profile, but the
value range narrows to ``[0, 4095]`` — ~10x inside AHN's ~5 cm vertical
accuracy on both production grids while cutting the compressed footprint."""

MAX_AXIS_ERROR_M = 0.025
"""The absolute height-error cap (metres). A tile whose exported bound
``z_scale / 2`` exceeds this is refused by the producer and the verifier — an
over-tall (therefore under-resolved) tile is a hard error, not a silently
lossy one. No genuine AHN tile approaches the equivalent 204.75 m extent."""

_PREFIX_FORMAT = "<4sIII" + "d" * 11 + "Q"
"""The header up to and including ``payload_len`` (bytes ``[0, 112)``): magic,
version, width, height, z_offset, z_scale, 3x rtc_centre, 6x region,
payload_len. This is the exact span ``header_crc32`` covers."""

_HEADER_FORMAT = _PREFIX_FORMAT + "II"
"""The full header: the CRC-covered prefix plus ``header_crc32`` and ``pad``."""

HEADER_SIZE = struct.calcsize(_HEADER_FORMAT)
"""Fixed header size in bytes (120): no padding under the ``<`` layout."""

_CRC_SPAN = struct.calcsize(_PREFIX_FORMAT)
"""Byte span ``header_crc32`` is computed over (112)."""

_FLOAT_FIELD_NAMES = (
    "z_offset",
    "z_scale",
    "rtc_centre[0]",
    "rtc_centre[1]",
    "rtc_centre[2]",
    "region[0]",
    "region[1]",
    "region[2]",
    "region[3]",
    "region[4]",
    "region[5]",
)
"""Names of the 11 ``float64`` header fields (struct indices 4..14), in order,
for the non-finite reject's message."""

_U16 = np.dtype("<u2")
_BYTES_PER_LEVEL = 2


@dataclass(frozen=True, eq=False)
class DecodedHeightfield:
    """One decoded ``.hf`` chunk: its header fields and height levels.

    Contract (fields):
        - ``version`` / ``width`` / ``height``: the header dims.
        - ``z_offset`` / ``z_scale``: the height-axis quantizer transform.
        - ``rtc_centre``: the A-profile ECEF y-up RTC centre anchor.
        - ``region``: the tile's own EPSG:4979 region (radians + metres).
        - ``z_ints``: the ``(height, width)`` uint16 quantized height
          plane, row-major top row first.

    ``eq=False``: wraps an array, so instances compare by identity.
    """

    version: int
    width: int
    height: int
    z_offset: float
    z_scale: float
    rtc_centre: tuple[float, float, float]
    region: Region
    z_ints: npt.NDArray[np.uint16]


def encode_heightfield(payload: TilePayload) -> bytes:
    """Encode a tile payload's height plane as a ``.hf`` chunk.

    Contract:
        - Quantizes ``payload.z`` (the ``(rows, cols)`` genuine NAP height
          samples) to ``uint16`` at :data:`MAX_LEVEL` via
          :func:`~ahn_cli.tiles3d.quantize.quantize_axis`, zstd-compresses
          the row-major (top row first) plane, and prepends the fixed
          header carrying the dims, the quantizer transform, the mesh's RTC
          centre and EPSG:4979 region, the frame length and a CRC over the
          preceding header bytes.
        - Deterministic: identical bytes for an identical payload.

    Failure modes:
        - Raises :class:`~ahn_cli.tiles3d.errors.Tiles3dError` if the tile's
          exported error bound ``z_scale / 2`` exceeds :data:`MAX_AXIS_ERROR_M`
          (the height extent is over-tall for the 12-bit range).
    """
    height, width = payload.z.shape
    quantized = quantize_axis(payload.z.reshape(-1), MAX_LEVEL)
    bound = axis_error_bound(quantized.scale)
    if bound > MAX_AXIS_ERROR_M:
        extent = quantized.scale * MAX_LEVEL
        msg = (
            f"heightfield tile height extent {extent:.4f} m exceeds the "
            f"{MAX_AXIS_ERROR_M} m error cap (z_scale/2 = {bound} > "
            f"{MAX_AXIS_ERROR_M})."
        )
        raise Tiles3dError(msg)
    frame = _compress(quantized.ints.astype(_U16).tobytes())
    prefix = struct.pack(
        _PREFIX_FORMAT,
        MAGIC,
        VERSION,
        width,
        height,
        quantized.offset,
        quantized.scale,
        *payload.mesh.center,
        *payload.mesh.region,
        len(frame),
    )
    crc = zlib.crc32(prefix) & 0xFFFFFFFF
    return prefix + struct.pack("<II", crc, 0) + frame


def decode_heightfield(data: bytes) -> DecodedHeightfield:
    """Decode a ``.hf`` chunk into its header fields and height plane.

    Contract:
        - Parses the fixed header, decompresses the trailing zstd frame and
          returns a :class:`DecodedHeightfield`. Inverse of
          :func:`encode_heightfield` for a well-formed chunk. ``header_crc32``
          is verified *before* any dimension or length is trusted.

    Failure modes (each a :class:`~ahn_cli.tiles3d.errors.Tiles3dError`,
    checked in an order that never trusts an unverified length):
        - input shorter than :data:`HEADER_SIZE`;
        - ``magic`` not :data:`MAGIC`; ``version`` not :data:`VERSION`;
        - ``header_crc32`` not matching the CRC-32 of bytes ``[0, 112)``;
        - ``pad`` not ``0``;
        - ``width`` or ``height`` equal to ``0``;
        - any non-finite ``float64`` header field;
        - ``z_scale`` not strictly positive;
        - the trailing bytes not exactly the header's ``payload_len``
          (truncated or trailing data);
        - a frame that fails to decompress (including a content-checksum
          mismatch);
        - a decompressed length not ``width * height * 2`` (evaluated in
          64-bit / unbounded Python integer arithmetic);
        - any stored level greater than :data:`MAX_LEVEL`.
    """
    header = _read_header(data)
    frame = data[HEADER_SIZE:]
    if len(frame) != header.payload_len:
        msg = (
            f"heightfield payload is {len(frame)} bytes but the header "
            f"declares {header.payload_len}."
        )
        raise Tiles3dError(msg)
    raw = _decompress(frame)
    expected = header.width * header.height * _BYTES_PER_LEVEL
    if len(raw) != expected:
        msg = (
            f"heightfield decompressed to {len(raw)} bytes, not "
            f"width*height*2 = {expected}."
        )
        raise Tiles3dError(msg)
    z_ints = np.frombuffer(raw, dtype=_U16).reshape(
        header.height, header.width
    )
    top_level = int(z_ints.max())
    if top_level > MAX_LEVEL:
        msg = (
            f"heightfield stored level {top_level} exceeds the 12-bit "
            f"maximum {MAX_LEVEL}."
        )
        raise Tiles3dError(msg)
    return DecodedHeightfield(
        version=header.version,
        width=header.width,
        height=header.height,
        z_offset=header.z_offset,
        z_scale=header.z_scale,
        rtc_centre=header.rtc_centre,
        region=header.region,
        z_ints=z_ints,
    )


@dataclass(frozen=True)
class _Header:
    """The parsed, integrity-checked fixed header (everything but the plane)."""

    version: int
    width: int
    height: int
    z_offset: float
    z_scale: float
    rtc_centre: tuple[float, float, float]
    region: Region
    payload_len: int


def _read_header(data: bytes) -> _Header:
    """Parse and integrity-check the fixed header, returning its fields.

    Runs every header-level reject in spec order — length, magic, version,
    ``header_crc32`` (before any dimension is trusted), ``pad``, zero dims,
    non-finite floats and non-positive ``z_scale`` — so the caller can trust
    the returned fields when sizing the payload. Each violation raises a
    :class:`~ahn_cli.tiles3d.errors.Tiles3dError`.
    """
    if len(data) < HEADER_SIZE:
        msg = (
            f"heightfield chunk is {len(data)} bytes, shorter than the "
            f"{HEADER_SIZE}-byte header."
        )
        raise Tiles3dError(msg)
    fields = struct.unpack(_HEADER_FORMAT, data[:HEADER_SIZE])
    magic = fields[0]
    if magic != MAGIC:
        msg = (
            f"heightfield chunk has bad magic {magic!r}; expected {MAGIC!r}."
        )
        raise Tiles3dError(msg)
    version = int(fields[1])
    if version != VERSION:
        msg = (
            f"heightfield chunk version {version} is not the supported "
            f"version {VERSION}."
        )
        raise Tiles3dError(msg)
    stored_crc = int(fields[16])
    actual_crc = zlib.crc32(data[:_CRC_SPAN]) & 0xFFFFFFFF
    if actual_crc != stored_crc:
        msg = (
            f"heightfield header CRC32 {stored_crc:#010x} does not match the "
            f"computed {actual_crc:#010x}; the header is corrupt."
        )
        raise Tiles3dError(msg)
    pad = int(fields[17])
    if pad != 0:
        msg = f"heightfield header pad field is {pad}, must be 0."
        raise Tiles3dError(msg)
    width = int(fields[2])
    height = int(fields[3])
    if width == 0 or height == 0:
        msg = (
            f"heightfield header declares a zero dimension: "
            f"width={width}, height={height}."
        )
        raise Tiles3dError(msg)
    for name, value in zip(_FLOAT_FIELD_NAMES, fields[4:15], strict=True):
        if not math.isfinite(value):
            msg = f"heightfield header field {name} is non-finite ({value})."
            raise Tiles3dError(msg)
    z_scale = float(fields[5])
    if z_scale <= 0.0:
        msg = f"heightfield header z_scale {z_scale} must be positive."
        raise Tiles3dError(msg)
    return _Header(
        version=version,
        width=width,
        height=height,
        z_offset=float(fields[4]),
        z_scale=z_scale,
        rtc_centre=(float(fields[6]), float(fields[7]), float(fields[8])),
        region=(
            float(fields[9]),
            float(fields[10]),
            float(fields[11]),
            float(fields[12]),
            float(fields[13]),
            float(fields[14]),
        ),
        payload_len=int(fields[15]),
    )


def zstandard_version() -> str:
    """Return the installed ``zstandard`` version string (for provenance)."""
    return zstd.__version__


def _compress(raw: bytes) -> bytes:
    """Compress ``raw`` deterministically: one-shot, size and checksum embedded.

    The one-shot ``compress`` call is normative — the streamed writer path
    emits a different frame for the same input, so it is never used here.
    """
    compressor = zstd.ZstdCompressor(
        level=ZSTD_LEVEL,
        write_content_size=True,
        write_checksum=True,
        threads=0,
    )
    return compressor.compress(raw)


def _decompress(frame: bytes) -> bytes:
    """Decompress a single embedded-size zstd frame, wrapping any error.

    A full-frame decode reaches and verifies the RFC 8878 content checksum,
    so a bit-flipped or truncated payload surfaces here as an error rather
    than as silent wrong bytes.
    """
    try:
        return zstd.ZstdDecompressor().decompress(frame)
    except zstd.ZstdError as exc:
        msg = f"heightfield payload is not a valid zstd frame: {exc}"
        raise Tiles3dError(msg) from exc
