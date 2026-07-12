"""The ``.hf`` heightfield chunk codec: header + zstd-framed height plane.

The heightfield profile (Approach C) stores each tile as a compact vendor
chunk instead of a glTF: a fixed 112-byte little-endian header followed by
one zstandard frame of the tile's quantized elevation plane. This module
is the only place that knows the ``.hf`` byte layout; the normative
specification the Rust runtime decoder codes against lives in
``docs/superpowers/specs/2026-07-12-heightfield-chunk-format.md`` and this
codec mirrors it exactly.

**Coordinate contract (load-bearing).** The stored plane is the *quantized
NAP height plane* — the genuine sampled source heights (``payload.z``),
quantized to ``uint16`` along that single axis via
:func:`~ahn_cli.tiles3d.quantize.quantize_axis`. It is **not** an
ECEF-swizzled mesh axis: a heightfield tile is a geodetic-grid product
whose footprint travels in the header ``region`` (bit-equal to the tile's
``tileset.json`` bounding volume), and whose implicit vertex X/Y, UVs and
connectivity a runtime reconstructs from that region. ``rtc_centre`` rides
in the header only as an A-profile alignment anchor. Every stored value is
a real source sample requantized — never averaged or infilled.

**Determinism.** zstandard is pinned (:data:`ZSTD_LEVEL` = 19,
single-threaded, content size embedded), so the same plane yields the same
bytes; :func:`zstandard_version` exposes the library version for
provenance. Pure module, no I/O beyond in-memory (de)compression.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import zstandard as zstd

from ahn_cli.tiles3d.errors import Tiles3dError
from ahn_cli.tiles3d.quantize import quantize_axis

if TYPE_CHECKING:
    import numpy.typing as npt

    from ahn_cli.tiles3d.mesh import Region
    from ahn_cli.tiles3d.payload import TilePayload

__all__ = [
    "HEADER_SIZE",
    "MAGIC",
    "VERSION",
    "ZSTD_LEVEL",
    "DecodedHeightfield",
    "decode_heightfield",
    "encode_heightfield",
    "zstandard_version",
]

MAGIC = b"AHNH"
"""The 4-byte chunk magic; any other leading bytes are a decode error."""

VERSION = 1
"""The format version this codec reads and writes."""

ZSTD_LEVEL = 19
"""Pinned zstandard level: the maximum stable ratio for write-once assets
(higher ``--ultra`` levels trade determinism/compat headroom for marginal
gains this pipeline does not need). Fixed here as a module constant."""

_HEADER_FORMAT = "<4sIII" + "d" * 11 + "Q"
"""Little-endian header: magic, version, width, height, z_offset, z_scale,
3x rtc_centre, 6x region, payload_len — 11 doubles between the three
uint32 dims and the trailing uint64 length."""

HEADER_SIZE = struct.calcsize(_HEADER_FORMAT)
"""Fixed header size in bytes (112): no padding under the ``<`` layout."""

_U16 = np.dtype("<u2")
_BYTES_PER_LEVEL = 2


@dataclass(frozen=True, eq=False)
class DecodedHeightfield:
    """One decoded ``.hf`` chunk: its header fields and height levels.

    Contract (fields):
        - ``version`` / ``width`` / ``height``: the header dims.
        - ``z_offset`` / ``z_scale``: the height-axis quantizer transform.
        - ``rtc_centre``: the A-profile ECEF y-up RTC centre anchor.
        - ``region``: the tile's EPSG:4979 region (radians + metres).
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
          samples) to ``uint16`` via
          :func:`~ahn_cli.tiles3d.quantize.quantize_axis`, zstd-compresses
          the row-major (top row first) plane, and prepends the fixed
          header carrying the dims, the quantizer transform, the mesh's RTC
          centre and EPSG:4979 region, and the frame length.
        - Deterministic: identical bytes for an identical payload.
    """
    height, width = payload.z.shape
    quantized = quantize_axis(payload.z.reshape(-1))
    frame = _compress(quantized.ints.astype(_U16).tobytes())
    header = struct.pack(
        _HEADER_FORMAT,
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
    return header + frame


def decode_heightfield(data: bytes) -> DecodedHeightfield:
    """Decode a ``.hf`` chunk into its header fields and height plane.

    Contract:
        - Parses the fixed header, decompresses the trailing zstd frame and
          returns a :class:`DecodedHeightfield`. Inverse of
          :func:`encode_heightfield` for a well-formed chunk.

    Failure modes (each a :class:`~ahn_cli.tiles3d.errors.Tiles3dError`):
        - input shorter than :data:`HEADER_SIZE`;
        - ``magic`` not :data:`MAGIC`; ``version`` not :data:`VERSION`;
        - the trailing bytes not exactly the header's ``payload_len``
          (truncated or trailing data);
        - a frame that fails to decompress;
        - a decompressed length not ``width * height * 2``.
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
    width = int(fields[2])
    height = int(fields[3])
    payload_len = int(fields[15])
    frame = data[HEADER_SIZE:]
    if len(frame) != payload_len:
        msg = (
            f"heightfield payload is {len(frame)} bytes but the header "
            f"declares {payload_len}."
        )
        raise Tiles3dError(msg)
    raw = _decompress(frame)
    expected = width * height * _BYTES_PER_LEVEL
    if len(raw) != expected:
        msg = (
            f"heightfield decompressed to {len(raw)} bytes, not "
            f"width*height*2 = {expected}."
        )
        raise Tiles3dError(msg)
    z_ints = np.frombuffer(raw, dtype=_U16).reshape(height, width)
    return DecodedHeightfield(
        version=version,
        width=width,
        height=height,
        z_offset=float(fields[4]),
        z_scale=float(fields[5]),
        rtc_centre=(float(fields[6]), float(fields[7]), float(fields[8])),
        region=(
            float(fields[9]),
            float(fields[10]),
            float(fields[11]),
            float(fields[12]),
            float(fields[13]),
            float(fields[14]),
        ),
        z_ints=z_ints,
    )


def zstandard_version() -> str:
    """Return the installed ``zstandard`` version string (for provenance)."""
    return zstd.__version__


def _compress(raw: bytes) -> bytes:
    """Compress ``raw`` deterministically at the pinned level (size embedded)."""
    compressor = zstd.ZstdCompressor(
        level=ZSTD_LEVEL, write_content_size=True, threads=0
    )
    return compressor.compress(raw)


def _decompress(frame: bytes) -> bytes:
    """Decompress a single embedded-size zstd frame, wrapping any error."""
    try:
        return zstd.ZstdDecompressor().decompress(frame)
    except zstd.ZstdError as exc:
        msg = f"heightfield payload is not a valid zstd frame: {exc}"
        raise Tiles3dError(msg) from exc
