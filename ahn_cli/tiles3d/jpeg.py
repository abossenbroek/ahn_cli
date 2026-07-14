"""Baseline JPEG texture codec for the game profile (Pillow-backed).

This is the *only* module in the tiles3d context allowed to know about
JPEG. The strict profile drapes tiles with hand-packed, byte-deterministic
PNG (:mod:`ahn_cli.tiles3d.png`); the game profile trades that lossless
exactness for far smaller textures by encoding the *same sampled ortho
pixels* as baseline sequential JPEG.

Pinned settings (stated once, here, as module constants) — every one is
fixed so that encoding is byte-deterministic per machine and per pinned
Pillow version:

* :data:`JPEG_QUALITY` = 85 — the quantization level.
* :data:`JPEG_SUBSAMPLING` = 2 — Pillow's 4:2:0 chroma subsampling.
* :data:`JPEG_PROGRESSIVE` = ``False`` — baseline sequential framing only.
* :data:`JPEG_OPTIMIZE` = ``False`` — fixed explicitly so Pillow never
  drifts the Huffman tables between encodes.

There is deliberately no quality knob (YAGNI): the game profile pins one
setting; revisit only on a stakeholder ask.

Because JPEG is lossy, the game profile guards authenticity with a decoded
fidelity floor (:data:`JPEG_MAX_MEAN_ABS_ERROR`, checked by
:func:`jpeg_fidelity_ok`): it asserts the encoder produced *this* image and
not garbage or a wrong tile — it is not a claim that JPEG is near-lossless.
Pillow's version is exposed via :func:`pillow_version` for provenance.
"""

from __future__ import annotations

import importlib.metadata
import io
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from ahn_cli.tiles3d.errors import Tiles3dError

if TYPE_CHECKING:
    import numpy.typing as npt

__all__ = [
    "JPEG_MAX_MEAN_ABS_ERROR",
    "JPEG_OPTIMIZE",
    "JPEG_PROGRESSIVE",
    "JPEG_QUALITY",
    "JPEG_SUBSAMPLING",
    "decode_jpeg",
    "encode_jpeg",
    "is_baseline_jpeg",
    "jpeg_fidelity_ok",
    "pillow_version",
]

JPEG_QUALITY = 85
"""Pinned quantization quality (Pillow's ``quality`` keyword)."""

JPEG_SUBSAMPLING = 2
"""Pinned chroma subsampling: Pillow's ``2`` is 4:2:0."""

JPEG_PROGRESSIVE = False
"""Baseline sequential framing only — never progressive."""

JPEG_OPTIMIZE = False
"""Huffman-optimization pinned off so tables never drift between encodes."""

JPEG_MAX_MEAN_ABS_ERROR = 65.0
"""Maximum tolerated mean absolute error (0-255 scale) between a source
tile and ``decode(encode(tile))``.

Chosen empirically at the pinned settings: worst-case seeded *random-noise*
tiles (the pathological input for JPEG) land at ~50 mean-abs-error, real
smooth ortho imagery at ~1, and comparing against a *different* tile at
~84. 65 clears noise with comfortable headroom yet rejects a wrong or
garbage image — the floor guards "the encoder produced this image", not
"JPEG is nearly lossless".
"""

_CHANNELS = 3

_SOI = b"\xff\xd8"
_MARKER = 0xFF
_SOF0 = 0xC0  # baseline sequential frame
_SOF2 = 0xC2  # progressive frame
_SOS = 0xDA  # start of scan
_SEGMENT_HEADER = 4  # marker byte pair + big-endian length


def is_baseline_jpeg(data: bytes) -> bool:
    """Return whether ``data`` is a baseline sequential JPEG.

    Contract:
        - ``True`` iff the stream starts with SOI and its header segments
          declare a baseline frame (SOF0) and no progressive frame
          (SOF2). Pillow reports ``format == "JPEG"`` for progressive
          streams too, so a progressive regression needs this marker-level
          check; the verifier uses it to pin the game profile's pinned
          baseline framing.
        - The header markers are walked by their big-endian segment
          lengths up to (not into) the entropy-coded scan (SOS); only the
          Pillow-shaped streams this codec produces or a test splices in
          reach it, so a non-marker byte or a length overrunning the
          buffer simply ends the walk.
    """
    if data[:2] != _SOI:
        return False
    markers = _header_markers(data)
    return _SOF0 in markers and _SOF2 not in markers


def _header_markers(data: bytes) -> list[int]:
    """Collect each header segment's marker byte, up to the scan (SOS)."""
    markers: list[int] = []
    pos = 2  # past SOI
    while pos + _SEGMENT_HEADER <= len(data):
        if data[pos] != _MARKER:  # not sitting on a marker: stop the walk
            break
        marker = data[pos + 1]
        if marker == _SOS:  # the entropy-coded scan follows -- stop
            break
        markers.append(marker)
        pos += 2 + ((data[pos + 2] << 8) | data[pos + 3])
    return markers


def encode_jpeg(rgb: npt.NDArray[np.uint8]) -> bytes:
    """Encode an ``(h, w, 3)`` uint8 image as baseline JPEG.

    Contract:
        - Input is the exact sampled ortho pixels — the same array the
          strict PNG path would write — as a contiguous ``(h, w, 3)``
          uint8 array. Encoding uses only the pinned module constants, so
          the same array yields byte-identical output in one process.

    Failure modes:
        - :class:`Tiles3dError` if ``rgb`` is not uint8 or not
          ``(h, w, 3)``, and if Pillow's encoder itself fails (its
          exception is wrapped, never raised raw).
    """
    _require_rgb_u8(rgb)
    try:
        buffer = io.BytesIO()
        Image.fromarray(rgb, mode="RGB").save(
            buffer,
            format="JPEG",
            quality=JPEG_QUALITY,
            subsampling=JPEG_SUBSAMPLING,
            progressive=JPEG_PROGRESSIVE,
            optimize=JPEG_OPTIMIZE,
        )
    except (OSError, ValueError) as exc:
        msg = f"JPEG encode failed: {exc}"
        raise Tiles3dError(msg) from exc
    return buffer.getvalue()


def decode_jpeg(data: bytes) -> npt.NDArray[np.uint8]:
    """Decode JPEG bytes to an ``(h, w, 3)`` uint8 image.

    Contract:
        - Always returns three channels: the decoded image is forced to
          RGB, so a grayscale-mode JPEG still yields ``(h, w, 3)``. Used
          by the verifier's fidelity check.

    Failure modes:
        - :class:`Tiles3dError` if the bytes are not a decodable image
          (Pillow's exception is wrapped, never raised raw).
    """
    try:
        with Image.open(io.BytesIO(data)) as image:
            rgb = np.array(image.convert("RGB"), dtype=np.uint8)
    except (OSError, ValueError) as exc:
        msg = f"JPEG decode failed: {exc}"
        raise Tiles3dError(msg) from exc
    return rgb


def jpeg_fidelity_ok(
    source_rgb: npt.NDArray[np.uint8],
    decoded_rgb: npt.NDArray[np.uint8],
) -> bool:
    """Return whether a decode is faithful to its source tile.

    Contract:
        - ``True`` iff the mean absolute error (0-255 scale) between the
          source tile and the decoded image is at or below
          :data:`JPEG_MAX_MEAN_ABS_ERROR`. Both arrays are the same-shape
          ``(h, w, 3)`` uint8 tile. This is the authenticity floor for the
          lossy game profile: it rejects a garbage or wrong-image decode.
    """
    error = np.abs(
        source_rgb.astype(np.int32) - decoded_rgb.astype(np.int32)
    ).mean()
    return bool(error <= JPEG_MAX_MEAN_ABS_ERROR)


def pillow_version() -> str:
    """Return the installed Pillow version string (for provenance)."""
    return importlib.metadata.version("pillow")


def _require_rgb_u8(rgb: npt.NDArray[np.uint8]) -> None:
    """Gate the encoder input to a contiguous ``(h, w, 3)`` uint8 array."""
    if rgb.dtype != np.uint8:
        msg = f"JPEG encoder needs a uint8 image, got dtype {rgb.dtype}."
        raise Tiles3dError(msg)
    if rgb.ndim != _CHANNELS or rgb.shape[2] != _CHANNELS:
        msg = f"JPEG encoder needs an (h, w, 3) image, got shape {rgb.shape}."
        raise Tiles3dError(msg)
