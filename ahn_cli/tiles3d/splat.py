"""The ``splat`` profile's codec: a binary 3DGS ``.ply`` blob, zstd-wrapped.

The splat profile (``content_kind = 2``) encodes each tile as a 3D
Gaussian Splatting cloud instead of a mesh or a height chunk: **one
isotropic gaussian per** :class:`~ahn_cli.tiles3d.payload.TilePayload`
**grid vertex** — the same stride-sampled grid the mesh and heightfield
profiles emit, so a splat tile aligns exactly with its mesh/heightfield
counterpart. This is a genuine, deterministic geometric encoding of our
own data (position + colour + a measured cell size); it is **not** a
trained radiance field (no view-dependent SH, no multi-view training).

**Per-gaussian construction:**

- ``position``: the payload's mesh vertex exactly as-is — tile-local RTC,
  glTF y-up, float32, bit-identical to :mod:`ahn_cli.tiles3d.mesh`'s
  ``TileMesh.positions``. No quantization.
- ``scale``: isotropic, one value per tile (not per vertex): the
  Euclidean RTC-frame distance between the first two column-adjacent
  vertices — a genuine geometric measurement of this tile's cell spacing
  at its LOD stride, applied uniformly to every gaussian and every axis.
  Every tile has at least two sampled columns and rows (the quadtree's
  2-sample-per-axis floor), so this is always well-defined.
- ``colour``: the sampled ortho pixel as an SH degree-0 coefficient,
  ``f_dc = (c/255 - 0.5) / SH_DC0``. The ortho byte value is used directly
  as ``c`` with **no sRGB-to-linear decode** — consistent with the game
  and heightfield profiles, which also drape tiles with the raw sampled
  bytes untouched (see :mod:`ahn_cli.tiles3d.jpeg` /
  :mod:`ahn_cli.tiles3d.png`); this profile does not introduce a
  colour-space conversion the others don't have either.
- ``opacity``: a fixed full-opacity constant for every gaussian (not
  data-derived).
- ``rotation``: the identity quaternion for every gaussian (axis-aligned,
  isotropic — there is no orientation to encode).

**Wire format.** A standard INRIA-convention binary little-endian 3DGS
``.ply``: ``x y z, f_dc_0..2, opacity, scale_0..2, rot_0..3`` (14 float32
properties per vertex, 56 bytes/vertex; ``~66k gaussians x 56 B ~= 3.7 MB``
raw per tile before compression — bounded by the LOD stride, same as every
other profile's per-tile footprint). Normals (``nx ny nz``) are **omitted**
(a loader either treats gaussians as normal-free or ignores them; this
producer never needs them). Per the INRIA convention, pre-activation
values are stored, not the activated ones: ``scale`` is stored as
``log(scale)`` (a loader applies ``exp``) and ``opacity`` as
``logit(opacity)`` (a loader applies ``sigmoid``); ``rot`` is a normalized
quaternion in ``(w, x, y, z)`` order; ``f_dc`` is linear (see above — no
further activation).

The whole ``.ply`` (header and body together) is then zstd-wrapped exactly
as :mod:`ahn_cli.tiles3d.heightfield` wraps its plane (one-shot, pinned
level, content size and RFC 8878 checksum embedded) — unlike the
heightfield chunk, there is no separate plaintext header outside the zstd
frame, so the frame's own content checksum already covers the whole blob.

This module is the only place in the tiles3d context that knows the PLY
field order or the zstd framing; :func:`decode_splat` is a Python
reference decoder for the verifier. Pure module, no I/O beyond in-memory
(de)compression.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import zstandard as zstd

from ahn_cli.tiles3d.errors import Tiles3dError

if TYPE_CHECKING:
    import numpy.typing as npt

    from ahn_cli.tiles3d.mesh import TileMesh
    from ahn_cli.tiles3d.payload import TilePayload

__all__ = [
    "OPACITY",
    "SH_DC0",
    "ZSTD_LEVEL",
    "DecodedSplat",
    "decode_splat",
    "encode_splat",
    "zstandard_version",
]

SH_DC0 = 0.28209479177387814
"""Degree-0 real spherical-harmonic normalization constant, ``1 / (2 sqrt(pi))``."""

OPACITY = 0.99
"""Fixed full-opacity probability every gaussian is assigned (not sampled)."""

ZSTD_LEVEL = 3
"""Pinned zstandard level, matching the heightfield chunk codec's pin."""

_ROT_IDENTITY = (1.0, 0.0, 0.0, 0.0)
"""Identity quaternion ``(w, x, y, z)``: every gaussian is axis-aligned."""

_OPACITY_LOGIT = math.log(OPACITY / (1.0 - OPACITY))
"""Stored pre-activation opacity: ``logit(OPACITY)``; a loader applies sigmoid."""

_PROPERTIES = (
    "x",
    "y",
    "z",
    "f_dc_0",
    "f_dc_1",
    "f_dc_2",
    "opacity",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
)
"""The pinned per-vertex PLY property order (14 float32 fields/gaussian)."""

_FIELD_COUNT = len(_PROPERTIES)
_F32 = np.dtype("<f4")
_BYTES_PER_VERTEX = _FIELD_COUNT * _F32.itemsize

_MAGIC_LINE = b"ply"
_FORMAT_LINE = b"format binary_little_endian 1.0"
_ELEMENT_PREFIX = b"element vertex "
_END_HEADER_LINE = b"end_header"
_END_HEADER = _END_HEADER_LINE + b"\n"
_HEADER_LINE_COUNT = 3 + _FIELD_COUNT + 1


@dataclass(frozen=True, eq=False)
class DecodedSplat:
    """One decoded splat ``.ply`` blob: its per-gaussian arrays.

    Contract (fields):
        - ``count``: number of gaussians (the tile's vertex count).
        - ``positions``: ``(count, 3)`` float32 tile-local RTC vertices,
          identical to the encoding tile's
          :class:`~ahn_cli.tiles3d.mesh.TileMesh.positions`.
        - ``f_dc``: ``(count, 3)`` float32 SH degree-0 colour coefficients.
        - ``opacity``: ``(count,)`` float32 pre-activation (logit) opacity.
        - ``scale``: ``(count, 3)`` float32 pre-activation (log) isotropic
          scale (every column equal for a genuine encode).
        - ``rot``: ``(count, 4)`` float32 ``(w, x, y, z)`` quaternions.

    ``eq=False``: wraps arrays, so instances compare by identity.
    """

    count: int
    positions: npt.NDArray[np.float32]
    f_dc: npt.NDArray[np.float32]
    opacity: npt.NDArray[np.float32]
    scale: npt.NDArray[np.float32]
    rot: npt.NDArray[np.float32]


def encode_splat(payload: TilePayload) -> bytes:
    """Encode a tile payload as a zstd-wrapped binary 3DGS ``.ply``.

    Contract:
        - One gaussian per ``payload.mesh`` vertex: position copied as-is,
          colour the sampled ortho pixel as an SH degree-0 coefficient,
          scale the tile's measured cell spacing (isotropic, log-stored),
          opacity :data:`OPACITY`'s logit, rotation the identity quaternion.
        - Deterministic: identical bytes for an identical payload.

    Failure modes:
        - Raises :class:`~ahn_cli.tiles3d.errors.Tiles3dError` if the
          tile's measured cell spacing is not a positive finite number
          (a degenerate or duplicate adjacent vertex pair).
    """
    positions = payload.mesh.positions
    count = positions.shape[0]
    rgb = payload.rgb.reshape(-1, 3).astype(np.float64) / 255.0
    f_dc = ((rgb - 0.5) / SH_DC0).astype(_F32)
    log_scale = np.float32(math.log(_cell_spacing(payload.mesh)))
    arr = np.empty((count, _FIELD_COUNT), dtype=_F32)
    arr[:, 0:3] = positions
    arr[:, 3:6] = f_dc
    arr[:, 6] = np.float32(_OPACITY_LOGIT)
    arr[:, 7:10] = log_scale
    arr[:, 10:14] = np.asarray(_ROT_IDENTITY, dtype=_F32)
    return _compress(_header(count) + arr.tobytes())


def decode_splat(data: bytes) -> DecodedSplat:
    """Decode a zstd-wrapped binary 3DGS ``.ply`` blob.

    Contract:
        - Inverse of :func:`encode_splat` for a well-formed blob: parses
          the fixed ASCII header (exact magic, format, vertex count and
          the pinned property order) and the binary vertex body, and
          returns a :class:`DecodedSplat`.

    Failure modes (each a :class:`~ahn_cli.tiles3d.errors.Tiles3dError`):
        - the frame is not a valid zstd frame (a bit-flip fails its RFC
          8878 content checksum);
        - the decompressed bytes are missing the ``end_header`` terminator;
        - the magic (``ply``), format, ``element vertex`` count or any
          ``property`` line does not match the pinned form;
        - the body length is not exactly ``count * 56`` bytes.
    """
    raw = _decompress(data)
    end = raw.find(_END_HEADER)
    if end == -1:
        msg = "splat ply is missing the end_header terminator."
        raise Tiles3dError(msg)
    header = raw[: end + len(_END_HEADER)]
    body = raw[end + len(_END_HEADER) :]
    count = _parse_header(header)
    expected_len = count * _BYTES_PER_VERTEX
    if len(body) != expected_len:
        msg = (
            f"splat ply body is {len(body)} bytes, not "
            f"count*{_BYTES_PER_VERTEX} = {expected_len}."
        )
        raise Tiles3dError(msg)
    arr = np.frombuffer(body, dtype=_F32).reshape(count, _FIELD_COUNT)
    return DecodedSplat(
        count=count,
        positions=arr[:, 0:3],
        f_dc=arr[:, 3:6],
        opacity=arr[:, 6],
        scale=arr[:, 7:10],
        rot=arr[:, 10:14],
    )


def zstandard_version() -> str:
    """Return the installed ``zstandard`` version string (for provenance)."""
    return zstd.__version__


def _cell_spacing(mesh: TileMesh) -> float:
    """Return the tile's measured cell spacing (metres), or raise.

    A genuine geometric measurement, not a formula: the Euclidean distance,
    in the tile's own RTC frame, between the first two column-adjacent
    vertices. Every tile has at least two sampled columns
    (:func:`~ahn_cli.tiles3d.quadtree.plan_quadtree`'s 2-sample-per-axis
    floor), so this is always defined for a genuine tile; applied uniformly
    (isotropic) as every gaussian's scale.
    """
    positions = mesh.positions.astype(np.float64)
    delta = positions[1] - positions[0]
    spacing = float(np.sqrt(np.sum(delta * delta)))
    if not math.isfinite(spacing) or spacing <= 0.0:
        msg = (
            f"splat tile cell spacing {spacing} is not a positive finite "
            "number (degenerate or duplicate adjacent vertices)."
        )
        raise Tiles3dError(msg)
    return spacing


def _header(count: int) -> bytes:
    """Build the fixed ASCII PLY header for ``count`` vertices."""
    lines = (
        _MAGIC_LINE,
        _FORMAT_LINE,
        _ELEMENT_PREFIX + str(count).encode("ascii"),
        *(f"property float {name}".encode("ascii") for name in _PROPERTIES),
        _END_HEADER_LINE,
    )
    return b"\n".join(lines) + b"\n"


def _parse_header(header: bytes) -> int:
    r"""Parse and strictly validate the fixed ASCII PLY header; return count.

    ``header`` is always the caller's ``raw[: end + len(_END_HEADER)]``
    slice, so it always ends with exactly ``b"end_header\n"`` — splitting
    on ``b"\n"`` therefore always yields a trailing empty element (dropped
    below) and a genuine final line of ``b"end_header"``; a redundant check
    of either would never fire and is not repeated here.
    """
    lines = header.split(b"\n")[:-1]
    if len(lines) != _HEADER_LINE_COUNT:
        msg = (
            f"splat ply header has {len(lines)} lines, expected "
            f"{_HEADER_LINE_COUNT}."
        )
        raise Tiles3dError(msg)
    if lines[0] != _MAGIC_LINE:
        msg = f"splat ply header magic is {lines[0]!r}, expected b'ply'."
        raise Tiles3dError(msg)
    if lines[1] != _FORMAT_LINE:
        msg = (
            f"splat ply header format line is {lines[1]!r}, expected "
            f"{_FORMAT_LINE!r}."
        )
        raise Tiles3dError(msg)
    element_line = lines[2]
    if not element_line.startswith(_ELEMENT_PREFIX):
        msg = f"splat ply header element line is {element_line!r}."
        raise Tiles3dError(msg)
    count_bytes = element_line[len(_ELEMENT_PREFIX) :]
    if not count_bytes.isdigit():
        msg = (
            f"splat ply vertex count {count_bytes!r} is not a decimal "
            "integer."
        )
        raise Tiles3dError(msg)
    for name, line in zip(
        _PROPERTIES, lines[3 : 3 + _FIELD_COUNT], strict=True
    ):
        expected = f"property float {name}".encode("ascii")
        if line != expected:
            msg = (
                f"splat ply property line is {line!r}, expected {expected!r}."
            )
            raise Tiles3dError(msg)
    return int(count_bytes)


def _compress(raw: bytes) -> bytes:
    """Compress ``raw`` deterministically: one-shot, size and checksum embedded."""
    compressor = zstd.ZstdCompressor(
        level=ZSTD_LEVEL,
        write_content_size=True,
        write_checksum=True,
        threads=0,
    )
    return compressor.compress(raw)


def _decompress(frame: bytes) -> bytes:
    """Decompress a single embedded-size zstd frame, wrapping any error."""
    try:
        return zstd.ZstdDecompressor().decompress(frame)
    except zstd.ZstdError as exc:
        msg = f"splat ply is not a valid zstd frame: {exc}"
        raise Tiles3dError(msg) from exc
