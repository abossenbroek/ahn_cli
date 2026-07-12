"""``EXT_meshopt_compression`` stream codec for the game profile.

This is the *only* module in the tiles3d context allowed to know about
meshopt. The game glTF writer (a later task) drives every vertex-attribute
and index bufferView through here; the verifier decodes the streams and
demands the pre-encode bytes back bit-for-bit. The extension is declared
**required** (no fallback buffer) — the Rust ``meshopt`` crate decodes it
at runtime.

Three streams, each with pinned mode/filter/stride (documented once, here):

* **POSITION** — the ``KHR_mesh_quantization`` ``uint16 x 3`` vertices from
  :mod:`ahn_cli.tiles3d.quantize`. meshopt's vertex codec requires the byte
  stride divisible by 4, so each 6-byte vertex is padded to
  :data:`POSITION_BYTE_STRIDE` (8) with two zero bytes. Mode
  :data:`MODE_ATTRIBUTES`, filter ``None`` — the values are already
  quantized integers, so no meshopt filter (OCTAHEDRAL/QUATERNION/EXPONENTIAL)
  applies.
* **TEXCOORD_0** — normalized ``uint16 x 2`` UVs, a natural
  :data:`UV_BYTE_STRIDE` (4) with no padding. Mode :data:`MODE_ATTRIBUTES`,
  filter ``None`` (quantized ints).
* **indices** — the ``uint32`` triangle list. Mode :data:`MODE_TRIANGLES`.
  Component size is deliberately **4 bytes** (:data:`INDEX_BYTE_STRIDE`):
  a full-resolution tile has up to ``257 x 257 = 66049`` vertices, past the
  ``uint16`` ceiling of 65535, so ``uint16`` indices would overflow — the
  codec's 2-byte TRIANGLES mode is not an option here.

The meshopt index codec canonicalizes triangle *rotation* (it preserves the
triangle set and winding but may rotate each triangle's in-triangle vertex
order). The mesh writer's ``(a, c, d), (a, d, b)`` grid pattern is already
rotation-canonical, so it round-trips exactly; :func:`encode_indices` proves
this per call and refuses any non-canonical buffer, so a future mesh change
that broke canonicality would fail loudly at encode time rather than silently
storing bytes that reproduce a different triangle list.

The pinned library version is exposed via :func:`meshoptimizer_version` for
provenance (wired in a later task); determinism is anchored by the
``uv.lock`` pin.
"""

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass
from typing import TYPE_CHECKING

import meshoptimizer as mo
import numpy as np

from ahn_cli.tiles3d.errors import Tiles3dError

if TYPE_CHECKING:
    import numpy.typing as npt

__all__ = [
    "INDEX_BYTE_STRIDE",
    "MODE_ATTRIBUTES",
    "MODE_TRIANGLES",
    "POSITION_BYTE_STRIDE",
    "UV_BYTE_STRIDE",
    "MeshoptStream",
    "decode_indices",
    "decode_positions",
    "decode_uvs",
    "encode_indices",
    "encode_positions",
    "encode_uvs",
    "meshoptimizer_version",
]

MODE_ATTRIBUTES = "ATTRIBUTES"
"""meshopt vertex-codec mode for vertex-attribute bufferViews."""

MODE_TRIANGLES = "TRIANGLES"
"""meshopt index-codec mode for index bufferViews."""

POSITION_BYTE_STRIDE = 8
"""``uint16 x 3`` (6 bytes) padded to 8 — meshopt needs stride % 4 == 0."""

UV_BYTE_STRIDE = 4
"""``uint16 x 2`` — already a multiple of 4, no padding."""

INDEX_BYTE_STRIDE = 4
"""``uint32`` indices — tiles exceed the uint16 vertex ceiling (65535)."""

_POSITION_COMPONENTS = 3
_UV_COMPONENTS = 2
_TRIANGLE = 3
_MATRIX_NDIM = 2


@dataclass(frozen=True, eq=False)
class MeshoptStream:
    """One ``EXT_meshopt_compression`` bufferView's compressed bytes + params.

    Contract (fields):
        - ``data``: the compressed stream, ready for the glb BIN chunk.
        - ``count``: element count — vertices for attributes, indices for
          the index stream (the glTF ``count`` / ``EXT_meshopt`` ``count``).
        - ``byte_stride``: the *pre-compression* element stride in bytes
          (8 padded POSITION, 4 UV, 4 index).
        - ``mode``: :data:`MODE_ATTRIBUTES` or :data:`MODE_TRIANGLES`.
        - ``filter``: the meshopt filter name, or ``None`` when none applies
          (quantized-integer attributes need no filter).

    ``eq=False``: wraps opaque bytes, so instances compare by identity.
    """

    data: bytes
    count: int
    byte_stride: int
    mode: str
    filter: str | None


def encode_positions(ints: npt.NDArray[np.uint16]) -> MeshoptStream:
    """Encode ``(n, 3)`` ``uint16`` POSITION vertices as a meshopt stream.

    Contract:
        - ``ints``: ``(n, 3)`` uint16 (``n >= 1``), the quantized RTC
          vertices from :func:`ahn_cli.tiles3d.quantize.quantize_positions`.
        - Each vertex's 6 little-endian bytes are padded to an 8-byte
          stride (two trailing zero bytes) and run through meshopt's vertex
          codec. Mode :data:`MODE_ATTRIBUTES`, filter ``None``, stride
          :data:`POSITION_BYTE_STRIDE`.

    Failure modes:
        - :class:`~ahn_cli.tiles3d.errors.Tiles3dError` if ``ints`` is not
          ``(n, 3)`` uint16 with ``n >= 1``.
    """
    data = _require_u16_matrix(ints, _POSITION_COMPONENTS, "positions")
    count = data.shape[0]
    padded = np.zeros((count, POSITION_BYTE_STRIDE), dtype=np.uint8)
    padded[:, :6] = data.view(np.uint8).reshape(count, 6)
    encoded = mo.encode_vertex_buffer(padded, count, POSITION_BYTE_STRIDE)
    return MeshoptStream(
        data=encoded,
        count=count,
        byte_stride=POSITION_BYTE_STRIDE,
        mode=MODE_ATTRIBUTES,
        filter=None,
    )


def encode_uvs(ints: npt.NDArray[np.uint16]) -> MeshoptStream:
    """Encode ``(n, 2)`` ``uint16`` TEXCOORD_0 UVs as a meshopt stream.

    Contract:
        - ``ints``: ``(n, 2)`` uint16 (``n >= 1``), the normalized UVs from
          :func:`ahn_cli.tiles3d.quantize.quantize_uvs`.
        - The 4-byte little-endian vertices run through meshopt's vertex
          codec directly (no padding). Mode :data:`MODE_ATTRIBUTES`, filter
          ``None``, stride :data:`UV_BYTE_STRIDE`.

    Failure modes:
        - :class:`~ahn_cli.tiles3d.errors.Tiles3dError` if ``ints`` is not
          ``(n, 2)`` uint16 with ``n >= 1``.
    """
    data = _require_u16_matrix(ints, _UV_COMPONENTS, "uvs")
    count = data.shape[0]
    buffer = data.view(np.uint8).reshape(count, UV_BYTE_STRIDE)
    encoded = mo.encode_vertex_buffer(buffer, count, UV_BYTE_STRIDE)
    return MeshoptStream(
        data=encoded,
        count=count,
        byte_stride=UV_BYTE_STRIDE,
        mode=MODE_ATTRIBUTES,
        filter=None,
    )


def encode_indices(indices: npt.NDArray[np.uint32]) -> MeshoptStream:
    """Encode a ``(t * 3,)`` ``uint32`` triangle list as a meshopt stream.

    Contract:
        - ``indices``: 1-D uint32, length a positive multiple of 3. Mode
          :data:`MODE_TRIANGLES`, filter ``None``, stride
          :data:`INDEX_BYTE_STRIDE`.
        - The buffer must be meshopt-canonical: ``decode(encode) ==
          indices`` exactly. The mesh writer's grid pattern already is; this
          check makes a future non-canonical pattern fail loudly rather than
          store bytes that reproduce a rotated triangle list.

    Failure modes:
        - :class:`~ahn_cli.tiles3d.errors.Tiles3dError` if ``indices`` is
          not 1-D uint32 with a positive length that is a multiple of 3, or
          if the buffer is not meshopt-canonical.
    """
    data = _require_index_buffer(indices)
    count = data.shape[0]
    vertex_count = int(data.max()) + 1
    encoded = mo.encode_index_buffer(data, count, vertex_count)
    roundtrip = _decode_index_bytes(encoded, count)
    if not np.array_equal(roundtrip, data):
        msg = (
            "meshopt index encode: buffer is not meshopt-canonical "
            "(decode(encode) rotated a triangle); refusing to store bytes "
            "that reproduce a different triangle list."
        )
        raise Tiles3dError(msg)
    return MeshoptStream(
        data=encoded,
        count=count,
        byte_stride=INDEX_BYTE_STRIDE,
        mode=MODE_TRIANGLES,
        filter=None,
    )


def decode_positions(
    stream: MeshoptStream | bytes, count: int
) -> npt.NDArray[np.uint16]:
    """Decode a POSITION stream back to ``(count, 3)`` ``uint16``.

    Contract:
        - Accepts a :class:`MeshoptStream` or its raw ``bytes``; strips the
          two pad bytes per 8-byte vertex and returns the ``(count, 3)``
          uint16 array — bit-identical to :func:`encode_positions`' input.

    Failure modes:
        - :class:`~ahn_cli.tiles3d.errors.Tiles3dError` if the bytes are not
          a decodable stream (the library error is wrapped, never raised
          raw).
    """
    raw = _decode_vertex_bytes(
        _stream_bytes(stream), count, POSITION_BYTE_STRIDE
    )
    padded = np.frombuffer(raw, dtype=np.uint8).reshape(
        count, POSITION_BYTE_STRIDE
    )
    unpadded = np.ascontiguousarray(padded[:, :6])
    return unpadded.view("<u2").reshape(count, _POSITION_COMPONENTS)


def decode_uvs(
    stream: MeshoptStream | bytes, count: int
) -> npt.NDArray[np.uint16]:
    """Decode a TEXCOORD_0 stream back to ``(count, 2)`` ``uint16``.

    Contract:
        - Accepts a :class:`MeshoptStream` or its raw ``bytes`` and returns
          the ``(count, 2)`` uint16 array — bit-identical to
          :func:`encode_uvs`' input.

    Failure modes:
        - :class:`~ahn_cli.tiles3d.errors.Tiles3dError` if the bytes are not
          a decodable stream (the library error is wrapped, never raised
          raw).
    """
    raw = _decode_vertex_bytes(_stream_bytes(stream), count, UV_BYTE_STRIDE)
    return np.frombuffer(raw, dtype="<u2").reshape(count, _UV_COMPONENTS)


def decode_indices(
    stream: MeshoptStream | bytes, index_count: int
) -> npt.NDArray[np.uint32]:
    """Decode an index stream back to a ``(index_count,)`` ``uint32`` list.

    Contract:
        - Accepts a :class:`MeshoptStream` or its raw ``bytes`` and returns
          the ``(index_count,)`` uint32 triangle list — bit-identical to
          :func:`encode_indices`' input.

    Failure modes:
        - :class:`~ahn_cli.tiles3d.errors.Tiles3dError` if the bytes are not
          a decodable stream (the library error is wrapped, never raised
          raw).
    """
    return _decode_index_bytes(_stream_bytes(stream), index_count)


def meshoptimizer_version() -> str:
    """Return the installed ``meshoptimizer`` version string (provenance)."""
    return importlib.metadata.version("meshoptimizer")


def _stream_bytes(stream: MeshoptStream | bytes) -> bytes:
    """Extract the compressed bytes from a stream or pass raw bytes through."""
    return stream.data if isinstance(stream, MeshoptStream) else stream


def _decode_vertex_bytes(data: bytes, count: int, stride: int) -> bytes:
    """Decode a meshopt vertex stream to its raw ``count * stride`` bytes."""
    try:
        decoded = mo.decode_vertex_buffer(count, stride, data)
    except (RuntimeError, ValueError) as exc:
        msg = f"meshopt vertex decode failed: {exc}"
        raise Tiles3dError(msg) from exc
    return decoded.tobytes()


def _decode_index_bytes(
    data: bytes, index_count: int
) -> npt.NDArray[np.uint32]:
    """Decode a meshopt index stream to a ``(index_count,)`` uint32 array."""
    try:
        decoded = mo.decode_index_buffer(index_count, INDEX_BYTE_STRIDE, data)
    except (RuntimeError, ValueError) as exc:
        msg = f"meshopt index decode failed: {exc}"
        raise Tiles3dError(msg) from exc
    return np.frombuffer(decoded.tobytes(), dtype="<u4").reshape(index_count)


def _require_u16_matrix(
    array: npt.NDArray[np.uint16], width: int, name: str
) -> npt.NDArray[np.uint16]:
    """Gate an ``(n, width)`` uint16 block with ``n >= 1``; return it LE-contiguous."""
    if array.dtype != np.uint16:
        msg = f"meshopt {name}: expected uint16, got dtype {array.dtype}."
        raise Tiles3dError(msg)
    if (
        array.ndim != _MATRIX_NDIM
        or array.shape[0] < 1
        or array.shape[1] != width
    ):
        msg = (
            f"meshopt {name}: expected an (n, {width}) array with n >= 1, "
            f"got shape {array.shape}."
        )
        raise Tiles3dError(msg)
    return np.ascontiguousarray(array, dtype="<u2")


def _require_index_buffer(
    indices: npt.NDArray[np.uint32],
) -> npt.NDArray[np.uint32]:
    """Gate a 1-D uint32 triangle list (positive multiple of 3); LE-contiguous."""
    if indices.dtype != np.uint32:
        msg = f"meshopt indices: expected uint32, got dtype {indices.dtype}."
        raise Tiles3dError(msg)
    if indices.ndim != 1 or indices.shape[0] < _TRIANGLE:
        msg = (
            "meshopt indices: expected a 1-D array of at least 3 elements, "
            f"got shape {indices.shape}."
        )
        raise Tiles3dError(msg)
    if indices.shape[0] % _TRIANGLE != 0:
        msg = (
            "meshopt indices: length must be a multiple of 3 (triangle "
            f"list), got {indices.shape[0]}."
        )
        raise Tiles3dError(msg)
    return np.ascontiguousarray(indices, dtype="<u4")
