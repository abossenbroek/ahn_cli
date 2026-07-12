"""Tests for the ``EXT_meshopt_compression`` stream codec (game profile).

These pin the codec's contract: deterministic (encode-twice byte-equal)
and bit-exact round-tripping for POSITION, TEXCOORD_0, and index streams,
the input gates, the library-exception wrapping into ``Tiles3dError``, and
the index canonicalization guard (the grid index pattern must survive the
meshopt index codec's rotation canonicalization untouched).
"""

from __future__ import annotations

import importlib.metadata
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

if TYPE_CHECKING:
    import numpy.typing as npt

from ahn_cli.tiles3d.errors import Tiles3dError
from ahn_cli.tiles3d.meshopt import (
    INDEX_BYTE_STRIDE,
    MODE_ATTRIBUTES,
    MODE_TRIANGLES,
    POSITION_BYTE_STRIDE,
    UV_BYTE_STRIDE,
    MeshoptStream,
    decode_indices,
    decode_positions,
    decode_uvs,
    encode_indices,
    encode_positions,
    encode_uvs,
    meshoptimizer_version,
)


def _grid_triangles(n_cols: int, n_rows: int) -> np.ndarray:
    """Replicate ``ahn_cli.tiles3d.mesh._grid_triangles`` exactly.

    The two-triangles-per-cell ``(a, c, d), (a, d, b)`` winding is the
    pattern the mesh writer emits; the meshopt index codec must round-trip
    it bit-for-bit (it is already rotation-canonical).
    """
    cell_col = np.arange(n_cols - 1, dtype=np.uint32)
    cell_row = np.arange(n_rows - 1, dtype=np.uint32)
    cc, rr = np.meshgrid(cell_col, cell_row)
    a = (rr * n_cols + cc).ravel()
    b = a + 1
    c = a + n_cols
    d = c + 1
    triangles = np.column_stack([a, c, d, a, d, b])
    return triangles.reshape(-1).astype(np.uint32)


def _rng_positions(n: int) -> np.ndarray:
    """Seeded ``(n, 3)`` uint16 stand-in for quantized RTC vertices."""
    rng = np.random.default_rng(1)
    return rng.integers(0, 65536, size=(n, 3), dtype=np.uint16)


def _rng_uvs(n: int) -> np.ndarray:
    """Seeded ``(n, 2)`` uint16 stand-in for quantized UVs."""
    rng = np.random.default_rng(2)
    return rng.integers(0, 65536, size=(n, 2), dtype=np.uint16)


# --- positions ------------------------------------------------------------


def test_encode_positions_roundtrip_bit_exact() -> None:
    """A POSITION stream decodes back to its exact uint16 input."""
    ints = _rng_positions(64)
    stream = encode_positions(ints)
    assert np.array_equal(decode_positions(stream, len(ints)), ints)


def test_encode_positions_stream_metadata() -> None:
    """A POSITION stream carries mode ATTRIBUTES, no filter, stride 8."""
    ints = _rng_positions(10)
    stream = encode_positions(ints)
    assert isinstance(stream, MeshoptStream)
    assert stream.count == 10
    assert stream.byte_stride == POSITION_BYTE_STRIDE == 8
    assert stream.mode == MODE_ATTRIBUTES
    assert stream.filter is None
    assert isinstance(stream.data, bytes)


def test_encode_positions_deterministic() -> None:
    """Encoding the same POSITION input twice is byte-identical."""
    ints = _rng_positions(40)
    assert encode_positions(ints).data == encode_positions(ints).data


def test_encode_positions_pad_bytes_are_zero() -> None:
    """The two pad bytes of every 8-byte vertex decode back to zero."""
    import meshoptimizer as mo  # noqa: PLC0415

    ints = _rng_positions(16)
    stream = encode_positions(ints)
    raw = mo.decode_vertex_buffer(len(ints), 8, stream.data)
    padded = np.frombuffer(raw.tobytes(), dtype=np.uint8).reshape(
        len(ints), 8
    )
    assert (padded[:, 6:8] == 0).all()


def test_encode_positions_rejects_wrong_dtype() -> None:
    """A non-uint16 POSITION array is refused."""
    wrong = cast("npt.NDArray[np.uint16]", np.zeros((4, 3), dtype=np.uint32))
    with pytest.raises(Tiles3dError):
        encode_positions(wrong)


def test_encode_positions_rejects_wrong_shape() -> None:
    """A POSITION array that is not (n, 3) is refused."""
    with pytest.raises(Tiles3dError):
        encode_positions(np.zeros((4, 2), dtype=np.uint16))


def test_encode_positions_rejects_empty() -> None:
    """An empty POSITION array is refused."""
    with pytest.raises(Tiles3dError):
        encode_positions(np.zeros((0, 3), dtype=np.uint16))


def test_decode_positions_wraps_library_error() -> None:
    """Corrupt POSITION bytes surface as Tiles3dError, not RuntimeError."""
    with pytest.raises(Tiles3dError):
        decode_positions(b"\x00\x01\x02", 10)


# --- uvs ------------------------------------------------------------------


def test_encode_uvs_roundtrip_bit_exact() -> None:
    """A TEXCOORD_0 stream decodes back to its exact uint16 input."""
    ints = _rng_uvs(50)
    stream = encode_uvs(ints)
    assert np.array_equal(decode_uvs(stream, len(ints)), ints)


def test_encode_uvs_stream_metadata() -> None:
    """A TEXCOORD_0 stream carries mode ATTRIBUTES, no filter, stride 4."""
    ints = _rng_uvs(7)
    stream = encode_uvs(ints)
    assert stream.count == 7
    assert stream.byte_stride == UV_BYTE_STRIDE == 4
    assert stream.mode == MODE_ATTRIBUTES
    assert stream.filter is None


def test_encode_uvs_deterministic() -> None:
    """Encoding the same UV input twice is byte-identical."""
    ints = _rng_uvs(30)
    assert encode_uvs(ints).data == encode_uvs(ints).data


def test_encode_uvs_rejects_wrong_dtype() -> None:
    """A non-uint16 UV array is refused."""
    wrong = cast("npt.NDArray[np.uint16]", np.zeros((4, 2), dtype=np.int16))
    with pytest.raises(Tiles3dError):
        encode_uvs(wrong)


def test_encode_uvs_rejects_wrong_shape() -> None:
    """A UV array that is not (n, 2) is refused."""
    with pytest.raises(Tiles3dError):
        encode_uvs(np.zeros((4, 3), dtype=np.uint16))


def test_decode_uvs_wraps_library_error() -> None:
    """Corrupt UV bytes surface as Tiles3dError, not RuntimeError."""
    with pytest.raises(Tiles3dError):
        decode_uvs(b"\x00", 10)


# --- indices --------------------------------------------------------------


@pytest.mark.parametrize(
    ("n_cols", "n_rows"),
    [(2, 2), (3, 2), (2, 3), (5, 4), (17, 17)],
)
def test_encode_indices_roundtrip_grid(n_cols: int, n_rows: int) -> None:
    """The mesh grid index pattern round-trips bit-exactly across sizes."""
    indices = _grid_triangles(n_cols, n_rows)
    count = 6 * (n_cols - 1) * (n_rows - 1)
    stream = encode_indices(indices)
    assert stream.count == count == len(indices)
    assert np.array_equal(decode_indices(stream, count), indices)


def test_encode_indices_stream_metadata() -> None:
    """An index stream carries mode TRIANGLES, no filter, stride 4."""
    indices = _grid_triangles(5, 4)
    stream = encode_indices(indices)
    assert stream.byte_stride == INDEX_BYTE_STRIDE == 4
    assert stream.mode == MODE_TRIANGLES
    assert stream.filter is None


def test_encode_indices_deterministic() -> None:
    """Encoding the same index input twice is byte-identical."""
    indices = _grid_triangles(6, 6)
    assert encode_indices(indices).data == encode_indices(indices).data


def test_encode_indices_rejects_non_canonical() -> None:
    """A rotation-non-canonical triangle is refused loudly.

    The meshopt index codec rotates ``[2, 3, 0]`` to ``[0, 2, 3]``, so
    ``decode(encode) != input``; the encoder must refuse rather than store
    bytes that reproduce a rotated triangle list. This is the guard that
    fires if a future mesh change breaks the grid pattern's canonicality.
    """
    non_canonical = np.array([2, 3, 0], dtype=np.uint32)
    with pytest.raises(Tiles3dError, match="canonical"):
        encode_indices(non_canonical)


def test_encode_indices_rejects_wrong_dtype() -> None:
    """A non-uint32 index array is refused."""
    wrong = cast(
        "npt.NDArray[np.uint32]", np.array([0, 1, 2], dtype=np.uint16)
    )
    with pytest.raises(Tiles3dError):
        encode_indices(wrong)


def test_encode_indices_rejects_non_multiple_of_three() -> None:
    """An index length that is not a multiple of 3 is refused."""
    with pytest.raises(Tiles3dError):
        encode_indices(np.array([0, 1, 2, 3], dtype=np.uint32))


def test_encode_indices_rejects_empty() -> None:
    """An empty index array is refused."""
    with pytest.raises(Tiles3dError):
        encode_indices(np.zeros((0,), dtype=np.uint32))


def test_encode_indices_rejects_non_1d() -> None:
    """A non-1-D index array is refused."""
    with pytest.raises(Tiles3dError):
        encode_indices(np.zeros((2, 3), dtype=np.uint32))


def test_decode_indices_wraps_library_error() -> None:
    """Corrupt index bytes surface as Tiles3dError, not RuntimeError."""
    with pytest.raises(Tiles3dError):
        decode_indices(b"\x00\x01", 6)


# --- decode accepts raw bytes or a stream --------------------------------


def test_decode_accepts_raw_bytes() -> None:
    """Every decoder accepts a MeshoptStream's raw ``.data`` bytes too."""
    pos = _rng_positions(12)
    uvs = _rng_uvs(12)
    idx = _grid_triangles(4, 4)
    assert np.array_equal(
        decode_positions(encode_positions(pos).data, len(pos)), pos
    )
    assert np.array_equal(decode_uvs(encode_uvs(uvs).data, len(uvs)), uvs)
    assert np.array_equal(
        decode_indices(encode_indices(idx).data, len(idx)), idx
    )


# --- provenance -----------------------------------------------------------


def test_meshoptimizer_version_matches_metadata() -> None:
    """The provenance helper returns the installed distribution version."""
    assert meshoptimizer_version() == importlib.metadata.version(
        "meshoptimizer"
    )
