"""Tests for the ``splat`` profile's binary 3DGS ``.ply`` + zstd codec.

Mirrors ``test_heightfield.py``'s rigor: encode/decode round-trip,
determinism, the per-field construction (position, SH deg-0 colour,
scale, rotation, opacity) and every documented decode reject.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import zstandard as zstd

from ahn_cli.tiles3d.errors import Tiles3dError
from ahn_cli.tiles3d.geodesy import Geodesy
from ahn_cli.tiles3d.mesh import TileMesh, build_tile_mesh
from ahn_cli.tiles3d.payload import TilePayload
from ahn_cli.tiles3d.quadtree import geometric_error, plan_quadtree
from ahn_cli.tiles3d.splat import (
    OPACITY,
    SH_DC0,
    ZSTD_LEVEL,
    decode_splat,
    encode_splat,
    zstandard_version,
)
from tests.tiles3d.conftest import make_terrain


def _payload(width: int = 6, height: int = 5, seed: int = 4) -> TilePayload:
    terrain = make_terrain(width, height, seed)
    tile = plan_quadtree(width, height).root
    mesh = build_tile_mesh(terrain, tile, Geodesy())
    grid = np.ix_(mesh.rows, mesh.cols)
    return TilePayload(
        level=tile.level,
        tx=tile.tx,
        ty=tile.ty,
        stride=tile.stride,
        geometric_error=geometric_error(tile.stride, 0.5),
        mesh=mesh,
        z=terrain.z[grid],
        rgb=terrain.rgb[grid],
    )


def _expected_f_dc(payload: TilePayload) -> np.ndarray:
    rgb = payload.rgb.reshape(-1, 3).astype(np.float64) / 255.0
    return ((rgb - 0.5) / SH_DC0).astype(np.float32)


def test_round_trip_reproduces_positions_and_count() -> None:
    """decode(encode(payload)) reproduces the tile's exact RTC positions."""
    payload = _payload()
    decoded = decode_splat(encode_splat(payload))
    assert decoded.count == payload.mesh.positions.shape[0]
    assert np.array_equal(decoded.positions, payload.mesh.positions)


def test_colour_is_sh_degree_zero_of_the_sampled_ortho() -> None:
    """``f_dc`` is ``(c/255 - 0.5) / SH_DC0`` per sampled ortho pixel."""
    payload = _payload()
    decoded = decode_splat(encode_splat(payload))
    assert np.array_equal(decoded.f_dc, _expected_f_dc(payload))


def test_opacity_is_the_fixed_logit_of_0_99() -> None:
    """Every gaussian stores ``logit(OPACITY)`` as its pre-activation opacity."""
    payload = _payload()
    decoded = decode_splat(encode_splat(payload))
    assert OPACITY == 0.99
    expected = np.float32(math.log(OPACITY / (1.0 - OPACITY)))
    assert bool((decoded.opacity == expected).all())


def test_scale_is_isotropic_and_positive() -> None:
    """Every axis of every gaussian's scale is the same finite log value."""
    payload = _payload()
    decoded = decode_splat(encode_splat(payload))
    assert decoded.scale.shape == (decoded.count, 3)
    first = decoded.scale[0, 0]
    assert bool(np.isfinite(decoded.scale).all())
    assert bool((decoded.scale == first).all())
    # exp(scale) is the actual metric spacing, which must be positive.
    assert math.exp(float(first)) > 0.0


def test_rotation_is_the_identity_quaternion() -> None:
    """Every gaussian's ``rot`` is the wxyz identity quaternion (1, 0, 0, 0)."""
    payload = _payload()
    decoded = decode_splat(encode_splat(payload))
    expected = np.zeros((decoded.count, 4), dtype=np.float32)
    expected[:, 0] = 1.0
    assert np.array_equal(decoded.rot, expected)


def test_encode_is_deterministic() -> None:
    """Encoding the same payload twice yields byte-identical output."""
    payload = _payload()
    assert encode_splat(payload) == encode_splat(payload)


def test_zstandard_version_is_a_string() -> None:
    """The provenance version helper returns the library version."""
    assert zstandard_version() == zstd.__version__


def test_zstd_level_is_pinned_to_three() -> None:
    """The module pins the same zstd level as the heightfield chunk codec."""
    assert ZSTD_LEVEL == 3


def test_decode_rejects_a_broken_zstd_frame() -> None:
    """A frame whose zstd magic is broken is refused."""
    data = bytearray(encode_splat(_payload()))
    data[0] ^= 0xFF
    with pytest.raises(Tiles3dError, match="not a valid zstd frame"):
        decode_splat(bytes(data))


def _decompressed(payload: TilePayload) -> bytes:
    return zstd.ZstdDecompressor().decompress(encode_splat(payload))


def _recompress(raw: bytes) -> bytes:
    return zstd.ZstdCompressor(
        level=ZSTD_LEVEL,
        write_content_size=True,
        write_checksum=True,
        threads=0,
    ).compress(raw)


def test_decode_rejects_a_missing_end_header() -> None:
    """A ply lacking the ``end_header`` terminator is refused."""
    raw = _decompressed(_payload()).replace(b"end_header\n", b"bogus\n")
    with pytest.raises(Tiles3dError, match="end_header"):
        decode_splat(_recompress(raw))


def test_decode_rejects_a_bad_magic_line() -> None:
    """A first header line other than ``ply`` is refused."""
    raw = _decompressed(_payload())
    raw = raw.replace(b"ply\n", b"nope\n", 1)
    with pytest.raises(Tiles3dError, match="magic"):
        decode_splat(_recompress(raw))


def test_decode_rejects_a_bad_format_line() -> None:
    """A format line other than binary_little_endian 1.0 is refused."""
    raw = _decompressed(_payload())
    raw = raw.replace(
        b"format binary_little_endian 1.0\n", b"format ascii 1.0\n"
    )
    with pytest.raises(Tiles3dError, match="format"):
        decode_splat(_recompress(raw))


def test_decode_rejects_a_non_numeric_vertex_count() -> None:
    """A non-decimal ``element vertex`` count is refused."""
    raw = _decompressed(_payload())
    end = raw.find(b"\n")
    end = raw.find(b"\n", end + 1)
    line_start = raw.find(b"element vertex ")
    line_end = raw.find(b"\n", line_start)
    raw = raw[:line_start] + b"element vertex x" + raw[line_end:]
    with pytest.raises(Tiles3dError, match="decimal integer"):
        decode_splat(_recompress(raw))


def test_decode_rejects_a_wrong_property_line() -> None:
    """A property line that doesn't match the pinned field order is refused."""
    raw = _decompressed(_payload())
    raw = raw.replace(b"property float x\n", b"property float w\n", 1)
    with pytest.raises(Tiles3dError, match="property"):
        decode_splat(_recompress(raw))


def test_decode_rejects_a_header_with_the_wrong_line_count() -> None:
    """A header carrying an extra line (still ending in end_header) fails."""
    raw = _decompressed(_payload())
    raw = raw.replace(b"ply\n", b"ply\nextra\n", 1)
    with pytest.raises(Tiles3dError, match="lines, expected"):
        decode_splat(_recompress(raw))


def test_decode_rejects_a_bad_element_line_prefix() -> None:
    """An ``element vertex`` line not starting with the pinned prefix fails."""
    raw = _decompressed(_payload())
    raw = raw.replace(b"element vertex ", b"elem_vertex ", 1)
    with pytest.raises(Tiles3dError, match="element line"):
        decode_splat(_recompress(raw))


def test_decode_rejects_a_body_length_mismatch() -> None:
    """A body whose length doesn't match ``count * bytes-per-vertex`` fails."""
    raw = _decompressed(_payload())
    with pytest.raises(Tiles3dError, match="body is"):
        decode_splat(_recompress(raw + b"\x00\x00\x00\x00"))


def test_cell_spacing_error_is_typed_for_a_degenerate_grid() -> None:
    """A payload whose grid can't yield a positive spacing raises typed."""
    payload = _payload()
    # Corrupt the mesh so the first two RTC positions coincide.
    positions = payload.mesh.positions.copy()
    positions[1] = positions[0]
    bad_mesh = TileMesh(
        positions=positions,
        uvs=payload.mesh.uvs,
        indices=payload.mesh.indices,
        center=payload.mesh.center,
        region=payload.mesh.region,
        cols=payload.mesh.cols,
        rows=payload.mesh.rows,
    )
    bad_payload = TilePayload(
        level=payload.level,
        tx=payload.tx,
        ty=payload.ty,
        stride=payload.stride,
        geometric_error=payload.geometric_error,
        mesh=bad_mesh,
        z=payload.z,
        rgb=payload.rgb,
    )
    with pytest.raises(Tiles3dError, match="cell spacing"):
        encode_splat(bad_payload)
