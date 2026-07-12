"""Tests for the ``.hf`` heightfield chunk codec.

The codec is exercised directly (encode/decode round-trip, determinism,
every documented decode reject) plus a size benchmark against the game
profile on one synthetic scene.
"""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING

import numpy as np
import pytest
import zstandard as zstd

from ahn_cli.tiles3d.build import build_tiles3d
from ahn_cli.tiles3d.errors import Tiles3dError
from ahn_cli.tiles3d.geodesy import Geodesy
from ahn_cli.tiles3d.heightfield import (
    HEADER_SIZE,
    VERSION,
    decode_heightfield,
    encode_heightfield,
    zstandard_version,
)
from ahn_cli.tiles3d.mesh import build_tile_mesh
from ahn_cli.tiles3d.payload import TilePayload
from ahn_cli.tiles3d.profile import Profile
from ahn_cli.tiles3d.quadtree import geometric_error, plan_quadtree
from ahn_cli.tiles3d.quantize import quantize_axis
from tests.tiles3d.conftest import (
    grid_for_ortho,
    make_ortho,
    make_terrain,
    synth_rgb,
    write_exr,
)

if TYPE_CHECKING:
    from pathlib import Path

# The header layout, restated from the normative doc so the codec is
# checked against the spec, not against its own source.
_HEADER_FMT = "<4sIII" + "d" * 11 + "Q"
_F_MAGIC, _F_VERSION, _F_WIDTH = 0, 1, 2


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


def test_header_size_is_112_bytes() -> None:
    """The fixed header is exactly the documented 112 bytes."""
    assert HEADER_SIZE == 112
    assert struct.calcsize(_HEADER_FMT) == HEADER_SIZE


def test_round_trip_reproduces_the_quantized_levels() -> None:
    """decode(encode(payload)) reproduces the quantized ints exactly."""
    payload = _payload()
    decoded = decode_heightfield(encode_heightfield(payload))
    height, width = payload.z.shape
    expected = quantize_axis(payload.z.reshape(-1)).ints.reshape(
        height, width
    )
    assert decoded.width == width
    assert decoded.height == height
    assert np.array_equal(decoded.z_ints, expected)


def test_header_carries_the_mesh_anchor_and_region() -> None:
    """The header stores the tile's quantizer, RTC centre and region."""
    payload = _payload()
    decoded = decode_heightfield(encode_heightfield(payload))
    quantized = quantize_axis(payload.z.reshape(-1))
    assert decoded.version == VERSION
    assert decoded.z_offset == quantized.offset
    assert decoded.z_scale == quantized.scale
    assert decoded.rtc_centre == payload.mesh.center
    assert decoded.region == payload.mesh.region


def test_payload_is_row_major_top_row_first() -> None:
    """Level (r, c) sits at flat index r*width + c (top row first)."""
    payload = _payload(width=7, height=4)
    decoded = decode_heightfield(encode_heightfield(payload))
    flat = quantize_axis(payload.z.reshape(-1)).ints
    assert decoded.z_ints[0, 0] == flat[0]
    assert decoded.z_ints[0, 1] == flat[1]
    assert decoded.z_ints[1, 0] == flat[decoded.width]


def test_encode_is_deterministic() -> None:
    """Encoding the same payload twice yields identical bytes."""
    payload = _payload()
    assert encode_heightfield(payload) == encode_heightfield(payload)


def test_flat_tile_stores_all_zero_levels() -> None:
    """A flat height plane quantizes to all zeros with the epsilon scale."""
    payload = _payload()
    flat_payload = TilePayload(
        level=payload.level,
        tx=payload.tx,
        ty=payload.ty,
        stride=payload.stride,
        geometric_error=payload.geometric_error,
        mesh=payload.mesh,
        z=np.full_like(payload.z, 3.0),
        rgb=payload.rgb,
    )
    decoded = decode_heightfield(encode_heightfield(flat_payload))
    assert decoded.z_offset == 3.0
    assert bool((decoded.z_ints == 0).all())


def test_zstandard_version_is_a_string() -> None:
    """The provenance version helper returns the library version."""
    assert zstandard_version() == zstd.__version__


def _corrupt(data: bytes, index: int, value: object) -> bytes:
    """Repack one header struct field with ``value``."""
    fields = list(struct.unpack(_HEADER_FMT, data[:HEADER_SIZE]))
    fields[index] = value
    return struct.pack(_HEADER_FMT, *fields) + data[HEADER_SIZE:]


def test_decode_rejects_a_short_buffer() -> None:
    """A buffer shorter than the header is refused."""
    with pytest.raises(Tiles3dError, match="shorter than"):
        decode_heightfield(encode_heightfield(_payload())[:50])


def test_decode_rejects_bad_magic() -> None:
    """A wrong magic is refused."""
    data = _corrupt(encode_heightfield(_payload()), _F_MAGIC, b"XXXX")
    with pytest.raises(Tiles3dError, match="bad magic"):
        decode_heightfield(data)


def test_decode_rejects_wrong_version() -> None:
    """An unsupported version is refused."""
    data = _corrupt(encode_heightfield(_payload()), _F_VERSION, VERSION + 1)
    with pytest.raises(Tiles3dError, match="not the supported version"):
        decode_heightfield(data)


def test_decode_rejects_truncated_payload() -> None:
    """A payload shorter than the declared length is refused."""
    with pytest.raises(Tiles3dError, match="declares"):
        decode_heightfield(encode_heightfield(_payload())[:-1])


def test_decode_rejects_trailing_bytes() -> None:
    """Any byte after the declared frame is refused."""
    with pytest.raises(Tiles3dError, match="declares"):
        decode_heightfield(encode_heightfield(_payload()) + b"\x00")


def test_decode_rejects_a_corrupt_zstd_frame() -> None:
    """A frame whose zstd magic is broken is refused."""
    data = bytearray(encode_heightfield(_payload()))
    data[HEADER_SIZE] ^= 0xFF  # break the zstd frame magic
    with pytest.raises(Tiles3dError, match="not a valid zstd frame"):
        decode_heightfield(bytes(data))


def test_decode_rejects_a_dims_product_mismatch() -> None:
    """Header dims whose product misses the plane length are refused."""
    payload = _payload(width=6, height=5)
    data = _corrupt(encode_heightfield(payload), _F_WIDTH, 7)  # 7*5 != 6*5
    with pytest.raises(Tiles3dError, match="decompressed to"):
        decode_heightfield(data)


def test_heightfield_geometry_is_far_smaller_than_game(
    tmp_path: Path,
) -> None:
    """Size benchmark: heightfield geometry B/px vs the game glb.

    Non-blocking — a loose sanity ceiling, printed for the record; the
    synthetic random-height scene is the worst case for compression.
    """
    rgb = synth_rgb(32, 32, seed=7)
    ortho = make_ortho(tmp_path / "ortho.tif", rgb)
    heights = write_exr(tmp_path / "r.exr", grid_for_ortho(rgb))
    pixels = 32 * 32

    def total(profile: Profile, suffix: str) -> int:
        out = tmp_path / profile.value
        build_tiles3d(ortho, heights, out, tile_pixels=16, profile=profile)
        return sum(
            p.stat().st_size
            for p in (out / "tiles").iterdir()
            if p.suffix == suffix
        )

    hf_bpp = total(Profile.HEIGHTFIELD, ".hf") / pixels
    glb_bpp = total(Profile.GAME, ".glb") / pixels
    print(  # noqa: T201 -- benchmark record
        f"geometry B/px: heightfield={hf_bpp:.3f} game-glb={glb_bpp:.3f}"
    )
    assert hf_bpp < glb_bpp  # sanity ceiling: far below the bundled glb
    assert hf_bpp < 8.0  # random-noise worst case stays well-bounded
