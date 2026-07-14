"""Tests for the ``.hf`` heightfield chunk codec (format version 3).

The codec is exercised directly (encode/decode round-trip, determinism,
the absolute-error cap, every documented decode reject) plus the normative
algorithm-conformance vectors and a size benchmark against the game profile
on one synthetic scene.
"""

from __future__ import annotations

import hashlib
import struct
import zlib
from typing import TYPE_CHECKING

import numpy as np
import pytest
import zstandard as zstd

from ahn_cli.tiles3d.build import build_tiles3d
from ahn_cli.tiles3d.errors import Tiles3dError
from ahn_cli.tiles3d.geodesy import Geodesy
from ahn_cli.tiles3d.heightfield import (
    HEADER_SIZE,
    MAX_LEVEL,
    VERSION,
    VERTICAL_DATUM,
    ZSTD_LEVEL,
    decode_heightfield,
    encode_heightfield,
    zstandard_version,
)
from ahn_cli.tiles3d.mesh import build_tile_mesh
from ahn_cli.tiles3d.pack import read_pack
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

    import numpy.typing as npt

# The header layout, restated from the normative doc so the codec is
# checked against the spec, not against its own source (v3): the CRC-covered
# prefix (magic, version, width, height, z_offset, z_scale, 3x rtc_centre,
# 6x region, payload_len) plus the vertical_datum field (offset 112, inside
# the CRC span) and header_crc32 (offset 116). The v2 trailing pad is gone.
_PREFIX_FMT = "<4sIII" + "d" * 11 + "Q"
_HEADER_FMT = _PREFIX_FMT + "II"
_CRC_SPAN = struct.calcsize(_PREFIX_FMT) + 4  # prefix + vertical_datum = 116
_F_MAGIC, _F_VERSION, _F_WIDTH, _F_HEIGHT = 0, 1, 2, 3
_F_Z_SCALE, _F_REGION_N = 5, 12
_F_PAYLOAD_LEN, _F_VERTICAL_DATUM, _F_CRC = 15, 16, 17

# The normative algorithm-conformance golden string (permanent, codec-agnostic).
_GOLDEN_STRING = b"AHN heightfield spike golden vector 0123456789"


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


def _replace_z(
    payload: TilePayload, z: npt.NDArray[np.float32]
) -> TilePayload:
    """Return a copy of ``payload`` carrying a new height plane."""
    return TilePayload(
        level=payload.level,
        tx=payload.tx,
        ty=payload.ty,
        stride=payload.stride,
        geometric_error=payload.geometric_error,
        mesh=payload.mesh,
        z=z,
        rgb=payload.rgb,
    )


def test_header_size_is_120_bytes() -> None:
    """The fixed header is exactly the documented 120 bytes."""
    assert HEADER_SIZE == 120
    assert struct.calcsize(_HEADER_FMT) == HEADER_SIZE


def test_version_is_three() -> None:
    """The codec reads and writes format version 3 (NAP-native)."""
    assert VERSION == 3


def test_round_trip_reproduces_the_quantized_levels() -> None:
    """decode(encode(payload)) reproduces the 12-bit quantized ints."""
    payload = _payload()
    decoded = decode_heightfield(encode_heightfield(payload))
    height, width = payload.z.shape
    expected = quantize_axis(payload.z.reshape(-1), MAX_LEVEL).ints.reshape(
        height, width
    )
    assert decoded.width == width
    assert decoded.height == height
    assert np.array_equal(decoded.z_ints, expected)
    assert int(decoded.z_ints.max()) <= MAX_LEVEL


def test_header_carries_the_mesh_anchor_and_region() -> None:
    """The header stores the tile's quantizer, RTC centre and region."""
    payload = _payload()
    decoded = decode_heightfield(encode_heightfield(payload))
    quantized = quantize_axis(payload.z.reshape(-1), MAX_LEVEL)
    assert decoded.version == VERSION
    assert decoded.z_offset == quantized.offset
    assert decoded.z_scale == quantized.scale
    assert decoded.rtc_centre == payload.mesh.center
    assert decoded.region == payload.mesh.region


def test_header_crc32_covers_the_first_116_bytes() -> None:
    """header_crc32 is the CRC-32 of bytes [0, 116), covering vertical_datum."""
    data = encode_heightfield(_payload())
    fields = struct.unpack(_HEADER_FMT, data[:HEADER_SIZE])
    assert _CRC_SPAN == 116
    assert fields[_F_VERTICAL_DATUM] == VERTICAL_DATUM
    assert fields[_F_CRC] == zlib.crc32(data[:_CRC_SPAN]) & 0xFFFFFFFF


def test_payload_is_row_major_top_row_first() -> None:
    """Level (r, c) sits at flat index r*width + c (top row first)."""
    payload = _payload(width=7, height=4)
    decoded = decode_heightfield(encode_heightfield(payload))
    flat = quantize_axis(payload.z.reshape(-1), MAX_LEVEL).ints
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
    flat_payload = _replace_z(payload, np.full_like(payload.z, 3.0))
    decoded = decode_heightfield(encode_heightfield(flat_payload))
    assert decoded.z_offset == 3.0
    assert bool((decoded.z_ints == 0).all())


def test_encode_refuses_a_tile_beyond_the_error_cap() -> None:
    """A height extent over the 12-bit cap is refused, naming the extent."""
    payload = _payload()
    # A 300 m extent gives z_scale/2 ~= 0.037 m > the 0.025 m cap.
    tall = np.zeros_like(payload.z)
    tall.reshape(-1)[0] = 300.0
    with pytest.raises(Tiles3dError, match="exceeds the 0.025 m error cap"):
        encode_heightfield(_replace_z(payload, tall))


def test_zstandard_version_is_a_string() -> None:
    """The provenance version helper returns the library version."""
    assert zstandard_version() == zstd.__version__


def test_crc32_conformance_vector() -> None:
    """CRC-32/ISO-HDLC of the normative golden string is the pinned value."""
    assert zlib.crc32(_GOLDEN_STRING) & 0xFFFFFFFF == 0xB4B41F5A


def test_sha256_conformance_vector() -> None:
    """SHA-256 of the normative golden string is the pinned value."""
    assert hashlib.sha256(_GOLDEN_STRING).hexdigest() == (
        "6a8998f8fab0139aaff77ccb9ab123907d58bc55d8b7244e2394f41418731b54"
    )


def _corrupt(
    data: bytes, index: int, value: object, *, fix_crc: bool = True
) -> bytes:
    """Repack one header struct field with ``value`` (v3 layout).

    With ``fix_crc`` (the default) the header_crc32 is recomputed over the
    patched ``[0, 116)`` span (prefix + ``vertical_datum``) so it stays
    valid — isolating a downstream check. With ``fix_crc=False`` the stored
    crc is left as-is (to exercise the crc check itself).
    """
    fields = list(struct.unpack(_HEADER_FMT, data[:HEADER_SIZE]))
    fields[index] = value
    body = struct.pack(_PREFIX_FMT, *fields[:16]) + struct.pack(
        "<I", fields[_F_VERTICAL_DATUM]
    )
    crc = zlib.crc32(body) & 0xFFFFFFFF if fix_crc else fields[_F_CRC]
    return body + struct.pack("<I", crc) + data[HEADER_SIZE:]


def _repack_levels(data: bytes, top: int) -> bytes:
    """Set the max level to ``top``, recompress, fix payload_len and crc."""
    decoded = decode_heightfield(data)
    ints = decoded.z_ints.copy()
    ints[0, 0] = top
    frame = zstd.ZstdCompressor(
        level=ZSTD_LEVEL,
        write_content_size=True,
        write_checksum=True,
        threads=0,
    ).compress(ints.astype("<u2").tobytes())
    fields = list(struct.unpack(_HEADER_FMT, data[:HEADER_SIZE]))
    fields[_F_PAYLOAD_LEN] = len(frame)
    body = struct.pack(_PREFIX_FMT, *fields[:16]) + struct.pack(
        "<I", fields[_F_VERTICAL_DATUM]
    )
    crc = zlib.crc32(body) & 0xFFFFFFFF
    return body + struct.pack("<I", crc) + frame


def test_decode_rejects_a_short_buffer() -> None:
    """A buffer shorter than the header is refused."""
    with pytest.raises(Tiles3dError, match="shorter than"):
        decode_heightfield(encode_heightfield(_payload())[:50])


def test_decode_rejects_bad_magic() -> None:
    """A wrong magic is refused (before the crc check)."""
    data = _corrupt(encode_heightfield(_payload()), _F_MAGIC, b"XXXX")
    with pytest.raises(Tiles3dError, match="bad magic"):
        decode_heightfield(data)


def test_decode_rejects_wrong_version() -> None:
    """An unsupported version is refused (before the crc check)."""
    data = _corrupt(encode_heightfield(_payload()), _F_VERSION, VERSION + 1)
    with pytest.raises(Tiles3dError, match="not the supported version"):
        decode_heightfield(data)


def test_decode_rejects_a_header_crc_mismatch() -> None:
    """A stale header_crc32 is refused before any dimension is trusted."""
    data = _corrupt(
        encode_heightfield(_payload()), _F_WIDTH, 999, fix_crc=False
    )
    with pytest.raises(Tiles3dError, match="CRC32"):
        decode_heightfield(data)


def test_decode_rejects_a_wrong_vertical_datum() -> None:
    """A vertical_datum other than EPSG:5709 is refused (crc re-signed).

    The datum tag is inside the CRC span, so ``_corrupt`` recomputes a valid
    crc over ``[0, 116)``; the failure therefore attributes to the datum
    reject, not the crc guard (which runs first and now passes).
    """
    data = _corrupt(encode_heightfield(_payload()), _F_VERTICAL_DATUM, 4979)
    with pytest.raises(Tiles3dError, match="vertical_datum 4979 is not"):
        decode_heightfield(data)


def test_decode_rejects_zero_width() -> None:
    """A zero width is refused (crc re-patched so only this fires)."""
    data = _corrupt(encode_heightfield(_payload()), _F_WIDTH, 0)
    with pytest.raises(Tiles3dError, match="zero dimension"):
        decode_heightfield(data)


def test_decode_rejects_zero_height() -> None:
    """A zero height is refused (crc re-patched so only this fires)."""
    data = _corrupt(encode_heightfield(_payload()), _F_HEIGHT, 0)
    with pytest.raises(Tiles3dError, match="zero dimension"):
        decode_heightfield(data)


def test_decode_rejects_a_non_finite_header_float() -> None:
    """A non-finite header float is refused (crc re-patched)."""
    data = _corrupt(encode_heightfield(_payload()), _F_REGION_N, float("inf"))
    with pytest.raises(Tiles3dError, match="region\\[3\\] is non-finite"):
        decode_heightfield(data)


def test_decode_rejects_a_non_positive_z_scale() -> None:
    """A finite but non-positive z_scale is refused (crc re-patched)."""
    data = _corrupt(encode_heightfield(_payload()), _F_Z_SCALE, -1.0)
    with pytest.raises(Tiles3dError, match="z_scale -1.0 must be positive"):
        decode_heightfield(data)


def test_decode_rejects_truncated_payload() -> None:
    """A payload shorter than the declared length is refused."""
    with pytest.raises(Tiles3dError, match="declares"):
        decode_heightfield(encode_heightfield(_payload())[:-1])


def test_decode_rejects_trailing_bytes() -> None:
    """Any byte after the declared frame is refused."""
    with pytest.raises(Tiles3dError, match="declares"):
        decode_heightfield(encode_heightfield(_payload()) + b"\x00")


def test_decode_rejects_a_broken_zstd_frame() -> None:
    """A frame whose zstd magic is broken is refused."""
    data = bytearray(encode_heightfield(_payload()))
    data[HEADER_SIZE] ^= 0xFF  # break the zstd frame magic
    with pytest.raises(Tiles3dError, match="not a valid zstd frame"):
        decode_heightfield(bytes(data))


def test_decode_rejects_a_payload_checksum_mismatch() -> None:
    """A bit flipped in the payload body fails the content checksum."""
    data = bytearray(encode_heightfield(_payload()))
    data[-8] ^= 0x01  # a byte inside the compressed body, not the frame magic
    with pytest.raises(Tiles3dError, match="not a valid zstd frame"):
        decode_heightfield(bytes(data))


def test_decode_rejects_a_dims_product_mismatch() -> None:
    """Header dims whose product misses the plane length are refused."""
    payload = _payload(width=6, height=5)
    data = _corrupt(encode_heightfield(payload), _F_WIDTH, 7)  # 7*5 != 6*5
    with pytest.raises(Tiles3dError, match="decompressed to"):
        decode_heightfield(data)


def test_decode_rejects_a_level_above_the_maximum() -> None:
    """A stored level outside the 12-bit range is refused."""
    data = _repack_levels(encode_heightfield(_payload()), MAX_LEVEL + 1)
    with pytest.raises(Tiles3dError, match="exceeds the 12-bit maximum"):
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

    def total(profile: Profile) -> int:
        out = tmp_path / profile.value
        build_tiles3d(ortho, heights, out, tile_pixels=16, profile=profile)
        # The per-tile geometry lives in the pack's primary blobs (.hf chunk
        # for heightfield, .glb for game); the texture is a separate concern.
        pack = read_pack(out / "tiles.hfp")
        return sum(entry.primary_size for entry in pack.entries)

    hf_bpp = total(Profile.HEIGHTFIELD) / pixels
    glb_bpp = total(Profile.GAME) / pixels
    print(  # noqa: T201 -- benchmark record
        f"geometry B/px: heightfield={hf_bpp:.3f} game-glb={glb_bpp:.3f}"
    )
    assert hf_bpp < glb_bpp  # sanity ceiling: far below the bundled glb
    assert hf_bpp < 8.0  # random-noise worst case stays well-bounded
