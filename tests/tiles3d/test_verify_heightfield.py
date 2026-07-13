"""Tests for the heightfield-profile per-tile verification.

Each negative test builds a valid heightfield tileset, corrupts exactly
the bytes one check guards through the real build-then-corrupt-then-verify
path, and asserts that check's message — proving each check fires
independently of the whole-file byte-identity backstop.
"""

from __future__ import annotations

import io
import struct
import zlib
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
import zstandard as zstd
from PIL import Image

import ahn_cli.tiles3d.build as build_module
import ahn_cli.tiles3d.verify_heightfield as verify_hf_module
from ahn_cli.tiles3d.build import build_tiles3d
from ahn_cli.tiles3d.errors import Tiles3dError
from ahn_cli.tiles3d.heightfield import (
    HEADER_SIZE,
    ZSTD_LEVEL,
    decode_heightfield,
)
from ahn_cli.tiles3d.pack import TileKey
from ahn_cli.tiles3d.profile import Profile
from ahn_cli.tiles3d.verify import verify_tiles3d
from tests.tiles3d.conftest import (
    grid_for_ortho,
    make_ortho,
    pack_blob,
    repack_one,
    synth_rgb,
    write_exr,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import numpy.typing as npt

_PREFIX_FMT = "<4sIII" + "d" * 11 + "Q"
_HEADER_FMT = _PREFIX_FMT + "II"
_F_Z_SCALE, _F_RTC_X, _F_REGION_W, _F_PAYLOAD_LEN, _F_PAD = 5, 6, 9, 15, 17

_LEAF_KEY = TileKey(2, 0, 0)
"""The leaf tile whose blobs the corruption tests target inside the pack."""


@pytest.fixture
def hf_site(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Build a valid two-level heightfield tileset; (out, ortho, heights)."""
    rgb = synth_rgb(20, 14, seed=13)
    ortho = make_ortho(tmp_path / "ortho.tif", rgb)
    heights = write_exr(tmp_path / "r.exr", grid_for_ortho(rgb))
    out = tmp_path / "out"
    build_tiles3d(
        ortho, heights, out, tile_pixels=8, profile=Profile.HEIGHTFIELD
    )
    return out, ortho, heights


def _verify(site: tuple[Path, Path, Path]) -> None:
    out, ortho, heights = site
    verify_tiles3d(
        out, ortho, heights, tile_pixels=8, profile=Profile.HEIGHTFIELD
    )


def _chunk_bytes(site: tuple[Path, Path, Path]) -> bytes:
    """Return the leaf tile's pristine ``.hf`` bytes from the pack."""
    return pack_blob(site[0] / "tiles.hfp", _LEAF_KEY)[0]


def _texture_bytes(site: tuple[Path, Path, Path]) -> bytes:
    """Return the leaf tile's pristine ``.jpg`` texture bytes from the pack."""
    return cast("bytes", pack_blob(site[0] / "tiles.hfp", _LEAF_KEY)[1])


def _repack_chunk(site: tuple[Path, Path, Path], chunk: bytes) -> None:
    """Repack ``tiles.hfp`` with the leaf tile's ``.hf`` replaced."""
    repack_one(
        site[0] / "tiles.hfp",
        _LEAF_KEY,
        lambda _primary, texture: (chunk, texture),
    )


def _repack_texture(site: tuple[Path, Path, Path], texture: bytes) -> None:
    """Repack ``tiles.hfp`` with the leaf tile's ``.jpg`` texture replaced."""
    repack_one(
        site[0] / "tiles.hfp",
        _LEAF_KEY,
        lambda primary, _texture: (primary, texture),
    )


def _rebuild(fields: list[object], frame: bytes) -> bytes:
    """Repack a header from ``fields`` with a valid crc, then the frame."""
    prefix = struct.pack(_PREFIX_FMT, *fields[:16])
    crc = zlib.crc32(prefix) & 0xFFFFFFFF
    return prefix + struct.pack("<II", crc, fields[_F_PAD]) + frame


def _patch_header(chunk: bytes, index: int, value: object) -> bytes:
    """Return ``chunk`` with one header field overwritten, CRC re-signed.

    The CRC is recomputed so a downstream verifier check (not the decoder's
    CRC guard) attributes the failure.
    """
    fields = list(struct.unpack(_HEADER_FMT, chunk[:HEADER_SIZE]))
    fields[index] = value
    return _rebuild(fields, chunk[HEADER_SIZE:])


def _repack_levels(
    chunk: bytes, mutate: Callable[[npt.NDArray[np.uint16]], None]
) -> bytes:
    """Decode, mutate the levels, recompress, fix the length and crc."""
    fields = list(struct.unpack(_HEADER_FMT, chunk[:HEADER_SIZE]))
    ints = decode_heightfield(chunk).z_ints.copy()
    mutate(ints)
    frame = zstd.ZstdCompressor(
        level=ZSTD_LEVEL,
        write_content_size=True,
        write_checksum=True,
        threads=0,
    ).compress(ints.astype("<u2").tobytes())
    fields[_F_PAYLOAD_LEN] = len(frame)
    return _rebuild(fields, frame)


def test_pristine_heightfield_build_verifies(
    hf_site: tuple[Path, Path, Path],
) -> None:
    """The verifier accepts what the heightfield builder just wrote."""
    _verify(hf_site)


def test_wrong_z_scale_is_refused(
    hf_site: tuple[Path, Path, Path],
) -> None:
    """A z_scale that is not the recomputed quantizer scale is caught."""
    chunk = _chunk_bytes(hf_site)
    decoded = decode_heightfield(chunk)
    _repack_chunk(
        hf_site, _patch_header(chunk, _F_Z_SCALE, decoded.z_scale * 2.0)
    )
    with pytest.raises(Tiles3dError, match="z_offset/z_scale"):
        _verify(hf_site)


def test_wrong_rtc_centre_is_refused(
    hf_site: tuple[Path, Path, Path],
) -> None:
    """An rtc_centre that is not the tile's RTC anchor is caught."""
    chunk = _chunk_bytes(hf_site)
    decoded = decode_heightfield(chunk)
    _repack_chunk(
        hf_site, _patch_header(chunk, _F_RTC_X, decoded.rtc_centre[0] + 1.0)
    )
    with pytest.raises(Tiles3dError, match="rtc_centre"):
        _verify(hf_site)


def test_wrong_region_is_refused(
    hf_site: tuple[Path, Path, Path],
) -> None:
    """A region that is not the tile's EPSG:4979 region is caught."""
    chunk = _chunk_bytes(hf_site)
    decoded = decode_heightfield(chunk)
    _repack_chunk(
        hf_site, _patch_header(chunk, _F_REGION_W, decoded.region[0] - 0.001)
    )
    with pytest.raises(Tiles3dError, match="region does not equal"):
        _verify(hf_site)


def test_off_by_one_level_fires_requantization(
    hf_site: tuple[Path, Path, Path],
) -> None:
    """An off-by-one level fires requantization, not the backstop.

    The frame is fully re-packed (valid zstd, declared length fixed) and
    the header dims/quantizer are left untouched, so only the
    requantization check attributes the failure.
    """

    def bump(ints: npt.NDArray[np.uint16]) -> None:
        # Stay inside the 12-bit range so requantization, not the level-cap
        # reject, attributes the failure.
        ints[0, 0] = 1 if int(ints[0, 0]) == 0 else 0

    _repack_chunk(hf_site, _repack_levels(_chunk_bytes(hf_site), bump))
    with pytest.raises(Tiles3dError, match="requantization of the source"):
        _verify(hf_site)


def test_corrupt_zstd_frame_is_refused(
    hf_site: tuple[Path, Path, Path],
) -> None:
    """A flipped byte in the zstd frame is caught at decode."""
    data = bytearray(_chunk_bytes(hf_site))
    data[HEADER_SIZE] ^= 0xFF  # break the zstd frame magic
    _repack_chunk(hf_site, bytes(data))
    with pytest.raises(Tiles3dError, match="not a valid zstd frame"):
        _verify(hf_site)


def test_dequant_bound_is_enforced(
    hf_site: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A negative error bound rejects the genuine dequantized heights.

    The dequant-bound assert guards the exported ``axis_error_bound``
    contract; forcing the bound below zero proves the verifier would
    reject any height that ever exceeded it.
    """

    def negative_bound(_scale: float) -> float:
        return -1.0

    monkeypatch.setattr(verify_hf_module, "axis_error_bound", negative_bound)
    with pytest.raises(
        Tiles3dError, match="exceeds the documented quantization error bound"
    ):
        _verify(hf_site)


def test_error_cap_is_enforced(
    hf_site: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bound above the absolute cap is refused.

    The genuine tile's bound is well under 0.025 m, so forcing the cap below
    it proves the verifier rejects any tile whose ``z_scale / 2`` ever
    exceeds :data:`~ahn_cli.tiles3d.heightfield.MAX_AXIS_ERROR_M`.
    """
    monkeypatch.setattr(verify_hf_module, "MAX_AXIS_ERROR_M", 1e-12)
    with pytest.raises(Tiles3dError, match="exceeds the .* m absolute cap"):
        _verify(hf_site)


def test_progressive_jpeg_texture_is_refused(
    hf_site: tuple[Path, Path, Path],
) -> None:
    """A progressive-framed sibling texture fails the baseline check."""
    with Image.open(io.BytesIO(_texture_bytes(hf_site))) as image:
        rgb = np.array(image.convert("RGB"))
    buffer = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(
        buffer, format="JPEG", progressive=True
    )
    _repack_texture(hf_site, buffer.getvalue())
    with pytest.raises(Tiles3dError, match="baseline sequential JPEG"):
        _verify(hf_site)


def test_jpeg_recompressed_at_a_different_quality_is_refused(
    hf_site: tuple[Path, Path, Path],
) -> None:
    """A texture re-encoded at another quality fails JPEG byte-equality."""
    with Image.open(io.BytesIO(_texture_bytes(hf_site))) as image:
        rgb = np.array(image.convert("RGB"))
    buffer = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buffer, format="JPEG", quality=60)
    _repack_texture(hf_site, buffer.getvalue())
    with pytest.raises(Tiles3dError, match="does not byte-equal"):
        _verify(hf_site)


def _perturbing_encode_jpeg(
    real: Callable[[npt.NDArray[np.uint8]], bytes],
) -> Callable[[npt.NDArray[np.uint8]], bytes]:
    def encode(rgb: npt.NDArray[np.uint8]) -> bytes:
        return real((rgb.astype(np.int32) ^ 3).astype(np.uint8))

    return encode


def test_rejected_heightfield_rebuild_restores_the_previous_deliverable(
    hf_site: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rebuild the verifier rejects restores the old build whole.

    Only the verifier's JPEG re-encode is perturbed, so the build writes a
    correct heightfield tile that its own verification then refuses — the
    swap machinery must remove the rejected write and restore the
    held-aside previous deliverable, provenance.json and every ``.hf`` /
    ``.jpg`` included.
    """
    out, ortho, heights = hf_site
    good = {
        p.relative_to(out): p.read_bytes()
        for p in out.rglob("*")
        if p.is_file()
    }
    monkeypatch.setattr(
        verify_hf_module,
        "encode_jpeg",
        _perturbing_encode_jpeg(verify_hf_module.encode_jpeg),
    )
    with pytest.raises(Tiles3dError, match="does not byte-equal"):
        build_tiles3d(
            ortho, heights, out, tile_pixels=8, profile=Profile.HEIGHTFIELD
        )
    after = {
        p.relative_to(out): p.read_bytes()
        for p in out.rglob("*")
        if p.is_file()
    }
    assert after == good
    assert (out / "provenance.json").is_file()
    assert build_module.BACKUP_SUBDIR not in {p.name for p in out.iterdir()}
