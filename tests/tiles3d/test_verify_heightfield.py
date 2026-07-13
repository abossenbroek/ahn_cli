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
from typing import TYPE_CHECKING

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
from ahn_cli.tiles3d.profile import Profile
from ahn_cli.tiles3d.verify import verify_tiles3d
from tests.tiles3d.conftest import (
    grid_for_ortho,
    make_ortho,
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


def _chunk(site: tuple[Path, Path, Path]) -> Path:
    return site[0] / "tiles" / "2-0-0.hf"


def _texture(site: tuple[Path, Path, Path]) -> Path:
    return site[0] / "tiles" / "2-0-0.jpg"


def _rebuild(fields: list[object], frame: bytes) -> bytes:
    """Repack a header from ``fields`` with a valid crc, then the frame."""
    prefix = struct.pack(_PREFIX_FMT, *fields[:16])
    crc = zlib.crc32(prefix) & 0xFFFFFFFF
    return prefix + struct.pack("<II", crc, fields[_F_PAD]) + frame


def _patch_header(path: Path, index: int, value: object) -> None:
    """Overwrite one header field in place, re-signing the header CRC.

    The CRC is recomputed so a downstream verifier check (not the decoder's
    CRC guard) attributes the failure.
    """
    data = bytearray(path.read_bytes())
    fields = list(struct.unpack(_HEADER_FMT, bytes(data[:HEADER_SIZE])))
    fields[index] = value
    path.write_bytes(_rebuild(fields, bytes(data[HEADER_SIZE:])))


def _repack_levels(
    path: Path, mutate: Callable[[npt.NDArray[np.uint16]], None]
) -> None:
    """Decode, mutate the levels, recompress, fix the length and crc."""
    data = bytearray(path.read_bytes())
    fields = list(struct.unpack(_HEADER_FMT, bytes(data[:HEADER_SIZE])))
    ints = decode_heightfield(bytes(data)).z_ints.copy()
    mutate(ints)
    frame = zstd.ZstdCompressor(
        level=ZSTD_LEVEL,
        write_content_size=True,
        write_checksum=True,
        threads=0,
    ).compress(ints.astype("<u2").tobytes())
    fields[_F_PAYLOAD_LEN] = len(frame)
    path.write_bytes(_rebuild(fields, frame))


def test_pristine_heightfield_build_verifies(
    hf_site: tuple[Path, Path, Path],
) -> None:
    """The verifier accepts what the heightfield builder just wrote."""
    _verify(hf_site)


def test_wrong_z_scale_is_refused(
    hf_site: tuple[Path, Path, Path],
) -> None:
    """A z_scale that is not the recomputed quantizer scale is caught."""
    decoded = decode_heightfield(_chunk(hf_site).read_bytes())
    _patch_header(_chunk(hf_site), _F_Z_SCALE, decoded.z_scale * 2.0)
    with pytest.raises(Tiles3dError, match="z_offset/z_scale"):
        _verify(hf_site)


def test_wrong_rtc_centre_is_refused(
    hf_site: tuple[Path, Path, Path],
) -> None:
    """An rtc_centre that is not the tile's RTC anchor is caught."""
    decoded = decode_heightfield(_chunk(hf_site).read_bytes())
    _patch_header(_chunk(hf_site), _F_RTC_X, decoded.rtc_centre[0] + 1.0)
    with pytest.raises(Tiles3dError, match="rtc_centre"):
        _verify(hf_site)


def test_wrong_region_is_refused(
    hf_site: tuple[Path, Path, Path],
) -> None:
    """A region that is not the tile's EPSG:4979 region is caught."""
    decoded = decode_heightfield(_chunk(hf_site).read_bytes())
    _patch_header(_chunk(hf_site), _F_REGION_W, decoded.region[0] - 0.001)
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

    _repack_levels(_chunk(hf_site), bump)
    with pytest.raises(Tiles3dError, match="requantization of the source"):
        _verify(hf_site)


def test_corrupt_zstd_frame_is_refused(
    hf_site: tuple[Path, Path, Path],
) -> None:
    """A flipped byte in the zstd frame is caught at decode."""
    path = _chunk(hf_site)
    data = bytearray(path.read_bytes())
    data[HEADER_SIZE] ^= 0xFF  # break the zstd frame magic
    path.write_bytes(bytes(data))
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


def test_missing_texture_is_refused(
    hf_site: tuple[Path, Path, Path],
) -> None:
    """A deleted sibling JPEG is caught by the link check."""
    _texture(hf_site).unlink()
    with pytest.raises(Tiles3dError, match="missing texture file"):
        _verify(hf_site)


def test_progressive_jpeg_texture_is_refused(
    hf_site: tuple[Path, Path, Path],
) -> None:
    """A progressive-framed sibling texture fails the baseline check."""
    path = _texture(hf_site)
    with Image.open(io.BytesIO(path.read_bytes())) as image:
        rgb = np.array(image.convert("RGB"))
    buffer = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(
        buffer, format="JPEG", progressive=True
    )
    path.write_bytes(buffer.getvalue())
    with pytest.raises(Tiles3dError, match="baseline sequential JPEG"):
        _verify(hf_site)


def test_jpeg_recompressed_at_a_different_quality_is_refused(
    hf_site: tuple[Path, Path, Path],
) -> None:
    """A texture re-encoded at another quality fails JPEG byte-equality."""
    path = _texture(hf_site)
    with Image.open(io.BytesIO(path.read_bytes())) as image:
        rgb = np.array(image.convert("RGB"))
    buffer = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buffer, format="JPEG", quality=60)
    path.write_bytes(buffer.getvalue())
    with pytest.raises(Tiles3dError, match="does not byte-equal"):
        _verify(hf_site)


def test_orphan_tile_file_is_refused(
    hf_site: tuple[Path, Path, Path],
) -> None:
    """An extra unreferenced file in tiles/ is caught."""
    (hf_site[0] / "tiles" / "stray.bin").write_bytes(b"junk")
    with pytest.raises(Tiles3dError, match="not referenced by tileset.json"):
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
