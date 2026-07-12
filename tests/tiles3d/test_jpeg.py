"""Tests for the baseline-JPEG texture codec (game profile only)."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, cast

import numpy as np
import PIL
import pytest
from PIL import Image

if TYPE_CHECKING:
    import numpy.typing as npt

from ahn_cli.tiles3d.errors import Tiles3dError
from ahn_cli.tiles3d.jpeg import (
    JPEG_MAX_MEAN_ABS_ERROR,
    JPEG_OPTIMIZE,
    JPEG_PROGRESSIVE,
    JPEG_QUALITY,
    JPEG_SUBSAMPLING,
    decode_jpeg,
    encode_jpeg,
    jpeg_fidelity_ok,
    pillow_version,
)
from tests.tiles3d.conftest import synth_rgb


def test_pinned_settings_are_the_stated_constants() -> None:
    """The module states the pinned encoder settings, once."""
    assert JPEG_QUALITY == 85
    assert JPEG_SUBSAMPLING == 2  # Pillow's 4:2:0
    assert JPEG_PROGRESSIVE is False
    assert JPEG_OPTIMIZE is False


def test_round_trip_shape_and_dtype() -> None:
    """decode(encode(x)) preserves (h, w, 3) uint8."""
    rgb = synth_rgb(7, 5, seed=9)
    out = decode_jpeg(encode_jpeg(rgb))
    assert out.shape == rgb.shape
    assert out.dtype == np.uint8


def test_encoding_is_deterministic() -> None:
    """Two encodings of the same array are byte-identical."""
    rgb = synth_rgb(16, 16, seed=10)
    assert encode_jpeg(rgb) == encode_jpeg(rgb)


def test_output_is_a_baseline_jpeg() -> None:
    """The bytes start with the JPEG SOI marker and decode as JPEG."""
    data = encode_jpeg(synth_rgb(8, 8))
    assert data[:2] == b"\xff\xd8"  # SOI
    with Image.open(io.BytesIO(data)) as image:
        assert image.format == "JPEG"


def test_fidelity_floor_passes_for_a_genuine_encode() -> None:
    """decode(encode(tile)) stays under the fidelity floor."""
    rgb = synth_rgb(64, 64, seed=1)
    decoded = decode_jpeg(encode_jpeg(rgb))
    assert jpeg_fidelity_ok(rgb, decoded) is True


def test_fidelity_floor_rejects_a_wrong_image() -> None:
    """A decode compared against a different tile fails the floor."""
    rgb = synth_rgb(64, 64, seed=1)
    other = synth_rgb(64, 64, seed=2)
    decoded = decode_jpeg(encode_jpeg(rgb))
    # The genuine encode of ``other`` decodes fine, but measured against
    # ``rgb`` (a different image) it is garbage and must be rejected.
    assert jpeg_fidelity_ok(rgb, decode_jpeg(encode_jpeg(other))) is False
    # Sanity: the genuine pairing does pass, so the gate is not always-false.
    assert jpeg_fidelity_ok(other, decode_jpeg(encode_jpeg(other))) is True
    assert jpeg_fidelity_ok(rgb, decoded) is True


def test_floor_constant_sits_between_noise_and_a_wrong_image() -> None:
    """The floor clears worst-case noise yet rejects a wrong image."""
    rgb = synth_rgb(64, 64, seed=1)
    other = synth_rgb(64, 64, seed=2)
    noise_error = float(
        np.mean(
            np.abs(
                rgb.astype(np.int32)
                - decode_jpeg(encode_jpeg(rgb)).astype(np.int32)
            )
        )
    )
    wrong_error = float(
        np.mean(np.abs(rgb.astype(np.int32) - other.astype(np.int32)))
    )
    assert noise_error < JPEG_MAX_MEAN_ABS_ERROR < wrong_error


def test_grayscale_input_is_forced_to_three_channels() -> None:
    """A grayscale-mode JPEG still decodes to (h, w, 3)."""
    buf = io.BytesIO()
    gray = synth_rgb(9, 6, seed=3)[:, :, 0]
    Image.fromarray(gray, mode="L").save(buf, format="JPEG")
    out = decode_jpeg(buf.getvalue())
    assert out.shape == (6, 9, 3)
    assert out.dtype == np.uint8


def test_non_uint8_input_is_refused() -> None:
    """A float array is refused before it reaches Pillow."""
    float_image = cast(
        "npt.NDArray[np.uint8]", np.zeros((4, 4, 3), np.float32)
    )
    with pytest.raises(Tiles3dError, match="uint8"):
        encode_jpeg(float_image)


def test_wrong_ndim_is_refused() -> None:
    """A 2-D array is refused."""
    with pytest.raises(Tiles3dError, match="h, w, 3"):
        encode_jpeg(np.zeros((4, 4), dtype=np.uint8))


def test_wrong_channel_count_is_refused() -> None:
    """An RGBA array is refused."""
    with pytest.raises(Tiles3dError, match="h, w, 3"):
        encode_jpeg(np.zeros((4, 4, 4), dtype=np.uint8))


def test_encoder_library_error_is_wrapped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Pillow save failure surfaces as a typed Tiles3dError."""

    def boom(*_args: object, **_kwargs: object) -> Image.Image:
        msg = "synthetic encoder failure"
        raise OSError(msg)

    monkeypatch.setattr("ahn_cli.tiles3d.jpeg.Image.fromarray", boom)
    with pytest.raises(Tiles3dError, match="JPEG encode"):
        encode_jpeg(synth_rgb(4, 4))


def test_undecodable_bytes_are_wrapped() -> None:
    """Garbage bytes surface as a typed Tiles3dError, not a Pillow error."""
    with pytest.raises(Tiles3dError, match="JPEG decode"):
        decode_jpeg(b"not a jpeg at all")


def test_pillow_version_matches_the_installed_library() -> None:
    """The exported version equals Pillow's own reported version."""
    version = pillow_version()
    assert isinstance(version, str)
    assert version == PIL.__version__
