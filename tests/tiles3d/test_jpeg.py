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
    is_baseline_jpeg,
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
    """The frame is baseline sequential (SOF0), never progressive (SOF2).

    Pillow reports ``format == "JPEG"`` for progressive streams too, so a
    ``JPEG_PROGRESSIVE = True`` regression would slip past a format check.
    Parsing the marker stream and asserting SOF0 present / SOF2 absent
    pins the pinned baseline framing.
    """
    data = encode_jpeg(synth_rgb(8, 8))
    assert data[:2] == b"\xff\xd8"  # SOI
    with Image.open(io.BytesIO(data)) as image:
        assert image.format == "JPEG"
    markers = _segment_markers(data)
    assert 0xC0 in markers  # SOF0: baseline sequential
    assert 0xC2 not in markers  # SOF2: progressive -- must be absent


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


def test_smooth_imagery_round_trips_almost_losslessly() -> None:
    """A smooth gradient (real-imagery-like) decodes near-losslessly.

    Every other fidelity test uses random noise (JPEG's worst case). Real
    ortho tiles are smooth, so this documents the near-lossless behaviour
    the module's docstring claims: a smooth gradient round-trips well under
    the floor, with a loose stable margin so it is not brittle.
    """
    gradient = _gradient_tile(64)
    decoded = decode_jpeg(encode_jpeg(gradient))
    mean_abs_error = float(
        np.mean(np.abs(gradient.astype(np.int32) - decoded.astype(np.int32)))
    )
    assert mean_abs_error < 5.0  # ~0.9 in practice; far below the floor
    assert jpeg_fidelity_ok(gradient, decoded) is True


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


def test_is_baseline_jpeg_accepts_the_pinned_encode() -> None:
    """The pinned encoder's output is recognised as baseline sequential."""
    assert is_baseline_jpeg(encode_jpeg(synth_rgb(8, 8))) is True


def test_is_baseline_jpeg_rejects_a_progressive_stream() -> None:
    """A progressive-framed JPEG (SOF2) is not baseline."""
    buffer = io.BytesIO()
    Image.fromarray(synth_rgb(8, 8), mode="RGB").save(
        buffer, format="JPEG", progressive=True
    )
    assert is_baseline_jpeg(buffer.getvalue()) is False


def test_is_baseline_jpeg_rejects_a_non_soi_stream() -> None:
    """A stream that does not start with SOI is refused up front."""
    assert is_baseline_jpeg(b"not a jpeg at all") is False


def test_is_baseline_jpeg_stops_on_a_non_marker_byte() -> None:
    """The header walk ends when a segment does not begin with 0xFF."""
    # SOI, an APP0 segment of length 4 (2 payload bytes), then a byte the
    # walk lands on that is not a marker -- no SOF0 is ever seen.
    data = b"\xff\xd8\xff\xe0\x00\x04ab\x01\x02\x03\x04"
    assert is_baseline_jpeg(data) is False


def test_is_baseline_jpeg_stops_when_a_segment_overruns() -> None:
    """The walk ends when a declared segment length overruns the buffer."""
    data = b"\xff\xd8\xff\xdb\x00\xff"  # DQT claims 255 bytes; buffer ends
    assert is_baseline_jpeg(data) is False


def _gradient_tile(size: int) -> npt.NDArray[np.uint8]:
    """Build a smooth (h, w, 3) uint8 gradient — real-imagery-like."""
    ramp = np.linspace(0, 255, size).astype(np.uint8)
    return np.stack(
        [
            np.tile(ramp, (size, 1)),
            np.tile(ramp[::-1], (size, 1)),
            np.tile(ramp.reshape(-1, 1), (1, size)),
        ],
        axis=2,
    )


def _segment_markers(data: bytes) -> list[int]:
    """Collect JPEG segment-marker bytes from SOI up to (excluding) SOS.

    Walks the marker stream: standalone markers (SOI/EOI/RST) carry no
    length, length-prefixed segments are skipped by their big-endian
    length, and the scan stops at SOS (start of the entropy-coded data).
    """
    markers: list[int] = []
    pos = 2  # past SOI
    end = len(data)
    while pos < end and data[pos] == 0xFF:
        while pos < end and data[pos] == 0xFF:  # skip fill bytes
            pos += 1
        if pos >= end:
            break
        marker = data[pos]
        pos += 1
        markers.append(marker)
        if marker == 0xD9 or 0xD0 <= marker <= 0xD7:  # EOI / RST: no length
            continue
        if marker == 0xDA:  # SOS: entropy data follows -- stop
            break
        length = (data[pos] << 8) | data[pos + 1]
        pos += length
    return markers
