"""Tests for the strict reconciled-EXR reader."""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING

import numpy as np
import pytest

from ahn_cli.tiles3d.errors import Tiles3dError
from ahn_cli.tiles3d.exr import read_reconciled_exr
from tests.tiles3d.conftest import corrupt, synth_grid, write_exr

if TYPE_CHECKING:
    from pathlib import Path

_W, _H = 5, 4


@pytest.fixture
def exr_path(tmp_path: Path) -> Path:
    """Write a valid 5x4 reconciled EXR with the real writer."""
    return write_exr(tmp_path / "r.exr", synth_grid(_W, _H))


def _attr_value_offset(path: Path, name: bytes, type_name: bytes) -> int:
    """Byte offset of the named header attribute's value."""
    data = path.read_bytes()
    marker = name + b"\x00" + type_name + b"\x00"
    return data.index(marker) + len(marker) + 4  # skip the size field


def _header_length(path: Path) -> int:
    """Byte offset of the scanline offset table (end of the header)."""
    end = _attr_value_offset(path, b"screenWindowWidth", b"float")
    return end + 4 + 1  # float value + the header's terminating NUL


def test_reads_back_exactly_what_the_writer_wrote(
    exr_path: Path,
) -> None:
    """Every plane round-trips bit-exact as float32."""
    grid = synth_grid(_W, _H)
    exr = read_reconciled_exr(exr_path)
    assert (exr.width, exr.height) == (_W, _H)
    assert np.array_equal(exr.x, grid[:, :, 0].astype(np.float32))
    assert np.array_equal(exr.y, grid[:, :, 1].astype(np.float32))
    assert np.array_equal(exr.z, grid[:, :, 2].astype(np.float32))
    assert np.array_equal(exr.r, (grid[:, :, 3] / 255.0).astype(np.float32))
    assert np.array_equal(exr.g, (grid[:, :, 4] / 255.0).astype(np.float32))
    assert np.array_equal(exr.b, (grid[:, :, 5] / 255.0).astype(np.float32))
    for plane in (exr.x, exr.y, exr.z, exr.r, exr.g, exr.b):
        assert plane.shape == (_H, _W)
        assert plane.dtype == np.float32


def test_missing_file_is_a_typed_error(tmp_path: Path) -> None:
    """An unreadable path raises Tiles3dError, not OSError."""
    with pytest.raises(Tiles3dError, match="not readable"):
        read_reconciled_exr(tmp_path / "absent.exr")


def test_bad_magic_is_refused(exr_path: Path) -> None:
    """A non-EXR file fails on the magic number."""
    corrupt(exr_path, 0, b"\xde\xad\xbe\xef")
    with pytest.raises(Tiles3dError, match="magic"):
        read_reconciled_exr(exr_path)


def test_bad_version_is_refused(exr_path: Path) -> None:
    """An unexpected EXR version is refused."""
    corrupt(exr_path, 4, struct.pack("<I", 3))
    with pytest.raises(Tiles3dError, match="version"):
        read_reconciled_exr(exr_path)


def test_unexpected_attribute_name_is_refused(exr_path: Path) -> None:
    """A renamed attribute breaks the exact expected attribute set."""
    data = exr_path.read_bytes()
    offset = data.index(b"lineOrder\x00lineOrder\x00")
    corrupt(exr_path, offset, b"lineOrdeR")
    with pytest.raises(Tiles3dError, match="attribute"):
        read_reconciled_exr(exr_path)


def test_unexpected_attribute_type_is_refused(exr_path: Path) -> None:
    """A retyped attribute breaks the expected name -> type mapping."""
    data = exr_path.read_bytes()
    offset = data.index(b"dataWindow\x00box2i\x00") + len(b"dataWindow\x00")
    corrupt(exr_path, offset, b"box2j")
    with pytest.raises(Tiles3dError, match="attribute"):
        read_reconciled_exr(exr_path)


def test_wrong_channel_name_is_refused(exr_path: Path) -> None:
    """The channel list must be exactly B, G, R, X, Y, Z."""
    offset = _attr_value_offset(exr_path, b"channels", b"chlist")
    corrupt(exr_path, offset, b"A")  # first channel name B -> A
    with pytest.raises(Tiles3dError, match="channel"):
        read_reconciled_exr(exr_path)


def test_wrong_pixel_type_is_refused(exr_path: Path) -> None:
    """A non-FLOAT channel is refused."""
    offset = _attr_value_offset(exr_path, b"channels", b"chlist")
    # after name "B\x00" comes the int32 pixel type
    corrupt(exr_path, offset + 2, struct.pack("<i", 1))  # HALF
    with pytest.raises(Tiles3dError, match="channel"):
        read_reconciled_exr(exr_path)


def test_compressed_exr_is_refused(exr_path: Path) -> None:
    """Any compression other than none is refused."""
    offset = _attr_value_offset(exr_path, b"compression", b"compression")
    corrupt(exr_path, offset, struct.pack("<B", 3))  # PIZ
    with pytest.raises(Tiles3dError, match="compression"):
        read_reconciled_exr(exr_path)


def test_nonzero_data_window_origin_is_refused(exr_path: Path) -> None:
    """The data window must start at (0, 0)."""
    offset = _attr_value_offset(exr_path, b"dataWindow", b"box2i")
    corrupt(exr_path, offset, struct.pack("<i", 1))
    with pytest.raises(Tiles3dError, match="data window"):
        read_reconciled_exr(exr_path)


def test_display_window_mismatch_is_refused(exr_path: Path) -> None:
    """The display window must equal the data window."""
    offset = _attr_value_offset(exr_path, b"displayWindow", b"box2i")
    corrupt(exr_path, offset + 8, struct.pack("<i", _W))
    with pytest.raises(Tiles3dError, match="display window"):
        read_reconciled_exr(exr_path)


def test_nonzero_line_order_is_refused(exr_path: Path) -> None:
    """Only increasing-Y line order is accepted."""
    offset = _attr_value_offset(exr_path, b"lineOrder", b"lineOrder")
    corrupt(exr_path, offset, struct.pack("<B", 1))
    with pytest.raises(Tiles3dError, match="line order"):
        read_reconciled_exr(exr_path)


def test_unexpected_pixel_aspect_ratio_is_refused(exr_path: Path) -> None:
    """The writer pins pixelAspectRatio to 1.0; anything else is refused."""
    offset = _attr_value_offset(exr_path, b"pixelAspectRatio", b"float")
    corrupt(exr_path, offset, struct.pack("<f", 2.0))
    with pytest.raises(Tiles3dError, match="pixelAspectRatio"):
        read_reconciled_exr(exr_path)


def test_broken_offset_table_is_refused(exr_path: Path) -> None:
    """A scanline offset that does not match the layout is refused."""
    corrupt(exr_path, _header_length(exr_path), struct.pack("<Q", 7))
    with pytest.raises(Tiles3dError, match="offset table"):
        read_reconciled_exr(exr_path)


def test_scanline_row_mismatch_is_refused(exr_path: Path) -> None:
    """A scanline whose y field is not its row index is refused."""
    first_block = _header_length(exr_path) + _H * 8
    corrupt(exr_path, first_block, struct.pack("<i", 2))
    with pytest.raises(Tiles3dError, match="scanline"):
        read_reconciled_exr(exr_path)


def test_scanline_size_mismatch_is_refused(exr_path: Path) -> None:
    """A scanline whose byte size is wrong is refused."""
    first_block = _header_length(exr_path) + _H * 8
    corrupt(exr_path, first_block + 4, struct.pack("<i", 12))
    with pytest.raises(Tiles3dError, match="scanline"):
        read_reconciled_exr(exr_path)


def test_truncated_file_is_refused(exr_path: Path) -> None:
    """A file cut short anywhere is refused as truncated."""
    data = exr_path.read_bytes()
    exr_path.write_bytes(data[:-4])
    with pytest.raises(Tiles3dError, match="truncated"):
        read_reconciled_exr(exr_path)


def test_truncated_header_is_refused(exr_path: Path) -> None:
    """A file ending inside the header is refused as truncated."""
    data = exr_path.read_bytes()
    exr_path.write_bytes(data[:20])
    with pytest.raises(Tiles3dError, match="truncated"):
        read_reconciled_exr(exr_path)


def test_trailing_bytes_are_refused(exr_path: Path) -> None:
    """Any bytes after the last scanline are refused."""
    data = exr_path.read_bytes()
    exr_path.write_bytes(data + b"\x00")
    with pytest.raises(Tiles3dError, match="trailing"):
        read_reconciled_exr(exr_path)
