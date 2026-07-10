"""Tests for the deterministic reconciled-cloud writers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import laspy
import numpy as np
import numpy.typing as npt
import pytest

from ahn_cli.reconcile.writers import (
    OutputFormat,
    open_writer,
    write_reconciled,
)

if TYPE_CHECKING:
    from pathlib import Path

_RGB_TO_UINT16 = 257


def _grid() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Return a 2x3x6 grid and a mask with a single void cell at (0, 2)."""
    grid = np.zeros((2, 3, 6), dtype=np.float64)
    grid[:, :, 0] = np.array([[10.0, 11.0, 12.0], [10.0, 11.0, 12.0]])
    grid[:, :, 1] = np.array([[20.0, 20.0, 20.0], [21.0, 21.0, 21.0]])
    grid[:, :, 2] = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    grid[:, :, 3] = 10.0  # red
    grid[:, :, 4] = 128.0  # green
    grid[:, :, 5] = 255.0  # blue
    mask = np.array([[True, True, False], [True, True, True]])
    return grid, mask


def test_pt_roundtrip(tmp_path: Path) -> None:
    """The .pt blob is raw float32 [N, 6] of the valid points, in order."""
    grid, mask = _grid()
    path = tmp_path / "out.pt"
    count = write_reconciled(OutputFormat.PT, grid, mask, path)
    assert count == 5
    data = np.frombuffer(path.read_bytes(), dtype="<f4").reshape(-1, 6)
    assert np.array_equal(data, grid[mask].astype(np.float32))


def test_pt_empty(tmp_path: Path) -> None:
    """An all-void mask writes a zero-length .pt and reports zero points."""
    grid, _ = _grid()
    mask = np.zeros((2, 3), dtype=np.bool_)
    path = tmp_path / "empty.pt"
    assert write_reconciled(OutputFormat.PT, grid, mask, path) == 0
    assert path.read_bytes() == b""


def test_ply_header_and_payload(tmp_path: Path) -> None:
    """The PLY header declares the vertex count and the payload is 27 B/vertex."""
    grid, mask = _grid()
    path = tmp_path / "out.ply"
    count = write_reconciled(OutputFormat.PLY, grid, mask, path)
    assert count == 5
    raw = path.read_bytes()
    header, payload = raw.split(b"end_header\n", 1)
    assert b"element vertex 5" in header
    assert b"property uchar red" in header
    assert len(payload) == 5 * (3 * 8 + 3)


def test_laz_roundtrip_and_rgb_scaling(tmp_path: Path) -> None:
    """The LAZ carries the valid points with RGB scaled uint8 -> uint16."""
    grid, mask = _grid()
    path = tmp_path / "out.laz"
    count = write_reconciled(OutputFormat.LAZ, grid, mask, path)
    assert count == 5
    with laspy.open(str(path)) as reader:
        las = reader.read()
    assert len(las.x) == 5
    assert int(np.asarray(las.red).max()) == 10 * _RGB_TO_UINT16
    assert int(np.asarray(las.blue).max()) == 255 * _RGB_TO_UINT16


def test_laz_empty_is_valid(tmp_path: Path) -> None:
    """An all-void mask writes a valid zero-point LAZ (offsets branch)."""
    grid, _ = _grid()
    mask = np.zeros((2, 3), dtype=np.bool_)
    path = tmp_path / "empty.laz"
    assert write_reconciled(OutputFormat.LAZ, grid, mask, path) == 0
    with laspy.open(str(path)) as reader:
        assert int(reader.header.point_count) == 0


def test_exr_reports_pixel_count(tmp_path: Path) -> None:
    """The EXR writer returns the full pixel count (dense image)."""
    grid, mask = _grid()
    path = tmp_path / "out.exr"
    assert write_reconciled(OutputFormat.EXR, grid, mask, path) == 6


def test_exr_void_z_is_sentinel(tmp_path: Path) -> None:
    """A void cell's Z is forced to 0, so its grid Z does not affect the bytes."""
    grid, mask = _grid()
    grid_changed = grid.copy()
    grid_changed[0, 2, 2] = 999.0  # (0, 2) is the void cell
    a = tmp_path / "a.exr"
    b = tmp_path / "b.exr"
    write_reconciled(OutputFormat.EXR, grid, mask, a)
    write_reconciled(OutputFormat.EXR, grid_changed, mask, b)
    assert a.read_bytes() == b.read_bytes()


def test_exr_valid_z_changes_bytes(tmp_path: Path) -> None:
    """Changing a valid cell's Z does change the EXR bytes (sanity companion)."""
    grid, mask = _grid()
    grid_changed = grid.copy()
    grid_changed[0, 0, 2] = 999.0  # (0, 0) is a valid cell
    a = tmp_path / "a.exr"
    b = tmp_path / "b.exr"
    write_reconciled(OutputFormat.EXR, grid, mask, a)
    write_reconciled(OutputFormat.EXR, grid_changed, mask, b)
    assert a.read_bytes() != b.read_bytes()


@pytest.mark.parametrize("output_format", list(OutputFormat))
def test_writers_are_deterministic(
    output_format: OutputFormat, tmp_path: Path
) -> None:
    """Every writer yields byte-identical output across two runs."""
    grid, mask = _grid()
    first = tmp_path / f"first.{output_format.value}"
    second = tmp_path / f"second.{output_format.value}"
    write_reconciled(output_format, grid, mask, first)
    write_reconciled(output_format, grid, mask, second)
    assert first.read_bytes() == second.read_bytes()


def _big_grid() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Return a 10x7 grid with a mixed mask (incl. a fully-void row block)."""
    rng = np.random.default_rng(0)
    grid = np.zeros((10, 7, 6), dtype=np.float64)
    grid[:, :, 0] = np.arange(7)[None, :] + 100.0
    grid[:, :, 1] = (200.0 - np.arange(10))[:, None]
    grid[:, :, 2] = rng.random((10, 7)) * 50.0
    grid[:, :, 3:6] = rng.integers(0, 256, (10, 7, 3))
    mask = rng.random((10, 7)) > 0.2
    mask[4:8] = False  # a fully-void block exercises the laz empty-block skip
    return grid, mask


@pytest.mark.parametrize("output_format", list(OutputFormat))
def test_streaming_matches_single_block(
    output_format: OutputFormat, tmp_path: Path
) -> None:
    """Streaming a grid in row-blocks equals writing it as one block.

    Uncompressed formats are byte-identical; LAZ (chunk-compressed) is compared
    by point read-back. A fully-void middle block exercises the empty-block path.
    """
    grid, mask = _big_grid()
    single = tmp_path / f"single.{output_format.value}"
    streamed = tmp_path / f"streamed.{output_format.value}"
    count_single = write_reconciled(output_format, grid, mask, single)

    x_offset = float(np.floor(grid[:, :, 0].min()))
    y_offset = float(np.floor(grid[:, :, 1].min()))
    writer = open_writer(output_format, streamed, 7, 10, x_offset, y_offset)
    for start in (0, 4, 8):
        writer.write_block(grid[start : start + 4], mask[start : start + 4])
    count_streamed = writer.close()

    assert count_streamed == count_single
    if output_format is OutputFormat.LAZ:
        with laspy.open(str(single)) as reader:
            one = reader.read()
        with laspy.open(str(streamed)) as reader:
            many = reader.read()
        assert np.array_equal(
            np.c_[one.x, one.y, one.z], np.c_[many.x, many.y, many.z]
        )
    else:
        assert single.read_bytes() == streamed.read_bytes()
