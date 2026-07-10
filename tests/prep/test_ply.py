"""Tests for the prep-context LAZ->PLY export transform (WP13).

Fixtures are small synthetic format-6 LAZ tiles built in-process with laspy --
no network, no large files on disk. They exercise the deterministic binary PLY
header/payload, bit-exact coordinate preservation at RD (EPSG:28992)
magnitudes, and -- via an instrumented reader spy -- the memory-bounded
streaming contract (the full record is never read; only bounded chunks are).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import laspy
import numpy as np
import numpy.typing as npt
import pytest
from typing_extensions import Self

from ahn_cli.prep.ply import PlyExportStats, export_ply

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

# Real (unpatched) laspy.open, captured so the streaming spy can delegate to it
# after ``monkeypatch`` has replaced ``laspy.open`` with the spy factory.
_REAL_OPEN = laspy.open

Point = tuple[float, float, float]  # (x, y, z) in EPSG:28992 metres


def _write_tile(
    path: Path,
    points: list[Point],
    *,
    offsets: tuple[float, float, float] = (190000.0, 440000.0, 0.0),
    scales: tuple[float, float, float] = (0.01, 0.01, 0.01),
) -> None:
    """Write a synthetic format-6 tile of ``points`` to ``path``."""
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.offsets = np.array(offsets, dtype=float)
    header.scales = np.array(scales, dtype=float)
    las = laspy.LasData(header)
    arr = np.array(points, dtype=float)
    las.x = arr[:, 0]
    las.y = arr[:, 1]
    las.z = arr[:, 2]
    las.write(str(path))


def _read_points(path: Path) -> laspy.LasData:
    """Read a LAZ/LAS file fully into memory (test-side reference)."""
    with laspy.open(str(path)) as reader:
        return reader.read()


def _read_ply(path: Path) -> tuple[int, npt.NDArray[np.float64]]:
    """Parse a binary little-endian PLY into its vertex count and XYZ array."""
    data = path.read_bytes()
    marker = b"end_header\n"
    header_end = data.index(marker) + len(marker)
    header = data[:header_end].decode("ascii")
    count = next(
        int(line.split()[-1])
        for line in header.splitlines()
        if line.startswith("element vertex ")
    )
    body = np.frombuffer(data[header_end:], dtype="<f8").reshape(-1, 3)
    return count, np.asarray(body, dtype=np.float64)


# RD-magnitude coordinates: an easting near 194000 needs double precision, so
# these bit-exact assertions would fail were the payload written as float32.
_RD_POINTS: list[Point] = [
    (194198.31, 443461.34, 12.57),
    (194200.19, 443500.02, 13.01),
    (194594.11, 443694.84, 9.88),
]


# --------------------------------------------------------------------------
# Value object
# --------------------------------------------------------------------------


def test_ply_export_stats_is_a_frozen_value_object() -> None:
    """PlyExportStats is hashable and equal by field value."""
    stats = PlyExportStats(point_count=5)

    assert stats == PlyExportStats(5)
    assert len({stats, PlyExportStats(5)}) == 1


# --------------------------------------------------------------------------
# Guard
# --------------------------------------------------------------------------


def test_export_rejects_non_positive_chunk_size(tmp_path: Path) -> None:
    """A non-positive streaming window is refused before any I/O."""
    src = tmp_path / "src.laz"
    _write_tile(src, _RD_POINTS)

    with pytest.raises(ValueError, match="positive"):
        export_ply(src, tmp_path / "out.ply", chunk_size=0)


# --------------------------------------------------------------------------
# Deterministic header + round-trip
# --------------------------------------------------------------------------


def test_ply_header_is_static_binary_little_endian_double(
    tmp_path: Path,
) -> None:
    """The header pins format, double x/y/z properties, and the vertex count."""
    src = tmp_path / "src.laz"
    _write_tile(src, _RD_POINTS)
    out = tmp_path / "out.ply"

    export_ply(src, out)

    data = out.read_bytes()
    assert data.startswith(b"ply\nformat binary_little_endian 1.0\n")
    assert b"element vertex 3\n" in data
    assert (
        b"property double x\nproperty double y\nproperty double z\n"
        b"end_header\n"
    ) in data


def test_export_preserves_point_count_and_coordinates(tmp_path: Path) -> None:
    """Every source point round-trips: count and bit-exact XYZ are preserved."""
    src = tmp_path / "src.laz"
    _write_tile(src, _RD_POINTS)
    out = tmp_path / "out.ply"

    stats = export_ply(src, out)

    source = _read_points(src)
    count, xyz = _read_ply(out)
    assert stats == PlyExportStats(point_count=len(_RD_POINTS))
    assert count == len(_RD_POINTS)
    assert np.array_equal(xyz[:, 0], np.asarray(source.x, dtype=np.float64))
    assert np.array_equal(xyz[:, 1], np.asarray(source.y, dtype=np.float64))
    assert np.array_equal(xyz[:, 2], np.asarray(source.z, dtype=np.float64))


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------


def test_export_is_byte_identical_on_repeat(tmp_path: Path) -> None:
    """Identical input yields byte-identical output across two writes."""
    src = tmp_path / "src.laz"
    _write_tile(src, _RD_POINTS)
    first = tmp_path / "first.ply"
    second = tmp_path / "second.ply"

    stats_first = export_ply(src, first)
    stats_second = export_ply(src, second)

    assert stats_first == stats_second
    assert first.read_bytes() == second.read_bytes()


# --------------------------------------------------------------------------
# Memory-bounded streaming (regression) -- instrumented reader spy
# --------------------------------------------------------------------------


class _StreamLog:
    """A ledger of how an :class:`_SpyReader` was consumed."""

    def __init__(self) -> None:
        self.chunk_sizes: list[int] = []
        self.read_calls: int = 0


class _SpyReader:
    """A laspy reader wrapper recording chunk sizes and full-read attempts.

    It delegates to a real reader but tallies every chunk yielded and every
    ``read()`` (full materialization) call, so a test can prove the export
    streams in bounded windows and never loads the whole cloud.
    """

    def __init__(self, path: str, log: _StreamLog) -> None:
        self._path = path
        self._log = log

    def __enter__(self) -> Self:
        self._cm = _REAL_OPEN(self._path)
        self._inner = self._cm.__enter__()
        return self

    def __exit__(self, *args: object) -> None:
        self._cm.__exit__(*args)

    @property
    def header(self) -> laspy.LasHeader:
        """Delegate the header the export reads its point count from."""
        return self._inner.header

    def read(self) -> laspy.ScaleAwarePointRecord:
        """Record and forward a full-record read (the anti-pattern to catch)."""
        self._log.read_calls += 1
        return self._inner.read()

    def chunk_iterator(
        self, points_per_iteration: int
    ) -> Iterator[laspy.ScaleAwarePointRecord]:
        """Yield the real reader's chunks, tallying each chunk's size."""
        for chunk in self._inner.chunk_iterator(points_per_iteration):
            self._log.chunk_sizes.append(len(chunk))
            yield chunk


def _spy_open_factory(log: _StreamLog) -> Callable[[str], _SpyReader]:
    """Return a drop-in ``laspy.open`` that produces logging readers."""

    def _spy_open(source: str) -> _SpyReader:
        return _SpyReader(source, log)

    return _spy_open


def test_export_streams_in_bounded_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Export consumes the reader in bounded chunks, never a full read.

    A 25-point cloud exported with ``chunk_size=10`` must be read as three
    windows of at most 10 points (10, 10, 5), summing to 25, with ``read()``
    (the full-materialization path) never called -- proving the memory bound.
    """
    src = tmp_path / "big.laz"
    points: list[Point] = [
        (194000.0 + i, 443000.0 + i, float(i)) for i in range(25)
    ]
    _write_tile(src, points)
    log = _StreamLog()
    monkeypatch.setattr(laspy, "open", _spy_open_factory(log))
    out = tmp_path / "out.ply"

    stats = export_ply(src, out, chunk_size=10)

    assert log.read_calls == 0
    assert log.chunk_sizes == [10, 10, 5]
    assert all(size <= 10 for size in log.chunk_sizes)
    assert sum(log.chunk_sizes) == 25
    assert stats.point_count == 25
