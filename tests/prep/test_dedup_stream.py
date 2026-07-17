"""Tests for out-of-core tile deduplication (`ahn_cli.prep.dedup_stream`).

The load-bearing contract is **byte-identity to the in-memory oracle**
:func:`ahn_cli.prep.dedup.deduplicate_tiles`: for every fixture the streaming
path's output file must hash identically to the oracle's, while never holding
more than one chunk of points in memory. The remaining tests lock the crop,
offset-reprojection, global-first-index survivor, determinism/invariance, the
disk-floor guard, and the bounded-memory (never ``reader.read()``) guarantee.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import laspy
import numpy as np
import pytest

from ahn_cli.prep import dedup_stream as dedup_stream_module
from ahn_cli.prep.dedup import CanonicalTile, DedupStats, deduplicate_tiles
from ahn_cli.prep.dedup_stream import stream_deduplicate_tiles
from ahn_cli.prep.spill import DiskFloorError

_SPILL_SUBDIR = "dedup_spill"  # mirrors dedup_stream._SPILL_SUBDIR
_FINALIZE_HEADROOM = (
    4 * 1024**2
)  # mirrors dedup_stream._FINALIZE_HEADROOM_BYTES

if TYPE_CHECKING:
    from pathlib import Path

    from ahn_cli.domain import BBox

Point = tuple[float, float, float, float]  # (x, y, z, gps_time)


def _write_tile(
    path: Path,
    points: list[Point],
    *,
    offsets: tuple[float, float, float] = (0.0, 0.0, 0.0),
    scales: tuple[float, float, float] = (0.01, 0.01, 0.01),
) -> None:
    """Write a synthetic format-6 (gps_time-bearing) tile to ``path``."""
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.offsets = np.array(offsets, dtype=float)
    header.scales = np.array(scales, dtype=float)
    las = laspy.LasData(header)
    arr = np.array(points, dtype=float)
    las.x = arr[:, 0]
    las.y = arr[:, 1]
    las.z = arr[:, 2]
    las.gps_time = arr[:, 3]
    las.write(str(path))


def _read(path: Path) -> laspy.LasData:
    """Read a LAZ/LAS file fully into memory (test-side only)."""
    with laspy.open(str(path)) as reader:
        return reader.read()


def _sha(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _world_xy(las: laspy.LasData) -> set[tuple[float, float]]:
    """Return the rounded world (x, y) coordinates present in ``las``."""
    return {
        (round(float(x), 3), round(float(y), 3))
        for x, y in zip(las.x, las.y, strict=True)
    }


# A pair of tiles whose canonical extents both claim the whole area, so every
# shared point survives the crop and is a genuine exact duplicate. Tile A's copy
# (the smaller global index) is the required survivor.
_EXT: BBox = (0.0, 0.0, 10.0, 10.0)
_TILE_A: list[Point] = [(2.0, 2.0, 0.0, 1.0), (5.0, 5.0, 0.0, 5.0)]
_TILE_B: list[Point] = [(5.0, 5.0, 0.0, 5.0), (8.0, 8.0, 0.0, 8.0)]


def _seam_tiles(tmp_path: Path) -> list[CanonicalTile]:
    """Two neighbouring tiles overlapping in a seam band (half-open crop)."""
    tile_a = tmp_path / "a.laz"
    tile_b = tmp_path / "b.laz"
    _write_tile(
        tile_a,
        [
            (1.0, 1.0, 0.0, 1.0),
            (5.0, 5.0, 0.0, 2.0),
            (9.0, 9.0, 0.0, 3.0),
            (10.0, 5.0, 0.0, 10.0),
            (11.0, 5.0, 0.0, 11.0),
        ],
    )
    _write_tile(
        tile_b,
        [
            (10.0, 5.0, 0.0, 10.0),
            (12.0, 5.0, 0.0, 12.0),
            (15.0, 5.0, 0.0, 15.0),
            (19.0, 9.0, 0.0, 19.0),
            (9.0, 5.0, 0.0, 9.0),
        ],
    )
    return [
        CanonicalTile(path=tile_a, extent=(0.0, 0.0, 10.0, 10.0)),
        CanonicalTile(path=tile_b, extent=(10.0, 0.0, 20.0, 10.0)),
    ]


def _dup_tiles(tmp_path: Path) -> list[CanonicalTile]:
    """Two tiles claiming the same extent, sharing an exact duplicate point."""
    tile_a = tmp_path / "a.laz"
    tile_b = tmp_path / "b.laz"
    _write_tile(tile_a, _TILE_A)
    _write_tile(tile_b, _TILE_B)
    return [
        CanonicalTile(path=tile_a, extent=_EXT),
        CanonicalTile(path=tile_b, extent=_EXT),
    ]


def _offset_tiles(tmp_path: Path) -> list[CanonicalTile]:
    """Two tiles with differing LAS offsets sharing one world point."""
    tile_a = tmp_path / "a.laz"
    tile_b = tmp_path / "b.laz"
    _write_tile(tile_a, _TILE_A, offsets=(0.0, 0.0, 0.0))
    _write_tile(
        tile_b,
        [(5.0, 5.0, 0.0, 5.0), (8.0, 8.0, 0.0, 8.0)],
        offsets=(1000.0, 1000.0, 0.0),
    )
    return [
        CanonicalTile(path=tile_a, extent=_EXT),
        CanonicalTile(path=tile_b, extent=_EXT),
    ]


def _disjoint_tiles(tmp_path: Path) -> list[CanonicalTile]:
    """Two disjoint tiles (no overlap, no duplicates)."""
    tile_a = tmp_path / "a.laz"
    tile_b = tmp_path / "b.laz"
    _write_tile(tile_a, [(1.0, 1.0, 0.0, 1.0), (2.0, 2.0, 0.0, 2.0)])
    _write_tile(tile_b, [(12.0, 5.0, 0.0, 12.0), (15.0, 5.0, 0.0, 15.0)])
    return [
        CanonicalTile(path=tile_a, extent=(0.0, 0.0, 10.0, 10.0)),
        CanonicalTile(path=tile_b, extent=(10.0, 0.0, 20.0, 10.0)),
    ]


def _same_tile_twice(tmp_path: Path) -> list[CanonicalTile]:
    """Build the same physical tile twice under two names (all duplicates)."""
    tile_a = tmp_path / "a.laz"
    tile_b = tmp_path / "b.laz"
    _write_tile(tile_a, _TILE_A)
    _write_tile(tile_b, _TILE_A)
    return [
        CanonicalTile(path=tile_a, extent=_EXT),
        CanonicalTile(path=tile_b, extent=_EXT),
    ]


_FIXTURES = {
    "seam": _seam_tiles,
    "dup": _dup_tiles,
    "offsets": _offset_tiles,
    "disjoint": _disjoint_tiles,
    "same-twice": _same_tile_twice,
}


# --------------------------------------------------------------------------
# F1 -- byte-identity to the in-memory oracle
# --------------------------------------------------------------------------


@pytest.mark.parametrize("fixture", list(_FIXTURES))
@pytest.mark.parametrize("chunk_points", [1, 2, 3, 1000])
def test_output_is_byte_identical_to_the_oracle(
    tmp_path: Path, fixture: str, chunk_points: int
) -> None:
    """sha256(streamed output) == sha256(deduplicate_tiles output), every case."""
    tiles = _FIXTURES[fixture](tmp_path)
    oracle_out = tmp_path / "oracle.laz"
    stream_out = tmp_path / "stream.laz"

    oracle_stats = deduplicate_tiles(tiles, oracle_out)
    stream_stats = stream_deduplicate_tiles(
        tiles,
        stream_out,
        workdir=tmp_path / "wd",
        chunk_points=chunk_points,
    )

    assert stream_stats == oracle_stats
    assert _sha(stream_out) == _sha(oracle_out)


def test_crop_before_merge_is_half_open(tmp_path: Path) -> None:
    """The seam band (including the shared x == 10 edge) is claimed by one tile."""
    tiles = _seam_tiles(tmp_path)
    out = tmp_path / "out.laz"

    stats = stream_deduplicate_tiles(tiles, out, workdir=tmp_path / "wd")

    assert stats == DedupStats(
        input_points=10,
        cropped_points=7,
        duplicates_removed=0,
        output_points=7,
    )
    world = _world_xy(_read(out))
    assert {(1.0, 1.0), (5.0, 5.0), (9.0, 9.0)} <= world
    assert {(12.0, 5.0), (15.0, 5.0), (19.0, 9.0), (10.0, 5.0)} <= world
    assert (11.0, 5.0) not in world  # A's seam band, cropped
    assert (9.0, 5.0) not in world  # B's seam band, cropped


def test_offset_reprojection_lands_tiles_on_one_lattice(
    tmp_path: Path,
) -> None:
    """A world point stored under two LAS offsets reduces to one after the sweep."""
    tiles = _offset_tiles(tmp_path)
    out = tmp_path / "out.laz"

    stats = stream_deduplicate_tiles(tiles, out, workdir=tmp_path / "wd")

    assert stats.duplicates_removed == 1
    assert _world_xy(_read(out)) == {(5.0, 5.0), (2.0, 2.0), (8.0, 8.0)}


def test_cross_tile_duplicate_keeps_the_global_first_index(
    tmp_path: Path,
) -> None:
    """Of a duplicate group the earlier tile's point (smaller idx) survives."""
    tiles = _dup_tiles(tmp_path)
    out = tmp_path / "out.laz"

    stats = stream_deduplicate_tiles(
        tiles, out, workdir=tmp_path / "wd", chunk_points=1
    )

    assert stats == DedupStats(
        input_points=4,
        cropped_points=4,
        duplicates_removed=1,
        output_points=3,
    )
    result = _read(out)
    # Ascending global index: A(2,2) idx0, A(5,5) idx1, B(8,8) idx3.
    assert [round(float(v), 3) for v in result.x] == [2.0, 5.0, 8.0]
    assert [round(float(v), 3) for v in result.gps_time] == [1.0, 5.0, 8.0]


def test_disjoint_tiles_pass_through_untouched(tmp_path: Path) -> None:
    """Disjoint tiles lose no points and gain no duplicate removals."""
    tiles = _disjoint_tiles(tmp_path)
    out = tmp_path / "out.laz"

    stats = stream_deduplicate_tiles(tiles, out, workdir=tmp_path / "wd")

    assert stats == DedupStats(4, 4, 0, 4)
    assert _world_xy(_read(out)) == {
        (1.0, 1.0),
        (2.0, 2.0),
        (12.0, 5.0),
        (15.0, 5.0),
    }


# --------------------------------------------------------------------------
# Determinism and invariance
# --------------------------------------------------------------------------


def test_default_workdir_is_used_when_none(tmp_path: Path) -> None:
    """Passing no workdir runs through a private temp dir and still succeeds."""
    tiles = _dup_tiles(tmp_path)
    out = tmp_path / "out.laz"

    stats = stream_deduplicate_tiles(tiles, out)

    assert stats.output_points == 3
    assert _read(out).gps_time.tolist() == [1.0, 5.0, 8.0]


def test_output_is_deterministic(tmp_path: Path) -> None:
    """Two independent runs produce byte-identical output."""
    tiles = _seam_tiles(tmp_path)
    out_a = tmp_path / "a_out.laz"
    out_b = tmp_path / "b_out.laz"

    stream_deduplicate_tiles(tiles, out_a, workdir=tmp_path / "wa")
    stream_deduplicate_tiles(tiles, out_b, workdir=tmp_path / "wb")

    assert _sha(out_a) == _sha(out_b)


def test_survivors_invariant_to_chunk_segment_and_partition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Output is byte-identical across chunk/segment/partition/read-block knobs.

    The property F1 demands: partition-count, chunk-size, and segment-roll are
    pure performance knobs -- the survivor set and their order never move -- and
    the result stays equal to the oracle regardless.
    """
    rng = np.random.default_rng(1)
    coords_a = rng.uniform(0.0, 10.0, size=(400, 3))
    coords_b = rng.uniform(0.0, 10.0, size=(400, 3))
    # Inject genuine cross-tile duplicates surviving the shared crop.
    dup = rng.choice(400, 60, replace=False)
    coords_b[:60] = coords_a[dup]
    gps_a = np.arange(400.0)
    gps_b = np.arange(400.0, 800.0)
    gps_b[:60] = gps_a[dup]
    pa = tmp_path / "a.laz"
    pb = tmp_path / "b.laz"
    _write_tile(pa, [(*coords_a[i], gps_a[i]) for i in range(400)])
    _write_tile(pb, [(*coords_b[i], gps_b[i]) for i in range(400)])
    tiles = [
        CanonicalTile(path=pa, extent=_EXT),
        CanonicalTile(path=pb, extent=_EXT),
    ]
    oracle_out = tmp_path / "oracle.laz"
    oracle_stats = deduplicate_tiles(tiles, oracle_out)
    expected = _sha(oracle_out)

    hashes: set[str] = set()
    for chunk_points in (7, 101, 10_000):
        for segment_bytes in (60, 10**9):
            for partition_bytes in (60, 10**9):
                monkeypatch.setattr(
                    dedup_stream_module, "_SEGMENT_BYTES", segment_bytes
                )
                monkeypatch.setattr(
                    dedup_stream_module,
                    "_PARTITION_TARGET_BYTES",
                    partition_bytes,
                )
                monkeypatch.setattr(
                    dedup_stream_module, "_PARTITION_READ_RECORDS", 32
                )
                out = tmp_path / f"o_{chunk_points}_{segment_bytes}.laz"
                stats = stream_deduplicate_tiles(
                    tiles,
                    out,
                    workdir=tmp_path / f"w_{chunk_points}_{segment_bytes}",
                    chunk_points=chunk_points,
                )
                assert stats == oracle_stats
                hashes.add(_sha(out))

    assert hashes == {expected}


def test_reduce_pass_handles_a_singleton_partition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A partition holding exactly one record skips the boundary-diff slice."""
    monkeypatch.setattr(dedup_stream_module, "_PARTITION_TARGET_BYTES", 1)
    tiles = _dup_tiles(tmp_path)
    out = tmp_path / "out.laz"

    stats = stream_deduplicate_tiles(tiles, out, workdir=tmp_path / "wd")

    assert stats.output_points == 3


def test_progress_is_reported_per_tile(tmp_path: Path) -> None:
    """The progress callback ticks once per streamed tile."""
    tiles = _disjoint_tiles(tmp_path)
    out = tmp_path / "out.laz"
    ticks: list[tuple[int, int]] = []

    stream_deduplicate_tiles(
        tiles,
        out,
        workdir=tmp_path / "wd",
        progress=lambda done, total: ticks.append((done, total)),
    )

    assert ticks == [(1, 2), (2, 2)]


# --------------------------------------------------------------------------
# Cropped-to-empty and per-chunk crop edges
# --------------------------------------------------------------------------


def test_a_fully_cropped_tile_contributes_nothing(tmp_path: Path) -> None:
    """A tile whose every point is outside its extent still matches the oracle."""
    tile_a = tmp_path / "a.laz"
    tile_b = tmp_path / "b.laz"
    _write_tile(tile_a, [(1.0, 1.0, 0.0, 1.0), (2.0, 2.0, 0.0, 2.0)])
    # Every B point lies outside B's own extent -> cropped away entirely.
    _write_tile(tile_b, [(1.0, 1.0, 0.0, 9.0), (2.0, 2.0, 0.0, 8.0)])
    tiles = [
        CanonicalTile(path=tile_a, extent=(0.0, 0.0, 10.0, 10.0)),
        CanonicalTile(path=tile_b, extent=(10.0, 0.0, 20.0, 10.0)),
    ]
    oracle_out = tmp_path / "oracle.laz"
    stream_out = tmp_path / "stream.laz"

    oracle_stats = deduplicate_tiles(tiles, oracle_out)
    stream_stats = stream_deduplicate_tiles(
        tiles, stream_out, workdir=tmp_path / "wd", chunk_points=1
    )

    assert stream_stats == oracle_stats
    assert stream_stats.cropped_points == 2
    assert _sha(stream_out) == _sha(oracle_out)


def test_everything_cropped_yields_empty_output(tmp_path: Path) -> None:
    """When the crop removes every point, an empty output is written."""
    tile_a = tmp_path / "a.laz"
    _write_tile(tile_a, [(50.0, 50.0, 0.0, 1.0), (60.0, 60.0, 0.0, 2.0)])
    tiles = [CanonicalTile(path=tile_a, extent=(0.0, 0.0, 10.0, 10.0))]
    oracle_out = tmp_path / "oracle.laz"
    stream_out = tmp_path / "stream.laz"

    oracle_stats = deduplicate_tiles(tiles, oracle_out)
    stream_stats = stream_deduplicate_tiles(
        tiles, stream_out, workdir=tmp_path / "wd"
    )

    assert stream_stats == oracle_stats
    assert stream_stats == DedupStats(2, 0, 0, 0)
    assert int(_read(stream_out).header.point_count) == 0
    assert _sha(stream_out) == _sha(oracle_out)


# --------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------


def test_rejects_an_empty_tile_list(tmp_path: Path) -> None:
    """With no tiles there is nothing to merge, so the call is refused."""
    with pytest.raises(ValueError, match="at least one tile"):
        stream_deduplicate_tiles([], tmp_path / "out.laz")


def test_rejects_non_positive_chunk_points(tmp_path: Path) -> None:
    """A non-positive chunk size is rejected before any I/O."""
    tiles = _dup_tiles(tmp_path)
    with pytest.raises(ValueError, match="chunk_points"):
        stream_deduplicate_tiles(tiles, tmp_path / "out.laz", chunk_points=0)


def test_rejects_a_format_without_gps_time(tmp_path: Path) -> None:
    """A point format lacking gps_time (PDRF < 6) is a clear ValueError."""
    path = tmp_path / "no_gps.laz"
    header = laspy.LasHeader(point_format=0, version="1.2")
    header.offsets = np.zeros(3)
    header.scales = np.array([0.01, 0.01, 0.01])
    las = laspy.LasData(header)
    las.x = np.array([1.0, 2.0])
    las.y = np.array([1.0, 2.0])
    las.z = np.array([0.0, 0.0])
    las.write(str(path))
    tiles = [CanonicalTile(path=path, extent=_EXT)]

    with pytest.raises(ValueError, match="gps_time"):
        stream_deduplicate_tiles(tiles, tmp_path / "out.laz")


# --------------------------------------------------------------------------
# Bounded memory
# --------------------------------------------------------------------------


def test_never_calls_reader_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With LasReader.read patched to raise, the pipeline still completes.

    Proves the whole path streams via ``chunk_iterator`` and never materialises
    a whole tile -- the bounded-memory guarantee.
    """

    def _boom(_self: object, *_a: object, **_k: object) -> object:
        msg = "reader.read() must not be called in the streaming path"
        raise AssertionError(msg)

    tiles = _dup_tiles(tmp_path)
    out = tmp_path / "out.laz"
    monkeypatch.setattr("laspy.lasreader.LasReader.read", _boom)

    stats = stream_deduplicate_tiles(tiles, out, workdir=tmp_path / "wd")

    monkeypatch.undo()  # restore read for the test-side verification below
    assert stats.output_points == 3
    assert _read(out).gps_time.tolist() == [1.0, 5.0, 8.0]


# --------------------------------------------------------------------------
# Stale-spill self-healing (mirrors voxel_stream)
# --------------------------------------------------------------------------


def test_supplied_workdir_clears_a_stale_spill_dir(tmp_path: Path) -> None:
    """A leftover spill directory from a crashed run is recreated empty."""
    workdir = tmp_path / "scratch"
    stale = workdir / _SPILL_SUBDIR
    stale.mkdir(parents=True)
    (stale / "seg_000001.bin").write_bytes(b"garbage from a dead run")
    tiles = _dup_tiles(tmp_path)
    out = tmp_path / "out.laz"

    stats = stream_deduplicate_tiles(tiles, out, workdir=workdir)

    assert stats.output_points == 3
    assert not stale.exists()


def test_supplied_workdir_clears_a_stale_spill_file(tmp_path: Path) -> None:
    """A stale plain FILE at the spill path is replaced, not a raw crash."""
    workdir = tmp_path / "scratch"
    workdir.mkdir()
    (workdir / _SPILL_SUBDIR).write_bytes(b"stale file, not a directory")
    tiles = _dup_tiles(tmp_path)
    out = tmp_path / "out.laz"

    stats = stream_deduplicate_tiles(tiles, out, workdir=workdir)

    assert stats.output_points == 3
    assert not (workdir / _SPILL_SUBDIR).exists()


def test_supplied_workdir_clears_a_stale_spill_symlink(
    tmp_path: Path,
) -> None:
    """A stale symlink at the spill path is unlinked, never followed."""
    workdir = tmp_path / "scratch"
    workdir.mkdir()
    target = tmp_path / "innocent"
    target.mkdir()
    (target / "keep.txt").write_text("do not delete")
    (workdir / _SPILL_SUBDIR).symlink_to(target)
    tiles = _dup_tiles(tmp_path)
    out = tmp_path / "out.laz"

    stats = stream_deduplicate_tiles(tiles, out, workdir=workdir)

    assert stats.output_points == 3
    assert (target / "keep.txt").exists()  # link removed, not the target
    assert not (workdir / _SPILL_SUBDIR).exists()


# --------------------------------------------------------------------------
# Disk floor -- breach at every write site cleans up and leaves output untouched
# --------------------------------------------------------------------------


def test_disk_floor_breach_at_startup_raises_and_cleans_up(
    tmp_path: Path,
) -> None:
    """A floor breach before any spill removes the spill dir; no output."""
    tiles = _dup_tiles(tmp_path)
    out = tmp_path / "out.laz"
    workdir = tmp_path / "scratch"

    with pytest.raises(DiskFloorError):
        stream_deduplicate_tiles(
            tiles, out, workdir=workdir, min_free_bytes=2**62
        )

    assert not out.exists()
    assert not (workdir / _SPILL_SUBDIR).exists()


def _fake_ensure_at(
    monkeypatch: pytest.MonkeyPatch, breach_call: int
) -> None:
    """Patch dedup_stream.ensure_free_disk to breach on the ``breach_call``-th call."""
    real = dedup_stream_module.ensure_free_disk
    calls = 0

    def _fake(
        directory: Path, incoming_bytes: int = 0, *, min_free_bytes: int
    ) -> None:
        nonlocal calls
        calls += 1
        if calls == breach_call:
            msg = "synthetic disk floor breach"
            raise DiskFloorError(msg)
        real(directory, incoming_bytes, min_free_bytes=min_free_bytes)

    monkeypatch.setattr(dedup_stream_module, "ensure_free_disk", _fake)


def test_disk_floor_breach_at_segment_spill_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A breach at the pass-1 segment append removes the spill; no output.

    Calls 1-2 are the workdir/output-dir guards; call 3 is the single segment
    flush for this small fixture.
    """
    _fake_ensure_at(monkeypatch, 3)
    tiles = _dup_tiles(tmp_path)
    out = tmp_path / "out.laz"
    workdir = tmp_path / "scratch"

    with pytest.raises(DiskFloorError):
        stream_deduplicate_tiles(tiles, out, workdir=workdir)

    assert not out.exists()
    assert not (workdir / _SPILL_SUBDIR).exists()


def test_disk_floor_breach_at_output_chunk_cleans_up_the_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A breach at the first pass-5 chunk write removes the partial temp output.

    Calls: 1-2 startup guards, 3 segment flush, then (partition/reduce/merge use
    spill's own guard, not this one) 4 the first output-chunk write.
    """
    _fake_ensure_at(monkeypatch, 4)
    tiles = _dup_tiles(tmp_path)
    out = tmp_path / "out.laz"
    workdir = tmp_path / "scratch"

    with pytest.raises(DiskFloorError):
        stream_deduplicate_tiles(tiles, out, workdir=workdir)

    assert not out.exists()
    assert not (tmp_path / "out.tmp.laz").exists()
    assert not (workdir / _SPILL_SUBDIR).exists()


def test_disk_floor_breach_at_finalisation_cleans_up_the_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A breach at the close-time finalisation guard still cleans the temp."""
    real = dedup_stream_module.ensure_free_disk

    def _fail_at_finalisation(
        directory: Path, incoming_bytes: int = 0, *, min_free_bytes: int
    ) -> None:
        if incoming_bytes == _FINALIZE_HEADROOM:
            msg = "synthetic finalisation floor breach"
            raise DiskFloorError(msg)
        real(directory, incoming_bytes, min_free_bytes=min_free_bytes)

    monkeypatch.setattr(
        dedup_stream_module, "ensure_free_disk", _fail_at_finalisation
    )
    tiles = _dup_tiles(tmp_path)
    out = tmp_path / "out.laz"

    with pytest.raises(DiskFloorError, match="finalisation"):
        stream_deduplicate_tiles(tiles, out, workdir=tmp_path / "wd")

    assert not out.exists()
    assert not (tmp_path / "out.tmp.laz").exists()


def test_finalisation_guard_error_wins_over_a_failing_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard's typed error is not masked by a close that also fails."""
    real = dedup_stream_module.ensure_free_disk

    def _fail_at_finalisation(
        directory: Path, incoming_bytes: int = 0, *, min_free_bytes: int
    ) -> None:
        if incoming_bytes == _FINALIZE_HEADROOM:
            msg = "synthetic finalisation floor breach"
            raise DiskFloorError(msg)
        real(directory, incoming_bytes, min_free_bytes=min_free_bytes)

    def _explode_close(_self: object) -> None:
        msg = "no space left on device"
        raise OSError(msg)

    tiles = _dup_tiles(tmp_path)
    out = tmp_path / "out.laz"
    monkeypatch.setattr(
        dedup_stream_module, "ensure_free_disk", _fail_at_finalisation
    )
    monkeypatch.setattr("laspy.laswriter.LasWriter.close", _explode_close)

    with pytest.raises(DiskFloorError, match="finalisation"):
        stream_deduplicate_tiles(tiles, out, workdir=tmp_path / "wd")

    assert not out.exists()
    assert not (tmp_path / "out.tmp.laz").exists()


def test_generic_write_failure_cleans_up_the_temp_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-floor mid-write failure removes the partial temp; output untouched."""

    def _explode_write(_self: object, _points: object) -> None:
        msg = "synthetic mid-write failure"
        raise RuntimeError(msg)

    tiles = _dup_tiles(tmp_path)
    out = tmp_path / "out.laz"
    monkeypatch.setattr(
        "laspy.laswriter.LasWriter.write_points", _explode_write
    )

    with pytest.raises(RuntimeError, match="synthetic mid-write failure"):
        stream_deduplicate_tiles(tiles, out, workdir=tmp_path / "wd")

    assert not out.exists()
    assert not (tmp_path / "out.tmp.laz").exists()
    assert not (tmp_path / "wd" / _SPILL_SUBDIR).exists()
