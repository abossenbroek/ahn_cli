"""Tests for out-of-core streaming voxel thinning (`ahn_cli.prep.voxel_stream`).

These lock the contract the streaming path shares with the in-memory reference
in :mod:`ahn_cli.prep.decimate`: within each occupied voxel the survivor is the
point with the smallest index in the class-filtered cloud, survivors are written
in ascending index order, every attribute is preserved, and identical input
yields byte-identical output -- all while never holding more than one chunk of
points in memory. Each correctness assertion is checked against a numpy oracle
that replays the intended semantics.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import laspy
import numpy as np
import pytest

from ahn_cli.prep import voxel_stream as voxel_stream_module
from ahn_cli.prep.decimate import (
    NumpyBackend,
    decimate_voxel,
    voxel_size_for_grade,
)
from ahn_cli.prep.spill import DiskFloorError
from ahn_cli.prep.voxel_stream import stream_voxel_thin

_SPILL_SUBDIR = "voxel_spill"  # mirrors voxel_stream._SPILL_SUBDIR
_FINALIZE_HEADROOM = (
    4 * 1024**2
)  # mirrors voxel_stream._FINALIZE_HEADROOM_BYTES

if TYPE_CHECKING:
    from pathlib import Path

Point = tuple[float, float, float, float, int]  # x, y, z, gps_time, class

_GRADE_1M = 3  # voxel edge length 1.0 m (see decimate._VOXEL_SIZES)


def _write_laz(
    path: Path, points: list[Point], *, scale: float = 0.01
) -> None:
    """Write a synthetic format-6 (gps_time + classification) LAZ file."""
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.offsets = np.array([0.0, 0.0, 0.0], dtype=float)
    header.scales = np.array([scale, scale, scale], dtype=float)
    las = laspy.LasData(header)
    arr = np.array(points, dtype=float)
    las.x = arr[:, 0]
    las.y = arr[:, 1]
    las.z = arr[:, 2]
    las.gps_time = arr[:, 3]
    las.classification = arr[:, 4].astype(np.uint8)
    las.write(str(path))


def _read(path: Path) -> laspy.LasData:
    """Read a LAZ file fully into memory."""
    with laspy.open(str(path)) as reader:
        return reader.read()


def _oracle_gps(
    points: list[Point],
    include: tuple[int, ...],
    exclude: tuple[int, ...],
    grade: int,
) -> list[float]:
    """Return the survivors' gps_time in written order, computed independently.

    Replays the intended semantics in numpy: class-filter (preserving order),
    then -- for a non-identity grade -- keep the smallest filtered index in each
    voxel anchored at the filtered coordinates' minimum, emitted in ascending
    filtered-index order. gps_time uniquely tags each input point.
    """
    arr = np.array(points, dtype=float)
    cls = arr[:, 4].astype(int)
    keep = np.ones(len(arr), dtype=bool)
    if include:
        keep &= np.isin(cls, list(include))
    if exclude:
        keep &= ~np.isin(cls, list(exclude))
    coords = arr[keep][:, :3]
    gps = arr[keep][:, 3]
    size = voxel_size_for_grade(grade)
    if size == 0.0:
        return gps.tolist()
    origin = coords.min(axis=0)
    cells = np.floor((coords - origin) / size).astype(np.int64)
    survivors: dict[tuple[int, int, int], int] = {}
    for index in range(len(coords)):
        cell = (
            int(cells[index, 0]),
            int(cells[index, 1]),
            int(cells[index, 2]),
        )
        survivors.setdefault(cell, index)  # first = smallest index
    return gps[sorted(survivors.values())].tolist()


# A voxel-A cluster (points within one 1 m cell) plus one isolated voxel-B point.
# Ordered so voxel B's survivor sits between voxel A's survivor and its dups.
_CLOUD: list[Point] = [
    (0.1, 0.1, 0.0, 100.0, 2),  # idx0: voxel A survivor
    (5.0, 5.0, 0.0, 101.0, 2),  # idx1: voxel B survivor
    (0.9, 0.2, 0.0, 102.0, 2),  # idx2: voxel A duplicate
    (0.2, 0.8, 0.0, 103.0, 6),  # idx3: voxel A duplicate (class 6)
]


def test_keeps_min_index_survivor_per_voxel(tmp_path: Path) -> None:
    """One survivor per occupied voxel, the smallest-index point, in order."""
    src = tmp_path / "src.laz"
    out = tmp_path / "out.laz"
    _write_laz(src, _CLOUD)

    count = stream_voxel_thin(src, out, _GRADE_1M, (), ())

    assert count == 2
    result = _read(out)
    assert result.gps_time.tolist() == _oracle_gps(_CLOUD, (), (), _GRADE_1M)
    assert result.gps_time.tolist() == [100.0, 101.0]
    assert result.classification.tolist() == [2, 2]  # attributes preserved


def test_in_place_thinning_replaces_source(tmp_path: Path) -> None:
    """Source and output may be the same path (the pipeline's usage)."""
    path = tmp_path / "cloud.laz"
    _write_laz(path, _CLOUD)

    count = stream_voxel_thin(path, path, _GRADE_1M, (), ())

    assert count == 2
    assert _read(path).gps_time.tolist() == [100.0, 101.0]


def test_include_filter_applies_before_grouping(tmp_path: Path) -> None:
    """An include filter drops other classes before voxel grouping."""
    src = tmp_path / "src.laz"
    out = tmp_path / "out.laz"
    _write_laz(src, _CLOUD)

    stream_voxel_thin(src, out, _GRADE_1M, (2,), ())

    assert _read(out).gps_time.tolist() == _oracle_gps(
        _CLOUD, (2,), (), _GRADE_1M
    )


def test_exclude_filter_applies_before_grouping(tmp_path: Path) -> None:
    """An exclude filter drops the listed class before voxel grouping."""
    src = tmp_path / "src.laz"
    out = tmp_path / "out.laz"
    _write_laz(src, _CLOUD)

    stream_voxel_thin(src, out, _GRADE_1M, (), (6,))

    assert _read(out).gps_time.tolist() == _oracle_gps(
        _CLOUD, (), (6,), _GRADE_1M
    )


def test_grade_zero_is_class_filtered_identity(tmp_path: Path) -> None:
    """Grade 0 keeps every class-included point unchanged (no thinning)."""
    src = tmp_path / "src.laz"
    out = tmp_path / "out.laz"
    _write_laz(src, _CLOUD)

    count = stream_voxel_thin(src, out, 0, (2,), ())

    assert count == 3  # the three class-2 points, voxel A not collapsed
    assert _read(out).gps_time.tolist() == _oracle_gps(_CLOUD, (2,), (), 0)


def test_survivor_spans_a_chunk_boundary(tmp_path: Path) -> None:
    """A voxel split across chunks still keeps the globally smallest index.

    With ``chunk_points=2`` the voxel-A survivor (idx0) and its duplicates land
    in different chunks; the second chunk holds only duplicates, exercising the
    empty-write branch, while the survivor from the first chunk is retained.
    """
    src = tmp_path / "src.laz"
    out = tmp_path / "out.laz"
    _write_laz(src, _CLOUD)

    count = stream_voxel_thin(src, out, _GRADE_1M, (), (), chunk_points=2)

    assert count == 2
    assert _read(out).gps_time.tolist() == [100.0, 101.0]


def test_fully_excluded_chunk_writes_no_spill(tmp_path: Path) -> None:
    """A chunk whose points are all class-excluded contributes nothing."""
    src = tmp_path / "src.laz"
    out = tmp_path / "out.laz"
    cloud: list[Point] = [
        (0.1, 0.1, 0.0, 100.0, 2),  # chunk 1
        (5.0, 5.0, 0.0, 101.0, 2),  # chunk 1
        (2.0, 2.0, 0.0, 102.0, 9),  # chunk 2 -- excluded
        (3.0, 3.0, 0.0, 103.0, 9),  # chunk 2 -- excluded
    ]
    _write_laz(src, cloud)

    count = stream_voxel_thin(src, out, _GRADE_1M, (), (9,), chunk_points=2)

    assert count == 2
    assert _read(out).gps_time.tolist() == [100.0, 101.0]


def test_everything_filtered_out_yields_empty_output(tmp_path: Path) -> None:
    """When the class filter removes every point, the output is empty."""
    src = tmp_path / "src.laz"
    out = tmp_path / "out.laz"
    _write_laz(
        src,
        [
            (0.1, 0.1, 0.0, 100.0, 9),
            (5.0, 5.0, 0.0, 101.0, 9),
        ],
    )

    count = stream_voxel_thin(src, out, _GRADE_1M, (), (9,))

    assert count == 0
    assert int(_read(out).header.point_count) == 0


def test_supplied_workdir_clears_a_stale_spill(tmp_path: Path) -> None:
    """A leftover spill dir from a crashed prior run is recreated empty."""
    src = tmp_path / "src.laz"
    out = tmp_path / "out.laz"
    workdir = tmp_path / "scratch"
    stale = workdir / _SPILL_SUBDIR
    stale.mkdir(parents=True)
    (stale / "seg_000001.bin").write_bytes(b"garbage from a dead run")
    _write_laz(src, _CLOUD)

    count = stream_voxel_thin(src, out, _GRADE_1M, (), (), workdir=workdir)

    assert count == 2
    assert _read(out).gps_time.tolist() == [100.0, 101.0]
    assert not stale.exists()  # spill is cleaned up on completion


def test_output_is_deterministic(tmp_path: Path) -> None:
    """Two independent runs produce byte-identical output."""
    src = tmp_path / "src.laz"
    _write_laz(src, _CLOUD)
    out_a = tmp_path / "a.laz"
    out_b = tmp_path / "b.laz"

    stream_voxel_thin(src, out_a, _GRADE_1M, (), ())
    stream_voxel_thin(src, out_b, _GRADE_1M, (), ())

    assert (
        hashlib.sha256(out_a.read_bytes()).hexdigest()
        == hashlib.sha256(out_b.read_bytes()).hexdigest()
    )


def test_progress_is_reported_per_chunk(tmp_path: Path) -> None:
    """The progress callback ticks once per streamed chunk to completion."""
    src = tmp_path / "src.laz"
    out = tmp_path / "out.laz"
    _write_laz(src, _CLOUD)
    ticks: list[tuple[int, int]] = []

    stream_voxel_thin(
        src,
        out,
        _GRADE_1M,
        (),
        (),
        chunk_points=2,
        progress=lambda done, total: ticks.append((done, total)),
    )

    assert ticks == [(1, 2), (2, 2)]


def test_rejects_non_positive_chunk_points(tmp_path: Path) -> None:
    """A non-positive chunk size is rejected before any I/O."""
    src = tmp_path / "src.laz"
    out = tmp_path / "out.laz"
    _write_laz(src, _CLOUD)

    with pytest.raises(ValueError, match="chunk_points"):
        stream_voxel_thin(src, out, _GRADE_1M, (), (), chunk_points=0)


def test_matches_the_in_memory_voxel_reference_on_a_random_cloud(
    tmp_path: Path,
) -> None:
    """Streaming survivors match `decimate.decimate_voxel` on the same data.

    Builds a random cloud, streams it through the class filter and voxel
    thin, then re-derives the expected survivors by reading the *stored*
    (scale-quantized) cloud back and running the in-memory reference over
    it -- an independent cross-check of the whole spill/partition/reduce
    pipeline against the algorithm it is required to match.
    """
    rng = np.random.default_rng(0)
    n = 2000
    coords = rng.uniform(0, 50, size=(n, 3))
    classes = rng.choice([2, 6, 9], size=n)
    points: list[Point] = [
        (
            float(coords[i, 0]),
            float(coords[i, 1]),
            float(coords[i, 2]),
            float(i),
            int(classes[i]),
        )
        for i in range(n)
    ]
    src = tmp_path / "src.laz"
    out = tmp_path / "out.laz"
    _write_laz(src, points)
    include = (2, 6)
    grade = 3

    stream_voxel_thin(src, out, grade, include, (), chunk_points=257)

    stored = _read(src)
    keep = np.isin(np.asarray(stored.classification), include)
    stored_coords = np.column_stack(
        [np.asarray(stored.x), np.asarray(stored.y), np.asarray(stored.z)]
    )[keep].astype(np.float64)
    stored_gps = np.asarray(stored.gps_time)[keep]
    expected_indices = decimate_voxel(
        stored_coords, grade, backend=NumpyBackend()
    )

    assert (
        _read(out).gps_time.tolist() == stored_gps[expected_indices].tolist()
    )


def test_multi_segment_and_multi_partition_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Shrinking the segment/partition size constants forces several of each."""
    monkeypatch.setattr(voxel_stream_module, "_SEGMENT_BYTES", 40)
    monkeypatch.setattr(voxel_stream_module, "_PARTITION_TARGET_BYTES", 40)
    src = tmp_path / "src.laz"
    out = tmp_path / "out.laz"
    _write_laz(src, _CLOUD)

    count = stream_voxel_thin(src, out, _GRADE_1M, (), (), chunk_points=1)

    assert count == 2
    assert _read(out).gps_time.tolist() == [100.0, 101.0]


def test_disk_floor_breach_raises_and_cleans_up(tmp_path: Path) -> None:
    """A disk-floor breach raises, removes the spill dir, and leaves output untouched."""
    src = tmp_path / "src.laz"
    out = tmp_path / "out.laz"
    workdir = tmp_path / "scratch"
    _write_laz(src, _CLOUD)

    with pytest.raises(DiskFloorError):
        stream_voxel_thin(
            src, out, _GRADE_1M, (), (), workdir=workdir, min_free_bytes=2**62
        )

    assert not out.exists()
    assert not (workdir / _SPILL_SUBDIR).exists()


def test_extreme_extent_raises_int32_cell_range_error(tmp_path: Path) -> None:
    """A voxel cell coordinate outside int32 range is a clear ValueError.

    Two points 3e9 m apart on x, with a 1.0 m LAS scale (so both raw LAS
    ints stay comfortably inside int32) but a 0.25 m voxel size (grade 1):
    the resulting cell index is far outside int32 range.
    """
    src = tmp_path / "src.laz"
    out = tmp_path / "out.laz"
    _write_laz(
        src,
        [
            (-1_500_000_000.0, 0.0, 0.0, 100.0, 2),
            (1_500_000_000.0, 0.0, 0.0, 101.0, 2),
        ],
        scale=1.0,
    )

    with pytest.raises(ValueError, match="int32 range"):
        stream_voxel_thin(src, out, 1, (), ())


def test_reduce_pass_handles_a_singleton_partition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A partition holding exactly one record skips the boundary-diff slice."""
    monkeypatch.setattr(voxel_stream_module, "_PARTITION_TARGET_BYTES", 1)
    src = tmp_path / "src.laz"
    out = tmp_path / "out.laz"
    _write_laz(src, _CLOUD)

    count = stream_voxel_thin(src, out, _GRADE_1M, (), ())

    assert count == 2
    assert _read(out).gps_time.tolist() == [100.0, 101.0]


def test_disk_floor_breach_during_output_write_cleans_up_the_temp_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A disk-floor breach mid pass-5 write removes the partial temp output.

    Forces the second ``ensure_free_disk`` call (the grade-0 identity path's
    only calls: the top-level output-dir guard, then one per written chunk)
    to raise, so the breach happens inside :func:`_write_pass`'s own
    try/finally rather than before it is even entered.
    """
    src = tmp_path / "src.laz"
    out = tmp_path / "out.laz"
    _write_laz(src, _CLOUD)
    real_ensure_free_disk = voxel_stream_module.ensure_free_disk
    calls = 0

    def _fake_ensure_free_disk(
        directory: Path, incoming_bytes: int = 0, *, min_free_bytes: int
    ) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            msg = "synthetic disk floor breach"
            raise DiskFloorError(msg)
        real_ensure_free_disk(
            directory, incoming_bytes, min_free_bytes=min_free_bytes
        )

    monkeypatch.setattr(
        voxel_stream_module, "ensure_free_disk", _fake_ensure_free_disk
    )

    with pytest.raises(DiskFloorError):
        stream_voxel_thin(
            src, out, 0, (), ()
        )  # grade 0: straight to _write_pass

    assert not out.exists()
    assert not (tmp_path / "out.tmp.laz").exists()


def test_generic_write_failure_cleans_up_the_temp_file(
    tmp_path: Path,
) -> None:
    """Any mid-write failure removes the partial temp and keeps the original.

    A non-:class:`DiskFloorError` failure (here: the progress callback
    raising) must exercise :func:`_write_pass`'s ``finally`` cleanup, not
    just the floor-breach path. Run in place (``source == output``) to also
    prove the original file survives byte-identical.
    """
    path = tmp_path / "cloud.laz"
    _write_laz(path, _CLOUD)
    before = path.read_bytes()

    def _explode(_done: int, _total: int) -> None:
        msg = "synthetic mid-write failure"
        raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="synthetic mid-write failure"):
        stream_voxel_thin(path, path, _GRADE_1M, (), (), progress=_explode)

    assert path.read_bytes() == before
    assert not (tmp_path / "cloud.tmp.laz").exists()


def test_supplied_workdir_clears_a_stale_spill_file(tmp_path: Path) -> None:
    """A stale plain FILE at the spill path is replaced, not a raw crash.

    Nothing in the pipeline writes a file at that exact path, but a crashed
    or foreign process might; ``rmtree`` alone would raise
    ``NotADirectoryError``.
    """
    src = tmp_path / "src.laz"
    out = tmp_path / "out.laz"
    workdir = tmp_path / "scratch"
    workdir.mkdir()
    (workdir / _SPILL_SUBDIR).write_bytes(b"stale file, not a directory")
    _write_laz(src, _CLOUD)

    count = stream_voxel_thin(src, out, _GRADE_1M, (), (), workdir=workdir)

    assert count == 2
    assert _read(out).gps_time.tolist() == [100.0, 101.0]
    assert not (workdir / _SPILL_SUBDIR).exists()


def test_disk_floor_breach_at_writer_finalisation_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A breach at the close-time finalisation guard still cleans the temp.

    The guard is recognised by its fixed headroom byte count -- the one
    ``ensure_free_disk`` call the per-chunk loop doesn't issue.
    """
    src = tmp_path / "src.laz"
    out = tmp_path / "out.laz"
    _write_laz(src, _CLOUD)
    real_ensure_free_disk = voxel_stream_module.ensure_free_disk

    def _fail_at_finalisation(
        directory: Path, incoming_bytes: int = 0, *, min_free_bytes: int
    ) -> None:
        if incoming_bytes == _FINALIZE_HEADROOM:
            msg = "synthetic finalisation floor breach"
            raise DiskFloorError(msg)
        real_ensure_free_disk(
            directory, incoming_bytes, min_free_bytes=min_free_bytes
        )

    monkeypatch.setattr(
        voxel_stream_module, "ensure_free_disk", _fail_at_finalisation
    )

    with pytest.raises(DiskFloorError, match="finalisation"):
        stream_voxel_thin(src, out, 0, (), ())

    assert not out.exists()
    assert not (tmp_path / "out.tmp.laz").exists()
