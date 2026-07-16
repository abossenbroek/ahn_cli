"""Tests for the dependency-free external-spill primitives in `ahn_cli.prep.spill`.

Drives `ensure_free_disk`, `advise_no_cache`, `PartitionWriter`,
`write_sorted_run`, `merge_sorted_runs`, and `iter_sorted_values` to full
branch coverage. Merge correctness is checked against a `numpy.sort` oracle
over the concatenation of every input run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import pytest

from ahn_cli.prep import spill as spill_module
from ahn_cli.prep.spill import (
    DiskFloorError,
    PartitionWriter,
    advise_no_cache,
    ensure_free_disk,
    iter_sorted_values,
    merge_sorted_runs,
    write_sorted_run,
)

if TYPE_CHECKING:
    from pathlib import Path

_RECORD_DTYPE = np.dtype([("idx", "<i8")])


def _records(values: list[int]) -> np.ndarray:
    """Build a tiny structured array of the shared test record dtype."""
    arr = np.zeros(len(values), dtype=_RECORD_DTYPE)
    arr["idx"] = values
    return arr


class _FakeUsage(NamedTuple):
    """Minimal stand-in for `shutil.disk_usage`'s return value."""

    total: int
    used: int
    free: int


def test_ensure_free_disk_passes_when_headroom_remains(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No error when free space minus the incoming write clears the floor."""

    def _fake_disk_usage(_directory: Path) -> _FakeUsage:
        return _FakeUsage(0, 0, 100)

    monkeypatch.setattr(spill_module.shutil, "disk_usage", _fake_disk_usage)

    ensure_free_disk(tmp_path, 10, min_free_bytes=50)  # 100 - 10 = 90 >= 50


def test_ensure_free_disk_raises_when_floor_would_be_breached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Raises when free space minus the incoming write dips below the floor."""

    def _fake_disk_usage(_directory: Path) -> _FakeUsage:
        return _FakeUsage(0, 0, 100)

    monkeypatch.setattr(spill_module.shutil, "disk_usage", _fake_disk_usage)

    with pytest.raises(DiskFloorError, match="below the 50 byte floor"):
        ensure_free_disk(
            tmp_path, 60, min_free_bytes=50
        )  # 100 - 60 = 40 < 50


def test_advise_no_cache_is_a_no_op_without_f_nocache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No fcntl call is made when the platform lacks F_NOCACHE."""
    monkeypatch.setattr(spill_module, "_F_NOCACHE", None)
    calls: list[tuple[object, ...]] = []

    def _fake_fcntl(*args: object) -> None:
        calls.append(args)

    monkeypatch.setattr(spill_module.fcntl, "fcntl", _fake_fcntl)
    with (tmp_path / "f.bin").open("wb") as handle:
        advise_no_cache(handle)

    assert calls == []


def test_advise_no_cache_issues_fcntl_when_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issues fcntl(fd, F_NOCACHE, 1) when the platform defines F_NOCACHE."""
    monkeypatch.setattr(spill_module, "_F_NOCACHE", 48)  # macOS's real value
    calls: list[tuple[int, int, int]] = []

    def _fake_fcntl(fd: int, cmd: int, arg: int) -> None:
        calls.append((fd, cmd, arg))

    monkeypatch.setattr(spill_module.fcntl, "fcntl", _fake_fcntl)
    with (tmp_path / "f.bin").open("wb") as handle:
        fd = handle.fileno()
        advise_no_cache(handle)

    assert calls == [(fd, 48, 1)]


def test_partition_writer_splits_records_across_appends(
    tmp_path: Path,
) -> None:
    """Records route to the correct partition file across multiple appends."""
    writer = PartitionWriter(tmp_path, partition_count=3)

    writer.append(np.array([0, 2, 1], dtype=np.int64), _records([10, 30, 20]))
    writer.append(np.array([1, 0], dtype=np.int64), _records([21, 11]))
    paths = writer.close()

    assert [path.name for path in paths] == [
        "partition_0000.bin",
        "partition_0001.bin",
        "partition_0002.bin",
    ]
    assert np.fromfile(paths[0], dtype=_RECORD_DTYPE)["idx"].tolist() == [
        10,
        11,
    ]
    assert np.fromfile(paths[1], dtype=_RECORD_DTYPE)["idx"].tolist() == [
        20,
        21,
    ]
    assert np.fromfile(paths[2], dtype=_RECORD_DTYPE)["idx"].tolist() == [30]


def test_partition_writer_flushes_at_the_buffer_cap(tmp_path: Path) -> None:
    """A tiny buffer_bytes cap forces a flush mid-run, appending across flushes."""
    writer = PartitionWriter(
        tmp_path, partition_count=1, buffer_bytes=_RECORD_DTYPE.itemsize
    )

    writer.append(
        np.array([0], dtype=np.int64), _records([1])
    )  # triggers a flush
    writer.append(np.array([0], dtype=np.int64), _records([2]))
    paths = writer.close()

    assert np.fromfile(paths[0], dtype=_RECORD_DTYPE)["idx"].tolist() == [
        1,
        2,
    ]


def test_partition_writer_close_omits_empty_partitions(
    tmp_path: Path,
) -> None:
    """A partition that never received a record has no file and is omitted."""
    writer = PartitionWriter(tmp_path, partition_count=3)

    writer.append(np.array([1], dtype=np.int64), _records([42]))
    paths = writer.close()

    assert [path.name for path in paths] == ["partition_0001.bin"]


def test_partition_writer_ignores_an_empty_append(tmp_path: Path) -> None:
    """Appending zero records is a no-op (covers the early-return branch)."""
    writer = PartitionWriter(tmp_path, partition_count=2)

    writer.append(np.empty(0, dtype=np.int64), _records([]))

    assert writer.close() == []


def test_partition_writer_flush_propagates_disk_floor_error(
    tmp_path: Path,
) -> None:
    """A flush that would breach the disk floor raises, not writes."""
    writer = PartitionWriter(
        tmp_path, partition_count=1, min_free_bytes=2**62
    )

    writer.append(np.array([0], dtype=np.int64), _records([1]))
    with pytest.raises(DiskFloorError):
        writer.close()


def test_write_sorted_run_writes_the_values_as_is(tmp_path: Path) -> None:
    """A sorted run is written verbatim as little-endian int64."""
    path = tmp_path / "run.bin"

    write_sorted_run(path, np.array([1, 4, 9], dtype=np.int64))

    assert np.fromfile(path, dtype="<i8").tolist() == [1, 4, 9]


def test_write_sorted_run_propagates_disk_floor_error(tmp_path: Path) -> None:
    """A run write that would breach the disk floor raises, not writes."""
    path = tmp_path / "run.bin"

    with pytest.raises(DiskFloorError):
        write_sorted_run(
            path, np.array([1, 2], dtype=np.int64), min_free_bytes=2**62
        )
    assert not path.exists()


def test_merge_sorted_runs_single_run_passes_through(tmp_path: Path) -> None:
    """A lone run is returned unchanged -- no merge pass, nothing deleted."""
    run = tmp_path / "run.bin"
    write_sorted_run(run, np.array([1, 2, 3], dtype=np.int64))

    result = merge_sorted_runs([run], tmp_path / "out")

    assert result == run
    assert run.exists()
    assert np.fromfile(result, dtype="<i8").tolist() == [1, 2, 3]


def test_merge_sorted_runs_merges_multiple_runs(tmp_path: Path) -> None:
    """Several runs merge into one sorted stream, matching a numpy oracle."""
    rng = np.random.default_rng(0)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    runs: list[Path] = []
    all_values: list[int] = []
    for i in range(5):
        values = np.sort(rng.choice(10_000, size=37, replace=False)).astype(
            np.int64
        )
        all_values.extend(values.tolist())
        path = tmp_path / f"run_{i}.bin"
        write_sorted_run(path, values)
        runs.append(path)

    result = merge_sorted_runs(runs, out_dir)

    assert np.fromfile(result, dtype="<i8").tolist() == sorted(all_values)
    for path in runs:
        assert not path.exists()  # consumed inputs are deleted


def test_merge_sorted_runs_forces_multiple_passes_with_small_fan_in(
    tmp_path: Path,
) -> None:
    """fan_in smaller than the run count requires more than one merge pass."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    runs: list[Path] = []
    all_values: list[int] = []
    for i, values in enumerate([[1, 4], [2, 5], [3, 6]]):
        all_values.extend(values)
        path = tmp_path / f"run_{i}.bin"
        write_sorted_run(path, np.array(values, dtype=np.int64))
        runs.append(path)

    result = merge_sorted_runs(runs, out_dir, fan_in=2)

    assert np.fromfile(result, dtype="<i8").tolist() == sorted(all_values)


def test_merge_sorted_runs_interleaved_values_hit_the_multi_piece_branch(
    tmp_path: Path,
) -> None:
    """Interleaved runs make a merge round draw from more than one buffer."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    run_a = tmp_path / "a.bin"
    run_b = tmp_path / "b.bin"
    write_sorted_run(run_a, np.array([1, 3, 5], dtype=np.int64))
    write_sorted_run(run_b, np.array([2, 4, 6], dtype=np.int64))

    result = merge_sorted_runs([run_a, run_b], out_dir)

    assert np.fromfile(result, dtype="<i8").tolist() == [1, 2, 3, 4, 5, 6]


def test_merge_sorted_runs_small_buffer_forces_mid_run_refill(
    tmp_path: Path,
) -> None:
    """A buffer_items smaller than a run's length forces a mid-run disk refill."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    run_a = tmp_path / "a.bin"
    run_b = tmp_path / "b.bin"
    write_sorted_run(run_a, np.arange(10, dtype=np.int64))
    write_sorted_run(run_b, np.arange(10, 20, dtype=np.int64))

    result = merge_sorted_runs([run_a, run_b], out_dir, buffer_items=2)

    assert np.fromfile(result, dtype="<i8").tolist() == list(range(20))


def test_merge_sorted_runs_empty_runs_yields_an_empty_file(
    tmp_path: Path,
) -> None:
    """No input runs produces a new, empty merged file."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    result = merge_sorted_runs([], out_dir)

    assert result.exists()
    assert np.fromfile(result, dtype="<i8").tolist() == []


def test_merge_sorted_runs_propagates_disk_floor_error(
    tmp_path: Path,
) -> None:
    """A merge-output write that would breach the disk floor raises."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    run_a = tmp_path / "a.bin"
    run_b = tmp_path / "b.bin"
    write_sorted_run(run_a, np.array([1], dtype=np.int64))
    write_sorted_run(run_b, np.array([2], dtype=np.int64))

    with pytest.raises(DiskFloorError):
        merge_sorted_runs([run_a, run_b], out_dir, min_free_bytes=2**62)


def test_iter_sorted_values_yields_sequential_blocks(tmp_path: Path) -> None:
    """Reading in small blocks yields every value, in order, across blocks."""
    path = tmp_path / "run.bin"
    write_sorted_run(path, np.arange(10, dtype=np.int64))

    blocks = list(iter_sorted_values(path, buffer_items=3))

    assert [block.tolist() for block in blocks] == [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [9],
    ]


def test_iter_sorted_values_empty_file_yields_nothing(tmp_path: Path) -> None:
    """An empty run file yields no blocks."""
    path = tmp_path / "run.bin"
    write_sorted_run(path, np.empty(0, dtype=np.int64))

    assert list(iter_sorted_values(path)) == []


def test_partition_writer_failed_flush_keeps_records_for_a_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mid-flush failure loses nothing and a retry writes each record once.

    The second partition's floor check fails: its file must not appear (the
    floor is checked before the file is even opened), its records must stay
    buffered, and a retried ``close()`` must write them exactly once while
    not duplicating the first partition's already-flushed records.
    """
    writer = PartitionWriter(tmp_path, 2)
    writer.append(np.array([0, 1], dtype=np.int64), _records([10, 11]))
    real_ensure_free_disk = spill_module.ensure_free_disk
    calls = 0

    def _fail_second_check(
        directory: Path, incoming_bytes: int = 0, *, min_free_bytes: int
    ) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            msg = "synthetic floor breach"
            raise DiskFloorError(msg)
        real_ensure_free_disk(
            directory, incoming_bytes, min_free_bytes=min_free_bytes
        )

    monkeypatch.setattr(spill_module, "ensure_free_disk", _fail_second_check)
    with pytest.raises(DiskFloorError):
        writer.close()
    assert not (tmp_path / "partition_0001.bin").exists()

    monkeypatch.setattr(
        spill_module, "ensure_free_disk", real_ensure_free_disk
    )
    paths = writer.close()

    assert [path.name for path in paths] == [
        "partition_0000.bin",
        "partition_0001.bin",
    ]
    assert np.fromfile(paths[0], dtype=_RECORD_DTYPE)["idx"].tolist() == [10]
    assert np.fromfile(paths[1], dtype=_RECORD_DTYPE)["idx"].tolist() == [11]
