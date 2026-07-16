"""Dependency-free external-spill machinery for out-of-core reductions.

Reusable numpy+stdlib primitives for spilling intermediate data to disk under
a scratch directory and reducing it back down, independent of any particular
caller's record layout. :mod:`ahn_cli.prep.voxel_stream` is the first (and
currently only) consumer -- see ``docs/specs/voxel-spill-design.md`` for the
design this module implements.

Two concerns are addressed:

1. **A hard free-space floor.** Every write site in a caller built on this
   module is expected to call :func:`ensure_free_disk` with the exact byte
   count it is about to write, before writing it. A breach raises
   :class:`DiskFloorError` instead of letting the run fill the volume --
   important on macOS, where a full system volume degrades and can break the
   machine.
2. **Bounded-memory partition/merge primitives.** :class:`PartitionWriter`
   fans a stream of records out to per-partition files with a capped
   in-memory buffer; :func:`write_sorted_run`, :func:`merge_sorted_runs`, and
   :func:`iter_sorted_values` write, k-way merge, and re-read sorted ``<i8``
   index runs -- the grace-hash partitioned aggregation's tiny survivor-index
   side of the work (the bulk records never need a second sort).
"""

from __future__ import annotations

import fcntl
import shutil
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path
    from typing import BinaryIO

    import numpy.typing as npt

MIN_FREE_DISK_BYTES = 20_000_000_000
"""Default free-space floor (decimal bytes, the unit macOS reports)."""

_RUN_DTYPE = np.dtype("<i8")
"""Element dtype of every sorted run/survivor file this module writes."""

_DEFAULT_BUFFER_BYTES = 256 * 1024**2
"""Default :class:`PartitionWriter` global buffered-bytes cap."""

_DEFAULT_MERGE_BUFFER_ITEMS = 1_000_000
"""Default per-run in-memory block size for the merge/read primitives."""

_F_NOCACHE = getattr(fcntl, "F_NOCACHE", None)
"""``fcntl.F_NOCACHE`` where the platform defines it (macOS only); else
``None``. A module-level guard so :func:`advise_no_cache` is a no-op on
platforms without it, and so both branches are monkeypatchable in tests
without needing a real unavailable-``fcntl`` platform."""


class DiskFloorError(RuntimeError):
    """Raised when a spill write would breach the configured free-space floor.

    Signals that the target volume does not have ``min_free_bytes`` free
    after accounting for the bytes about to be written. Callers should treat
    this as fatal for the run: abandon the spill and leave the destination
    untouched rather than risk filling the volume.
    """


def ensure_free_disk(
    directory: Path,
    incoming_bytes: int = 0,
    *,
    min_free_bytes: int = MIN_FREE_DISK_BYTES,
) -> None:
    """Raise :class:`DiskFloorError` if writing ``incoming_bytes`` would breach the floor.

    Contract:
        - ``directory`` names any path on the target volume (need not exist
          as a file; :func:`shutil.disk_usage` resolves the containing
          filesystem).
        - ``incoming_bytes`` is the exact size of the write about to happen;
          ``0`` checks the volume's current headroom only.
        - Passes silently when
          ``shutil.disk_usage(directory).free - incoming_bytes >= min_free_bytes``.

    Failure modes:
        - :class:`DiskFloorError` otherwise, reporting the directory, the
          free bytes remaining after the write, and the configured floor.
    """
    free = shutil.disk_usage(directory).free
    remaining = free - incoming_bytes
    if remaining < min_free_bytes:
        msg = (
            f"writing {incoming_bytes} bytes under {directory} would leave "
            f"{remaining} bytes free, below the {min_free_bytes} byte floor."
        )
        raise DiskFloorError(msg)


def advise_no_cache(fileobj: BinaryIO) -> None:
    """Best-effort request that ``fileobj``'s I/O bypass the page cache.

    Contract:
        - Issues ``fcntl.fcntl(fileobj.fileno(), fcntl.F_NOCACHE, 1)`` on
          platforms that define ``F_NOCACHE`` (macOS), so large sequential
          spill writes do not evict the working set from the page cache --
          the same technique SQLite uses on macOS.
        - A silent no-op wherever ``F_NOCACHE`` is unavailable (Linux and
          other platforms); never raises.
    """
    if _F_NOCACHE is None:
        return
    fcntl.fcntl(fileobj.fileno(), _F_NOCACHE, 1)


class PartitionWriter:
    """Buffered fan-out writer spilling structured records to per-partition files.

    Contract:
        - Writes to ``directory / f"{prefix}_%04d.bin"`` for partition ids in
          ``[0, partition_count)``; ``directory`` must already exist.
        - :meth:`append` splits a batch of records by partition id (stable
          argsort + searchsorted) into per-partition in-memory buffers.
        - The buffered byte total across every partition is capped at
          ``buffer_bytes``; reaching or exceeding it flushes every buffer.
        - Each flush opens one partition file at a time in append mode,
          checks the disk floor for the exact byte count about to be
          written, applies :func:`advise_no_cache`, writes with
          :meth:`numpy.ndarray.tofile`, and closes -- so at most one spill
          file handle is ever open, regardless of ``partition_count``.
        - :meth:`close` flushes any remaining buffered records and returns
          the partition file paths that were actually written, in ascending
          partition-id order (a partition that received no records has no
          file and is omitted).

    Failure modes:
        - :class:`DiskFloorError` from a flush that would breach the floor.
    """

    def __init__(
        self,
        directory: Path,
        partition_count: int,
        *,
        prefix: str = "partition",
        buffer_bytes: int = _DEFAULT_BUFFER_BYTES,
        min_free_bytes: int = MIN_FREE_DISK_BYTES,
    ) -> None:
        """Set up an empty writer over ``partition_count`` partitions."""
        self._partition_count = partition_count
        self._buffer_bytes = buffer_bytes
        self._min_free_bytes = min_free_bytes
        self._paths = [
            directory / f"{prefix}_{index:04d}.bin"
            for index in range(partition_count)
        ]
        self._buffers: list[list[npt.NDArray[np.void]]] = [
            [] for _ in range(partition_count)
        ]
        self._buffered_bytes = 0

    def append(
        self,
        partition_ids: npt.NDArray[np.int64],
        records: npt.NDArray[np.void],
    ) -> None:
        """Route ``records`` to their partition buffers by ``partition_ids``.

        ``partition_ids[i]`` names the partition of ``records[i]``; both
        arrays share length. Flushes every buffer once the global buffered
        total reaches ``buffer_bytes``.
        """
        if partition_ids.shape[0] == 0:
            return
        order = np.argsort(partition_ids, kind="stable")
        sorted_ids = partition_ids[order]
        sorted_records = records[order]
        boundaries = np.searchsorted(
            sorted_ids, np.arange(self._partition_count + 1)
        )
        for partition_id in range(self._partition_count):
            begin, end = (
                boundaries[partition_id],
                boundaries[partition_id + 1],
            )
            if begin == end:
                continue
            chunk = sorted_records[begin:end]
            self._buffers[partition_id].append(chunk)
            self._buffered_bytes += chunk.nbytes
        if self._buffered_bytes >= self._buffer_bytes:
            self._flush_all()

    def _flush_all(self) -> None:
        """Flush every partition's buffer to disk, one file handle at a time."""
        for partition_id in range(self._partition_count):
            self._flush_partition(partition_id)
        self._buffered_bytes = 0

    def _flush_partition(self, partition_id: int) -> None:
        """Append ``partition_id``'s buffered records to its file and clear it."""
        chunks = self._buffers[partition_id]
        if not chunks:
            return
        combined = chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
        self._buffers[partition_id] = []
        path = self._paths[partition_id]
        with path.open("ab") as handle:
            ensure_free_disk(
                path.parent,
                combined.nbytes,
                min_free_bytes=self._min_free_bytes,
            )
            advise_no_cache(handle)
            combined.tofile(handle)

    def close(self) -> list[Path]:
        """Flush remaining buffers and return the partition paths that exist."""
        self._flush_all()
        return [path for path in self._paths if path.exists()]


def write_sorted_run(
    path: Path,
    values: npt.NDArray[np.int64],
    *,
    min_free_bytes: int = MIN_FREE_DISK_BYTES,
) -> None:
    """Write a sorted ``<i8`` array to ``path`` as a single disk-floor-checked run.

    Contract:
        - ``values`` must already be sorted ascending; this function does not
          sort. Writes a single new file (overwrites any existing file at
          ``path``).

    Failure modes:
        - :class:`DiskFloorError` if the write would breach the floor.
    """
    prepared = np.ascontiguousarray(values, dtype=_RUN_DTYPE)
    ensure_free_disk(
        path.parent, prepared.nbytes, min_free_bytes=min_free_bytes
    )
    with path.open("wb") as handle:
        advise_no_cache(handle)
        prepared.tofile(handle)


def merge_sorted_runs(
    runs: Sequence[Path],
    out_dir: Path,
    *,
    fan_in: int = 64,
    buffer_items: int = _DEFAULT_MERGE_BUFFER_ITEMS,
    min_free_bytes: int = MIN_FREE_DISK_BYTES,
) -> Path:
    """K-way merge sorted ``<i8`` run files into a single sorted file.

    Contract:
        - ``runs`` are paths to sorted ``<i8`` files (as :func:`write_sorted_run`
          produces); values within and across runs are unique (they are
          indices), so no stability concern arises when interleaving them.
        - Merges at most ``fan_in`` runs per pass, bounding open file handles;
          repeats passes over the intermediate merged files until one file
          remains. Every input run consumed by a pass is deleted afterwards.
        - A single input run is returned unchanged (no pass runs, nothing is
          deleted). An empty ``runs`` sequence produces a new, empty file.
        - Returns the path to the final merged file, under ``out_dir`` (or,
          for the single-run case, wherever that run already lived).

    Failure modes:
        - :class:`DiskFloorError` from a merge-output write that would breach
          the floor.
    """
    if not runs:
        empty = out_dir / "merged_empty.bin"
        ensure_free_disk(out_dir, 0, min_free_bytes=min_free_bytes)
        empty.touch()
        return empty
    pending: list[Path] = list(runs)
    pass_no = 0
    while len(pending) > 1:
        next_pending: list[Path] = []
        for group_no, start in enumerate(range(0, len(pending), fan_in)):
            group = pending[start : start + fan_in]
            out_path = out_dir / f"merge_p{pass_no:03d}_{group_no:04d}.bin"
            _merge_group(
                group,
                out_path,
                buffer_items=buffer_items,
                min_free_bytes=min_free_bytes,
            )
            for path in group:
                path.unlink()
            next_pending.append(out_path)
        pending = next_pending
        pass_no += 1
    return pending[0]


class _RunReadBuffer:
    """One sorted run file, read sequentially in bounded-size blocks.

    Backs :func:`_merge_group`'s vectorized merge: exposes the current
    in-memory block's last value (a safe lower bound on every value not yet
    read from this run, since the file is sorted ascending) and lets the
    caller pop off a sorted prefix at a time, refilling from disk as needed.
    """

    def __init__(self, path: Path, buffer_items: int) -> None:
        """Open ``path`` and load its first block."""
        self._handle: BinaryIO = path.open("rb")
        self._buffer_bytes: int = buffer_items * _RUN_DTYPE.itemsize
        self._values: npt.NDArray[np.int64] = np.empty(0, dtype=_RUN_DTYPE)
        self._exhausted: bool = False
        self._refill()

    def _refill(self) -> None:
        """Read the next block from disk, closing the handle once it is spent."""
        raw = self._handle.read(self._buffer_bytes)
        if not raw:
            self._exhausted = True
            self._handle.close()
            return
        self._values = np.frombuffer(raw, dtype=_RUN_DTYPE)

    @property
    def has_values(self) -> bool:
        """Whether this run still has unconsumed buffered values."""
        return self._values.size > 0

    @property
    def last(self) -> int:
        """The current block's maximum (last, since sorted) buffered value."""
        return int(self._values[-1])

    def take_upto(self, bound: int) -> npt.NDArray[np.int64]:
        """Pop and return this run's buffered values ``<= bound``, refilling if drained."""
        cut = int(np.searchsorted(self._values, bound, side="right"))
        taken = self._values[:cut]
        self._values = self._values[cut:]
        if self._values.size == 0 and not self._exhausted:
            self._refill()
        return taken


def _merge_group(
    paths: Sequence[Path],
    out_path: Path,
    *,
    buffer_items: int,
    min_free_bytes: int,
) -> None:
    """Vectorized k-way merge of ``paths`` (at most ``fan_in`` runs) into ``out_path``."""
    buffers: list[_RunReadBuffer] = [
        _RunReadBuffer(path, buffer_items) for path in paths
    ]
    active: list[_RunReadBuffer] = [
        buffer for buffer in buffers if buffer.has_values
    ]
    ensure_free_disk(out_path.parent, 0, min_free_bytes=min_free_bytes)
    with out_path.open("wb") as handle:
        advise_no_cache(handle)
        while active:
            bound = min(buffer.last for buffer in active)
            pieces: list[npt.NDArray[np.int64]] = [
                buffer.take_upto(bound) for buffer in active
            ]
            pieces = [piece for piece in pieces if piece.size]
            merged: npt.NDArray[np.int64] = (
                pieces[0]
                if len(pieces) == 1
                else np.sort(np.concatenate(pieces))
            )
            ensure_free_disk(
                out_path.parent, merged.nbytes, min_free_bytes=min_free_bytes
            )
            merged.tofile(handle)
            active = [buffer for buffer in active if buffer.has_values]


def iter_sorted_values(
    path: Path, *, buffer_items: int = _DEFAULT_MERGE_BUFFER_ITEMS
) -> Iterator[npt.NDArray[np.int64]]:
    """Yield ``path``'s sorted ``<i8`` contents sequentially, in bounded-size blocks.

    Contract:
        - Reads ``path`` in blocks of up to ``buffer_items`` values, yielding
          each block in file order (which is ascending, since the file is a
          merged/sorted run). The last block may be shorter.
    """
    buffer_bytes = buffer_items * _RUN_DTYPE.itemsize
    with path.open("rb") as handle:
        while True:
            raw = handle.read(buffer_bytes)
            if not raw:
                return
            yield np.frombuffer(raw, dtype=_RUN_DTYPE)
