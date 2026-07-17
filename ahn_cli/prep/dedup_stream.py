"""Prep-context out-of-core tile deduplication (dependency-free external spill).

The in-memory oracle in :mod:`ahn_cli.prep.dedup` (:func:`~ahn_cli.prep.dedup.
deduplicate_tiles`) reads every tile whole (``reader.read()``), concatenates the
cropped records into one merged array, and sweeps exact duplicates with a single
``np.unique`` -- so a national-scale merge (billions of points) holds roughly
twice the whole cloud in memory and the process is killed (SIGKILL / exit 137).
This module is the memory-bounded replacement: it never holds more than one chunk
of points, plus a few bounded-size spill buffers, at a time, and yet emits output
that is **byte-identical** to the oracle.

Semantics are the dedup contract of :mod:`ahn_cli.prep.dedup`:

1. **Crop before merge.** Each tile is cropped to its canonical extent with the
   half-open ``[minx, maxx) x [miny, maxy)`` rule, so a point on a shared edge is
   claimed by exactly one tile and the seam band never enters the merge.
2. **Offset reprojection.** Every cropped point is cast onto the harmonized
   header's scale/offset grid (``header.offsets - source.offsets`` correction),
   exactly as the oracle's ``_crop_and_reproject``, so tiles written with
   different LAS offsets share one integer lattice before keys are compared.
3. **Exact-duplicate sweep.** Two points coincide when their reprojected integer
   ``X/Y/Z`` and ``gps_time`` all match; of each such group the survivor is the
   one with the smallest *global* index -- its position in the merged record,
   assigned continuously across tiles in input order, then in-tile file order.
   Survivors are written in ascending global-index order, exactly the oracle's
   ``np.sort`` over ``np.unique``'s first-occurrence indices.

Internals implement the grace-hash partitioned aggregation of
``docs/specs/dedup-spill-design.md`` (mirroring
``docs/specs/voxel-spill-design.md``), using only :mod:`ahn_cli.prep.spill`'s
numpy+stdlib primitives. The flow is five streaming passes over a scratch spill
directory:

1. **Spill.** Stream every tile in chunks (never ``reader.read()``), apply the
   half-open crop and the offset reprojection, and append each surviving point's
   reprojected ``(X, Y, Z, gps_time)`` key plus its dense global index to segment
   files, rolled at ``_SEGMENT_BYTES``. Counts the pre-crop input and the cropped
   (kept) totals.
2. **Partition.** Hash-partition every spilled record by its exact key (the
   ``gps_time`` f8 reinterpreted to u64 bits) into ``partition_count`` files, so
   every point of a given key lands in one partition (a grace hash join).
3. **Reduce.** Per partition: ``np.lexsort`` by key then index and keep the
   smallest index in each key run -- the partitioning guarantees every point of a
   key lands in the same partition, so this local reduction is exact. Writes the
   partition's surviving indices as a sorted run.
4. **Merge.** K-way merge the per-partition runs into one ascending stream of
   every surviving global index.
5. **Write.** Re-stream the tiles, re-crop/re-reproject each chunk identically,
   consume the merged survivor stream over that chunk's global-index window, and
   write survivors through a temp file swapped into ``output`` at the end.

Every write site checks the target volume's free space first via
:func:`~ahn_cli.prep.spill.ensure_free_disk` and raises
:class:`~ahn_cli.prep.spill.DiskFloorError` rather than risk filling the disk; on
that error (or any other) the spill directory and any partial output are removed
and ``output`` is left untouched.

Determinism: tiles are streamed in input order, chunks in file order, the global
index is assigned in that order, and the per-key minimum-index reduction is
order-independent (partition count, segment boundaries, chunk size, and merge
fan-in do not affect the survivor set), so identical input yields byte-identical
output -- byte-identical, specifically, to the in-memory oracle.
"""

from __future__ import annotations

import contextlib
import math
import shutil
import tempfile
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import laspy
import numpy as np

from ahn_cli.prep.spill import (
    MIN_FREE_DISK_BYTES,
    PartitionWriter,
    advise_no_cache,
    ensure_free_disk,
    iter_sorted_values,
    merge_sorted_runs,
    write_sorted_run,
)

# Header harmonization is reused from the grandfathered ``process`` module via
# its public ``harmonize_headers`` re-export, mirroring :mod:`ahn_cli.prep.dedup`.
# The import emits a module-load ``DeprecationWarning``; wrapping it in
# ``catch_warnings`` keeps that warning from leaking into this gated context,
# while staying a module-top-level import.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from ahn_cli.process import harmonize_headers

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from ahn_cli.domain import BBox, ProgressCallback

from ahn_cli.prep.dedup import DedupStats

if TYPE_CHECKING:
    from ahn_cli.prep.dedup import CanonicalTile

DEFAULT_CHUNK_POINTS = 1_000_000
"""Points held in memory per streamed chunk (matches the PLY export window)."""

_SPILL_SUBDIR = "dedup_spill"
"""Scratch subdirectory (under the workdir) holding every intermediate spill."""

_SEGMENT_BYTES = 256 * 1024**2
"""Pass 1 spill-segment roll size."""

_PARTITION_TARGET_BYTES = 128 * 1024**2
"""Target in-memory size of a single pass-2 partition file."""

_PARTITION_MIN = 1
_PARTITION_MAX = 4096

_PARTITION_READ_RECORDS = 2_000_000
"""Records read per pass-2 segment-read block (bounds transient memory)."""

_FINALIZE_HEADROOM_BYTES = 4 * 1024**2
"""Free-space headroom demanded before the LAZ writer finalises on close.

laspy rewrites the header and chunk table and flushes its compressor when the
writer closes -- a small but otherwise unguarded write. 4 MiB bounds it."""

_RECORD_DTYPE = np.dtype(
    [
        ("X", "<i4"),
        ("Y", "<i4"),
        ("Z", "<i4"),
        ("gps_time", "<f8"),
        ("idx", "<i8"),
    ]
)
"""Spill/partition record: reprojected key (X, Y, Z, gps_time) + global index."""

# Odd 64-bit constants (Knuth/xxhash-style) for the pass-2 partition hash.
_HASH_K1 = np.uint64(0x9E3779B185EBCA87)
_HASH_K2 = np.uint64(0xC2B2AE3D27D4EB4F)
_HASH_K3 = np.uint64(0x165667B19E3779F9)
_HASH_K4 = np.uint64(0x27D4EB2F165667C5)


def _no_op_progress(_done: int, _total: int) -> None:
    """Report nothing; the default when the caller supplies no callback."""


def _require_gps_time(header: laspy.LasHeader) -> None:
    """Reject a harmonized header without a ``gps_time`` dimension.

    The exact-duplicate key includes ``gps_time`` (LAS point format 6+); a
    format that lacks it (PDRF < 6, e.g. formats 0/2) cannot be deduplicated
    on this contract.

    Failure modes:
        - :class:`ValueError` if ``gps_time`` is absent from the format.
    """
    if "gps_time" not in header.point_format.dimension_names:
        msg = (
            "stream_deduplicate_tiles requires a gps_time dimension "
            "(LAS point format 6+); the harmonized header has none."
        )
        raise ValueError(msg)


def stream_deduplicate_tiles(
    tiles: Sequence[CanonicalTile],
    output_path: Path,
    *,
    workdir: Path | None = None,
    progress: ProgressCallback | None = None,
    chunk_points: int = DEFAULT_CHUNK_POINTS,
    min_free_bytes: int = MIN_FREE_DISK_BYTES,
) -> DedupStats:
    """Crop, merge, and exact-duplicate-sweep ``tiles`` into ``output_path``, out of core.

    The memory-bounded replacement for
    :func:`~ahn_cli.prep.dedup.deduplicate_tiles`, producing byte-identical
    output while never holding more than one chunk of points at a time.

    Contract:
        - ``tiles`` is a non-empty sequence of
          :class:`~ahn_cli.prep.dedup.CanonicalTile`; each is cropped to its
          canonical extent with the half-open ``[min, max)`` rule, cast onto one
          harmonized header, and exact reprojected-XYZ+gps_time duplicates are
          swept, keeping the smallest global index of each group in ascending
          global-index order -- byte-identical to the in-memory oracle.
        - Every input tile must carry a ``gps_time`` dimension (LAS point format
          6+), which forms part of the duplicate key.
        - Returns a :class:`~ahn_cli.prep.dedup.DedupStats` ledger of the point
          counts.
        - ``workdir`` is the scratch directory for the binary spill files; when
          ``None`` a private temp dir is created and removed afterwards. The
          spill lives in a dedicated subdirectory recreated empty each run and
          removed when the dedup finishes.
        - Calls ``progress(tiles_done, total_tiles)`` once per tile as it is
          streamed and spilled; defaults to a no-op.
        - ``chunk_points`` bounds how many points are read at once (must be
          positive). ``min_free_bytes`` is the free-space floor checked before
          every spill and output write; both keyword-only knobs mirror
          :func:`~ahn_cli.prep.voxel_stream.stream_voxel_thin` and do not change
          the (byte-identical) output.

    Invariants:
        - Memory-bounded: never more than one chunk of points plus a few
          bounded-size spill buffers are resident, regardless of tile count.
        - Deterministic: identical input yields byte-identical output, matching
          the oracle byte-for-byte.

    Failure modes:
        - :class:`ValueError` if ``tiles`` is empty, if ``chunk_points`` is not
          positive, or if the harmonized header lacks ``gps_time``.
        - :class:`~ahn_cli.prep.spill.DiskFloorError` if a spill or output write
          would leave the target volume below ``min_free_bytes`` free; the spill
          directory and any partial output are removed and ``output`` is left
          untouched.
        - :class:`OSError` on an I/O failure while streaming, spilling, or
          writing, with the same cleanup guarantees as a floor breach.
    """
    if not tiles:
        msg = "stream_deduplicate_tiles requires at least one tile."
        raise ValueError(msg)
    if chunk_points <= 0:
        msg = f"chunk_points must be a positive point count; got {chunk_points}."
        raise ValueError(msg)
    report = progress if progress is not None else _no_op_progress

    files = [str(tile.path) for tile in tiles]
    header = harmonize_headers(files)
    _require_gps_time(header)

    if workdir is None:
        with tempfile.TemporaryDirectory(prefix="ahn_cli_dedup_") as tmp:
            return _dedup_with_spill(
                tiles,
                output_path,
                header,
                Path(tmp),
                chunk_points,
                report,
                min_free_bytes,
            )
    return _dedup_with_spill(
        tiles,
        output_path,
        header,
        workdir,
        chunk_points,
        report,
        min_free_bytes,
    )


def _dedup_with_spill(
    tiles: Sequence[CanonicalTile],
    output: Path,
    header: laspy.LasHeader,
    workdir: Path,
    chunk_points: int,
    report: ProgressCallback,
    min_free_bytes: int,
) -> DedupStats:
    """Run the spill -> partition -> reduce -> merge -> write dedup.

    ``cropped == 0`` (every point cropped away) short-circuits passes 2-4: there
    is nothing to partition, reduce, or merge, so pass 5 writes an empty output
    directly (survivor stream ``None``).
    """
    spill = workdir / _SPILL_SUBDIR
    if spill.is_symlink() or spill.is_file():
        # A stale symlink or plain file at the spill path (a crashed or foreign
        # process might leave one) would make rmtree raise -- and a symlink must
        # be removed as a link, never followed into its target.
        spill.unlink()
    elif spill.is_dir():
        shutil.rmtree(spill)
    spill.mkdir(parents=True)
    try:
        ensure_free_disk(workdir, min_free_bytes=min_free_bytes)
        ensure_free_disk(output.parent, min_free_bytes=min_free_bytes)
        segments, input_points, cropped = _spill_pass(
            tiles, header, spill, chunk_points, report, min_free_bytes
        )
        survivors_path: Path | None = None
        if cropped > 0:
            partition_dir = spill / "partitions"
            partition_dir.mkdir()
            partition_paths = _partition_pass(
                segments,
                _partition_count_for(cropped),
                partition_dir,
                min_free_bytes,
            )
            runs_dir = spill / "runs"
            runs_dir.mkdir()
            run_paths = _reduce_pass(
                partition_paths, runs_dir, min_free_bytes
            )
            merge_dir = spill / "merge"
            merge_dir.mkdir()
            survivors_path = merge_sorted_runs(
                run_paths, merge_dir, min_free_bytes=min_free_bytes
            )
        output_points = _write_pass(
            tiles,
            output,
            header,
            survivors_path,
            chunk_points,
            min_free_bytes,
        )
        return DedupStats(
            input_points=input_points,
            cropped_points=cropped,
            duplicates_removed=cropped - output_points,
            output_points=output_points,
        )
    finally:
        shutil.rmtree(spill, ignore_errors=True)


def _crop_and_reproject(
    chunk: laspy.ScaleAwarePointRecord,
    extent: BBox,
    source_header: laspy.LasHeader,
    header: laspy.LasHeader,
) -> laspy.ScaleAwarePointRecord:
    """Crop one chunk to ``extent`` and cast it onto ``header``'s grid.

    An exact per-chunk mirror of :func:`ahn_cli.prep.dedup._crop_and_reproject`:
    the crop is half-open ``[minx, maxx) x [miny, maxy)`` on the source's own
    world coordinates, then the raw integer coordinates are reprojected onto the
    harmonized grid via the ``header.offsets - source.offsets`` correction.
    Because both are elementwise, chunking a tile yields records bit-identical to
    cropping/reprojecting the whole tile at once.
    """
    minx, miny, maxx, maxy = extent
    x = np.asarray(chunk.x, dtype=float)
    y = np.asarray(chunk.y, dtype=float)
    kept = (x >= minx) & (x < maxx) & (y >= miny) & (y < maxy)
    cropped = chunk[kept]

    out = laspy.ScaleAwarePointRecord.zeros(len(cropped), header=header)
    # The harmonized header is a superset of every tile's dimensions, so a
    # straight per-dimension copy is total (no missing-field guard needed).
    for name in cropped.point_format.dimension_names:
        out[name] = cropped[name]

    offset_correction = header.offsets - source_header.offsets
    out.x = out.x - offset_correction[0]
    out.y = out.y - offset_correction[1]
    out.z = out.z - offset_correction[2]
    return out


def _spill_pass(
    tiles: Sequence[CanonicalTile],
    header: laspy.LasHeader,
    spill: Path,
    chunk_points: int,
    report: ProgressCallback,
    min_free_bytes: int,
) -> tuple[list[Path], int, int]:
    """Stream ``tiles``, spilling each cropped point's reprojected key + index.

    ``idx`` is the point's dense index in the merged (cropped, in-input-order)
    cloud, assigned continuously across tiles and, within a tile, in streamed
    file order -- exactly the oracle's concatenation index. Buffers accumulate in
    memory and flush to a new segment file once ``_SEGMENT_BYTES`` is reached.

    Returns the segment paths in write order, the total pre-crop input count, and
    the total cropped (kept) count. Ticks ``report(tile, total_tiles)`` once per
    tile.
    """
    input_points = 0
    cropped = 0
    global_idx = 0
    segments: list[Path] = []
    buffer: list[npt.NDArray[np.void]] = []
    buffered_bytes = 0
    seg_no = 0

    def flush() -> None:
        nonlocal buffer, buffered_bytes, seg_no
        if not buffer:
            return
        path = spill / f"seg_{seg_no:06d}.bin"
        ensure_free_disk(spill, buffered_bytes, min_free_bytes=min_free_bytes)
        with path.open("ab") as handle:
            advise_no_cache(handle)
            for record in buffer:
                record.tofile(handle)
        segments.append(path)
        buffer = []
        buffered_bytes = 0
        seg_no += 1

    for tile_no, tile in enumerate(tiles, start=1):
        with laspy.open(str(tile.path)) as reader:
            source_header = reader.header
            for chunk in reader.chunk_iterator(chunk_points):
                input_points += len(chunk)
                out = _crop_and_reproject(
                    chunk, tile.extent, source_header, header
                )
                count = len(out)
                if count == 0:
                    continue
                record = np.zeros(count, dtype=_RECORD_DTYPE)
                record["X"] = np.asarray(out.X)
                record["Y"] = np.asarray(out.Y)
                record["Z"] = np.asarray(out.Z)
                record["gps_time"] = np.asarray(out.gps_time)
                record["idx"] = np.arange(
                    global_idx, global_idx + count, dtype=np.int64
                )
                buffer.append(record)
                buffered_bytes += record.nbytes
                global_idx += count
                cropped += count
                if buffered_bytes >= _SEGMENT_BYTES:
                    flush()
        report(tile_no, len(tiles))
    flush()
    return segments, input_points, cropped


def _partition_count_for(cropped: int) -> int:
    """Return the pass-2 partition count for ``cropped`` spilled points.

    ``ceil(cropped * record_size / _PARTITION_TARGET_BYTES)``, clamped to
    ``[_PARTITION_MIN, _PARTITION_MAX]`` -- enough partitions that each fits
    comfortably in memory during pass 3's reduction.
    """
    raw = math.ceil(
        cropped * _RECORD_DTYPE.itemsize / _PARTITION_TARGET_BYTES
    )
    return min(_PARTITION_MAX, max(_PARTITION_MIN, raw))


def _partition_pass(
    segments: list[Path],
    partition_count: int,
    partition_dir: Path,
    min_free_bytes: int,
) -> list[Path]:
    """Hash-partition every spilled record onto its exact key (a grace hash join).

    Processes one segment at a time -- read in bounded-size blocks, hashed by the
    exact ``(X, Y, Z, gps_time)`` key, routed through a
    :class:`~ahn_cli.prep.spill.PartitionWriter` -- deleting the segment once
    every block is routed. Returns the partition file paths
    :meth:`~ahn_cli.prep.spill.PartitionWriter.close` reports.
    """
    writer = PartitionWriter(
        partition_dir, partition_count, min_free_bytes=min_free_bytes
    )
    for segment in segments:
        record_count = segment.stat().st_size // _RECORD_DTYPE.itemsize
        with segment.open("rb") as handle:
            remaining = record_count
            while remaining > 0:
                block = min(_PARTITION_READ_RECORDS, remaining)
                records = np.fromfile(
                    handle, dtype=_RECORD_DTYPE, count=block
                )
                remaining -= block
                partition_ids = _partition_ids(records, partition_count)
                writer.append(partition_ids, records)
        segment.unlink()
    return writer.close()


def _partition_ids(
    records: npt.NDArray[np.void], partition_count: int
) -> npt.NDArray[np.int64]:
    """Hash each record's exact key to a partition id in ``[0, partition_count)``.

    The ``gps_time`` f8 is reinterpreted to its u64 bit pattern before hashing,
    so the hash is a pure integer function of the exact key. Two float-equal but
    bit-differing gps values (``-0.0`` vs ``0.0``, or distinct NaN payloads)
    would therefore hash apart -- an edge the oracle's float-equality
    ``np.unique`` would collapse, but one that never arises in real AHN gps_time
    (large positive GPS seconds).
    """
    hx = records["X"].astype(np.int64).astype(np.uint64)
    hy = records["Y"].astype(np.int64).astype(np.uint64)
    hz = records["Z"].astype(np.int64).astype(np.uint64)
    hg = np.ascontiguousarray(records["gps_time"]).view(np.uint64)
    hashed = hx * _HASH_K1 ^ hy * _HASH_K2 ^ hz * _HASH_K3 ^ hg * _HASH_K4
    return (hashed % np.uint64(partition_count)).astype(np.int64)


def _reduce_pass(
    partition_paths: list[Path], runs_dir: Path, min_free_bytes: int
) -> list[Path]:
    """Reduce each partition to its surviving (smallest-idx-per-key) indices.

    Every point of a given exact key lands in the same partition (by construction
    of the pass-2 hash), so this local reduction -- sort by key then index, keep
    the first (smallest-idx) record per key run -- is exact. Deletes each
    partition file once read; writes each partition's sorted survivor indices as
    a run.
    """
    run_paths: list[Path] = []
    for index, path in enumerate(partition_paths):
        records = np.fromfile(path, dtype=_RECORD_DTYPE)
        path.unlink()
        order = np.lexsort(
            (
                records["idx"],
                records["gps_time"],
                records["Z"],
                records["Y"],
                records["X"],
            )
        )
        ordered = records[order]
        is_start = np.ones(len(ordered), dtype=np.bool_)
        if len(ordered) > 1:
            is_start[1:] = (
                (ordered["X"][1:] != ordered["X"][:-1])
                | (ordered["Y"][1:] != ordered["Y"][:-1])
                | (ordered["Z"][1:] != ordered["Z"][:-1])
                | (ordered["gps_time"][1:] != ordered["gps_time"][:-1])
            )
        survivors = np.sort(ordered["idx"][is_start])
        run_path = runs_dir / f"run_{index:06d}.bin"
        write_sorted_run(run_path, survivors, min_free_bytes=min_free_bytes)
        run_paths.append(run_path)
    return run_paths


class _SurvivorCursor:
    """Sequential consumer of the merged global survivor-index stream.

    Wraps :func:`~ahn_cli.prep.spill.iter_sorted_values`, buffering blocks and
    handing back the prefix below a caller-supplied bound. Pass 5 calls
    :meth:`take_window` with a strictly increasing bound once per chunk, so each
    call only ever looks forward from where the previous left off. ``path`` of
    ``None`` yields a cursor that always returns empty (the empty-survivor-set
    path -- every point cropped away).
    """

    def __init__(self, path: Path | None, buffer_items: int) -> None:
        """Wrap ``path``'s sorted stream (or nothing, for ``path is None``)."""
        self._source = (
            iter_sorted_values(path, buffer_items=buffer_items)
            if path is not None
            else iter(())
        )
        self._pending: npt.NDArray[np.int64] = np.empty(0, dtype=np.int64)

    def take_window(self, end: int) -> npt.NDArray[np.int64]:
        """Pop and return every buffered survivor index strictly below ``end``."""
        parts: list[npt.NDArray[np.int64]] = []
        while True:
            if self._pending.size == 0:
                self._pending = next(
                    self._source, np.empty(0, dtype=np.int64)
                )
                if self._pending.size == 0:
                    break
            cut = int(np.searchsorted(self._pending, end, side="left"))
            if cut == 0:
                break
            parts.append(self._pending[:cut])
            self._pending = self._pending[cut:]
        return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)


def _write_pass(
    tiles: Sequence[CanonicalTile],
    output: Path,
    header: laspy.LasHeader,
    survivors_path: Path | None,
    chunk_points: int,
    min_free_bytes: int,
) -> int:
    """Re-stream ``tiles`` and write the surviving points to ``output``.

    Re-crops and re-reprojects each chunk identically to pass 1 (so the global
    index alignment holds), consuming the merged survivor stream window-by-window
    against each chunk's global-index range via :class:`_SurvivorCursor`.
    ``survivors_path`` of ``None`` means the empty survivor set (every point
    cropped away), so nothing is written. The write goes to a sibling temp file
    swapped into ``output`` at the end. Returns the written count.

    Failure modes:
        - :class:`~ahn_cli.prep.spill.DiskFloorError` if an output chunk write --
          or the writer's close-time finalisation, guarded with
          ``_FINALIZE_HEADROOM_BYTES`` -- would breach the floor. On *any*
          failure the partial temp file is removed in a ``finally`` and
          ``output`` is left untouched: the swap into ``output`` is the last
          operation inside the guarded block.
    """
    tmp_out = output.with_name(f"{output.stem}.tmp{output.suffix}")
    cursor = _SurvivorCursor(survivors_path, chunk_points)
    written = 0
    filtered = 0
    point_size = header.point_format.size
    try:
        writer = laspy.open(str(tmp_out), mode="w", header=header)
        finalised = False
        try:
            for tile in tiles:
                with laspy.open(str(tile.path)) as reader:
                    source_header = reader.header
                    for chunk in reader.chunk_iterator(chunk_points):
                        out = _crop_and_reproject(
                            chunk, tile.extent, source_header, header
                        )
                        count = len(out)
                        local = (
                            cursor.take_window(filtered + count) - filtered
                        )
                        filtered += count
                        selected = out[local]
                        if len(selected) > 0:
                            ensure_free_disk(
                                tmp_out.parent,
                                len(selected) * point_size,
                                min_free_bytes=min_free_bytes,
                            )
                            writer.write_points(selected)
                            written += len(selected)
            # Closing the writer finalises the LAZ header/chunk table -- the one
            # write the per-chunk guards above don't cover. The close is
            # explicit, directly after its guard: a context-managed writer would
            # finalise in ``__exit__`` even when the guard raises, and a
            # genuinely failing close would supersede the guard's typed error.
            ensure_free_disk(
                tmp_out.parent,
                _FINALIZE_HEADROOM_BYTES,
                min_free_bytes=min_free_bytes,
            )
            writer.close()
            finalised = True
        finally:
            if not finalised:
                # Best-effort teardown; the temp is discarded anyway and a close
                # failure must never mask the original error.
                with contextlib.suppress(Exception):
                    writer.close()
        tmp_out.replace(output)
    finally:
        # No-op after a successful replace (the temp no longer exists); removes
        # the partial temp on every failure path.
        tmp_out.unlink(missing_ok=True)
    return written
