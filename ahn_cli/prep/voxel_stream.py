"""Prep-context out-of-core voxel thinning (dependency-free external spill).

The in-memory voxel path in :mod:`ahn_cli.prep.decimate` materialises the whole
cloud at once -- ``reader.read()`` plus a full ``(n, 3)`` float64 coordinate copy
plus the ``np.unique(cells, axis=0)`` sort temporaries -- so a national-scale
cloud (hundreds of millions of points) exhausts RAM and the process is killed
(SIGKILL / exit 137). This module is the memory-bounded alternative used by the
prep pipeline for :class:`~ahn_cli.prep.decimate.VoxelThinning` requests: it never
holds more than one chunk of points, plus a few bounded-size spill buffers, at a
time, so peak memory is independent of the point count.

Semantics are the voxel contract of :mod:`ahn_cli.prep.decimate`: within each
occupied voxel exactly one point survives -- the one with the smallest index in
the *class-filtered* cloud -- and the survivors are emitted in ascending index
order. The voxel grid is anchored at the per-cloud coordinate minimum and its
edge length comes from :func:`~ahn_cli.prep.decimate.voxel_size_for_grade`, so a
given grade yields the identical grid the in-memory reference uses.

Internals implement the grace-hash partitioned aggregation described in
``docs/specs/voxel-spill-design.md``, using only :mod:`ahn_cli.prep.spill`'s
numpy+stdlib primitives (no third-party out-of-core engine). For grade > 0 the
flow is five streaming passes over a scratch spill directory:

1. **Spill.** Stream the source in chunks, apply the classification filter, and
   append each kept point's *raw* LAS integer coordinates plus its dense
   class-filtered index to segment files, rolled at ``_SEGMENT_BYTES``. Tracks
   the running per-axis minimum of the *scaled* kept coordinates (the voxel
   grid's origin) and the total kept count.
2. **Partition.** Hash-partition every spilled point by its voxel cell (a grace
   hash join) into ``partition_count`` files, sized so each partition fits
   comfortably in memory. Processes one segment at a time, deleting it once its
   records are routed.
3. **Reduce.** Per partition (deleted once read): sort by voxel cell then index
   and keep the smallest index in each occupied cell -- the partitioning
   guarantees every point of a given cell lands in the same partition, so this
   local reduction is exact. Writes the partition's surviving indices as a
   sorted run. The pass-2 partition count is clamped at
   :data:`_PARTITION_MAX` (a file-handle/overhead budget, not a hard memory
   ceiling): a partition still too large to read at once past that clamp is
   re-hash-partitioned further here, recursively, before being reduced --
   see :func:`_reduce_partition`.
4. **Merge.** K-way merge the per-partition sorted runs into one sorted stream
   of every surviving index, via :func:`~ahn_cli.prep.spill.merge_sorted_runs`.
5. **Write.** Re-stream the source a second time and, per chunk, consume the
   merged survivor stream over that chunk's class-filtered index window,
   writing survivors through a temp file swapped into ``output`` at the end.

Grade 0 is the identity (no spill): only the class filter is applied, in a
single streaming pass. Every write site checks the target volume's free space
first via :func:`~ahn_cli.prep.spill.ensure_free_disk` and raises
:class:`~ahn_cli.prep.spill.DiskFloorError` rather than risk filling the disk;
on that error the spill directory and any partial output are removed and
``output`` is left untouched.

Determinism: chunks are read in file order, the filtered index is assigned in
that order, and the per-voxel minimum-index reduction is order-independent
(partition count, segment boundaries, and merge fan-in do not affect the
survivor set), so identical input yields byte-identical output.
"""

from __future__ import annotations

import contextlib
import math
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import laspy
import numpy as np

from ahn_cli.prep.decimate import voxel_size_for_grade
from ahn_cli.prep.spill import (
    MIN_FREE_DISK_BYTES,
    PartitionWriter,
    advise_no_cache,
    ensure_free_disk,
    iter_sorted_values,
    merge_sorted_runs,
    write_sorted_run,
)

if TYPE_CHECKING:
    import numpy.typing as npt

    from ahn_cli.domain.progress import ProgressCallback

DEFAULT_CHUNK_POINTS = 1_000_000
"""Points held in memory per streamed chunk (matches the PLY export window)."""

_SPILL_SUBDIR = "voxel_spill"
"""Scratch subdirectory (under the workdir) holding every intermediate spill."""

_SEGMENT_BYTES = 256 * 1024**2
"""Pass 1 spill-segment roll size."""

_PARTITION_TARGET_BYTES = 128 * 1024**2
"""Target in-memory size of a single pass-2 partition file."""

_PARTITION_MIN = 1
_PARTITION_MAX = 4096
"""Initial pass-2 fan-out cap (file-handle/overhead budget, not a hard ceiling:
see :data:`_PARTITION_MAX_BYTES` -- a partition that is still oversized after
this cap bites is re-split further in pass 3, so arbitrarily large inputs
still spill within :data:`~ahn_cli.prep.spill.MIN_FREE_DISK_BYTES`)."""

_PARTITION_MAX_BYTES = 4 * _PARTITION_TARGET_BYTES
"""Safety ceiling on a single partition file's size before pass 3 re-splits it
further instead of reading it whole into memory. Set well above
``_PARTITION_TARGET_BYTES`` so ordinary partitions (sized to that target, with
some hash-skew slack) are read directly; a partition breaching this only
happens once ``_partition_count_for`` has clamped at ``_PARTITION_MAX`` for a
cloud too large for that many partitions to keep each one small."""

_RESPLIT_FACTOR = 64
"""Sub-partitions an oversized partition is re-hashed into, per recursion
level (``_RESPLIT_FACTOR ** depth`` cumulative fan-out)."""

_MAX_RESPLIT_DEPTH = 6
"""Recursion depth safety valve for :func:`_reduce_partition`.

Past this depth a still-oversized partition is reduced in memory regardless.
Re-splitting hashes only a partition's voxel-cell coordinates, so it can only
separate *distinct* cells that collided into the same bucket; it can never
divide a single cell's own points across sub-partitions (they always hash
together), so unbounded recursion could not rescue the pathological case of
one voxel cell alone holding enough points to exceed
``_PARTITION_MAX_BYTES`` -- that working set is irreducible by partitioning."""

_RESPLIT_SALT_BASE = np.uint64(0x2545F4914F6CDD1D)
"""Odd 64-bit constant salting each re-split recursion level's hash distinctly
from the pass-2 hash and from every other level, so distinct cells that
collided at one level are spread apart at the next."""

_PARTITION_READ_RECORDS = 2_000_000
"""Records read per pass-2 segment-read block (bounds transient memory).

Sized so the block (~40 MB of records plus the float64 conversion
temporaries) stays small next to the reduce pass's partition slab while
keeping the number of :meth:`PartitionWriter.append` calls -- each a Python
loop over the partitions present in the block -- low."""

_FINALIZE_HEADROOM_BYTES = 4 * 1024**2
"""Free-space headroom demanded before the LAZ writer finalises on close.

laspy rewrites the header and chunk table and flushes its compressor when
the writer closes -- a small (measured: tens of bytes to ~10 KB) but
otherwise unguarded write. 4 MiB is a deliberately generous bound on it."""

_RAW_DTYPE = np.dtype(
    [("X", "<i4"), ("Y", "<i4"), ("Z", "<i4"), ("idx", "<i8")]
)
"""Pass-1 spill record: raw (unscaled) LAS integer coords + filtered index."""

_PARTITION_DTYPE = np.dtype(
    [("cx", "<i4"), ("cy", "<i4"), ("cz", "<i4"), ("idx", "<i8")]
)
"""Pass-2 partition record: voxel cell coords + filtered index."""

_INT32_MIN = np.iinfo(np.int32).min
_INT32_MAX = np.iinfo(np.int32).max

# Odd 64-bit constants (Knuth/xxhash-style) for the pass-2 partition hash.
_HASH_K1 = np.uint64(0x9E3779B185EBCA87)
_HASH_K2 = np.uint64(0xC2B2AE3D27D4EB4F)
_HASH_K3 = np.uint64(0x165667B19E3779F9)


def _no_op_progress(_done: int, _total: int) -> None:
    """Report nothing; the default when the caller supplies no callback."""


def stream_voxel_thin(
    source: Path,
    output: Path,
    grade: int,
    include_classes: tuple[int, ...],
    exclude_classes: tuple[int, ...],
    *,
    workdir: Path | None = None,
    chunk_points: int = DEFAULT_CHUNK_POINTS,
    progress: ProgressCallback | None = None,
    min_free_bytes: int = MIN_FREE_DISK_BYTES,
) -> int:
    """Class-filter and voxel-thin ``source`` into ``output``, out of core.

    Contract:
        - ``source`` is a readable LAS/LAZ; ``output`` receives the class-filtered,
          voxel-thinned cloud (``source`` and ``output`` may be the same path --
          the write goes through a temp file swapped in at the end).
        - ``grade`` is a voxel grade in ``[GRADE_MIN, GRADE_MAX]``; grade 0 is the
          identity (every class-kept point survives, no thinning).
        - ``include_classes`` / ``exclude_classes`` are the classification filter;
          empty tuples mean "no filter on that side".
        - Within each occupied voxel the surviving point is the one with the
          smallest index in the class-filtered cloud; survivors are written in
          ascending index order, every source attribute preserved. Returns the
          surviving point count.
        - ``workdir`` is the scratch directory for the binary spill files; when
          ``None`` a private temp dir is created and removed afterwards. The
          spill lives in a dedicated subdirectory that is recreated empty each
          run and removed when the thin finishes.
        - ``chunk_points`` bounds how many points are read at once (must be
          positive). ``progress`` ticks ``(chunk, total_chunks)`` as the output
          is written; defaults to a no-op.
        - ``min_free_bytes`` is the free-space floor checked before every spill
          and output write (see :mod:`ahn_cli.prep.spill`); defaults to
          :data:`~ahn_cli.prep.spill.MIN_FREE_DISK_BYTES`.

    Invariants:
        - Memory-bounded: never more than one chunk of points plus a few
          bounded-size spill buffers are resident, regardless of cloud size.
        - Deterministic: identical input and parameters yield byte-identical
          output.

    Failure modes:
        - :class:`ValueError` if ``chunk_points`` is not positive, or if ``grade``
          is out of range (via :func:`voxel_size_for_grade`), or if a voxel cell
          coordinate does not fit in int32 (the cloud's extent is too large for
          the requested voxel size).
        - :class:`~ahn_cli.prep.spill.DiskFloorError` if a spill or output write
          would leave the target volume below ``min_free_bytes`` free; the spill
          directory and any partial output are removed and ``output`` is left
          untouched.
        - :class:`OSError` on an I/O failure while streaming, spilling, or
          writing -- an anticipated outcome on real hardware, with the same
          cleanup guarantees as a floor breach; the prep pipeline wraps it
          (with ``ValueError`` and ``DiskFloorError``) into its typed
          :class:`~ahn_cli.prep.transform.PrepError` at the ``prepare()``
          boundary.
    """
    if chunk_points <= 0:
        msg = f"chunk_points must be a positive point count; got {chunk_points}."
        raise ValueError(msg)
    report = progress if progress is not None else _no_op_progress
    size = voxel_size_for_grade(grade)
    if size == 0.0:
        # Grade 0 is the identity: apply only the class filter, keep everything
        # else. No voxel grouping, so no spill is needed.
        ensure_free_disk(output.parent, min_free_bytes=min_free_bytes)
        return _write_pass(
            source,
            output,
            include_classes,
            exclude_classes,
            None,
            chunk_points,
            report,
            min_free_bytes,
        )
    if workdir is None:
        with tempfile.TemporaryDirectory(prefix="ahn_cli_voxel_") as tmp:
            return _thin_with_spill(
                source,
                output,
                size,
                include_classes,
                exclude_classes,
                Path(tmp),
                chunk_points,
                report,
                min_free_bytes,
            )
    return _thin_with_spill(
        source,
        output,
        size,
        include_classes,
        exclude_classes,
        workdir,
        chunk_points,
        report,
        min_free_bytes,
    )


def _thin_with_spill(
    source: Path,
    output: Path,
    size: float,
    include_classes: tuple[int, ...],
    exclude_classes: tuple[int, ...],
    workdir: Path,
    chunk_points: int,
    report: ProgressCallback,
    min_free_bytes: int,
) -> int:
    """Run the spill -> partition -> reduce -> merge -> write voxel thinning.

    ``kept == 0`` (every point class-filtered out) short-circuits passes 2-4:
    there is nothing to partition, reduce, or merge, so pass 5 writes an empty
    output directly.
    """
    spill = workdir / _SPILL_SUBDIR
    if spill.is_symlink() or spill.is_file():
        # A stale symlink or plain file at the spill path (not something
        # this pipeline writes, but a crashed or foreign process might
        # leave one) would make rmtree raise -- and a symlink must be
        # removed as a link, never followed into its target.
        spill.unlink()
    elif spill.is_dir():
        shutil.rmtree(spill)
    spill.mkdir(parents=True)
    try:
        ensure_free_disk(workdir, min_free_bytes=min_free_bytes)
        ensure_free_disk(output.parent, min_free_bytes=min_free_bytes)
        segments, kept, scale, offset, origin = _spill_pass(
            source,
            include_classes,
            exclude_classes,
            spill,
            chunk_points,
            min_free_bytes,
        )
        survivors_path: Path | None = None
        if kept > 0:
            partition_dir = spill / "partitions"
            partition_dir.mkdir()
            partition_paths = _partition_pass(
                segments,
                scale,
                offset,
                origin,
                size,
                _partition_count_for(kept),
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
        return _write_pass(
            source,
            output,
            include_classes,
            exclude_classes,
            survivors_path,
            chunk_points,
            report,
            min_free_bytes,
        )
    finally:
        shutil.rmtree(spill, ignore_errors=True)


def _class_keep(
    classification: npt.NDArray[np.generic],
    include: tuple[int, ...],
    exclude: tuple[int, ...],
) -> npt.NDArray[np.bool_]:
    """Return the boolean keep-mask for the classification filter over a chunk.

    A point is kept when its class is in ``include`` (or ``include`` is empty) and
    not in ``exclude``. Empty on both sides keeps every point. Mirrors the
    in-memory ``transform._class_mask`` on a per-chunk classification array.
    """
    keep = np.ones(classification.shape[0], dtype=np.bool_)
    if include:
        keep &= np.isin(classification, np.asarray(include))
    if exclude:
        keep &= ~np.isin(classification, np.asarray(exclude))
    return keep


def _spill_pass(
    source: Path,
    include: tuple[int, ...],
    exclude: tuple[int, ...],
    spill: Path,
    chunk_points: int,
    min_free_bytes: int,
) -> tuple[
    list[Path],
    int,
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Stream ``source``, spilling each class-kept point's raw coords + index.

    ``idx`` is the point's dense index in the class-filtered cloud, assigned in
    streamed (file order) across chunks. Buffers accumulate in memory and are
    flushed to a new segment file once ``_SEGMENT_BYTES`` is reached, so each
    segment is written in one open+append+close.

    Returns the segment paths in write order, the total kept count, the
    header's per-axis scale/offset (needed to reconstruct scaled coordinates in
    pass 2), and the running per-axis minimum of the *scaled* kept coordinates
    (the voxel grid's origin) -- computed from ``chunk.x/y/z``, exactly the
    values the in-memory reference sees.
    """
    kept = 0
    origin = np.full(3, np.inf, dtype=np.float64)
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
            # Written record-array by record-array: concatenating first
            # would transiently double the segment's memory.
            for record in buffer:
                record.tofile(handle)
        segments.append(path)
        buffer = []
        buffered_bytes = 0
        seg_no += 1

    with laspy.open(str(source)) as reader:
        scale = np.asarray(reader.header.scales, dtype=np.float64)
        offset = np.asarray(reader.header.offsets, dtype=np.float64)
        for chunk in reader.chunk_iterator(chunk_points):
            cls_keep = _class_keep(
                np.asarray(chunk.classification), include, exclude
            )
            count = int(cls_keep.sum())
            if count == 0:
                continue
            record = np.zeros(count, dtype=_RAW_DTYPE)
            record["X"] = np.asarray(chunk.X)[cls_keep]
            record["Y"] = np.asarray(chunk.Y)[cls_keep]
            record["Z"] = np.asarray(chunk.Z)[cls_keep]
            record["idx"] = np.arange(kept, kept + count, dtype=np.int64)
            xs = np.asarray(chunk.x, dtype=np.float64)[cls_keep]
            ys = np.asarray(chunk.y, dtype=np.float64)[cls_keep]
            zs = np.asarray(chunk.z, dtype=np.float64)[cls_keep]
            origin[0] = min(origin[0], float(xs.min()))
            origin[1] = min(origin[1], float(ys.min()))
            origin[2] = min(origin[2], float(zs.min()))
            buffer.append(record)
            buffered_bytes += record.nbytes
            kept += count
            if buffered_bytes >= _SEGMENT_BYTES:
                flush()
    flush()
    return segments, kept, scale, offset, origin


def _partition_count_for(kept: int) -> int:
    """Return the pass-2 partition count for ``kept`` spilled points.

    ``ceil(kept * record_size / _PARTITION_TARGET_BYTES)``, clamped to
    ``[_PARTITION_MIN, _PARTITION_MAX]`` -- enough partitions that each fits
    comfortably in memory during pass 3's reduction.
    """
    raw = math.ceil(
        kept * _PARTITION_DTYPE.itemsize / _PARTITION_TARGET_BYTES
    )
    return min(_PARTITION_MAX, max(_PARTITION_MIN, raw))


def _partition_pass(
    segments: list[Path],
    scale: npt.NDArray[np.float64],
    offset: npt.NDArray[np.float64],
    origin: npt.NDArray[np.float64],
    size: float,
    partition_count: int,
    partition_dir: Path,
    min_free_bytes: int,
) -> list[Path]:
    """Hash-partition every spilled point onto its voxel cell (a grace hash join).

    Processes one segment at a time -- read in bounded-size blocks,
    reconstructed to scaled coordinates, quantized to a voxel cell, routed
    through a :class:`~ahn_cli.prep.spill.PartitionWriter` keyed by a hash of
    the cell -- deleting the segment once every block is routed. Returns the
    partition file paths :meth:`~ahn_cli.prep.spill.PartitionWriter.close`
    reports.

    Failure modes:
        - :class:`ValueError` if a voxel cell coordinate does not fit in
          int32 (the cloud's extent is too large for ``size``).
    """
    writer = PartitionWriter(
        partition_dir, partition_count, min_free_bytes=min_free_bytes
    )
    for segment in segments:
        record_count = segment.stat().st_size // _RAW_DTYPE.itemsize
        with segment.open("rb") as handle:
            remaining = record_count
            while remaining > 0:
                block = min(_PARTITION_READ_RECORDS, remaining)
                records = np.fromfile(handle, dtype=_RAW_DTYPE, count=block)
                remaining -= block
                partition_records, partition_ids = _partition_block(
                    records,
                    scale,
                    offset,
                    origin,
                    size,
                    partition_count,
                    segment,
                )
                writer.append(partition_ids, partition_records)
        segment.unlink()
    return writer.close()


def _partition_block(
    records: npt.NDArray[np.void],
    scale: npt.NDArray[np.float64],
    offset: npt.NDArray[np.float64],
    origin: npt.NDArray[np.float64],
    size: float,
    partition_count: int,
    segment: Path,
) -> tuple[npt.NDArray[np.void], npt.NDArray[np.int64]]:
    """Quantize one block of raw records to voxel cells and their partition ids."""
    x = records["X"].astype(np.float64) * scale[0] + offset[0]
    y = records["Y"].astype(np.float64) * scale[1] + offset[1]
    z = records["Z"].astype(np.float64) * scale[2] + offset[2]
    cx = np.floor((x - origin[0]) / size).astype(np.int64)
    cy = np.floor((y - origin[1]) / size).astype(np.int64)
    cz = np.floor((z - origin[2]) / size).astype(np.int64)
    _check_cell_range(cx, cy, cz, segment)

    hashed = (
        cx.astype(np.uint64) * _HASH_K1
        ^ cy.astype(np.uint64) * _HASH_K2
        ^ cz.astype(np.uint64) * _HASH_K3
    )
    partition_ids = (hashed % np.uint64(partition_count)).astype(np.int64)

    out = np.zeros(records.shape[0], dtype=_PARTITION_DTYPE)
    out["cx"] = cx
    out["cy"] = cy
    out["cz"] = cz
    out["idx"] = records["idx"]
    return out, partition_ids


def _check_cell_range(
    cx: npt.NDArray[np.int64],
    cy: npt.NDArray[np.int64],
    cz: npt.NDArray[np.int64],
    segment: Path,
) -> None:
    """Reject voxel cell coordinates that do not fit in int32.

    ``cx``/``cy``/``cz`` are never empty: every pass-2 read block holds at
    least one record (the segment-reading loop only calls this on a
    positive block size).

    Failure modes:
        - :class:`ValueError` if any axis's cell index falls outside
          ``[INT32_MIN, INT32_MAX]``.
    """
    lo = min(int(cx.min()), int(cy.min()), int(cz.min()))
    hi = max(int(cx.max()), int(cy.max()), int(cz.max()))
    if lo < _INT32_MIN or hi > _INT32_MAX:
        msg = (
            f"voxel cell coordinate out of int32 range while partitioning "
            f"{segment} (cell range [{lo}, {hi}]); the cloud's extent is too "
            "large for its voxel size."
        )
        raise ValueError(msg)


def _reduce_pass(
    partition_paths: list[Path], runs_dir: Path, min_free_bytes: int
) -> list[Path]:
    """Reduce each partition to its surviving (smallest-idx-per-cell) indices.

    Every point of a given voxel cell lands in the same partition (by
    construction of the pass-2 hash), so this local reduction is exact
    regardless of how many further re-split levels a given partition needed
    (see :func:`_reduce_partition`): a cell's points can never be separated
    across sub-partitions, since every re-split level hashes the same cell
    coordinates. Deletes each partition file once it is read; writes the
    sorted survivor indices of each partition (or re-split sub-partition) as
    its own run.
    """
    run_paths: list[Path] = []
    for index, path in enumerate(partition_paths):
        run_paths.extend(
            _reduce_partition(
                path, runs_dir, f"{index:06d}", min_free_bytes, depth=0
            )
        )
    return run_paths


def _reduce_partition(
    path: Path,
    runs_dir: Path,
    label: str,
    min_free_bytes: int,
    *,
    depth: int,
) -> list[Path]:
    """Reduce one partition file to its surviving sorted-index run(s).

    A partition whose file size still exceeds :data:`_PARTITION_MAX_BYTES`
    (only reachable once :func:`_partition_count_for` has clamped at
    :data:`_PARTITION_MAX` for an extremely large cloud) is re-hash-
    partitioned into :data:`_RESPLIT_FACTOR` further sub-partitions with a
    depth-salted hash -- via :func:`_resplit_partition`, itself streamed in
    bounded blocks -- and each sub-partition is reduced recursively, up to
    :data:`_MAX_RESPLIT_DEPTH`. An in-budget partition (the common case) is
    read whole and reduced directly, exactly as before this recursion
    existed.
    """
    size = path.stat().st_size
    if size > _PARTITION_MAX_BYTES and depth < _MAX_RESPLIT_DEPTH:
        sub_paths = _resplit_partition(
            path, runs_dir, label, depth, min_free_bytes
        )
        path.unlink()
        run_paths: list[Path] = []
        for sub_index, sub_path in enumerate(sub_paths):
            run_paths.extend(
                _reduce_partition(
                    sub_path,
                    runs_dir,
                    f"{label}_{sub_index:04d}",
                    min_free_bytes,
                    depth=depth + 1,
                )
            )
        return run_paths
    records = np.fromfile(path, dtype=_PARTITION_DTYPE)
    path.unlink()
    order = np.lexsort(
        (records["idx"], records["cz"], records["cy"], records["cx"])
    )
    ordered = records[order]
    is_start = np.ones(len(ordered), dtype=np.bool_)
    if len(ordered) > 1:
        is_start[1:] = (
            (ordered["cx"][1:] != ordered["cx"][:-1])
            | (ordered["cy"][1:] != ordered["cy"][:-1])
            | (ordered["cz"][1:] != ordered["cz"][:-1])
        )
    survivors = np.sort(ordered["idx"][is_start])
    run_path = runs_dir / f"run_{label}.bin"
    write_sorted_run(run_path, survivors, min_free_bytes=min_free_bytes)
    return [run_path]


def _resplit_partition(
    path: Path,
    runs_dir: Path,
    label: str,
    depth: int,
    min_free_bytes: int,
) -> list[Path]:
    """Stream-read an oversized partition and re-hash it into fresh sub-files.

    Reads ``path`` in ``_PARTITION_READ_RECORDS``-record blocks (never the
    whole oversized file at once) and routes each block through a fresh
    :class:`~ahn_cli.prep.spill.PartitionWriter` keyed by a depth-salted
    rehash of each record's already-computed voxel cell -- a different salt
    per depth so cells that collided at one level spread apart at the next.
    Returns the sub-partition file paths :meth:`PartitionWriter.close`
    reports.
    """
    sub_dir = runs_dir / f"resplit_{label}_{depth:02d}"
    sub_dir.mkdir()
    salt = _RESPLIT_SALT_BASE * np.uint64(depth + 1)
    writer = PartitionWriter(
        sub_dir, _RESPLIT_FACTOR, prefix="sub", min_free_bytes=min_free_bytes
    )
    record_count = path.stat().st_size // _PARTITION_DTYPE.itemsize
    with path.open("rb") as handle:
        remaining = record_count
        while remaining > 0:
            block = min(_PARTITION_READ_RECORDS, remaining)
            records = np.fromfile(handle, dtype=_PARTITION_DTYPE, count=block)
            remaining -= block
            hashed = (
                records["cx"].astype(np.int64).astype(np.uint64) * _HASH_K1
                ^ records["cy"].astype(np.int64).astype(np.uint64) * _HASH_K2
                ^ records["cz"].astype(np.int64).astype(np.uint64) * _HASH_K3
                ^ salt
            )
            sub_ids = (hashed % np.uint64(_RESPLIT_FACTOR)).astype(np.int64)
            writer.append(sub_ids, records)
    return writer.close()


class _SurvivorCursor:
    """Sequential consumer of the merged global survivor-index stream.

    Wraps :func:`~ahn_cli.prep.spill.iter_sorted_values`, buffering blocks and
    handing back the prefix below a caller-supplied bound. Pass 5 calls
    :meth:`take_window` with a strictly increasing bound once per chunk, so
    each call only ever needs to look forward from where the previous call
    left off. ``path`` of ``None`` yields a cursor that always returns empty
    (used for the identity/empty-survivor-set paths, which need no cursor).
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
    source: Path,
    output: Path,
    include: tuple[int, ...],
    exclude: tuple[int, ...],
    survivors_path: Path | None,
    chunk_points: int,
    report: ProgressCallback,
    min_free_bytes: int,
) -> int:
    """Re-stream ``source`` and write the surviving points to ``output``.

    ``survivors_path`` is ``None`` for the grade-0 identity (every class-kept
    point is written) or an empty survivor set (``kept == 0``); otherwise it
    is the merged global survivor-index stream from :func:`merge_sorted_runs`,
    consumed window-by-window against each chunk's class-filtered index range
    via :class:`_SurvivorCursor`. The write goes to a sibling temp file
    swapped into ``output`` at the end, so a source-equals-output in-place
    thin is safe. Ticks ``report(chunk, total_chunks)`` per streamed chunk --
    the only pass that reports progress. Returns the written count.

    Failure modes:
        - :class:`~ahn_cli.prep.spill.DiskFloorError` if an output chunk
          write -- or the writer's close-time header/chunk-table
          finalisation, guarded with ``_FINALIZE_HEADROOM_BYTES`` -- would
          breach the floor. On *any* failure -- floor breach, source read
          error, interrupt -- the partial temp file is removed in a
          ``finally`` and ``output`` is left untouched: the swap into
          ``output`` is the last operation inside the guarded block.
    """
    tmp_out = output.with_name(f"{output.stem}.tmp{output.suffix}")
    cursor = _SurvivorCursor(survivors_path, chunk_points)
    written = 0
    filtered = 0
    try:
        with laspy.open(str(source)) as reader:
            writer = laspy.open(str(tmp_out), mode="w", header=reader.header)
            finalised = False
            try:
                total = int(reader.header.point_count)
                total_chunks = max(1, -(-total // chunk_points))
                point_size = reader.header.point_format.size
                for chunk_no, chunk in enumerate(
                    reader.chunk_iterator(chunk_points), start=1
                ):
                    cls_keep = _class_keep(
                        np.asarray(chunk.classification), include, exclude
                    )
                    count = int(cls_keep.sum())
                    if survivors_path is None:
                        point_keep = cls_keep
                    else:
                        local = (
                            cursor.take_window(filtered + count) - filtered
                        )
                        point_keep = np.zeros(len(chunk), dtype=np.bool_)
                        point_keep[np.flatnonzero(cls_keep)[local]] = True
                    filtered += count
                    selected = chunk[point_keep]
                    if len(selected) > 0:
                        ensure_free_disk(
                            tmp_out.parent,
                            len(selected) * point_size,
                            min_free_bytes=min_free_bytes,
                        )
                        writer.write_points(selected)
                        written += len(selected)
                    report(chunk_no, total_chunks)
                # Closing the writer finalises the LAZ header/chunk table --
                # the one write the per-chunk guards above don't cover. The
                # close is explicit, directly after its guard: a
                # context-managed writer would finalise in ``__exit__`` even
                # when the guard raises, and a genuinely failing close would
                # supersede the guard's typed error.
                ensure_free_disk(
                    tmp_out.parent,
                    _FINALIZE_HEADROOM_BYTES,
                    min_free_bytes=min_free_bytes,
                )
                writer.close()
                finalised = True
            finally:
                if not finalised:
                    # Best-effort teardown; the temp is discarded anyway and
                    # a close failure must never mask the original error.
                    with contextlib.suppress(Exception):
                        writer.close()
        tmp_out.replace(output)
    finally:
        # No-op after a successful replace (the temp no longer exists);
        # removes the partial temp on every failure path, including ones the
        # floor guard doesn't raise (source read errors, interrupts).
        tmp_out.unlink(missing_ok=True)
    return written
