# Out-of-core voxel thinning: dependency-free external spill design

Status: accepted (2026-07-15). Supersedes the Polars/Parquet internals of
`ahn_cli/prep/voxel_stream.py` introduced by `feat/prep-voxel-out-of-core`.

## Problem

The out-of-core voxel path currently spills `(x, y, z, idx)` to per-chunk
Parquet files and delegates the group-by-voxel → min-index reduction to
Polars' streaming engine. That works, but:

1. It adds a heavyweight dependency (`polars`, ~30 MB wheel, its own Rust
   runtime and thread pool) for one group-by.
2. Its disk usage is opaque: Polars' streaming engine spills to its own
   temporary locations with no hook to bound or observe it, so a
   national-scale run can fill the volume. On macOS, filling the system
   volume degrades and can break the machine (APFS needs free space for
   snapshots, swap, and purgeable management).

## Goals

- Same public contract: `stream_voxel_thin(source, output, grade,
  include_classes, exclude_classes, *, workdir, chunk_points, progress)`
  produces byte-identical output to today's implementation (which itself
  matches the in-memory reference in `prep/decimate.py`): per occupied voxel
  the smallest class-filtered index survives, survivors written in ascending
  index order, all attributes preserved, deterministic.
- Dependency-free internals: numpy + stdlib only. `polars` leaves
  `pyproject.toml`.
- Bounded, observable disk usage under the scratch `workdir`, with a hard
  free-space floor: **every** write site checks the target volume's free
  space first and raises a typed error when it would leave less than
  20 GB (decimal, `20_000_000_000` bytes — the unit macOS reports to users)
  free. The run aborts cleanly (spill removed, output untouched) instead of
  filling the disk.
- Peak memory independent of point count (as today).

## Non-goals

- No change to the Poisson path, grade semantics, CLI surface, or provenance.
- No resumability of interrupted runs (spill is scratch; a rerun restarts).
- No cross-platform direct-I/O tuning beyond a best-effort macOS
  `F_NOCACHE` hint.

## Approaches considered

1. **External merge sort of full records by voxel key** (classic): sort
   20-byte records in memory-sized runs, k-way merge, then a sequential
   grouped scan keeps the first (min-idx) record per cell. Correct, but the
   bulk data is comparison-sorted and rewritten log-fan-in times; the final
   survivors then still need re-sorting by idx.
2. **Grace-hash partitioned aggregation** (chosen — what DuckDB-class
   engines do for out-of-core group-by): hash-partition records by cell so
   every cell lands wholly in one partition; reduce each partition in
   memory; only the tiny survivor-index streams (8 B each) go through a
   k-way external merge. Past the initial spill the bulk data sees one
   rewrite and two sequential reads (repartition read+write, then the
   reduce read) — never a comparison sort on disk — and the in-memory
   reduce is a vectorized sort of a bounded slab.
3. **Keep Polars** — rejected by directive (dependency-free) and by goal 3
   (unobservable spill).

## Architecture

Two new/changed modules:

- **`ahn_cli/prep/spill.py` (new)** — reusable, numpy+stdlib external-spill
  machinery, independently unit-tested:
  - `MIN_FREE_DISK_BYTES = 20_000_000_000` and
    `ensure_free_disk(directory, incoming_bytes, *, min_free_bytes)` →
    raises `DiskFloorError(RuntimeError)` when
    `shutil.disk_usage(directory).free - incoming_bytes < min_free_bytes`.
    The message reports the directory, the free bytes, and the floor.
  - `advise_no_cache(fileobj)` — best-effort
    `fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)` so large spill I/O does not evict
    the page cache (what SQLite does on macOS); silently a no-op where
    `fcntl`/`F_NOCACHE` is unavailable.
  - `PartitionWriter` — buffered fan-out writer: `append(partition_ids,
    records)` splits a structured-array batch by partition id (stable
    argsort + searchsorted) into per-partition in-memory buffers, visiting
    only the partitions present in the batch and copying each slice (a view
    would pin the whole batch, breaking the byte accounting); when the
    global buffered total exceeds `buffer_bytes` (default 256 MB) all
    buffers flush. A flush checks the disk floor with the exact byte count
    about to be written, then opens the partition file in append mode,
    writes the buffered chunks sequentially with `ndarray.tofile` (never
    concatenated first — concatenation would transiently double the
    buffered memory), and closes — so at most one spill file
    handle is ever open regardless of partition count, and a breach cannot
    create a stray empty file. A partition's buffer is cleared (and the
    byte accounting decremented) only after its write succeeds, so a
    floor-breach flush failure is exactly-once retry-safe (the check
    precedes the open); a failure inside a write keeps the buffer but may
    leave a partial append, so callers treat the spill as poisoned (the
    voxel pipeline removes it wholesale). `close()` flushes and returns
    the per-partition paths that exist.
  - `write_sorted_run(path, values)` — disk-floor-checked `<i8` run writer.
  - `merge_sorted_runs(runs, out_dir, *, fan_in=64, buffer_items)` — k-way
    merge of sorted `<i8` run files into one sorted file, merging at most
    `fan_in` runs per pass (bounded file handles), repeating passes until
    one file remains; inputs are deleted as they are consumed. The merge is
    vectorized: per round, take `bound = min` over the open runs of their
    current buffer's last value, gather every buffered value `<= bound`,
    `np.sort` the concatenation, emit, refill exhausted buffers — O(fan_in)
    Python operations per ~buffer-sized emission, never itemwise.
  - `iter_sorted_values(path, *, buffer_items)` — buffered sequential
    reader yielding numpy blocks of the merged survivor file.
- **`ahn_cli/prep/voxel_stream.py` (rewritten internals)** — same public
  function and docstring contract, plus a keyword-only
  `min_free_bytes: int = MIN_FREE_DISK_BYTES` (threaded to every guard;
  tests lower/raise it without monkeypatching the OS). All intermediates
  live under `<workdir>/voxel_spill/` exactly as today (recreated empty at
  start, removed in a `finally`).

`ahn_cli/prep/transform.py` maps `DiskFloorError` to the context's typed
`PrepError` at the `prepare()` boundary (no import cycle: `spill.py` imports
nothing from `transform.py`).

### Record formats (packed little-endian numpy dtypes)

- Raw spill record (pass 1): `("X","<i4"), ("Y","<i4"), ("Z","<i4"),
  ("idx","<i8")` — 20 B. X/Y/Z are the **unscaled LAS int32** coordinates
  (`chunk.X` …), not float64, saving 37.5 % spill versus float records.
- Partition record (pass 2): `("cx","<i4"), ("cy","<i4"), ("cz","<i4"),
  ("idx","<i8")` — 20 B.
- Survivor runs and merged survivors: plain `<i8`.

### Passes (grade > 0)

1. **Spill.** Stream the LAZ in `chunk_points` chunks (default 1 M, as
   today), apply the class filter, and append raw records to segment files
   `seg_%06d.bin` rolled at `SEGMENT_BYTES = 256 MB`; track the running
   float64 per-axis minimum of the *scaled* kept coordinates (from
   `chunk.x/y/z`, exactly the values the in-memory reference sees) and the
   kept count. Disk floor checked before every segment append.
2. **Partition.** `partition_count = clamp(ceil(kept * 20 /
   PARTITION_TARGET_BYTES), 1, 4096)` with `PARTITION_TARGET_BYTES =
   128 MB`. For each segment in order (deleted immediately after
   processing, so peak scratch ≈ one dataset copy + one segment): read
   record blocks, reconstruct `x = X * scale + offset` in float64 (the
   identical expression laspy's `ScaledArrayView` evaluates, so
   bit-identical to pass 1 / the reference), quantize `cell = floor((x -
   origin) / size)` as int64, verify each axis fits int32 (raise
   `ValueError` on absurd extents), and route `(cx, cy, cz, idx)` through
   `PartitionWriter` keyed by `(cx·K1 ^ cy·K2 ^ cz·K3) mod
   partition_count` on uint64 (Knuth/xxhash-style odd constants; wraparound
   is well-defined and deterministic).
3. **Reduce.** Per partition file (deleted after use): load the whole slab
   (≤ ~128 MB by construction), `np.lexsort((idx, cz, cy, cx))`, keep the
   first record of every cell run (boundary diff) — that is the cell's
   minimum idx — then `np.sort` those survivor idxs and
   `write_sorted_run`.
4. **Merge.** `merge_sorted_runs` → a single sorted `survivors.bin`.
5. **Write.** Re-stream the LAZ, class-filter each chunk, consume the
   sorted survivor stream for the chunk's filtered-index window
   `[filtered, filtered + kept_in_chunk)`, scatter onto the chunk mask via
   `np.flatnonzero(cls_keep)`, and append to a sibling temp file swapped
   into `output` at the end (unchanged). Progress ticks `(chunk,
   total_chunks)` in this pass only — unchanged UX and tests.

Grade 0 stays the existing single filtered pass (no spill). `kept == 0`
short-circuits passes 2–4 (empty survivor set).

### Disk-floor guard sites

Start of run (workdir volume and output volume), every segment append,
every partition-buffer flush, every run write, every merge output block,
before each output chunk in pass 5, and — with a generous fixed headroom
(`_FINALIZE_HEADROOM_BYTES`, 4 MiB) — before the LAZ writer's close-time
header/chunk-table finalisation, the one write that has no per-chunk
guard. The finalisation is performed as an explicit `writer.close()`
immediately after its guard: a context-managed close would run even when
the guard raises, and its own failure would supersede the guard's typed
error; on failure paths the writer is instead closed best-effort with its
error suppressed, so the original error always reaches the caller. A
breach raises `DiskFloorError`; the `finally` removes the spill dir; the
temp output (if any) is removed; `output` is never replaced by a partial
file. At the `prepare()` boundary, `OSError` and `ValueError`
from the voxel path are wrapped into `PrepError` alongside
`DiskFloorError`, so no streaming failure reaches the CLI as a raw
traceback.

A scratch `workdir` must not be shared by concurrent prep runs: every run
claims the same `voxel_spill/` subdirectory and clears it (directory,
stale file, or stale symlink — removed as a link, never followed) at
start, which is also what makes crashed-run state self-healing.

### Determinism and bit-exactness

- The survivor **set** (min idx per cell) is independent of segmentation,
  partition count, hash constants, buffer sizes, and merge order — so the
  output is byte-identical across machines and parameter overrides.
- Cell assignment is bit-exact with `decimate._voxel_group_ids`: same
  float64 `X*scale + offset` evaluation, same float64 running min (min is
  associative and order-insensitive for these totals), same
  `floor((coords - origin)/size)` expression.
- All spill I/O uses explicit little-endian packed dtypes.

## Testing

- Keep every existing test in `tests/prep/test_voxel_stream.py` (the
  numpy-oracle contract is unchanged); the stale-spill test drops its
  `.parquet` filename for a neutral one.
- New parity test: randomized cloud (fixed seed), survivors equal to
  `decimate.decimate_voxel(coords, grade, backend=NumpyBackend())` applied
  to the class-filtered coords.
- Multi-segment, multi-partition, and multi-merge-pass paths exercised by
  shrinking the module constants (monkeypatch) / passing small parameters
  to `spill.py` directly.
- Disk floor: unit tests for `ensure_free_disk` (monkeypatched
  `shutil.disk_usage`) plus an end-to-end `stream_voxel_thin` run with
  `min_free_bytes` set above the real free space, asserting `DiskFloorError`,
  no output mutation, and spill cleanup; `prepare()` maps it to `PrepError`.
- Cell int32 range check covered via a `scale=1.0` header with extreme
  coordinates.
- `advise_no_cache` covered on both the available and unavailable branches
  (monkeypatch).
- 100 % branch coverage on both modules, pyright strict, ruff ALL — the
  repo gates.

## Implementation plan (staged agents, sequential)

1. **Haiku — mechanical sweep.** Remove `polars==1.42.1` from
   `pyproject.toml`, run `uv lock`; update the two CLAUDE.md bullets
   (`voxel_stream.py` description, Key-libraries list) and
   `transform.py`'s `workdir` docstring wording to the design above (spill
   is "packed binary segment/partition files", not Parquet/Polars).
2. **Sonnet — core.** Implement `prep/spill.py` + rewrite
   `prep/voxel_stream.py` internals + `transform.py` error mapping + full
   test suites per this spec.
3. **Opus — adversarial verify.** Independent review against this spec
   (bit-exactness, guard-site completeness, cleanup on every failure path,
   handle bounds), then drive `make check` to green including 100 %
   coverage.
