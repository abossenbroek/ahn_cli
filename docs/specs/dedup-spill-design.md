# Out-of-core tile deduplication: dependency-free external spill design

Status: accepted (2026-07-16). Adds `ahn_cli/prep/dedup_stream.py` alongside the
in-memory oracle `ahn_cli/prep/dedup.py`, mirroring the external-spill machinery
of `docs/specs/voxel-spill-design.md`.

## Problem

The in-memory oracle `dedup.deduplicate_tiles` reads every tile whole
(`reader.read()`), concatenates the cropped records into one merged array, and
sweeps exact duplicates with a single `np.unique` over `(X, Y, Z, gps_time)`.
Merging a national-scale AOI (billions of points, hundreds of overlapping AHN
sheets) therefore holds roughly twice the whole cloud in memory — the audited
`prep dedup` OOM — and the process is killed (SIGKILL / exit 137).

## Goals

- **Byte-identical output to the oracle.** `stream_deduplicate_tiles(tiles,
  output_path, *, workdir, progress, chunk_points, min_free_bytes)` produces a
  LAZ whose bytes hash identically to `deduplicate_tiles(tiles, output_path)`:
  same crop, same offset reprojection, same survivor set, same order.
- **Dependency-free internals:** numpy + stdlib only, reusing
  `ahn_cli/prep/spill.py`.
- **Bounded, observable disk usage** under the scratch `workdir`, with the same
  hard 20 GB free-space floor (`spill.MIN_FREE_DISK_BYTES`) checked before every
  write site.
- **Peak memory independent of point count** — never more than one chunk of
  points plus a few bounded spill buffers resident; never `reader.read()`.

## Non-goals

- No change to the oracle (it stays the reference), the crop/sweep semantics, the
  CLI surface, or provenance.
- No resumability of interrupted runs (spill is scratch; a rerun restarts).
- No pipeline `dedup` stage adapter — that is a later workstream; this module is
  the algorithm only.

## Oracle semantics being matched

1. **Crop before merge.** Each tile is cropped to its canonical extent with the
   half-open `[minx, maxx) x [miny, maxy)` rule on the tile's own world
   coordinates, so a point on a shared edge is claimed by exactly one tile.
2. **Offset reprojection.** Each cropped point is cast onto the harmonized
   header (`harmonize_headers`, first tile's scale/offset) via the
   `header.offsets - source.offsets` correction, expressed through laspy's
   scaled-array round-trip so the reprojected int32 `X/Y/Z` are bit-identical.
3. **Exact-duplicate sweep.** The key is the reprojected `(X, Y, Z, gps_time)`.
   Of each duplicate group the survivor is the one with the smallest **global**
   index — its position in the merged record, assigned continuously across tiles
   in input (sequence) order, then in-tile file order. Survivors are emitted in
   ascending global-index order (the oracle's `np.sort` over `np.unique`'s
   stable first-occurrence indices).

## Architecture

Reuses `ahn_cli/prep/spill.py` unchanged (`PartitionWriter`, `write_sorted_run`,
`merge_sorted_runs`, `iter_sorted_values`, `ensure_free_disk`, `DiskFloorError`,
`advise_no_cache`). All intermediates live under `<workdir>/dedup_spill/`,
recreated empty at start (directory, stale file, or stale symlink removed as a
link, never followed) and removed in a `finally`.

### Record format (packed little-endian numpy dtype)

Spill and partition records share one 28-byte dtype:
`("X","<i4"), ("Y","<i4"), ("Z","<i4"), ("gps_time","<f8"), ("idx","<i8")`.
`X/Y/Z` are the **reprojected** LAS int32 coordinates (the exact sweep key);
`idx` is the dense global merged index.

### Passes (`cropped > 0`)

1. **Spill.** Stream each tile in `chunk_points` chunks (never `reader.read()`),
   apply the half-open crop and the offset reprojection per chunk (elementwise,
   so chunking is bit-identical to whole-tile), assign the global index
   continuously, and append records to `seg_%06d.bin` rolled at
   `_SEGMENT_BYTES` (256 MB). Counts pre-crop input and cropped totals. Ticks
   `progress(tile, total_tiles)` once per tile.
2. **Partition.** `partition_count = clamp(ceil(cropped * 28 /
   _PARTITION_TARGET_BYTES), 1, 4096)`. For each segment (deleted after use),
   read record blocks and route each record through `PartitionWriter` keyed by
   `(X·K1 ^ Y·K2 ^ Z·K3 ^ gps_bits·K4) mod partition_count` on uint64
   (Knuth/xxhash-style odd constants; `gps_time` reinterpreted to its u64 bit
   pattern). Every point of a given exact key lands in one partition.
3. **Reduce.** Per partition file (deleted after use): load the slab,
   `np.lexsort((idx, gps_time, Z, Y, X))`, keep the first record of every key
   run (boundary diff over all four key fields) — the key's minimum idx — then
   `np.sort` those survivor idxs and `write_sorted_run`.
4. **Merge.** `merge_sorted_runs` → one ascending `survivors.bin` of every
   surviving global index.
5. **Write.** Re-stream the tiles, re-crop/re-reproject each chunk identically
   (so the per-chunk global-index windows align with pass 1), consume the
   survivor stream over each chunk's `[filtered, filtered + count)` window via a
   `_SurvivorCursor`, scatter the selected local positions, and `write_points`
   to a sibling temp file swapped into `output` at the end. The output writer is
   opened once with the **harmonized** header, spanning every tile's chunks —
   verified byte-identical to the oracle's one-shot `LasData.write`.

`cropped == 0` (every point cropped away) short-circuits passes 2–4: the
survivor stream is `None`, so pass 5 writes an empty output.

### Disk-floor guard sites

Start of run (workdir volume and output volume), every segment append, every
partition-buffer flush and run/merge write (inside `spill.py`), before each
output chunk write, and — with `_FINALIZE_HEADROOM_BYTES` (4 MiB) — before the
LAZ writer's close-time finalisation. The finalisation is an explicit
`writer.close()` immediately after its guard (a context-managed close would run
even when the guard raises and its own failure would supersede the typed error);
on failure paths the writer is closed best-effort with its error suppressed. A
breach raises `DiskFloorError`; the `finally` removes the spill dir; the temp
output (if any) is removed; `output` is never replaced by a partial file.

### Determinism and bit-exactness

- The survivor **set** (min idx per key) and its ascending-idx order are
  independent of chunk size, segment boundaries, partition count, hash
  constants, buffer sizes, and merge fan-in — so the output is byte-identical
  across machines and parameter overrides, and equal to the oracle.
- Crop and reprojection reuse laspy's scaled-array evaluation exactly as the
  oracle, so the reprojected int keys and the written records are bit-identical.
- All spill I/O uses explicit little-endian packed dtypes.

### gps_time hashing subtlety (for downstream consumers)

The partition hash reinterprets `gps_time` as raw f8 → u64 bits. Two
float-equal but bit-differing gps values (`-0.0` vs `0.0`, or distinct NaN
payloads) would therefore hash into different partitions and be treated as
distinct keys — unlike the oracle's float-equality `np.unique`, which would
collapse them. Real AHN `gps_time` is a large positive GPS-seconds value, so
this edge never arises; it is documented here because a caller synthesising
degenerate gps values could observe a one-point divergence from the oracle.

## Testing

`tests/prep/test_dedup_stream.py`, 100 % branch coverage, oracle =
`deduplicate_tiles`:

- **Byte-identity** (`sha256` equal) across overlapping-seam, exact-duplicate,
  differing-offset, disjoint-passthrough, and same-tile-twice fixtures, each at
  several chunk sizes.
- Half-open crop; offset-reprojection onto one lattice; cross-tile duplicate →
  global-first-index survivor and ascending order.
- **Property: chunk / segment / partition / read-block invariance** — one hash
  across every knob combination, equal to the oracle.
- Empty tile list, non-positive `chunk_points`, and missing `gps_time`
  (PDRF < 6) → typed `ValueError`.
- **Bounded memory** — `LasReader.read` patched to raise, the pipeline still
  completes.
- Stale-spill self-healing (dir / plain file / symlink).
- **Disk floor** breached at startup, at the segment spill, at an output chunk,
  and at finalisation (plus a failing-close variant) → `DiskFloorError`, output
  untouched, spill removed, no partial temp; and a generic mid-write failure
  with the same cleanup.
