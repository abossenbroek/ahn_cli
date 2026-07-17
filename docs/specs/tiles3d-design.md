# 3D Tiles export (`tiles3d`) + hard dimension gates — design

Date: 2026-07-11
Status: approved for implementation (autonomous goal directive)
Standard: OGC 22-025r4 — 3D Tiles 1.1 (https://docs.ogc.org/cs/22-025r4/22-025r4.html)

## Goal

1. The output steps gain an orthophoto-map conversion to OGC 3D Tiles 1.1:
   a new `tiles3d` bounded context + CLI subcommand that drapes the site's
   orthophoto over the reconciled per-pixel heights and emits a
   `tileset.json` + glb quadtree.
2. All verification and reconcile steps ensure the two data dimensions
   (imagery grid and height grid) **match perfectly**; any missing data —
   missing file, missing pixel, void interpolation estimate, NaN height,
   grid mismatch — is a hard, typed error. Data is never infilled.
3. The 3D Tiles output is verified with the strictest checks possible:
   every artifact is re-read from disk after writing and validated
   byte-for-byte against an independent recomputation from the sources;
   any failure deletes the outputs and raises.

## Alternatives considered

- **Depend on `py3dtiles`/`pygltflib`/`trimesh`** — rejected. None are in
  the dependency set; they give up control over byte determinism and make
  the "recompute independently and compare bit-exact" verifier impossible
  to state. Hand-packed binary writers are the house style (EXR, PT, COPC).
- **Single-tile tileset (one glb)** — rejected. A city ortho is ~10⁸
  pixels; a single mesh is unusable and unloadable. A quadtree with
  REPLACE refinement is the standard-intended shape.
- **Quadtree with stride-sampled LODs, hand-written glb/PNG writers, and
  a full post-write verifier** — chosen.

## Inputs and the dimension-match contract

`ahn_cli tiles3d --ortho <ortho.tif> --heights <reconciled.exr> --out <dir>`

- `--ortho`: the fetched orthophoto (RGB GeoTIFF, EPSG:28992). Texture
  ground truth.
- `--heights`: the `reconcile` EXR output (6-channel B,G,R,X,Y,Z float32,
  EPSG:7415 heights). Geometry ground truth. The pipeline is
  `fetch → prep → reconcile --format exr → tiles3d`.

Input gates (all → `Tiles3dError`):

- ortho unreadable / <3 bands / `uniform_image` sample → error (same gates
  as reconcile's `open_ortho`).
- heights EXR unreadable, wrong magic/version/compression/channel-set,
  truncated scanlines, or offset-table inconsistency → error (strict
  reader for our own deterministic uncompressed format).
- **Perfect dimension match**: EXR `dataWindow` width/height must equal the
  ortho's `PixelGrid` width/height exactly; the EXR X and Y planes must
  equal `float32` of the ortho pixel-centre coordinates derived from the
  ortho geotransform, bit-exact, at every pixel; the EXR B,G,R planes must
  equal `float32(ortho_band / 255.0)` bit-exact at every pixel. Any single
  mismatching pixel → error naming the first offending pixel.
- **No missing data**: any non-finite Z anywhere → error. `flat_surface`
  on the Z plane → error (placeholder guard, consistent with dsm /
  positions gates).

## Reconcile tightening (existing context)

- `_verify_cloud_overlaps_grid` becomes a **coverage** gate: the cloud XY
  bounding box must contain the full pixel-centre extent of the ortho grid
  (`cloud.min_x <= centre_min_x`, etc. on all four sides). Partial overlap
  is no longer accepted; the error names each uncovered side and the
  shortfall in metres.
- **Void estimates are errors**: after each interpolated block, if any
  pixel's `valid` mask is False, raise `ReconcileError` ("N pixel(s) have
  no genuine estimate; missing data is an error, never infilled"),
  removing all partial outputs (existing never-leave-rejected-artifacts
  pattern). The EXR `0.0` void sentinel path becomes unreachable from
  `reconcile()`; the writers keep their mask API (independently
  unit-tested).

## The `tiles3d` bounded context

```
ahn_cli/tiles3d/
  __init__.py   # docstring: "3D Tiles 1.1 export bounded context …"
  errors.py     # Tiles3dError(Exception)
  exr.py        # strict reader for reconcile's uncompressed EXR
  sources.py    # load ortho + heights, all input/dimension gates → TerrainGrid
  geodesy.py    # pyproj: EPSG:7415 → EPSG:4978 (ECEF) and → EPSG:4979 (region)
  quadtree.py   # tiling plan: pixel spans per tile, LOD strides, geometric errors
  mesh.py       # sampled grid → positions (RTC float32, y-up swizzle), UVs, indices
  png.py        # deterministic stdlib-zlib PNG encoder (RGB8, filter 0)
  gltf.py       # glb assembly: JSON chunk + BIN chunk, exact accessor min/max
  tileset.py    # tileset.json emission (sorted keys, deterministic)
  verify.py     # the strict post-write verifier (see below)
  build.py      # build_tiles3d() orchestrator; cleanup on any failure
```

- New declared dependency: `pyproj>=3.7` (already resolved transitively;
  promoted to a direct dependency because `geodesy.py` imports it).

### Tiling plan (`quadtree.py`)

- Leaf tiles span up to 256×256 pixels; adjacent tiles share their
  boundary pixel column/row so edge vertices coincide (no cracks within a
  level, no fabricated vertices).
- Level k above the leaves samples every `2^k`-th pixel of its span,
  always including the last column/row so the full extent is preserved.
  Every vertex at every level is a genuine source sample — no averaging,
  no synthesis. Textures are sampled with the same stride (nearest).
- Levels: `ceil(log2(max(ceil(w/256), ceil(h/256))))`; a grid ≤256 px is a
  single root leaf.
- `refine: "REPLACE"` on the root; children inherit.
- `geometricError`: leaves 0; level k tiles `stride_k × max(|px|, |py|) × 4`;
  tileset geometric error = 2× the root tile's. Monotone non-increasing
  root→leaf by construction.

### Geometry (`mesh.py`, `geodesy.py`, `gltf.py`)

- Vertex positions: `(x, y, z)` EPSG:7415 → EPSG:4978 ECEF (float64,
  pyproj), minus the tile's RTC centre (mean of the tile's ECEF min/max),
  quantised to float32, swizzled to glTF y-up as `(x, z, −y)`. The RTC
  centre is carried as the glTF root-node `translation` (also swizzled),
  so runtime y-up→z-up rotation (§spec) reproduces ECEF exactly.
- One mesh, one primitive: POSITION + TEXCOORD_0 (float32), uint32
  indices, two CCW triangles per grid cell. Material:
  `pbrMetallicRoughness` with `baseColorTexture`, `metallicFactor 0`,
  `doubleSided: true`; no extensions. Texture: embedded PNG bufferView,
  sampler CLAMP_TO_EDGE / NEAREST-safe filtering; texel centres map to
  vertex UVs `(i + 0.5)/n`.
- POSITION accessor `min`/`max` are the exact componentwise extremes of
  the written float32 data.
- glb: little-endian, 4-byte-aligned JSON + BIN chunks, `asset.version
  "2.0"`, deterministic JSON (sorted keys, no timestamps).
- `boundingVolume.region` per tile: `[west, south, east, north, minH,
  maxH]` in EPSG:4979 radians/metres, computed as the exact min/max of the
  tile's own vertices transformed to EPSG:4979 — containment by
  construction; parent regions additionally expanded to enclose all
  descendant content (spec: content of children fully inside parent's
  volume).

### tileset.json (`tileset.py`)

`asset: {"version": "1.1", "generator": "ahn_cli tiles3d"}`, root as
above, `content.uri` relative glb paths (`tiles/<level>-<x>-<y>.glb`).
Deterministic serialisation: sorted keys, `indent=2`, trailing newline.

## The strict output verifier (`verify.py`)

Runs unconditionally as the final step of `build_tiles3d()`. It re-reads
**everything from disk** and validates against an **independent
recomputation from the input rasters**. Any failure → outputs deleted →
`Tiles3dError`. Checks:

1. **Tileset structure**: parses; exact expected key sets at every level
   (unknown keys are errors — we authored the file); `asset.version ==
   "1.1"`; root `refine == "REPLACE"`; every `geometricError` finite,
   ≥ 0, and every child's ≤ its parent's; tileset GE ≥ root GE.
2. **Regions**: `west < east`, `south < north`, lon ∈ [−π, π], lat ∈
   [−π/2, π/2], `minH ≤ maxH`; every child region ⊆ parent region.
3. **Content links**: every `content.uri` resolves to an existing file
   strictly inside the output directory; **every** glb file under the
   output directory is referenced exactly once (no orphans, no reuse).
4. **glb container**: magic/version/declared length == actual file size;
   chunk alignment; JSON chunk first, BIN chunk second, nothing after.
5. **glTF internals**: buffer byteLength == BIN chunk length; every
   bufferView within the buffer; every accessor within its bufferView;
   POSITION `min`/`max` recomputed from the binary payload and compared
   **bit-exact**; index count divisible by 3; every index < vertex count;
   no degenerate triangle (three distinct indices); every UV ∈ [0, 1].
6. **PNG**: signature, chunk layout, **CRC-32 of every chunk verified**,
   IHDR dims equal the tile's expected sample dims, zlib stream inflates,
   decoded byte count == 3 × w × h (+ filter bytes), decoded pixels
   compared bit-exact to the sampled ortho pixels.
7. **Geometry ground truth**: for every tile, the verifier independently
   re-samples the sources, re-runs the EPSG:7415→ECEF transform, re-derives
   RTC-relative float32 positions and compares them **bit-exact** to the
   glb payload; UVs and indices likewise.
8. **Containment**: every vertex's EPSG:4979 coordinate lies inside its
   tile's region and inside every ancestor's region.
9. **Full coverage, nothing missing**: leaf pixel spans tile the full grid
   exactly — every pixel belongs to a leaf, overlaps only on shared
   boundary rows/columns; any gap → error.
10. **Cross-artifact re-check**: the ortho/EXR dimension-match gates from
    `sources.py` are re-run inside the verifier from fresh disk reads.

## Orchestrator, CLI, provenance

- `build_tiles3d(ortho: Path, heights: Path, out: Path, *, tile_pixels:
  int = 256, profile: Profile = STRICT, progress: ProgressCallback | None =
  None, workers: int | None = None, pool_factory: PoolFactory | None = None)
  -> Tiles3dBuildResult` (frozen result: paths, tile count, levels, vertex /
  triangle totals). `workers` defaults to the CPU count; `pool_factory` is the
  test seam. On any failure, all partially written outputs are removed (copc
  `build.py` pattern).
- CLI subcommand `tiles3d` in `cli/app.py`: options `--ortho`
  (exists), `--heights` (exists), `--out` (dir), `--profile`, `--workers`
  (`IntRange(min=1)`, default all cores), tqdm progress, and
  `Tiles3dError → click.ClickException` (exit 1); Click arg errors exit 2.
- Provenance: parity with `copc` (result object returned; no sidecar).

## Testing

- `tests/tiles3d/` mirroring the package; synthetic fixtures: small
  (6×6 and 20×14) ortho GeoTIFF + matching EXR built by a conftest factory
  (reusing `reconcile`'s writer to guarantee format fidelity), plus
  deliberately corrupted variants for every negative gate.
- Every verifier check gets a negative test that corrupts the specific
  bytes it guards (flipped CRC, truncated chunk, off-by-one index,
  perturbed POSITION min, orphan glb, mismatched X plane, NaN Z, …).
- Reconcile tightening: new tests for partial-coverage error and
  void-estimate error; existing void-tolerant tests updated.
- CLI tests per house convention (exit 0 / 1 / 2, progress spy).
- External conformance (manual, not in `make test`): render the Moerkapelle
  output in CesiumJS and run the Cesium 3d-tiles-validator; documented in
  the doc's follow-up section, mirroring how copc used copc-validator.
- Gates unchanged: 100% branch coverage, pyright strict, ruff ALL;
  `tests/test_coverage_gate.py`'s `FUTURE_GATED` gains `"tiles3d"`.

## Bookkeeping

- `pyproject.toml`: add `pyproj>=3.7`.
- `cli/__init__.py` docstring + CLAUDE.md: document the new subcommand and
  the `reconcile --format exr` prerequisite, and the stricter reconcile
  contract (full coverage required; voids are errors).

## Follow-up: parallel per-tile encode (W10, 2026-07-17)

**Finding.** A real 2×2 km Westland run showed the standalone `tiles3d`
build is single-threaded — ~47 min for 21,845 splat tiles. The pack writer
(`pack.write_pack`) drove one CPU-bound `blob_source(key)` encode per tile
*inline* on the calling thread; the writer's own work is pure I/O, so one
core did all the encoding (gaussian build + zstd for splat;
quantize/meshopt/JPEG for game; `.hf`/JPEG for heightfield; float32 glTF +
PNG for strict) while the rest sat idle. Every tile is an independent,
deterministic pure function of the terrain, so the encode fans out trivially.

**Design — bounded reorder window (`parallel.py`).** `ordered_encode(keys,
encode, *, workers, window, pool_factory)` encodes up to `window` tiles
concurrently but yields results in the **exact** `keys` order, so the writer
keeps streaming to disk in canonical order with the output bytes unchanged.
Two properties hold at once — parallel encode *and* bounded on-disk streaming:
at most `window` encoded blobs are ever resident (default `window = 2 *
workers`, a small multiple of the worker count), independent of the tile
count. It never "encodes everything into RAM, then writes". `workers == 1`
runs inline with no pool — the byte-for-byte serial reference.

The driver primes `window` futures, then for each result it drains (in
submission = key order) it submits one more, so the in-flight/resident set is
capped at `window` for any tile count. `write_pack` and the strict loose-file
writer (`build._write_strict`) both drain this iterator; `emit.compute_build`
was made lazy to match `compute_packed_build` (it holds a per-tile
`blob_source`, not a materialised glb dict), so the strict path is bounded too.

**Thread pool, not process pool.** The heavy per-tile work is numpy array
math plus C-extension codecs (zstd, Pillow JPEG, pyproj) that all release the
GIL, so a `ThreadPoolExecutor` parallelises the hot sections without pickling
anything. A process pool would have to pickle the large in-memory terrain grid
to every worker and pickle the `blob_source` closure — which is not picklable
at all — for no gain over threads on GIL-releasing codecs. The thread pool
shares the one terrain grid and keeps the encode a pure function, so
byte-identity and determinism are automatic: results are consumed in key order
regardless of finish order. The pool + worker count are injectable seams
(`PoolFactory`, `workers`) for deterministic tests.

**Byte-identity guarantee (enforced by construction).** The build writes with
`workers = cpu_count`; the unconditional post-write verifier
(`verify._verify_packed_byte_identity` / `_verify_byte_identity`) rebuilds
every artifact **serially** (`workers = 1`) and byte-compares. Any divergence
between the parallel and serial encodings would fail verification and reject
the build — so parallel == serial is checked on every single build, not just
in tests. `--workers` (CLI) / `workers=` (`build_tiles3d`) select the count;
the free-disk floor is reused (`pack.require_free_disk` wraps
`prep.spill.ensure_free_disk`, translating `DiskFloorError → Tiles3dError`) and
checked at every blob write, so a breach leaves the crash-safe swap's held
previous deliverable untouched.

**Measured speedup.** Isolating the encode via `write_pack` (BytesIO sink,
serial `workers=1` vs `workers=cpu_count`) over an 85-tile plan of real 256px
production leaves (a 2048×2048 smooth scene, 4 levels) on a 12-core host —
`dataset_id` byte-identical between the two runs in every case:

| profile | serial | parallel (12) | speedup |
|---|---|---|---|
| splat | 8.00 s | 1.61 s | 4.97× |
| game | 7.69 s | 1.37 s | 5.62× |
| heightfield | 7.06 s | 1.33 s | 5.32× |

That is near-linear on the CPU-bound codecs (the sub-12× reflects Amdahl —
the serial tree-walk plan, the pack writer's own I/O, and short-stream codec
overhead). The win requires substantial per-tile work: at 32px tiles the encode
is dominated by Python-level array setup that holds the GIL, so threads add
overhead and *lose* (~0.6×) — which is why the production leaf is 256px and the
encode-dominated Westland splat case is where the gain is largest. `workers=1`
is byte-for-byte the pre-W10 serial path. See
`tests/tiles3d/test_parallel_build.py` for the
byte-identity / determinism / bounded-window / disk-floor / crash-safety locks.

## Follow-up: conformance results (2026-07-11)

- Real-data smoke: the Amsterdam LFS fixtures (256x256 Beeldmateriaal
  ortho window + AHN5 cloud) ran `reconcile --format exr` ->
  `tiles3d`; the built-in strict verifier passed.
- External conformance: Cesium's `3d-tiles-validator` on that output
  reported **0 errors, 0 warnings, 0 infos**.
- The Moerkapelle site has no fetched ortho on disk (only the uniform
  placeholder grid, which the gates refuse by design); run
  `ahn_cli fetch --ortho` there before a full-site conversion.
