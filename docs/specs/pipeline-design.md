# `pipeline` verb: fuse the verbs into a tile-streaming pipeline

Status: accepted (2026-07-16). Implements Workstream W9 of the `pipeline`
scale epic (`epic/pipeline-scale`). Depends on W0–W8 (the `ahn_cli/pipeline/`
foundation, executor, spec, machine sensing, and the per-tile stage adapters).

## Problem

Every standalone verb processes the **whole area of interest at once** and
round-trips a full intermediate artifact through disk to the next verb
(`pointcloud.laz` → `reconciled.exr` → tileset). Two audits found this causes
OOM in ~10 places and makes a Westland-scale run (~90 km², billions of points,
a ~14-billion-pixel ortho) impossible. The chosen fix is a **`pipeline` verb**
that streams one **spatial tile** end-to-end through fused stages *in RAM*: a
tile (an AHN sheet plus a correctness halo) is small, so a tile-scoped stage
reuses the existing in-memory verb logic unchanged and no whole-area
intermediate is ever materialized.

W0–W8 built the foundation: the `Stage` protocol and `TilePayload` value objects
(`model.py`), the resumable tile-streaming executor (`executor.py`), the
PDAL-style YAML/JSON spec (`spec.py`), the RAM-adaptive tiling/halo sizer
(`tiling.py`), and the per-tile `dedup`/`thin`/`reconcile`/`tiles3d` stage
adapters — each proven byte-identical to its standalone verb at the stage level.
W9 is the **integration seam**: it wires a parsed spec into those seams, drives
the executor, and assembles the sink's deliverable.

## Goals

- One `ahn_cli pipeline run <spec.(yaml|json)>` verb: parse the spec, wire the
  planner/source/stages/sink, call `run_pipeline`, translate `PipelineError`
  into a `click.ClickException`.
- **Fusion identity**: a single-tile run is byte-identical to the standalone
  verbs in sequence (`reconcile` → `tiles3d`, plus each sink profile's
  sidecars).
- **Tiling invariance**: a multi-tile run stitches back to a whole-area
  standalone run — the halo floor makes edge kNN identical — and is byte
  identical across two different injected free-RAM budgets (RAM-adaptive sizing
  is a pure performance knob).
- **Resumability**: a kill at any tile boundary, then a resume, reproduces an
  uninterrupted run's deliverable with no partial/temp survivor.
- **Bounded memory**: peak RSS flat as the tile count grows; no whole-AOI
  `pointcloud.laz` / `reconciled.exr` / ortho mosaic ever written.
- 100% branch coverage on all new code, no `# pragma: no cover`; RAM, clock and
  network always injected, never live.

## Non-goals (this workstream)

- **The live Westland pod run is deferred** (the RunPod is down). The local
  multi-tile validation below is the proof of scale-correctness; the exact
  command + a sample `westland.yaml` are documented at the end for the user to
  run when a pod is available.
- **A `fetch` source** (network acquisition per tile). Only the `read` source
  is wired — a pre-populated on-disk site. Pre-fetch with the `fetch` verb and
  point a `read` stage at the site. `run_spec` rejects a `fetch` source with a
  clear typed error naming this.
- **End-to-end byte-identity for a multi-*level* `tiles3d` quadtree.** The
  parent-LOD path needs a strided ortho window + a strided-mesh sink the current
  W7 `Tiles3dSink` does not implement (it reconstructs a tile via a uniform
  `rasterio.from_bounds` transform, which cannot reproduce the quadtree's
  `sample_indices` remainders). The single-tile (`levels == 0`) `tiles3d` path
  is byte-identical and verified; the cloud/`write` sink covers the multi-tile
  gates. See "Known limitation" below.

## Architecture

The verb is a thin adapter over `run_spec` (`ahn_cli/pipeline/run.py`), which
composes six new pieces around the existing executor:

### 1. Concrete planners — `ahn_cli/pipeline/planners.py`

The sink chooses the output grid (a pure function of the area of interest, never
of RAM — the two-budget invariant depends on it):

- **`QuadtreePlanner`** (tiles3d sink): measures the area in pixels at the
  ortho's native resolution (`aoi_pixel_dims`), reuses
  `tiles3d.quadtree.plan_quadtree` unchanged, and emits one `TileContext` per
  node (root first, then children depth-first). A leaf's bbox is the exact
  pixel-edge extent of its inclusive pixel span, so the tiles3d sink
  reconstructs it pixel-for-pixel.
- **`GridTilePlanner`** (cloud/`write` sink, re-exported from `tiling.py`): a
  clean, pixel-aligned partition of the area with no shared boundaries — the
  shape the reconcile stage's halo-kNN identity is proven against.

### 2. Concrete source + ortho windows — `ahn_cli/pipeline/sources.py`

Network-free, on-disk seams:

- **`ReadSource`** (`TileSource`): reads each AHN sheet's extent from its LAZ
  header once at construction; per tile, selects the sheets overlapping the
  tile's bbox grown by `ctx.halo_m`, reads and crops their points, and returns a
  `PointTile` plus a **cheap content hash** of the source files' identities
  (name/size/mtime — never a bulk re-serialization of the points). A tile only
  loads the sheets it overlaps, so peak memory is bounded by the tile.
- **`WindowedOrtho`** (`OrthoWindows`): reads the global ortho GeoTIFF windowed
  per tile and returns a **pixel-aligned** sub-window — the sub-grid transform
  shifts only the translation coefficients (`c' = a·col0 + c`,
  `f' = e·row0 + f`), keeping the pixel size — so a tiled estimate lands on
  exactly the global grid's pixel centres (the W6 CRITICAL note). If tile edges
  did not fall on pixel boundaries byte-identity would break.

### 3. Spec → stage wiring — `ahn_cli/pipeline/wiring.py`

Pure translation of the spec's self-explanatory keys into the verbs' own value
objects:

- **`grade_for_voxel_size`**: the spec stores `voxel_size_m` (metres); the
  `prep` voxel thinner is graded. Only an **exact** grade edge length resolves
  (`0.0`→0, `0.25`→1, `1.0`→3, …, `64.0`→9); a size with no exact grade is a
  hard `PipelineError` (the gap is reconciled explicitly, never silently
  rounded).
- **`neighbors_for`**: the halo floor's neighbour count — the method's `k` for
  IDW/kriging; `LinearInterp` carries none, so a conservative
  `DEFAULT_LINEAR_NEIGHBORS = 6` (twice a triangle's three vertices) is used.

### 4. Dedup + write stage adapters — `ahn_cli/pipeline/stages/{dedup,write}.py`

- **`DedupStage`**: the in-memory tile-local adapter over the `prep` dedup
  contract — class filter, then the exact `(x, y, z, gps_time)` duplicate sweep
  keeping the smallest index in ascending order (the cross-sheet crop-before-
  merge is the source's job). `halo_m == 0`.
- **`GridWriteSink`**: the cloud sink — encodes a tile's reconciled `GridTile`
  into one deterministic, self-describing `"grid"` blob (heights `float32` then
  the RGB `uint8` planes). A later assembly places each tile's grid at its pixel
  offset.

### 5. Cross-tile tiles3d assembly — `ahn_cli/pipeline/assemble.py`

The executor streams one tile at a time and persists each tile's blobs; nothing
in that stream knows the tileset's tree shape, and `ManifestEntry` deliberately
carries no region field (the W7 note). So a `Tiles3dSink` is wrapped by
`run.py`'s `_RegionRecordingSink`, which appends each tile's own region as a
small `region.json` blob (persisted and resume-skipped like the geometry/texture
blobs — no executor or manifest change). `assemble_tiles3d` then stitches the
persisted per-tile blobs into the standalone verb's exact deliverable:

- folds regions **children-first** (`union_region`), so a parent's bounding
  volume contains all descendants (mirroring `tiles3d.emit`);
- strict: writes `tiles/<level>-<tx>-<ty>.glb` + a loose `tileset.json`;
- packed: writes `tiles.hfp` via `tiles3d.pack.write_pack` + `tileset.json` +
  `provenance.json` + `manifest.json`, reproducing
  `tiles3d.build._write_packed` byte for byte.

### 6. Orchestration — `ahn_cli/pipeline/run.py`

`run_spec(spec, *, point_spacing_m, probe, cpu_count, pool_factory, fault)`:

1. Resolve the `read` source's `(cloud_dir, ortho_path)` from the site
   (`<site>/ahn/*.laz` or `<site>/*.laz`; `<site>/ortho.tif` or
   `<site>/ortho/ortho.tif`).
2. Build `WindowedOrtho`; the area of interest is the ortho's full extent and
   the pixel size its native resolution.
3. Build the middle stages (`dedup`/`thin`/`reconcile`) and the sink; compute
   `halo_floor_m = max(stage.halo_m())` (the reconcile floor dominates).
4. Call `run_pipeline` (resumable, bounded, injectable RAM/CPU/pool/fault). A
   tiles3d sink stages its blobs under `workdir/_tiles3d_store` then assembles
   into `output`; a cloud sink writes straight into `output`.

### The CLI verb — `ahn_cli/cli/app.py`

A `pipeline` group with a `run` subcommand: `ahn_cli pipeline run <spec>`
(`.yaml`/`.yml` parsed as YAML, everything else as JSON), plus an optional
`--point-spacing-m`. It parses the spec, calls `run_spec`, and translates
`PipelineError` into a `click.ClickException` — no logic of its own.

## Invariants and how they hold

| Invariant | Mechanism |
|---|---|
| **Fusion identity** | The stage adapters are byte-identical to their standalone verbs (W5–W7); the assembler reproduces `build._write_strict`/`_write_packed` exactly. |
| **Tiling invariance** | The planner grid is a pure function of the AOI; the reconcile halo floor makes every tile-edge kNN set equal a global run's, so stitched output == whole-area output. |
| **Two-budget identity** | RAM only sizes the halo (always ≥ floor) and the cross-tile concurrency; neither changes a tile's output. The resume input hash excludes `halo_m`. |
| **Resumability** | The executor's per-tile two-phase commit (`os.replace` marker) — a kill leaves an uncommitted tile that reprocesses cleanly; the region blob is persisted with the tile, so resume needs no recompute for the assembler. |
| **Bounded memory** | One tile (+halo) resident; the source loads only overlapping sheets; no whole-AOI intermediate is written (asserted). |
| **Determinism** | Planner order, injected RAM/CPU/pool, and sorted manifests make a run byte-identical across machines, worker counts and interrupt/resume histories. |

## Known limitation: multi-level `tiles3d`

Byte-identical end-to-end `tiles3d` is proven and wired for a **single-tile**
(`levels == 0`) area. A multi-*level* quadtree needs a strided ortho window and a
strided-mesh sink; the current `Tiles3dSink` (W7) reconstructs a tile's grid
through a uniform `from_bounds` transform, which cannot reproduce
`quadtree.sample_indices`' non-uniform remainder columns for a parent LOD. The
cloud/`write` sink (a clean partition, no LODs) carries the multi-tile /
two-budget / resumability gates instead. Completing the parent-LOD path (a
`stride`-aware `WindowedOrtho` + a `Tiles3dSink` that meshes at the tile's
stride) is the follow-up to make `tiles3d` scale past one tile.

## Test matrix (all network-free, RAM/CPU/pool/fault injected)

Foundation contracts (W0–W8) keep their own suites; W9 adds:

- **`test_run.py`** — the load-bearing gates:
  - `test_single_tile_fusion_is_byte_identical` (strict, splat, heightfield):
    `run_spec` == standalone `reconcile` → `build_tiles3d`, hash-tree equal.
  - `test_multi_tile_with_halo_matches_whole_area`: a 2×2 `write` run stitches
    back to the whole-area standalone `reconcile` grid (heights + colour).
  - `test_two_ram_budgets_are_byte_identical`: tiny vs generous injected free
    RAM → identical deliverable grid blobs.
  - `test_resume_after_fault_is_byte_identical`: a fault at a tile boundary,
    then a resume, reproduces a clean run's grid blobs; `processed`/`skipped`
    reflect the partial resume.
  - `test_no_whole_area_intermediate_is_written`: no `pointcloud.laz` /
    `reconciled.exr` / `reconciled.laz` anywhere.
  - full-chain (dedup + thin) wiring, misplaced-middle-stage error, fetch-source
    error, read-site-layout variants, non-directory read path.
- **`test_planners.py`** — `aoi_pixel_dims` (+ bad pixel size / sub-pixel /
  degenerate bbox), single-tile and multi-level quadtree, too-small-for-a-
  surface, `tree_for`/`levels_for`.
- **`test_sources.py`** — `find_ahn_sheets` errors; `ReadSource` load/crop,
  content hash, empty tile, RGB carry; `WindowedOrtho` pixel-alignment, band-
  count and out-of-bounds errors.
- **`test_wiring.py`** — `grade_for_voxel_size` (exact + off-grid error),
  `thinning_for` (voxel/poisson), `neighbors_for` (idw/kriging/linear).
- **`test_assemble.py`** — region blob round trip, missing-region error, strict
  multi-tile children-first region union.
- **`stages/test_dedup.py`**, **`stages/test_write.py`** — the two new stage
  adapters (filter, exact-duplicate sweep, grid-blob codec, error paths).
- **`tests/cli/test_pipeline_cli.py`** — the verb over YAML and JSON,
  `--point-spacing-m`, and `PipelineError` → `ClickException` translation.

## Deferred: running Westland on a pod

When a pod is available, pre-fetch the site then run the pipeline. The spec
below assumes the AHN sheets and the mosaicked ortho already live under
`data/westland` (the `fetch` verb's output layout: `data/westland/ahn/*.LAZ` +
`data/westland/ortho.tif`):

```bash
# 1. Acquire the site (network; per-tile parallel downloads).
uv run ahn_cli fetch --out data/westland -c westland --ortho --source geotiles -j 8

# 2. Stream the fused pipeline over it (bounded RSS, resumable).
uv run ahn_cli pipeline run westland.yaml
```

`westland.yaml`:

```yaml
aoi:      { geojson: westland.geojson }   # or { bbox: "x0,y0,x1,y1" }
tiling:   { tile_pixels: 256, halo: auto }
workdir:  /workspace/scratch
output:   data/westland/tiles3d
stages:
  - { type: read,      path: data/westland }
  - { type: dedup,     include_classes: [1, 2, 6] }
  - { type: thin,      method: voxel, voxel_size_m: 1.0 }   # or voxel_grade: 3
  - { type: reconcile, method: idw, idw: { power: 2, neighbors: 12 } }
  - { type: tiles3d,   profile: splat }
```

Notes for the pod run:

- The `read` source is offline: `fetch` populates `data/westland` first; the
  pipeline never touches the network.
- Kill-safe: re-running the same command resumes, recomputing only unfinished
  tiles.
- A cloud/`write` sink (`- { type: write, path: cloud }`) exercises the same
  streaming path and, unlike `tiles3d`, is not limited to a single tile today —
  use it to validate bounded RSS at Westland scale while the multi-level
  `tiles3d` parent-LOD path (see "Known limitation") is completed.
```
