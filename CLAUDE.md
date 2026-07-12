# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Development Setup
```bash
# Install dependencies with uv (preferred)
make install

# Install pre-commit hooks (strict ruff lint + format, typos, pyright)
uv run pre-commit install

# Update dependencies
make update
```

### Testing
```bash
# Run all tests (network-free; 100% branch coverage required on non-legacy code)
make test

# Run the nightly suite (hits real PDOK/Beeldmateriaal endpoints; not run by default)
make test-nightly

# Run a specific test file
uv run pytest tests/path/to/test_file.py

# Run a specific test
uv run pytest tests/path/to/test_file.py::test_function_name

# Run tests with verbose output
uv run pytest -v
```

### Linting and Formatting
```bash
# Run linter (ruff check)
make lint

# Check for typos in Python files
make typos

# Type-check (pyright, strict mode)
make typecheck

# Format code (ruff format)
make format

# Check formatting without changing files
make format-check

# Fix linting issues automatically
make fix

# Run all checks (lint, typos, typecheck, test, format-check) — exactly what CI runs
make check
```

### Running the CLI
`ahn_cli` is a Click **group**; every invocation names a subcommand (`fetch`, `prep`, `reconcile`, `copc`, `tiles3d`, `import-viirs`, `export-positions`). Running it with no subcommand prints usage to stderr and exits with code 2 — that is expected `click.Group` behavior, not a bug.

```bash
# Run the CLI with arguments (ARGS must start with a subcommand)
make run ARGS="fetch --out data/delft -c delft"

# Or directly with uv
uv run ahn_cli fetch --out data/delft -c delft

# Acquire raw AHN + DSM + orthophoto tiles for a city, latest generation auto-selected
uv run ahn_cli fetch --out data/delft -c delft --dsm --ortho

# Acquire with a bounding box instead of a city, pinned to AHN4, via GeoTiles fallback
uv run ahn_cli fetch --out data/utrecht -b 194198.0,443461.0,194594.0,443694.0 --ahn AHN4 --source geotiles

# Acquire with a GeoJSON polygon
uv run ahn_cli fetch --out data/area -g my_area.geojson

# Transform a fetched site: filter classes, dedup, export pointcloud.laz (+ .ply)
uv run ahn_cli prep --data data/delft -i 2,6 --points

# Transform with graded voxel/Poisson-disk thinning instead of raw class filtering only
uv run ahn_cli prep --data data/delft --thin-method voxel --thin-grade 3
uv run ahn_cli prep --data data/delft --thin-method poisson --thin-radius 1.5 --thin-seed 0

# Export the fetched DSM to a deterministic position map for TouchDesigner
uv run ahn_cli export-positions --data data/delft

# Import an externally-produced VIIRS GeoTIFF into the site directory
uv run ahn_cli import-viirs --out data/delft path/to/viirs.tif

# Interpolate the AHN cloud onto the ortho's pixel grid (IDW, power=2, k=12), emit a coloured cloud
uv run ahn_cli reconcile --ortho data/delft/ortho.tif --cloud data/delft/pointcloud.laz --out data/delft/reconciled --method idw --idw "2,12"

# Convert a pipeline LAZ into a validator-green Cloud-Optimized Point Cloud (streaming, 0.5m dedup)
uv run ahn_cli copc --cloud data/delft/reconciled/reconciled.laz --out data/delft/reconciled/reconciled.copc.laz

# Convert the ortho map + reconciled heights into an OGC 3D Tiles 1.1 tileset (requires `reconcile --format exr`)
uv run ahn_cli tiles3d --ortho data/delft/ortho/ortho.tif --heights data/delft/reconciled/reconciled.exr --out data/delft/tiles3d

# Same conversion in the compact game profile (quantized glTF + meshopt + JPEG, plus a provenance.json)
uv run ahn_cli tiles3d --ortho data/delft/ortho/ortho.tif --heights data/delft/reconciled/reconciled.exr --out data/delft/tiles3d --profile game

# Same conversion in the heightfield profile (vendor .hf height chunks + sibling JPEG, plus a provenance.json)
uv run ahn_cli tiles3d --ortho data/delft/ortho/ortho.tif --heights data/delft/reconciled/reconciled.exr --out data/delft/tiles3d --profile heightfield
```

A typical end-to-end run is `fetch` → `prep` → (`export-positions` and/or `reconcile`) → optionally `copc` and/or `tiles3d` (which needs `reconcile --format exr`): each step reads the previous step's output from the site directory on disk and writes its own outputs plus an updated `provenance.json`; there is no in-memory handoff between subcommands.

## Architecture

AHN CLI acquires and transforms Dutch elevation data (AHN — Actueel Hoogtebestand Nederland), plus matched DSM/orthophoto/VIIRS layers, for a given site (city, bbox, or GeoJSON area of interest). The codebase is organized as a set of **bounded contexts** behind a thin CLI adapter, each owning one stage of the pipeline; the `ahn_cli/cli/__init__.py` docstring states this explicitly, and `tests/test_bounded_contexts.py` enforces the boundary as a contract test.

### Core Components

1. **CLI adapter** (`ahn_cli/cli/app.py`): the `ahn_cli` Click group and its five subcommands (`fetch`, `prep`, `import-viirs`, `export-positions`, `reconcile`). Registered via `pyproject.toml`'s `[project.scripts] ahn_cli = "ahn_cli.cli:cli"`. This layer owns argument parsing/validation and translates each context's typed errors (`AcquisitionError`, `PrepError`, `ViirsImportError`, `PositionsExportError`, `ReconcileError`) into `click.ClickException`; it holds no acquisition or transform logic of its own. `fetch`, `prep`, and `reconcile` map one-to-one onto their own context directories below; `import-viirs` and `export-positions` don't have their own directories — they're implemented inside `fetch/viirs.py` and `prep/positions.py` respectively, since they belong to the acquisition and transform contexts respectively.

2. **`domain/`**: pure value objects shared by every context, with no I/O — `Tile`/`BBox` (identity, EPSG:28992), `PixelGrid`/`GeoTransform` (pixel ↔ world coords), `Generation` (AHN3/4/5…), `Product` (ahn/dsm/ortho/viirs), `Vintage` (acquisition year), `Provenance` (in-memory acquisition record), and `authenticity.py`'s pure data-authenticity predicates (`uniform_image` / `flat_surface` / `degenerate_cloud`) shared by every verb's output-step gate.

3. **`fetch/`** (acquisition bounded context): turns an area of interest into raw, cached source tiles. Never transforms or exports pixel/point data.
   - `acquisition.py` — orchestrates resolve-AOI → download-through-cache → record-provenance for AHN point cloud tiles.
   - `pdok.py` — primary distribution source (PDOK INSPIRE ATOM feeds).
   - `geotiles_source.py` — fallback distribution source (GeoTiles.nl).
   - `generation.py` — the AHN generation registry backing `--ahn` (including `auto` probing).
   - `dsm.py` — windowed COG fetch + clip to `dsm.tif`.
   - `ortho.py` — Beeldmateriaal orthophoto fetch (mosaic + clip + CC-BY provenance).
   - `viirs.py` — imports an externally-produced VIIRS GeoTIFF into `<site>/viirs/`.
   - `source.py` — shared `FetchSource` value objects and EPSG:28992 ↔ 4326 helpers.

4. **`prep/`** (transform/export bounded context): turns cached raw source tiles into finished deliverables. Never reaches out to a distribution portal itself.
   - `transform.py` — `prepare()` orchestrates dedup → class filter → thin → provenance → export.
   - `dedup.py` — tile de-duplication (crop-before-merge, then an exact XYZ + GPS-time sweep); reuses `harmonize_headers` from the legacy `process.py` module (still a live dependency — see Legacy modules below).
   - `decimate.py` — graded thinning: voxel-grid and Poisson-disk methods, pure-numpy reference backend with an optional Apple-silicon MLX GPU accelerator (`uv sync --extra mlx`, arm64 macOS only); CPU and GPU backends are required to produce identical voxel output.
   - `ply.py` — exports `pointcloud.ply` for TouchDesigner (`-p/--points`).
   - `positions.py` — exports `dsm.tif` to a deterministic `positions.exr` (3-channel float32 OpenEXR).

5. **`reconcile/`** (interpolation bounded context, added after the fetch/prep epic closed): interpolates the AHN point cloud onto the orthophoto's pixel grid and emits a single coloured cloud. The ortho is EPSG:28992; the AHN DSM/LAZ is EPSG:7415 (EPSG:28992 horizontally + NAP height vertically) — the two grids coincide exactly in X/Y, so no reprojection is needed and only Z (NAP height) is semantically distinct.
   - `reconcile.py` — orchestrates block-streamed interpolation and writes output.
   - `clean.py` — class filter + XY de-duplication of the source cloud before interpolation.
   - `method.py` — `LinearInterp` / `IdwInterp` / `KrigingInterp` / `Variogram` value objects.
   - `neighbors.py` — deterministic kNN via `scipy.spatial.cKDTree` (an MLX/Metal GPU spike was built and benchmarked but removed in favor of this CPU reference — see `docs/superpowers/specs/2026-07-10-reconcile-design.md`).
   - `raster.py` — raster/point-cloud IO (rasterio + laspy).
   - `writers.py` — deterministic `laz`/`ply`/`pt`/`exr` output writers.

6. **`copc/`** (COPC export bounded context, added to resolve `docs/bugs/2026-07-11-pdal-copc-xyz-bounds-flat-terrain.md`): turns a pipeline LAZ (`prep`'s or `reconcile`'s output) into a `.copc.laz` whose LAS-header bounds and COPC octree cube are consistent **by construction** — PDAL's `writers.copc` computes them through two float64 paths that disagree by an epsilon on flat, horizontally-huge Dutch terrain (every point pinned to the cube's Z-min face), failing `copc-validator`'s `xyz` check. Fully streaming (chunked reads → on-disk XY bucket spill → one bucket in memory at a time), so nationwide-scale inputs work. Design doc: `docs/superpowers/specs/2026-07-11-copc-design.md`.
   - `octree.py` — `CopcError`, `plan_build()` (whole-metre cube anchored ≥1 m outside the data, below-NAP Z included), copc.js-exact node bounds (`min + (max - min) / 2` midpoint halving, matching `Bounds.stepTo` bit for bit) and the `LodSampler` top-down grid-occupancy sampler that assigns each point to exactly one node via those same midpoints.
   - `dedup.py` — 0.5 m-voxel de-duplication preserving AHN's native coarseness: only voxels holding >1 point collapse, survivor picked by outlier reasoning (median/MAD on Z, nearest-to-median, index tie-break); points are never moved or synthesised.
   - `scatter.py` — pass-1 streaming scatter into per-column bucket record files; normalizes attributes (scan_angle_rank→scan_angle, return numbers lifted to 1..15, LAS bit-fields — synthetic/key_point/withheld/overlap, scanner_channel, scan_direction_flag, edge_of_flight_line — packed into the PDRF 6 flags byte and carried through).
   - `writer.py` — typed façade over `copclib` (vendored stub in `typings/copclib/`): nodes handed over as raw pre-packed int32 PDRF 6/7 bytes (no second quantization path), header min/max set from the written quantized extremes, per-node GPS sort, WKT1 SRS (the validator's proj4js can't parse WKT2), and a post-Close binary patch of the COPC info VLR's `gpstime_minimum/maximum` (the copclib binding never fills them).
   - `build.py` — `build_copc()` orchestrator: plan → scatter → per-bucket dedup/LOD-sample/write, ancestors above the bucket level held back and written last; RGB policy (no/black RGB → PDRF 6, 8-bit-looking RGB widened ×257, real 16-bit passthrough).
   - Verified: the real 46.3M-point Moerkapelle site passes `npx copc-validator -d` 24/24 green; `_typos.toml` carries the `lod`/`LinearNDInterpolator`/legacy-geojson spell-check exceptions this work surfaced.

7. **`tiles3d/`** (3D Tiles export bounded context): converts the orthophoto map plus `reconcile`'s EXR heights into an OGC 3D Tiles 1.1 tileset (OGC 22-025r4) — a quadtree of binary glTF terrain tiles draped with the ortho, all coordinates in ECEF with region bounding volumes in EPSG:4979 radians. The two inputs must match **perfectly**: equal dimensions, EXR X/Y planes bit-equal to the ortho's pixel centres, EXR colour planes bit-equal to this ortho's bands, every elevation finite — any mismatch or missing value is a hard `Tiles3dError`, and data is never infilled. Every vertex/texel at every LOD is a genuine source sample (stride subsampling, no averaging). The `--profile` flag selects the on-disk representation: `strict` (default) is the byte-frozen lossless float32-glTF + PNG profile and writes no sidecar, `game` emits quantized (`KHR_mesh_quantization`) + `EXT_meshopt_compression` glTF draped with baseline JPEG plus a deterministic `provenance.json`, and `heightfield` emits the vendor `.hf` height chunks (fixed header + zstd-framed `uint16` NAP-height plane; normative spec in `docs/superpowers/specs/2026-07-12-heightfield-chunk-format.md`) with sibling baseline JPEGs plus a `provenance.json`. Both lossy profiles record the pinned quantization/JPEG/encoder/zstd settings in the sidecar.
   - `errors.py` — `Tiles3dError`, the context's single typed error.
   - `profile.py` / `encoders.py` / `payload.py` — the encoder seam: `Profile` (strict|game|heightfield, `Profile.parse` at the CLI boundary, `Profile.encoder()`, `Profile.content_suffix()`/`Profile.texture_suffix()`) selects a `TileEncoder` (`StrictEncoder` / `GameEncoder` / `HeightfieldEncoder`) that turns a sampled `TilePayload` into an `EncodedTile`; emission and the swap machinery stay agnostic to the packing.
   - `quantize.py` / `jpeg.py` / `meshopt.py` / `gltf_quant.py` — the game profile's encoder layer (pure `KHR_mesh_quantization` quantizer, baseline-JPEG codec, `EXT_meshopt_compression` stream codec, quantized-glb writer); each owns and exports the constants/version helpers the provenance sidecar records.
   - `heightfield.py` — the heightfield profile's `.hf` codec (fixed little-endian header + pinned-level zstd frame of the tile's `uint16` NAP-height plane; `quantize.py`'s `quantize_axis` on the height axis only) with a Python reference decoder; the normative byte layout lives in `docs/superpowers/specs/2026-07-12-heightfield-chunk-format.md`. `verify_heightfield.py` is its per-tile verifier (chunk decode + header/requantization/dequant-bound + sibling-JPEG checks, run before the byte-identity backstop).
   - `provenance.py` — deterministic game-profile `provenance.json` codec (sorted-key JSON, no timestamps), sourcing every field from the encoder-layer modules; the verifier recomputes and byte-compares it.
   - `exr.py` — strict byte-level reader for reconcile's uncompressed EXR (exact attribute set, offset table, scanline framing; refuses truncation/trailing bytes).
   - `sources.py` — `load_terrain()` with the perfect-dimension-match gates and the `uniform_image`/`flat_surface` authenticity guards.
   - `geodesy.py` — pyproj EPSG:7415 → EPSG:4978 (ECEF) and → EPSG:4979 (radians); deterministic per machine (PROJ grid availability affects absolute heights, never self-consistency).
   - `quadtree.py` — tiling plan: shared-boundary pixel spans, per-level strides, `geometric_error` (leaves are 0).
   - `mesh.py` — RTC float32 vertex grids swizzled to glTF y-up (`(x, z, -y)`), texel-centre UVs, exact per-tile EPSG:4979 regions.
   - `png.py` / `gltf.py` / `tileset.py` — hand-packed deterministic writers (stdlib zlib PNG, glb container, sorted-key tileset.json).
   - `emit.py` — pure in-memory emission shared by build and verify (children-first, so parent regions contain all descendant content by construction).
   - `build.py` — `build_tiles3d()` orchestrator; a failed or verification-rejected build removes everything it wrote, and a previous build in the same `--out` (the tool-owned `tileset.json` + `tiles/` subtree + the game profile's `provenance.json`) is held aside during a rebuild and restored on any failure — it is only dropped once the new build passes verification. The swap is two-phase with an accept-marker file as its commit point, so re-runs are safe and a good deliverable is never destroyed, even across hard kills (SIGKILL/power loss) at any moment.
   - `verify.py` — the **strictest post-write verifier**, run unconditionally as the build's final step: re-reads every artifact from disk and checks exact tileset key sets and 1.1 rules, region validity/containment, content-link integrity (no orphans/escapes/duplicates), glb container framing, accessor bounds with bit-exact POSITION extremes, index/UV validity, CRC-verified PNG textures bit-equal to the sampled ortho, vertex containment in every enclosing region, full leaf coverage, and whole-file **byte identity** against an independent rebuild from the sources.

8. **`provenance/`**: `sidecar.py` — deterministic `provenance.json` codec shared by every fetcher/transform step.

9. **`cache/`**: `store.py`'s `ContentAddressedCache` + `key.py`'s `CacheKey` — a checksum-verified, idempotent cache keyed by (product, generation/vintage, tile id), making `fetch` safe to re-run.

### Legacy / deprecated modules

`ahn_cli/main.py`, `process.py`, `config.py`, `kwargs.py`, `validator.py`, `fetcher/`, and `manipulator/` predate the bounded-context refactor and are **not** part of the live CLI surface — `main.py`'s single-command Click interface (the old `-c/-o/-i/-e/-d/-b/-g/-p` flags) is dead code, unreferenced by `pyproject.toml`'s entry point or any other module. Each carries a `DEPRECATED` banner and a module-level `DeprecationWarning`, and each is explicitly grandfathered out of `make lint`/`make typecheck`/coverage (kept in sync across `[tool.ruff.lint.per-file-ignores]`, `[tool.coverage.run] omit`, and `[tool.pyright] exclude` in `pyproject.toml`) — a module may only be de-grandfathered by removing it from all three lists and bringing it to 100% coverage and strict typecheck.

**Exception**: `process.py` is not fully dead — `prep/dedup.py` imports `harmonize_headers` from it (with the deprecation warning explicitly suppressed) and reuses it inside the new `prep` context. Don't delete `process.py` when cleaning up the rest of the legacy modules.

### Key Design Patterns

- Each bounded context (`fetch`, `prep`, `reconcile`, `copc`, `tiles3d`) owns one pipeline stage and communicates through `domain/` value objects and the `provenance/` sidecar — no context reaches into another's internals.
- `fetch` is idempotent via the content-addressed `cache/`; `prep`, `reconcile`, `copc`, and `tiles3d` are pure transforms over already-cached/fetched inputs with no network access.
- Point cloud processing streams/block-processes rather than loading whole tiles where practical (DSM windowed COG reads, `reconcile`'s block-streamed interpolation) to manage memory on large areas.
- Deterministic outputs are a first-class requirement: `provenance.json`, the `reconcile` writers, and Poisson-disk thinning (via `--thin-seed`) are all designed to be reproducible given the same inputs.
- Every verb's output step hard-verifies data authenticity (via the `domain/authenticity.py` predicates) before emitting: only a genuine AHN cloud / genuine imagery passes — placeholders, degenerate clouds, and flat surfaces are refused with the verb's typed error. Dimensions must match perfectly and missing data is a hard error: `reconcile` requires the cloud's XY bbox to cover every ortho pixel centre and refuses any void estimate (removing partial outputs), and `tiles3d` requires bit-exact ortho/EXR grid+colour agreement and all-finite heights. Data is never infilled to satisfy a step.
- The `cli/` layer is a thin adapter: argument parsing, validation, and error translation only. All business logic lives in the bounded-context packages, never in `cli/app.py`.

### Important Constants

- **AHN classification classes**: currently only defined in the deprecated `ahn_cli/validator.py` (`AHN_CLASSES = [0, 1, 2, 6, 9, 14, 26]`) and were not migrated into any new bounded-context module:
  - 0: Created, never classified
  - 1: Unclassified
  - 2: Ground
  - 6: Building
  - 9: Water
  - 14: High tension
  - 26: Civil structure

  `prep`'s `-i/--include-class`/`-e/--exclude-class` and `reconcile`'s `--classes` options parse raw comma-separated integers with no canonical validation against this list — if you're touching class filtering, be aware there is no first-class enum for it yet in the new code.

- **Data source URLs**: no longer centralized; each `fetch/` module defines its own and deliberately does not import the deprecated `config.py`:
  - `fetch/pdok.py` — PDOK INSPIRE ATOM base (`https://service.pdok.nl/rws/ahn/atom/`) with per-generation feed URLs.
  - `fetch/geotiles_source.py` — GeoTiles.nl AHN4/AHN5 bases (`https://geotiles.citg.tudelft.nl/AHN{4,5}_T/`), the fallback source.
  - `fetch/generation.py` — a GeoTiles.nl AHN4 endpoint used by generation auto-probing.

### Coordinate Systems

The tool works primarily in the Dutch national grid, EPSG:28992 (RD New / Amersfoort), for orthophotos, bboxes, and tile identity. The AHN DSM/LAZ data is natively EPSG:7415 (EPSG:28992 horizontally, NAP height vertically) — `reconcile` relies on this to interpolate without reprojection, since only the Z axis differs semantically between the ortho grid and the AHN cloud. `fetch/source.py` provides the EPSG:28992 ↔ 4326 conversion helpers used when reading city/GeoJSON boundaries.

### Testing Structure

Tests are organized to mirror the source layout:
- `tests/domain/`, `tests/fetch/`, `tests/prep/`, `tests/reconcile/`, `tests/copc/`, `tests/tiles3d/`, `tests/provenance/`, `tests/cache/`, `tests/cli/` — one directory per bounded context / adapter, matching `ahn_cli/`.
- `tests/nightly/` — live checks against the real PDOK/Beeldmateriaal endpoints; marked `nightly`, excluded from `make test` by default (`addopts = "-m 'not nightly'"`), run explicitly via `make test-nightly`.
- `tests/test_bounded_contexts.py` — contract test asserting the `fetch`/`prep` docstrings and `domain.__all__` match the bounded-context framing described above.
- `tests/test_integration_vertical_slice.py` — end-to-end acceptance test for the full fetch → prep pipeline.
- `tests/fetcher/`, `tests/manipulator/`, and a handful of loose `tests/test_*.py` files (`test_pipeline.py`, `test_rasterize.py`, `test_validator.py`, `test_geojson_integration.py`, `test_extra_bytes_harmonization.py`) — grandfathered tests for the legacy modules listed above.

### Dependencies

The project uses modern Python tooling:
- `uv` for dependency management (pyproject.toml)
- `ruff` for linting and formatting (`select = ["ALL"]` minus formatter-conflicting rules; legacy modules exempted per-file)
- `pyright` (strict mode) for type checking
- `pytest` for testing (`--cov=ahn_cli --cov-branch`, `fail_under = 100` on non-legacy code)
- Python 3.10–3.12 supported
- Key libraries: laspy (point clouds), copclib (COPC container writing), geopandas/shapely (geometry), rasterio (rasters/COGs), scipy (kNN), pyproj (EPSG:7415 → ECEF/EPSG:4979 for 3D Tiles), pdal (optional), mlx (optional, Apple-silicon GPU decimation)

### CI

`.github/workflows/ci.yml` runs `make lint`, `make typos`, `make format-check`, `make typecheck`, `make test` on push/PR across Python 3.10/3.11/3.12, plus a separately scheduled nightly job running `make test-nightly`.
