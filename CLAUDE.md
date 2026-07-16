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

# Run the nightly suite (hits real PDOK endpoints; not run by default)
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

# Same conversion in the compact game profile: packed tiles.hfp (quantized glTF + meshopt + JPEG blobs) + tileset.json/provenance.json/manifest.json sidecars
uv run ahn_cli tiles3d --ortho data/delft/ortho/ortho.tif --heights data/delft/reconciled/reconciled.exr --out data/delft/tiles3d --profile game

# Same conversion in the heightfield profile: packed tiles.hfp (vendor .hf height chunks + sibling JPEG blobs) + tileset.json/provenance.json/manifest.json sidecars
uv run ahn_cli tiles3d --ortho data/delft/ortho/ortho.tif --heights data/delft/reconciled/reconciled.exr --out data/delft/tiles3d --profile heightfield

# Same conversion in the splat profile: packed tiles.hfp (one isotropic 3D Gaussian Splatting .ply blob per tile, no texture) + tileset.json/provenance.json/manifest.json sidecars
uv run ahn_cli tiles3d --ortho data/delft/ortho/ortho.tif --heights data/delft/reconciled/reconciled.exr --out data/delft/tiles3d --profile splat
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
   - `ortho.py` — Beeldmateriaal Nederland orthophoto fetch: resolves tiles from a GeoJSON tile index published by `basisdata.nl` (pinned to the 2025 HRL vintage; the Beeldmateriaal open-data ATOM feed this module used before is retired), verifies each download's SHA-256 against the index, then mosaics + clips (CC BY 4.0 provenance).
   - `viirs.py` — imports an externally-produced VIIRS GeoTIFF into `<site>/viirs/`.
   - `source.py` — shared `FetchSource` value objects and EPSG:28992 ↔ 4326 helpers.

4. **`prep/`** (transform/export bounded context): turns cached raw source tiles into finished deliverables. Never reaches out to a distribution portal itself.
   - `transform.py` — `prepare()` orchestrates dedup → class filter → thin → provenance → export.
   - `dedup.py` — tile de-duplication (crop-before-merge, then an exact XYZ + GPS-time sweep); reuses `harmonize_headers` from the legacy `process.py` module (still a live dependency — see Legacy modules below).
   - `decimate.py` — graded thinning: voxel-grid and Poisson-disk methods, pure-numpy reference backend with an optional Apple-silicon MLX GPU accelerator (`uv sync --extra mlx`, arm64 macOS only); CPU and GPU backends are required to produce identical voxel output. This is the **in-memory reference** for the voxel semantics; `transform.py` routes prep's voxel thinning through `voxel_stream.py` instead (the in-memory voxel path OOMs on national-scale clouds), while Poisson stays in-memory here.
   - `voxel_stream.py` — **out-of-core** voxel thinning (`stream_voxel_thin`): the memory-bounded path prep actually uses for a `VoxelThinning` request. Streams the LAZ in chunks (never `reader.read()`), spills each class-kept point's `(x, y, z, idx)` to per-chunk Parquet in a scratch `--workdir`, offloads the group-by-voxel → min-index reduction to Polars' streaming engine, then re-streams to write survivors through a temp-file swap. Same voxel contract as `decimate.py` (smallest filtered index per voxel, ascending order, deterministic) but with peak memory independent of point count.
   - `ply.py` — exports `pointcloud.ply` for TouchDesigner (`-p/--points`).
   - `positions.py` — exports `dsm.tif` to a deterministic `positions.exr` (3-channel float32 OpenEXR).

5. **`reconcile/`** (interpolation bounded context, added after the fetch/prep epic closed): interpolates the AHN point cloud onto the orthophoto's pixel grid and emits a single coloured cloud. The ortho is EPSG:28992; the AHN DSM/LAZ is EPSG:7415 (EPSG:28992 horizontally + NAP height vertically) — the two grids coincide exactly in X/Y, so no reprojection is needed and only Z (NAP height) is semantically distinct.
   - `reconcile.py` — orchestrates block-streamed interpolation and writes output.
   - `clean.py` — class filter + XY de-duplication of the source cloud before interpolation.
   - `method.py` — `LinearInterp` / `IdwInterp` / `KrigingInterp` / `Variogram` value objects.
   - `neighbors.py` — deterministic kNN via `scipy.spatial.cKDTree` (an MLX/Metal GPU spike was built and benchmarked but removed in favor of this CPU reference — see `docs/specs/reconcile-design.md`).
   - `raster.py` — raster/point-cloud IO (rasterio + laspy).
   - `writers.py` — deterministic `laz`/`ply`/`pt`/`exr` output writers.

6. **`copc/`** (COPC export bounded context, added to resolve `docs/bugs/pdal-copc-xyz-bounds-flat-terrain.md`): turns a pipeline LAZ (`prep`'s or `reconcile`'s output) into a `.copc.laz` whose LAS-header bounds and COPC octree cube are consistent **by construction** — PDAL's `writers.copc` computes them through two float64 paths that disagree by an epsilon on flat, horizontally-huge Dutch terrain (every point pinned to the cube's Z-min face), failing `copc-validator`'s `xyz` check. Fully streaming (chunked reads → on-disk XY bucket spill → one bucket in memory at a time), so nationwide-scale inputs work. Design doc: `docs/specs/copc-design.md`.
   - `octree.py` — `CopcError`, `plan_build()` (whole-metre cube anchored ≥1 m outside the data, below-NAP Z included), copc.js-exact node bounds (`min + (max - min) / 2` midpoint halving, matching `Bounds.stepTo` bit for bit) and the `LodSampler` top-down grid-occupancy sampler that assigns each point to exactly one node via those same midpoints.
   - `dedup.py` — 0.5 m-voxel de-duplication preserving AHN's native coarseness: only voxels holding >1 point collapse, survivor picked by outlier reasoning (median/MAD on Z, nearest-to-median, index tie-break); points are never moved or synthesised.
   - `scatter.py` — pass-1 streaming scatter into per-column bucket record files; normalizes attributes (scan_angle_rank→scan_angle, return numbers lifted to 1..15, LAS bit-fields — synthetic/key_point/withheld/overlap, scanner_channel, scan_direction_flag, edge_of_flight_line — packed into the PDRF 6 flags byte and carried through).
   - `writer.py` — typed façade over `copclib` (vendored stub in `typings/copclib/`): nodes handed over as raw pre-packed int32 PDRF 6/7 bytes (no second quantization path), header min/max set from the written quantized extremes, per-node GPS sort, WKT1 SRS (the validator's proj4js can't parse WKT2), and a post-Close binary patch of the COPC info VLR's `gpstime_minimum/maximum` (the copclib binding never fills them).
   - `build.py` — `build_copc()` orchestrator: plan → scatter → per-bucket dedup/LOD-sample/write, ancestors above the bucket level held back and written last; RGB policy (no/black RGB → PDRF 6, 8-bit-looking RGB widened ×257, real 16-bit passthrough).
   - Verified: the real 46.3M-point Moerkapelle site passes `npx copc-validator -d` 24/24 green; `_typos.toml` carries the `lod`/`LinearNDInterpolator`/legacy-geojson spell-check exceptions this work surfaced.

7. **`tiles3d/`** (3D Tiles export bounded context): converts the orthophoto map plus `reconcile`'s EXR heights into an OGC 3D Tiles 1.1 tileset (OGC 22-025r4) — a quadtree of binary glTF terrain tiles draped with the ortho, all coordinates in ECEF with region bounding volumes in EPSG:4979 radians. The two inputs must match **perfectly**: equal dimensions, EXR X/Y planes bit-equal to the ortho's pixel centres, EXR colour planes bit-equal to this ortho's bands, every elevation finite — any mismatch or missing value is a hard `Tiles3dError`, and data is never infilled. Every vertex/texel at every LOD is a genuine source sample (stride subsampling, no averaging). The `--profile` flag selects the on-disk representation: `strict` (default) is the byte-frozen lossless float32-glTF + PNG profile, writing a loose `tileset.json` + `tiles/` directory and no other sidecar. `game`, `heightfield` and `splat` are all **packed** profiles — every tile's content blobs (quantized `KHR_mesh_quantization` + `EXT_meshopt_compression` glTF for `game`; the vendor `.hf` height chunk for `heightfield`; a zstd-wrapped binary 3D Gaussian Splatting `.ply` cloud for `splat`; `game`/`heightfield` draped with baseline JPEG, `splat` untextured — colour lives in the gaussians' SH coefficients) are bundled into one self-describing `tiles.hfp` **AHNP pack** (a binary scene index that is the runtime's only input besides its own blobs; normative spec `docs/specs/hfp-pack-format.md`, `content_kind` 0/1/2 for heightfield/game/splat), alongside a demoted `tileset.json` debug/interop sidecar, a deterministic `provenance.json` (pinned quantization/JPEG/encoder/zstd settings plus a `pack` block carrying the pack's `dataset_id`), and a `manifest.json` integrity sidecar hashing every loose file plus the pack. The `.hf` chunk itself is v3: a 120-byte CRC'd header (CRC span `[0,116)`, covering a `vertical_datum` EPSG field at offset 112) followed by one zstd level-3, checksummed frame of 12-bit-quantized (25 mm absolute-error-capped) `uint16` **NAP**-height levels; normative spec `docs/specs/heightfield-chunk-format.md`. The heightfield profile is **NAP-native** (a deliberate Netherlands-specific choice): its stored plane *and* its region heights (`.hf` header + `tileset.json` + pack index) are NAP (EPSG:5709), tagged in the header's `vertical_datum` and in `provenance.json` — so the region contains its own geometry, but heightfield tiles sit ~43 m off a WGS84 globe and do not co-register with the ellipsoidal `strict`/`game`/`splat` profiles (which store ECEF geometry and stay globe-correct). The NAP-vs-ellipsoidal region datum is the one place the region's height component is profile-dependent (via the encoder-seam `TileEncoder.region_of`). A companion Rust crate, `rust/ahn-heightfield` (MSRV 1.77), provides a two-layer decode API (chunk + archive) for both artifacts plus an optional `encode` feature for the chunk layer, coding against the two normative specs rather than the Python source (the archive layer also opens `content_kind = 2` splat packs — no texture, `decode_tile` typed-errors on them since they are not heightfields); its CI (`.github/workflows/rust.yml`, mirrored locally by `make rust-check`/`make rust-lint`/`make rust-test`) runs lint/clippy/cargo-deny/doc, a 3-OS × {stable, 1.77} test matrix, a non-blocking fuzz smoke pass, and a 3-OS Python↔Rust cross-language round-trip against committed fixtures.
   - `errors.py` — `Tiles3dError`, the context's single typed error.
   - `profile.py` / `encoders.py` / `payload.py` — the encoder seam: `Profile` (strict|game|heightfield|splat, `Profile.parse` at the CLI boundary, `Profile.encoder()`, `Profile.content_suffix()`/`Profile.texture_suffix()`) selects a `TileEncoder` (`StrictEncoder` / `GameEncoder` / `HeightfieldEncoder` / `SplatEncoder`) that turns a sampled `TilePayload` into an `EncodedTile`; emission and the swap machinery stay agnostic to the packing.
   - `quantize.py` / `jpeg.py` / `meshopt.py` / `gltf_quant.py` — the game profile's encoder layer (pure `KHR_mesh_quantization` quantizer, baseline-JPEG codec, `EXT_meshopt_compression` stream codec, quantized-glb writer); each owns and exports the constants/version helpers the provenance sidecar records.
   - `heightfield.py` — the heightfield profile's `.hf` v2 codec (120-byte little-endian CRC'd header + zstd level-3 checksummed frame of the tile's 12-bit-quantized `uint16` NAP-height plane, 25 mm absolute-error cap enforced; `quantize.py`'s `quantize_axis` on the height axis only) with a Python reference decoder; the normative byte layout lives in `docs/specs/heightfield-chunk-format.md`. `verify_heightfield.py` is its per-tile verifier (chunk decode + header/requantization/dequant-bound + sibling-JPEG checks, run before the byte-identity backstop).
   - `splat.py` — the splat profile's codec: one isotropic gaussian per tile vertex (position copied as-is from the RTC mesh, colour the sampled ortho pixel as an SH degree-0 coefficient with no sRGB decode, scale the tile's measured cell spacing log-stored, opacity a fixed logit constant, rotation the identity quaternion), serialised as a standard binary little-endian 3DGS `.ply` (14 float32 properties/gaussian, no normals) and zstd-wrapped whole (mirroring `heightfield.py`'s framing, but with no separate plaintext header — the zstd frame's own content checksum covers everything) — a Python reference decoder backs `verify_splat.py`, its per-tile verifier (decode + position/colour/opacity/scale/rotation recompute, run before the byte-identity backstop).
   - `pack.py` — the `AHNP` pack container (`tiles.hfp`): `write_pack` is a single-pass, bounded-memory streaming writer (one tile's blobs resident at a time) and `read_pack` is the reference validating reader (header/index CRCs, `dataset_id` recomputation, per-blob SHA-256), both mirroring `docs/specs/hfp-pack-format.md` exactly — the 96-byte index entries carry region + `geometric_error`, a level directory fronts them, a cold hash section anchors `dataset_id` (the pack's content version, a Merkle-style SHA-256 root over the hash section), and every key is the unified `(level, tx, ty, tz=0)` shape.
   - `manifest.py` — deterministic `manifest.json` codec: sorted-key SHA-256 + size over every loose sidecar plus `tiles.hfp`, tying the on-disk deliverable to the pack's `dataset_id`.
   - `provenance.py` — deterministic game/heightfield/splat-profile `provenance.json` codec (sorted-key JSON, no timestamps), sourcing every field from the encoder-layer modules plus the pack's `pack`/`producer` blocks; the verifier recomputes and byte-compares it.
   - `exr.py` — strict byte-level reader for reconcile's uncompressed EXR (exact attribute set, offset table, scanline framing; refuses truncation/trailing bytes).
   - `sources.py` — `load_terrain()` with the perfect-dimension-match gates and the `uniform_image`/`flat_surface` authenticity guards.
   - `geodesy.py` — pyproj EPSG:7415 → EPSG:4978 (ECEF) and → EPSG:4979 (radians); deterministic per machine (PROJ grid availability affects absolute heights, never self-consistency).
   - `quadtree.py` — tiling plan: shared-boundary pixel spans, per-level strides, `geometric_error` (leaves are 0).
   - `mesh.py` — RTC float32 vertex grids swizzled to glTF y-up (`(x, z, -y)`), texel-centre UVs, exact per-tile EPSG:4979 regions.
   - `png.py` / `gltf.py` / `tileset.py` — hand-packed deterministic writers (stdlib zlib PNG, glb container, sorted-key tileset.json, LF newlines on every platform).
   - `emit.py` — pure in-memory emission shared by build and verify (children-first, so parent regions contain all descendant content by construction); for the packed profiles this computes a `PackedBuild` (the tileset document plus one lazily-encoded `PackEntry` per tile) rather than holding every blob resident.
   - `build.py` — `build_tiles3d()` orchestrator; a failed or verification-rejected build removes everything it wrote, and a previous build in the same `--out` (the tool-owned artifact set — `tileset.json` + `tiles/` for `strict`, or `tiles.hfp` + `tileset.json` + `provenance.json` + `manifest.json` for the packed profiles) is held aside during a rebuild and restored on any failure — it is only dropped once the new build passes verification. The swap is two-phase with an accept-marker file as its commit point, so re-runs are safe and a good deliverable is never destroyed, even across hard kills (SIGKILL/power loss) at any moment.
   - `verify.py` / `verify_game.py` / `verify_heightfield.py` / `verify_splat.py` — the **strictest post-write verifier**, run unconditionally as the build's final step: re-reads every artifact from disk (materializing a packed build's blobs from `tiles.hfp` via `read_pack` first) and checks exact tileset key sets and 1.1 rules, region validity/containment, content-link integrity (no orphans/escapes/duplicates), glb container framing, accessor bounds with bit-exact POSITION extremes, index/UV validity, CRC-verified textures bit-equal to the sampled ortho, vertex containment in every enclosing region, full leaf coverage, the pack/tileset two-encodings witness (region and `geometricError` bit-agreement, one-to-one URI↔key mapping), and whole-file **byte identity** against an independent rebuild from the sources.

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
  - `fetch/ortho.py` — a pinned `basisdata.nl` HRL GeoJSON tile index URL (2025 vintage; `opendata.beeldmateriaal.nl`'s ATOM feed this module used before is retired).

### Coordinate Systems

The tool works primarily in the Dutch national grid, EPSG:28992 (RD New / Amersfoort), for orthophotos, bboxes, and tile identity. The AHN DSM/LAZ data is natively EPSG:7415 (EPSG:28992 horizontally, NAP height vertically) — `reconcile` relies on this to interpolate without reprojection, since only the Z axis differs semantically between the ortho grid and the AHN cloud. `fetch/source.py` provides the EPSG:28992 ↔ 4326 conversion helpers used when reading city/GeoJSON boundaries.

### Testing Structure

Tests are organized to mirror the source layout:
- `tests/domain/`, `tests/fetch/`, `tests/prep/`, `tests/reconcile/`, `tests/copc/`, `tests/tiles3d/`, `tests/provenance/`, `tests/cache/`, `tests/cli/` — one directory per bounded context / adapter, matching `ahn_cli/`.
- `tests/nightly/` — live checks against the real PDOK endpoints (`tests/nightly/test_portal_contract.py`; it does not exercise the Beeldmateriaal/basisdata.nl orthophoto feed); marked `nightly`, excluded from `make test` by default (`addopts = "-m 'not nightly'"`), run explicitly via `make test-nightly`.
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
- Key libraries: laspy (point clouds), copclib (COPC container writing), polars (out-of-core voxel-thinning group-by, pinned for determinism), geopandas/shapely (geometry), rasterio (rasters/COGs), scipy (kNN), pyproj (EPSG:7415 → ECEF/EPSG:4979 for 3D Tiles), pdal (optional), mlx (optional, Apple-silicon GPU decimation)

### CI

`.github/workflows/ci.yml` runs `make lint`, `make typos`, `make format-check`, `make typecheck`, `make test` on push/PR across Python 3.10/3.11/3.12, plus a separately scheduled nightly job running `make test-nightly`.
