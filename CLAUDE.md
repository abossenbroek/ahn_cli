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
`ahn_cli` is a Click **group**; every invocation names a subcommand (`fetch`, `prep`, `reconcile`, `import-viirs`, `export-positions`). Running it with no subcommand prints usage to stderr and exits with code 2 — that is expected `click.Group` behavior, not a bug.

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
```

A typical end-to-end run is `fetch` → `prep` → (`export-positions` and/or `reconcile`): each step reads the previous step's output from the site directory on disk and writes its own outputs plus an updated `provenance.json`; there is no in-memory handoff between subcommands.

## Architecture

AHN CLI acquires and transforms Dutch elevation data (AHN — Actueel Hoogtebestand Nederland), plus matched DSM/orthophoto/VIIRS layers, for a given site (city, bbox, or GeoJSON area of interest). The codebase is organized as a set of **bounded contexts** behind a thin CLI adapter, each owning one stage of the pipeline; the `ahn_cli/cli/__init__.py` docstring states this explicitly, and `tests/test_bounded_contexts.py` enforces the boundary as a contract test.

### Core Components

1. **CLI adapter** (`ahn_cli/cli/app.py`): the `ahn_cli` Click group and its five subcommands (`fetch`, `prep`, `import-viirs`, `export-positions`, `reconcile`). Registered via `pyproject.toml`'s `[project.scripts] ahn_cli = "ahn_cli.cli:cli"`. This layer owns argument parsing/validation and translates each context's typed errors (`AcquisitionError`, `PrepError`, `ViirsImportError`, `PositionsExportError`, `ReconcileError`) into `click.ClickException`; it holds no acquisition or transform logic of its own. `fetch`, `prep`, and `reconcile` map one-to-one onto their own context directories below; `import-viirs` and `export-positions` don't have their own directories — they're implemented inside `fetch/viirs.py` and `prep/positions.py` respectively, since they belong to the acquisition and transform contexts respectively.

2. **`domain/`**: pure value objects shared by every context, with no I/O — `Tile`/`BBox` (identity, EPSG:28992), `PixelGrid`/`GeoTransform` (pixel ↔ world coords), `Generation` (AHN3/4/5…), `Product` (ahn/dsm/ortho/viirs), `Vintage` (acquisition year), `Provenance` (in-memory acquisition record).

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

6. **`provenance/`**: `sidecar.py` — deterministic `provenance.json` codec shared by every fetcher/transform step.

7. **`cache/`**: `store.py`'s `ContentAddressedCache` + `key.py`'s `CacheKey` — a checksum-verified, idempotent cache keyed by (product, generation/vintage, tile id), making `fetch` safe to re-run.

### Legacy / deprecated modules

`ahn_cli/main.py`, `process.py`, `config.py`, `kwargs.py`, `validator.py`, `fetcher/`, and `manipulator/` predate the bounded-context refactor and are **not** part of the live CLI surface — `main.py`'s single-command Click interface (the old `-c/-o/-i/-e/-d/-b/-g/-p` flags) is dead code, unreferenced by `pyproject.toml`'s entry point or any other module. Each carries a `DEPRECATED` banner and a module-level `DeprecationWarning`, and each is explicitly grandfathered out of `make lint`/`make typecheck`/coverage (kept in sync across `[tool.ruff.lint.per-file-ignores]`, `[tool.coverage.run] omit`, and `[tool.pyright] exclude` in `pyproject.toml`) — a module may only be de-grandfathered by removing it from all three lists and bringing it to 100% coverage and strict typecheck.

**Exception**: `process.py` is not fully dead — `prep/dedup.py` imports `harmonize_headers` from it (with the deprecation warning explicitly suppressed) and reuses it inside the new `prep` context. Don't delete `process.py` when cleaning up the rest of the legacy modules.

### Key Design Patterns

- Each bounded context (`fetch`, `prep`, `reconcile`) owns one pipeline stage and communicates through `domain/` value objects and the `provenance/` sidecar — no context reaches into another's internals.
- `fetch` is idempotent via the content-addressed `cache/`; `prep` and `reconcile` are pure transforms over already-cached/fetched inputs with no network access.
- Point cloud processing streams/block-processes rather than loading whole tiles where practical (DSM windowed COG reads, `reconcile`'s block-streamed interpolation) to manage memory on large areas.
- Deterministic outputs are a first-class requirement: `provenance.json`, the `reconcile` writers, and Poisson-disk thinning (via `--thin-seed`) are all designed to be reproducible given the same inputs.
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
- `tests/domain/`, `tests/fetch/`, `tests/prep/`, `tests/reconcile/`, `tests/provenance/`, `tests/cache/`, `tests/cli/` — one directory per bounded context / adapter, matching `ahn_cli/`.
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
- Key libraries: laspy (point clouds), geopandas/shapely (geometry), rasterio (rasters/COGs), scipy (kNN), pdal (optional), mlx (optional, Apple-silicon GPU decimation)

### CI

`.github/workflows/ci.yml` runs `make lint`, `make typos`, `make format-check`, `make typecheck`, `make test` on push/PR across Python 3.10/3.11/3.12, plus a separately scheduled nightly job running `make test-nightly`.
