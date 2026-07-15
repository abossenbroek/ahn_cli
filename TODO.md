# TODO — ahn_cli → 7rad §10 Data Acquisition

Target reader: senior engineer. No hand-holding. Terse by design, but every item
carries enough context to act on without re-reading the spec.

Source: `7rad_10_data_acquisition.md` §0–§6, reconciled against current repo via
gap analysis + stakeholder Q&A. Cross-references below use item titles, not numbers.

---

## Guardrails (apply to every item, no exceptions)

- **DDD.** Model the domain first: `Product` (AHN/Ortho/DSM/VIIRS), `Generation`,
  `Vintage`, `Tile`, `Provenance` as explicit types/value-objects. No stringly-typed
  product/generation switches. Bounded contexts: `fetch` (acquisition) vs `prep`
  (transform/export) — don't blur them.
- **TDD.** Red-green-refactor, test written before implementation, always. PR
  without a failing-test-first commit history gets bounced.
- **100% test coverage.** Line + branch. No `# pragma: no cover` without a
  documented reason in the PR description. Hard CI gate below 100%.
- **Documentation.** Every public function/class: docstring stating contract
  (inputs, outputs, invariants, failure modes) — not what the code obviously does.
  README/CHANGELOG updated in the same PR as the feature.
- **No speculative code.** Build exactly what's specified below. Flag anything
  you think is missing — don't silently add it.
- **Determinism.** Same input → same output, byte-identical, always. Load-bearing
  for "Provenance Sidecar", "Checksum / Content-Addressed Caching", and the
  "Full Integration Test Suite" — not aspirational.
- **Extensibility only where specified.** The generation registry in "AHN
  Generation Selection" must be open for extension. Nowhere else — don't add
  abstraction the spec doesn't call for.

---

## CLI Restructure: `fetch` / `prep` Verbs

Today's CLI is one monolithic `@click.command()` — fetch, clip, filter,
decimate, verify, preview all fused into one call. Spec wants two verbs so
acquisition and transform are separable stages.

**Requirement:** Click group, two subcommands: `7rad fetch --bbox <rd> --out
data/<site>/` (acquisition only — downloads/caches raw tiles) and `7rad prep
--data data/<site>/ [--points]` (everything downstream: clip, classification
filter, dedup, decimate, mosaic, export, provenance write).

**Also fix while touching this file:** `-e` flag collision — currently both
`--exclude-class` and `--epsg` bind to `-e` (`main.py:55` and `main.py:75`),
a pre-existing bug unrelated to the new verbs but in the same blast radius.

**Decide at implementation time (non-blocking):** drop the old single command,
or keep it as a deprecated alias.

**Definition of done:**
- Click group with `fetch`, `prep` subcommands
- `data/<site>/{ahn,ortho,viirs}/` directory layout enforced
- `-e` collision fixed, regression test added
- 100% branch coverage on arg parsing incl. all mutual-exclusivity paths

---

## AHN Generation Selection

AHN4 is hardcoded today (`config.py:7`). AHN5 is now live for the western
Netherlands and changes DSM computation (highest-point-per-cell vs IDW mean)
and classification (class 6 = roofs only) — better for this project's roof
extraction. User wants forward compatibility for AHN6, so this can't be a
two-way if/else.

**Requirement:** `--ahn ahn5|ahn4|auto` flag, default `auto`. `auto` probes
AHN5 coverage for the AOI, falls back to AHN4 if uncovered.

**Design constraint:** generation is a registry entry (base URL, coverage-probe
function, semantics note) — not a branch. Adding AHN6 later must be a pure
registry addition; write a test that proves this (add a fixture generation,
touch zero production call sites).

**Explicitly descoped:** generation-aware classification (the AHN4↔AHN5
semantic shift in class 6) — user doesn't need this handled. Keep the current
static class list. Only record which generation was used, in provenance.

**Definition of done:**
- `Generation` value object + registry
- `auto` coverage probe (AHN5 covered → AHN5, else AHN4)
- Extensibility test: new generation added via registry only, no other code touched
- Generation used is recorded in the provenance sidecar

---

## Distribution Path: PDOK ATOM Primary, GeoTiles.nl Fallback

Verified via web search: PDOK's ATOM download service is the canonical
distribution — native 5×6.25km sheets, no tile overlap, both LAZ point cloud
and DSM/DTM COGs available through it. GeoTiles.nl (TU Delft) re-tiles to
1×1.25km sub-tiles with a documented ~20–25m overlap between neighbours, for
parallel-processing convenience — that overlap is the direct cause of the
dedup problem (see "Tile Dedup: Crop-Before-Merge + Post-Merge Sweep").
Today's code uses GeoTiles.nl only.

**Requirement:** `--source pdok|geotiles` flag, default `pdok`. Keep the
existing GeoTiles.nl fetcher (`fetcher/request.py`) intact as the fallback —
don't delete it.

**Open question, resolve before implementation:** does PDAL support a
partial/windowed LAZ read for PDOK's larger native sheets, or is full-sheet
download acceptable? DSM COG windowed reads are confirmed feasible (HTTP
range requests) regardless.

**Definition of done:**
- PDOK ATOM feed client (parse feed, resolve tile URLs intersecting the AOI)
- `--source` flag wired, default pdok, geotiles fallback path unchanged
- Portal-contract test (nightly, live): ATOM feeds parse, expected
  products/vintages exist, tile-index checksums stable
- Mocked-ATOM unit tests for tile resolution logic

---

## Tile Dedup: Crop-Before-Merge + Post-Merge Sweep

The spec calls this "the real engineering of this stage." Currently absent —
`process.py` merges tiles by straight point-array append, so every GeoTiles.nl
25m overlap band is duplicated in the merged output today.

**Requirement, two-stage, both mandatory:**
1. `filters.crop` to each tile's nominal, non-overlapping boundary, before merge.
2. Post-merge exact-duplicate sweep: hash on XYZ + GPS-time.

Applies regardless of which distribution path is active — PDOK sheets don't
overlap by construction, but the post-merge sweep is cheap insurance and also
catches the same tile being ingested twice under two different names.

**Definition of done:**
- Crop-before-merge stage in the PDAL pipeline
- Post-merge dedup sweep (XYZ+time hash)
- Spec's own integration test, verbatim: synthetic overlapping tile pair with
  a known seam band → merged point count equals the analytic expectation,
  zero exact XYZ+time duplicates remain, interior points fully preserved (no
  over-crop)

---

## Decimation: Voxel-Grid + Poisson-Disk, Graded, GPU-Accelerated

Today's only decimation is uniform nth-point step selection (`ptc_handler.py`)
— not spatially uniform, no named presets. Spec wants named density tiers;
user wants a finer graded scale plus a second sampling method, both GPU-backed.

**Requirement:**
- Voxel-grid thinning, graded 0–9 (0 = full density, 9 = coarsest), grade maps
  to voxel size.
- Poisson-disk sampling as an alternate method: `--thin-method voxel|poisson`.
- **GPU acceleration is mandatory, not optional.** Target platform is Mac (per
  the spec's own stack table). No perf target is fixed yet — get one before
  committing to a specific backend (candidates to research: Metal Performance
  Shaders via a Python-accessible binding, `mlx`, PyTorch's MPS backend).
- Keep the existing `--decimate <step>` nth-point option for backward
  compatibility — additive only, don't touch its behavior.

**Definition of done:**
- Voxel-grid grade 0–9 implemented and tested (density strictly decreases with
  grade; spatial-uniformity check vs nth-point)
- Poisson-disk implemented and tested (minimum-distance property holds)
- GPU backend selected and wired; CPU-vs-GPU output-equivalence test to guard
  against the acceleration path silently changing results
- Perf benchmark test (informational, non-blocking on CI)
- Old `--decimate` behavior unchanged, regression-tested

---

## DSM Fetch + Clip

Entirely absent today — `rasterizer.py` only rasterizes clip polygons to
masks, it doesn't fetch or read any elevation raster. This is also a
prerequisite for the `positions.exr` output (see "TouchDesigner Outputs").

**Requirement:** new module. PDOK ATOM DSM COG, windowed HTTP-range read,
clipped to the AOI. Output `dsm.tif`.

**Rule carried over from the spec:** glass-roof lidar voids and spikes are
recorded, never repaired here — fill-vs-keep is a downstream "look" decision.
Don't build repair logic; flag it if later asked for.

**Definition of done:**
- DSM COG fetch + windowed clip implemented
- `dsm.tif` output produced
- Spec's own integration test, verbatim: window extent/transform match the
  bbox, nodata preserved (not filled)
- Voids/spikes recorded as an artefact note (provenance or a QA field) — not fixed

---

## Orthophoto Fetcher (Beeldmateriaal, D20 Vintage Logic)

**Update (2026-07):** implemented and since revised — see `ahn_cli/fetch/ortho.py`.
The Beeldmateriaal open-data ATOM feed this section originally specified
(`opendata.beeldmateriaal.nl`) was retired; tiles are now resolved from a
GeoJSON tile index published by `basisdata.nl`, pinned to the 2025 HRL
vintage, with each download verified against the index's SHA-256 digest.
Mosaicking uses `rasterio.merge` (never `gdalbuildvrt`, contrary to the
Requirement below). The rest of this section is kept for historical context.

Zero orthophoto code exists anywhere in the repo today. This is a full new
module, built into the same `fetch`/`prep` CLI as AHN — not a separate repo.

**Requirement:**
- Beeldmateriaal open-data tile enumeration and download (source GeoTIFF
  orthomosaic tiles, never WMS `GetMap` crops).
- D20 vintage/zone selection: prefer the 5cm-resolution zone for any
  acceptable year, otherwise take 8cm (7.5cm for pre-2025 vintages). Pin the
  chosen vintage in config — never float to "Actueel" (the newest-year layer).
- CC-BY 4.0 attribution string recorded, feeding into "Provenance Sidecar".
- Content-addressed fetch, keyed by `(product, vintage, tile-id)` — shared
  design with "Checksum / Content-Addressed Caching".
- Mosaic via `gdalbuildvrt`, then clip to the AOI. Tiles are assumed
  edge-aligned by construction — verify this with a test, don't assume it.
- Record mosaic seamlines and building lean ("omvalling") as a provenance
  note — survey facts, not defects to hide, per the spec.

**Definition of done:**
- Beeldmateriaal fetcher implemented
- Vintage/zone selection logic implemented, chosen vintage pinned in config
- CC-BY attribution string recorded into provenance
- `gdalbuildvrt` mosaic + AOI clip implemented
- Spec's own integration test, verbatim: mocked tile server → mosaic pixel
  count equals bbox area ÷ px², no source row/column contributes twice, seam
  pixels bit-identical to a single-image reference
- Vintage/zone selection test covering both the 5cm-preferred and
  8cm-fallback cases

---

## VIIRS Registration — Integration Only, Not a Rebuild

User has **already implemented** VIIRS-to-GeoTIFF via Google Earth Engine,
outside this repo. This item is integration, not a build-from-scratch.

**Requirement:** wire the existing GEE output into the `data/<site>/viirs/`
convention and the shared "Provenance Sidecar". Verify the GeoTIFF opens,
record CRS/extent/band structure, compute a content checksum, copy the file
untouched — no resampling, no re-colormapping, no normalisation, ever.

**Open question, blocking — resolve with the user before writing any code:**
exact integration interface. Does `ahn_cli` shell out to the existing GEE
script, or does it just consume an already-produced file path handed to it?

**Definition of done:**
- Integration point identified and confirmed with the user
- Thin wrapper: verify-open, record metadata, checksum, copy-untouched into
  the directory convention
- Spec's own integration test, verbatim: float and RGB fixtures register,
  output bytes equal input bytes, checksum recorded

---

## TouchDesigner Outputs: `positions.exr` + `pointcloud.ply`

Currently only a disconnected, untracked script exists
(`assets/laz_to_ply_massive.py`) — memory-efficient LAZ→PLY export, but
standalone, hardcoded filenames, not wired into the CLI, oriented at
Blender/UE5 rather than TouchDesigner's actual contract.

**Requirement:**
- `positions.exr`: float32 XYZ map of the DSM grid — depends on "DSM Fetch +
  Clip" existing first.
- `pointcloud.ply`: LAZ → PLY export, reusing the memory-efficient approach
  already prototyped in `laz_to_ply_massive.py` (laspy + plyfile direct,
  bypassing PDAL, needed for very large point counts), but moved into the
  `ahn_cli` package/CLI/test suite — no hardcoded filenames.

**Out of scope:** the Blender/UE5-oriented `assets/*.json` PDAL pipelines and
`ue5_pipeline*.py` scripts. Different downstream tool, different offset/scale
conventions. Don't fold these into the TouchDesigner export path.

**Definition of done:**
- `positions.exr` export implemented, matches DSM grid dimensions/values
- `pointcloud.ply` export implemented, round-trip tested (point count and
  coordinates preserved from source LAZ)
- Large-file memory-efficiency regression test (carries forward the design
  intent of `laz_to_ply_massive.py`)
- `assets/laz_to_ply_massive.py` retired once its logic is absorbed into the package

---

## Provenance Sidecar

Zero provenance code exists today — no sidecar file is written anywhere, and
`config.py`/`validator.py` carry no licence/vintage/zone fields. Build the
schema and writer first — every other fetcher ("AHN Generation Selection",
"Distribution Path", "DSM Fetch + Clip", "Orthophoto Fetcher", "VIIRS
Registration") populates it, so it can't be built last.

**Requirement:** machine-written `provenance.json` per dataset, fields per
spec (exhaustive): source portal, product ID, vintage, acquisition zone,
resolution tier obtained, AHN generation used, licence + CC-BY attribution
string, bbox, request keys, download timestamps, input and output checksums,
tool version.

**Definition of done:**
- Schema defined (JSON Schema or dataclass + serializer)
- Writer implemented, called by every fetcher listed above
- Spec's own integration test, verbatim: schema-validated, attribution string
  present, checksums verify

---

## Checksum / Content-Addressed Caching

Every `fetch()` call today downloads to a fresh temp file with no checksum and
no cache lookup (`fetcher/request.py`); temp files are deleted after
processing, so nothing is idempotent across runs. This is cross-cutting —
shared by "AHN Generation Selection"/"Distribution Path", "DSM Fetch + Clip",
and "Orthophoto Fetcher".

**Requirement:** every fetch keyed by `(product, generation/vintage,
tile-id)`, content-addressed, checksummed. Re-running `fetch` on the same
input must perform zero network calls and change zero bytes on disk.

**Definition of done:**
- Cache keyed by `(product, vintage/generation, tile-id)`
- Checksum verification on read
- Spec's own integration test, verbatim: second fetch performs zero network
  calls (cassette assertion), changes zero bytes on disk

---

## Full Integration Test Suite — 100%, No Sampling

Per explicit user instruction: every category the spec lists gets built, none
skipped, except the one explicit descope noted below.

- Portal contract (nightly, live) — see "Distribution Path"
- Mosaic overlap-free — see "Orthophoto Fetcher"
- Cache idempotence — see "Checksum / Content-Addressed Caching"
- LAZ dedup — see "Tile Dedup: Crop-Before-Merge + Post-Merge Sweep"
- ~~Generation-aware classification~~ — **descoped**, user decision, see
  "AHN Generation Selection"
- DSM clip — see "DSM Fetch + Clip"
- VIIRS registration — see "VIIRS Registration — Integration Only, Not a Rebuild"
- Provenance completeness — see "Provenance Sidecar"
- Property-based (`hypothesis`): random bboxes → tile enumeration is always
  exact-cover, never overlapping, never missing

**Automation:** all of the above except the portal-contract test run in the
fast CI tier, against fixtures and `vcrpy` cassettes — no network, no GPU.
Fixtures and cassettes live in git-LFS.

---

## Blocking Questions — Resolve Before Writing Code

1. VIIRS/GEE integration interface — see "VIIRS Registration — Integration
   Only, Not a Rebuild".
2. GPU backend for decimation — need a perf target first, see "Decimation:
   Voxel-Grid + Poisson-Disk, Graded, GPU-Accelerated".
3. PDAL partial/windowed LAZ read feasibility for PDOK sheets — see
   "Distribution Path: PDOK ATOM Primary, GeoTiles.nl Fallback".
4. Old single-command CLI: drop entirely, or keep as a deprecated alias —
   see "CLI Restructure: `fetch` / `prep` Verbs".

---

## Explicitly Out of Scope — Do Not Touch

- Generation-aware classification logic (the class-6 AHN4↔AHN5 semantic shift).
- Blender/UE5 `assets/` pipelines — separate downstream concern, leave as-is.
- WMS/WMTS fetch paths — quick-look verification only per the spec, never
  pipeline input.
