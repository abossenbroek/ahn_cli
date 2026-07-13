# AHN CLI

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Version: 0.3.5](https://img.shields.io/badge/Version-0.3.5-green.svg)](https://github.com/abossenbroek/ahn_cli/releases)
[![CI](https://github.com/abossenbroek/ahn_cli/actions/workflows/ci.yml/badge.svg)](https://github.com/abossenbroek/ahn_cli/actions/workflows/ci.yml)

## Description

AHN CLI acquires and prepares Dutch elevation data — AHN (Actueel Hoogtebestand Nederland) point clouds, plus matched DSM and orthophoto layers — for a site defined by a city name, a bounding box, or a GeoJSON polygon. It produces deterministic, ready-to-use deliverables: filtered/thinned point clouds, position maps, and — via `reconcile` — a single point cloud coloured from the orthophoto.

> **New to Dutch geodata?** [`docs/overview.md`](docs/overview.md) explains the domain concepts (AHN, orthophotos, NAP, RD New) and the output formats (including the `AHNP` terrain pack) and where the package fits — a good starting point for understanding the broader use.

The CLI is organized as a small pipeline of subcommands rather than one big command:

```
fetch ┬→ prep → reconcile
      └→ export-positions
```

`fetch` downloads raw tiles into a **site directory**; every later step reads from and writes back into that same directory, plus a `provenance.json` sidecar recording what was done.

## Features

- Acquire AHN point cloud tiles for a city, bounding box, or GeoJSON area, from PDOK (primary) or GeoTiles.nl (fallback), auto-selecting the newest AHN generation or pinning one explicitly
- Optionally fetch the matching DSM raster and Beeldmateriaal orthophoto for the same area in the same `fetch` call
- Import an externally-produced VIIRS GeoTIFF into a site
- Filter AHN tiles by classification class, with automatic tile de-duplication
- Graded point-cloud thinning — voxel-grid or Poisson-disk — with optional Apple Silicon (MLX) GPU acceleration
- Export `pointcloud.ply` for TouchDesigner and `positions.exr` (a DSM-derived position map) for the same
- Interpolate the AHN cloud onto the orthophoto's pixel grid (linear, IDW, or ordinary kriging) and write a coloured cloud as `laz`/`ply`/`pt`/`exr`
- A deterministic `provenance.json` sidecar recorded at every step

## Prerequisites

Python 3.10–3.12. The core dependencies (`rasterio`, `geopandas`, `laspy`, `shapely`, `scipy`) ship as prebuilt wheels on common platforms (Linux x86_64, macOS, Windows), so a plain `pip install` does not require GDAL/GEOS/PROJ to already be on your system. Two optional extras pull in more:

- `--extra pdal` enables PDAL-based LAZ verification and requires a working PDAL installation.
- `--extra mlx` enables GPU-accelerated point-cloud thinning on Apple Silicon (arm64 macOS only); `prep`'s thinning falls back to a numpy CPU backend without it.

## Installation

```bash
pip install ahn_cli
```

Or with uv:

```bash
uv tool install ahn_cli
```

## Usage

`ahn_cli` is a command **group** — every invocation names one of the subcommands below. Running `ahn_cli` with no subcommand prints usage and exits with status 2; use `ahn_cli --help` or `ahn_cli <command> --help` for the full option reference at any time.

### `fetch` — acquire raw source tiles for a site

```
Options:
  -o, --out DIRECTORY   Site directory to populate, e.g. data/delft. [required]
  -c, --city TEXT        Acquire the area of a named municipality.
  -b, --bbox TEXT        Acquire an EPSG:28992 bounding box 'minx,miny,maxx,maxy'.
  -g, --geojson TEXT      Acquire the area of the polygon(s) in a GeoJSON file.
  --ahn [auto|ahn5|ahn4]         AHN generation to fetch; 'auto' picks the newest available. [default: auto]
  --source [pdok|geotiles]      Distribution source; 'pdok' is primary, 'geotiles' the fallback. [default: pdok]
  --dsm                          Also fetch the DSM raster, windowed-clipped to <out>/dsm.tif.
  --ortho                        Also fetch the Beeldmateriaal orthophoto (CC-BY) for the AOI.
```

Exactly one of `-c/--city`, `-b/--bbox`, or `-g/--geojson` is required.

```bash
# Download AHN point cloud tiles for Delft (auto-selects the newest generation)
ahn_cli fetch --out data/delft -c delft

# Also fetch the DSM and orthophoto for the same area
ahn_cli fetch --out data/delft -c delft --dsm --ortho

# Pin AHN4 explicitly and use the GeoTiles.nl fallback source
ahn_cli fetch --out data/utrecht -b 194198.0,443461.0,194594.0,443694.0 --ahn ahn4 --source geotiles

# Acquire a custom area from a GeoJSON file
ahn_cli fetch --out data/area -g my_area.geojson
```

### `prep` — transform and export a fetched site

```
Options:
  -d, --data DIRECTORY    Site directory produced by a prior fetch. [required]
  -i, --include-class TEXT   Keep only these classes (comma-separated integers).
  -e, --exclude-class TEXT   Drop these classes (comma-separated integers).
  -p, --points                Export the point cloud (pointcloud.ply).
  --thin-method [voxel|poisson]  Graded thinning method.
  --thin-grade INTEGER          Voxel thinning grade 0-9 (0 keeps all; higher is coarser).
  --thin-radius FLOAT            Poisson-disk minimum spacing in metres.
  --thin-seed INTEGER             Poisson-disk RNG seed (deterministic sampling). [default: 0]
```

`-i/--include-class` and `-e/--exclude-class` are mutually exclusive per class code. `--thin-grade` only applies with `--thin-method voxel`; `--thin-radius`/`--thin-seed` only apply with `--thin-method poisson`. `prep` always deduplicates overlapping tiles before filtering and thinning, and writes `<data>/pointcloud.laz` plus an updated `provenance.json`.

```bash
# Keep only ground (2) and building (6) classes, and also export pointcloud.ply
ahn_cli prep --data data/delft -i 2,6 --points

# Graded voxel-grid thinning (0-9; higher is coarser)
ahn_cli prep --data data/delft --thin-method voxel --thin-grade 3

# Poisson-disk thinning with a 1.5 m minimum spacing (deterministic; seed defaults to 0)
ahn_cli prep --data data/delft --thin-method poisson --thin-radius 1.5 --thin-seed 0
```

### `export-positions` — export the DSM to a position map

```
Options:
  --data DIRECTORY   Site directory produced by a prior fetch (must contain dsm.tif). [required]
```

Reads `<data>/dsm.tif` (from `fetch --dsm`) and writes a byte-deterministic 3-channel float32 OpenEXR position map (R=easting, G=northing, B=elevation) to `<data>/positions.exr`, for use in TouchDesigner.

```bash
ahn_cli export-positions --data data/delft
```

### `import-viirs` — import an externally-produced VIIRS GeoTIFF

```
Options:
  --out DIRECTORY   Site directory to populate, e.g. data/delft. [required]

Arguments:
  GEOTIFF   Path to the VIIRS GeoTIFF to import. [required, must exist]
```

Copies the raster byte-for-byte into `<out>/viirs/` and records its CRS/extent/bands and a content checksum in a provenance sidecar. No reprojection or resampling is performed.

```bash
ahn_cli import-viirs --out data/delft path/to/viirs.tif
```

### `reconcile` — interpolate the AHN cloud onto the ortho grid

```
Options:
  --ortho FILE     Orthophoto GeoTIFF defining the target (e.g. 8 cm) grid. [required]
  --cloud FILE      AHN point-cloud LAZ whose elevation is interpolated onto the grid. [required]
  --out DIRECTORY    Directory to write reconciled.<ext> output(s) into. [required]
  --method [linear|idw|kriging]   Interpolation method for the elevation. [default: idw]
  --idw TEXT          IDW parameters as 'power,k' (used when --method idw). [default: 2.0,12]
  --kriging TEXT       Kriging parameters as 'model,nugget,sill,range,k' (used when --method kriging).
                       [default: spherical,0.0,1.0,50.0,16]
  --classes TEXT        Class filter 'keep:2,6' or 'drop:7,18' (LAS codes); default keeps all.
  --format [laz|ply|pt|exr]   Output format(s); repeatable. Default: all four.
```

Estimates an elevation at every ortho pixel centre from the AHN cloud, colours each pixel from the ortho, and writes a coloured cloud as `reconciled.<ext>` for each requested format. Coincident-XY returns are always de-duplicated (highest Z kept) before interpolation; output is byte-deterministic.

```bash
# IDW interpolation (the default), all four output formats
ahn_cli reconcile --ortho data/delft/ortho/ortho.tif --cloud data/delft/pointcloud.laz --out data/delft/reconciled

# Ordinary kriging, keep ground+building classes only, write LAZ only
ahn_cli reconcile --ortho data/delft/ortho/ortho.tif --cloud data/delft/pointcloud.laz --out data/delft/reconciled \
  --method kriging --kriging "spherical,0.0,1.0,50.0,16" --classes keep:2,6 --format laz
```

## Exporting to COPC

`ahn_cli copc` converts any pipeline LAZ (`prep`'s `pointcloud.laz` or `reconcile`'s `reconciled.laz`) into a [COPC](https://copc.io/) (Cloud Optimized Point Cloud) — the octree-indexed LAZ that viewers (Potree, CesiumJS, QGIS, etc.) can stream and LOD instead of loading whole:

```bash
ahn_cli copc --cloud data/delft/reconciled/reconciled.laz --out data/delft/reconciled/reconciled.copc.laz
```

The command exists because external COPC writers break on Dutch-shaped data (see `docs/bugs/2026-07-11-pdal-copc-xyz-bounds-flat-terrain.md`: PDAL's `writers.copc` declares cube and header bounds through two different float64 paths, and on flat, horizontally-huge terrain — where every point is pinned to the octree cube's Z-minimum face — the resulting sub-millimetre epsilon fails `copc-validator`'s `xyz` check on hundreds of nodes, including the root). `ahn_cli copc` instead:

- **streams in bounded memory** (chunked reads → on-disk XY buckets → one bucket at a time), so nationwide-scale inputs work;
- **preserves AHN's native 0.5 m coarseness**: it never thins below the source grid, and de-duplicates only when multiple points share one 0.5 m voxel — the survivor is chosen by outlier reasoning (median/MAD on Z, nearest-to-median wins), never synthesised;
- **builds the octree for Netherlands-shaped data by construction**: whole-metre cube anchor at least 1 m below/left of the data (below-NAP Z included), header bounds computed from the same quantized int32 → float64 path every reader decodes, and point→node assignment descending through the exact double-precision midpoints `copc.js` uses — so no boundary epsilon can exist;
- normalises the attribute zoo (EPSG:28992 WKT1 SRS, return numbers lifted to the LAS-valid range, 8-bit-looking RGB widened to 16-bit, GPS-sorted nodes, GPS range in the COPC info VLR).

Verify the result with `copc-validator` (no install needed, `npx` fetches it on demand) — a 46.3M-point real-world Zuidplaspolder site (Z from −8.57 m NAP) passes all 24 checks green:

```bash
npx copc-validator -d reconciled.copc.laz
```

## Exporting to 3D Tiles

`ahn_cli tiles3d` drapes the orthophoto over `reconcile`'s EXR heights and writes an [OGC 3D Tiles 1.1](https://www.ogc.org/standard/3dtiles/) tileset — a quadtree of binary glTF terrain tiles that Cesium, deck.gl and other viewers stream and LOD. The ortho and EXR must match perfectly (bit-exact pixel grid and colours; every height finite), and every written artifact is re-verified from disk against an independent rebuild before the tileset is accepted:

```bash
ahn_cli tiles3d --ortho data/delft/ortho/ortho.tif --heights data/delft/reconciled/reconciled.exr --out data/delft/tiles3d
```

`--profile` selects the on-disk representation:

- `strict` (default) — lossless float32 glTF with embedded PNG textures, written as a loose `tileset.json` + `tiles/` directory; writes no other sidecar.
- `game` — the compact runtime profile: quantized (`KHR_mesh_quantization`) geometry, `EXT_meshopt_compression` streams and baseline JPEG textures.
- `heightfield` — the vendor Approach-C profile: each tile is a self-describing `.hf` chunk (120-byte header + a zstd level-3, checksummed frame of 12-bit-quantized `uint16` NAP-height levels, 25mm absolute-error cap; format spec in `docs/specs/2026-07-12-heightfield-chunk-format.md`) with a sibling baseline JPEG.

Both `game` and `heightfield` bundle every tile's blobs into a single binary `tiles.hfp` **AHNP pack** — a self-describing scene index that is the runtime's only input besides its own blobs (format spec `docs/specs/2026-07-12-hfp-pack-format.md`) — plus a demoted `tileset.json` debug/interop sidecar, a deterministic `provenance.json` (pinned quantization/JPEG/encoder/zstd settings and the pack's content-version `dataset_id`), and a `manifest.json` integrity sidecar hashing every loose file plus the pack. `.hf`/`tiles.hfp` are vendor content types the generic OGC validator does not recognize, so a `heightfield` tileset is validated by `ahn_cli`'s own post-write verifier rather than the generic one.

```bash
ahn_cli tiles3d --ortho data/delft/ortho/ortho.tif --heights data/delft/reconciled/reconciled.exr --out data/delft/tiles3d --profile game
```

### Rust consumer crate

`rust/ahn-heightfield` is a `#![forbid(unsafe_code)]` Rust decoder for both the `.hf` chunk and the `AHNP` pack, coding against the two normative specs above rather than the Python source: a chunk layer (`Heightfield::decode`) and an archive layer (`Archive::open` over any positioned-read backing store) plus an optional, off-by-default `encode` feature for the chunk layer. MSRV is `1.77`, enforced in CI (`.github/workflows/rust.yml`) across Ubuntu/macOS/Windows at both `stable` and `1.77`, alongside a 3-OS Python↔Rust cross-language round-trip against committed fixtures; run the same gates locally with `make rust-check`. See `rust/ahn-heightfield/README.md` for the full API and its MSRV policy.

## AHN classification classes

Class codes used by `-i/--include-class`, `-e/--exclude-class`, and `--classes` are the standard AHN/LAS codes:

| Code | Meaning |
|---|---|
| 0 | Created, never classified |
| 1 | Unclassified |
| 2 | Ground |
| 6 | Building |
| 9 | Water |
| 14 | High tension |
| 26 | Civil structure |

## Coordinate systems

Orthophotos, bounding boxes, and tile identity use EPSG:28992 (Dutch RD New / Amersfoort). AHN point cloud/DSM data is natively EPSG:7415 (EPSG:28992 horizontally, NAP height vertically) — `reconcile` relies on this to interpolate without reprojection, since the ortho and AHN grids already coincide in X/Y.

## Reporting Issues

Encountering issues or bugs? We greatly appreciate your feedback. Please report any problems by [opening an issue](https://github.com/abossenbroek/ahn_cli/issues). Be as detailed as possible in your report, including steps to reproduce the issue, the expected outcome, and the actual result. This information will help us address and resolve the issue more efficiently.

## Contributing

Your contributions are welcome! If you're looking to contribute to the AHN CLI project, please first review our Contribution Guidelines. Whether it's fixing bugs, adding new features, or improving documentation, we value your help.

### Local development setup

```bash
# Install dependencies (creates the uv-managed virtualenv)
make install

# Install the pre-commit hooks (strict ruff lint + format, typos, pyright);
# they run automatically on every `git commit`.
uv run pre-commit install

# Run the full gate locally (lint, typos, pyright, tests + 100% coverage,
# format-check) — this is exactly what CI runs:
make check
```

To get started:

- Fork the repository on GitHub.
- Clone your forked repository to your local machine.
- Create a new branch for your contribution.
- Make your changes and commit them with clear, descriptive messages.
  Push your changes to your fork.
- Submit a pull request to our repository, providing details about your changes and the value they add to the project.
- We look forward to reviewing your contributions and potentially merging them into the project!
