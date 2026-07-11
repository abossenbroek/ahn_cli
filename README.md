# AHN CLI

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Version: 0.3.5](https://img.shields.io/badge/Version-0.3.5-green.svg)](https://github.com/abossenbroek/ahn_cli/releases)
[![CI](https://github.com/abossenbroek/ahn_cli/actions/workflows/ci.yml/badge.svg)](https://github.com/abossenbroek/ahn_cli/actions/workflows/ci.yml)

## Description

AHN CLI acquires and prepares Dutch elevation data — AHN (Actueel Hoogtebestand Nederland) point clouds, plus matched DSM and orthophoto layers — for a site defined by a city name, a bounding box, or a GeoJSON polygon. It produces deterministic, ready-to-use deliverables: filtered/thinned point clouds, position maps, and — via `reconcile` — a single point cloud coloured from the orthophoto.

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

## Exporting `reconcile` output to COPC

`reconcile --format laz` writes an ordinary LAZ file, not a [COPC](https://copc.io/) (Cloud Optimized Point Cloud) — COPC adds a spatial octree index that lets viewers (Potree, CesiumJS, QGIS, etc.) stream and LOD large clouds instead of loading the whole file. Building a valid COPC from `reconcile`'s output takes an extra [PDAL](https://pdal.io/) pass, and a naive pipeline will silently produce a COPC file that *looks* fine but fails strict validation (checked here with [`copc-validator`](https://github.com/hobuinc/copc-validator), the reference validator). These are the gotchas we hit converting a real `reconcile` output, and the pipeline that verifiably passes all 24 validator checks with no warnings.

**Known gaps in `reconcile`'s own writers** (as of this writing) that you have to compensate for in the PDAL pipeline:
- **No CRS is embedded** in any `reconcile` output (`laz`/`ply`/`pt`/`exr`) — you must pass the CRS explicitly. `reconcile` operates in EPSG:28992 (see [Coordinate systems](#coordinate-systems) below), so use `a_srs: "EPSG:28992"`, not the default WGS84 you'd get from an unset/guessed SRS.
- **`ReturnNumber`/`NumberOfReturns` are unset** (all zero, in both `laz` and `ply`) — a strict COPC validator flags this as a `pointCountByReturn` mismatch. Since `reconcile` output is a single-return synthetic cloud (not real lidar), set both to `1` for every point.
- **`ply`'s RGB is 8-bit** (`uchar red/green/blue`, correct for plain PLY) but LAS/COPC's RGB fields are 16-bit — PDAL cannot widen a dimension's type once `readers.ply` has registered it from the file header, so 8-bit-range color survives straight through to the COPC and a strict validator warns about it. Fix this *before* PDAL sees the file (script below).

**PDAL also needs help of its own**, independent of `reconcile`:
- `writers.copc`'s coordinate scale defaults to 1 cm (`0.01`) regardless of the source file's precision. At 1 cm, points near the tightly auto-fit octree cube edge can round *outside* the cube and fail the validator's `xyz` (bounds) check. Set `scale_x`/`scale_y`/`scale_z` to `0.001` (1 mm) explicitly.

We verified the pipeline below against a real `reconcile` output (1,048,576 points) with `copc-validator` — 24/24 checks pass, no warnings, reproducibly across repeated runs. It starts from `reconciled.ply` because that route is the one we could get to pass cleanly and repeatably; building directly from `reconciled.laz` still tripped the `xyz` check in our testing even with the scale/CRS/return-number fixes applied, so we don't recommend it until that's root-caused.

First, widen `ply`'s 8-bit RGB to 16-bit (requires `pip install plyfile`, a one-off scripting dependency — not an `ahn_cli` requirement):

```python
import numpy as np
from plyfile import PlyData, PlyElement

ply = PlyData.read("reconciled.ply")
v = ply["vertex"].data
out = np.empty(len(v), dtype=[("x", "f8"), ("y", "f8"), ("z", "f8"),
                               ("red", "u2"), ("green", "u2"), ("blue", "u2")])
out["x"], out["y"], out["z"] = v["x"], v["y"], v["z"]
out["red"] = v["red"].astype(np.uint16) * 256
out["green"] = v["green"].astype(np.uint16) * 256
out["blue"] = v["blue"].astype(np.uint16) * 256
PlyData([PlyElement.describe(out, "vertex")], text=False, byte_order="<").write("reconciled_rgb16.ply")
```

Then run this PDAL pipeline (`reconciled_to_copc.json`):

```json
{
  "pipeline": [
    { "type": "readers.ply", "filename": "reconciled_rgb16.ply" },
    { "type": "filters.assign", "value": ["ReturnNumber = 1", "NumberOfReturns = 1"] },
    {
      "type": "writers.copc",
      "filename": "reconciled.copc.laz",
      "a_srs": "EPSG:28992",
      "scale_x": 0.001,
      "scale_y": 0.001,
      "scale_z": 0.001
    }
  ]
}
```

```bash
pdal pipeline reconciled_to_copc.json
```

Verify the result with `copc-validator` (no install needed, `npx` fetches it on demand):

```bash
npx copc-validator -d reconciled.copc.laz
```

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
