# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Development Setup
```bash
# Install dependencies with uv (preferred)
make install

# Update dependencies
make update
```

### Testing
```bash
# Run all tests
make test

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

# Format code (ruff format)
make format

# Check formatting without changing files
make format-check

# Fix linting issues automatically
make fix

# Run all checks (lint, typos, test, format-check)
make check
```

### Running the CLI
```bash
# Run the CLI with arguments
make run ARGS="-c delft -o ./delft.laz"

# Or directly with uv
uv run ahn_cli -c delft -o ./delft.laz

# Example commands:
# Download all classes for a city
uv run ahn_cli -c amsterdam -o amsterdam.laz

# Download only ground and building classes
uv run ahn_cli -c utrecht -o utrecht.laz -i 2,6

# Download with bounding box instead of city
uv run ahn_cli -o output.laz -b 194198.0,443461.0,194594.0,443694.0

# Download with GeoJSON polygon(s)
uv run ahn_cli -g my_area.geojson -o output.laz

# Download with GeoJSON and filter classes
uv run ahn_cli -g my_area.geojson -o output.laz -i 2,6

# Download with GeoJSON and enable bbox verification
uv run ahn_cli -g my_area.geojson -o output.laz --bbox-tolerance 15.0

# Download with strict bbox verification (fails if bbox mismatch)
uv run ahn_cli -g my_area.geojson -o output.laz --strict-bbox-check

# Skip verification entirely
uv run ahn_cli -g my_area.geojson -o output.laz --no-verify

# Enable PDAL verification (requires PDAL installed)
uv run ahn_cli -g my_area.geojson -o output.laz --verify-pdal

# Preview point cloud in 3D viewer
uv run ahn_cli -c delft -o delft.laz -p
```

## Architecture

AHN CLI is a command-line tool for downloading Dutch elevation point cloud data (AHN - Actueel Hoogtebestand Nederland). The codebase follows a modular architecture:

### Core Components

1. **Entry Point** (`main.py`): Click-based CLI interface that validates arguments and invokes the processing pipeline.

2. **Fetcher Module** (`fetcher/`):
   - `geotiles.py`: Determines which AHN tiles cover the requested area
   - `municipality.py`: Handles city polygon data and municipality boundaries
   - `request.py`: Downloads LAZ files from the AHN service

3. **Manipulator Module** (`manipulator/`):
   - `ptc_handler.py`: Core point cloud processing - filtering by classification, clipping to boundaries, decimation
   - `transformer.py`: Coordinate transformations between different EPSG systems
   - `preview.py`: 3D visualization using polyscope
   - `rasterizer.py`: Converting point cloud data to raster format

4. **Processing Pipeline** (`process.py`): Orchestrates the entire workflow:
   - Fetches required tiles based on city/bbox
   - Processes each tile (filter classes, clip, decimate)
   - Merges results into single output file
   - Handles coordinate offsets for large areas

### Key Design Patterns

- Uses laspy for LAZ file handling with streaming to handle large files
- Processes tiles incrementally to manage memory usage
- Applies transformations in a specific order: filter classes → clip → decimate
- Maintains global header information when merging multiple tiles

### Important Constants

- **AHN Classes** (defined in `validator.py`):
  - 0: Created, never classified
  - 1: Unclassified
  - 2: Ground
  - 6: Building
  - 9: Water
  - 14: High tension
  - 26: Civil structure

- **Data Sources** (defined in `config.py`):
  - Base URL: `https://geotiles.citg.tudelft.nl/AHN4_T/`
  - Municipality data: `ahn_cli/fetcher/data/municipality_simple.geojson`

### Coordinate Systems

The tool works primarily with the Dutch national grid (EPSG:28992) but supports coordinate transformations through the `-e/--epsg` option when using custom clip files.

### Data Download Flow

The tool downloads pre-tiled AHN LAZ files based on your area of interest:

1. **Input Methods** (choose one):
   - **City name**: `-c amsterdam` (uses pre-stored municipality boundaries)
   - **Bounding box**: `-b minx,miny,maxx,maxy` (coordinates in EPSG:28992)
   - **GeoJSON file**: `-g my_area.geojson` (arbitrary polygon(s) for custom areas)

2. **Tile Discovery**:
   - The Netherlands is divided into a grid of AHN tiles (stored in `ahn_cli/fetcher/data/ahn_subunit.geojson`)
   - Each tile has an identifier (e.g., "37FN2", "37FZ1")
   - System finds which tiles intersect with your area

3. **Download Process**:
   - Tile IDs are converted to URLs: `https://geotiles.citg.tudelft.nl/AHN4_T/{tile_id}.LAZ`
   - Multiple tiles downloaded concurrently (up to 8 threads)
   - Files temporarily stored during processing

4. **Processing Pipeline** (per tile):
   - Filter by classification classes (if `-i` or `-e` specified)
   - Clip to exact boundary (city polygon or bbox)
   - Decimate points (if `-d` specified)
   - Merge all tiles into single output file

**Important**: AHN data is pre-tiled - the tool downloads complete tiles that overlap your area, then clips to your exact boundary. This means it may download more data than ultimately needed.

### GeoJSON Polygon Flow

When using GeoJSON files for custom area selection:

1. **CLI Entry** (`main.py`):
   - Accepts GeoJSON file path with `-g/--geojson` option
   - File must have `.geojson` or `.json` extension

2. **Validation** (`validator.py`):
   - Checks file exists and has correct extension
   - Enforces mutual exclusivity with city and bbox options
   - Exactly one of `--city`, `--bbox`, or `--geojson` must be specified

3. **Tile Selection** (`fetcher/geotiles.py:ahn_subunit_indices_of_geojson`):
   - Reads GeoJSON file using GeoPandas
   - Automatically transforms to EPSG:28992 if in different CRS
   - Uses efficient spatial join (`gpd.sjoin`) to find intersecting tiles
   - Performance warning if >50 tiles selected

4. **Point Cloud Clipping** (`manipulator/ptc_handler.py:clip_by_arbitrary_polygon`):
   - Handles multiple polygons by creating a union (`unary_union`)
   - Supports MultiPolygon geometries
   - Respects CRS information in GeoJSON
   - Applies same rasterization-based clipping as other methods

**Features**:
- Supports any valid GeoJSON with Polygon or MultiPolygon geometries
- Multiple features are automatically combined into a single area
- CRS transformations handled automatically
- Compatible with all other CLI options (class filtering, decimation, etc.)

### Bounding Box (bbox) Flow

When using bbox instead of city name:

1. **CLI Entry** (`main.py`): 
   - Accepts bbox as comma-separated string: "minx,miny,maxx,maxy"
   - Parsed into list of floats: `[minx, miny, maxx, maxy]`

2. **Validation** (`validator.py`):
   - Must have exactly 4 coordinates
   - minx < maxx and miny < maxy
   - Cannot be used together with city parameter (mutually exclusive)

3. **Tile Selection** (`fetcher/geotiles.py:ahn_subunit_indices_of_bbox`):
   - Bbox coordinates are in EPSG:28992 (Dutch national grid)
   - Transformed to EPSG:4326 for spatial indexing
   - Uses GeoPandas spatial index (`cx[minx:maxx, miny:maxy]`) to find intersecting tiles

4. **Point Cloud Clipping** (`manipulator/ptc_handler.py:clip_by_bbox`):
   - After downloading tiles, points are filtered to bbox extent
   - Simple coordinate comparison: keeps points where `minx <= x <= maxx` and `miny <= y <= maxy`
   - Applied before other processing steps

### Testing Structure

Tests are organized by module:
- `tests/fetcher/` - Tests for data fetching components
- `tests/manipulator/` - Tests for point cloud processing
- `tests/test_pipeline.py` - Integration tests for the full pipeline
- `tests/testdata/` - Test fixtures and sample data

### Dependencies

The project uses modern Python tooling:
- `uv` for dependency management (pyproject.toml)
- `ruff` for linting and formatting (configured in pyproject.toml)
- `pytest` for testing
- Python 3.10-3.12 supported
- Key libraries: laspy (point clouds), geopandas/shapely (geometry), rasterio (rasters), polyscope (3D visualization)

### Bounding Box Verification (GeoJSON)

When using GeoJSON input, the tool can verify that the output LAZ file's bounding box matches the input polygon:

1. **Verification Process** (`manipulator/verifier.py`):
   - Reads GeoJSON and extracts unified polygon geometry
   - Transforms polygon to EPSG:28992 if needed (handles any CRS)
   - Calculates expected bounding box from transformed polygon
   - Compares with actual LAZ file bounding box
   - Reports coverage percentage and coordinate differences

2. **CLI Options**:
   - `--bbox-tolerance`: Maximum allowed difference in meters (default: 10.0)
   - `--strict-bbox-check`: Fail if bbox mismatch exceeds tolerance
   - `--no-verify`: Disable verification entirely
   - `--verify-pdal`: Enable PDAL validation (requires PDAL installed)

3. **CRS Handling**:
   - Automatically detects CRS from GeoJSON file
   - Falls back to `--epsg` parameter if no CRS in file (default: 4326)
   - All comparisons done in EPSG:28992 (Dutch national grid)

4. **Verification Output**:
   ```
   Verifying output bounding box against GeoJSON input...
   GeoJSON bbox: [194198.30, 443461.34, 194594.11, 443694.84]
   LAZ bbox:     [194201.45, 443463.12, 194591.23, 443692.56]
   Bbox difference: max 3.15m (within 10.0m tolerance) ✓
   Coverage: 98.7% of input area covered ✓
   ```

5. **Use Cases**:
   - Quality assurance for data processing
   - Detecting CRS transformation issues
   - Ensuring complete area coverage
   - Validating clipping accuracy