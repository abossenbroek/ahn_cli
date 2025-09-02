# GeoJSON Polygon Support with Verification - Implementation Plan

## Overview
Add support for arbitrary GeoJSON polygon input to the AHN CLI tool with comprehensive verification to ensure output LAZ file validity and polygon agreement.

## Implementation Commits

### Commit 1: Fix critical header merging bug
**Files**: `ahn_cli/process.py`
**Description**: Fix the header merging issue that causes corruption when tiles have different headers
- Add `_validate_headers()` function to check header compatibility
- Validate point formats and extra dimensions match across tiles
- Update main processing loop to use validated header
- Add proper error handling with clear messages
- Remove the unsafe `if i == 0: global_header = las.header` pattern

### Commit 2: Add GeoJSON CLI option and validation
**Files**: `ahn_cli/main.py`, `ahn_cli/validator.py`, `ahn_cli/kwargs.py`
**Description**: Add CLI support for GeoJSON input with proper validation
- Add `--geojson/-g` option to main.py
- Add `validate_geojson()` function to check file exists and has .geojson/.json extension
- Update `validate_exclusive_args()` to enforce mutual exclusivity between city, bbox, and geojson
- Update `validate_all()` to include geojson validation
- Add geojson parameter to CLIArgs type

### Commit 3: Implement GeoJSON tile selection
**Files**: `ahn_cli/fetcher/geotiles.py`, `tests/fetcher/test_geotiles.py`
**Description**: Add function to find AHN tiles that intersect with GeoJSON polygons
- Add `ahn_subunit_indices_of_geojson()` function
- Use GeoPandas for efficient spatial join
- Handle CRS transformations to EPSG:28992
- Support MultiPolygon geometries
- Add comprehensive unit tests

### Commit 4: Update fetcher to support GeoJSON
**Files**: `ahn_cli/fetcher/request.py`
**Description**: Modify fetcher to handle GeoJSON-based tile selection
- Add geojson_file parameter to Fetcher constructor
- Update `_construct_urls()` to use GeoJSON tile selection when provided
- Add tile count validation and warnings for large downloads

### Commit 5: Fix polygon clipping for GeoJSON
**Files**: `ahn_cli/manipulator/ptc_handler.py`
**Description**: Fix the _arbitrary_polygon method to handle multiple polygons correctly
- Use `unary_union` to combine multiple polygons
- Handle MultiPolygon geometries properly
- Improve CRS transformation logic
- Add better error handling

### Commit 6: Create verification module
**Files**: `ahn_cli/manipulator/verifier.py`, `tests/manipulator/test_verifier.py`
**Description**: Create comprehensive verification functions for output validation
- Add `verify_laz_integrity()` for basic LAZ file validity
- Add `verify_bounds()` to compare LAZ bounds with GeoJSON polygon
- Add `verify_with_pdal()` for optional PDAL validation
- Handle CRS transformations in bounds verification
- Add unit tests for all verification functions

### Commit 7: Add verification CLI options
**Files**: `ahn_cli/main.py`, `ahn_cli/validator.py`, `ahn_cli/kwargs.py`
**Description**: Add CLI flags for controlling verification behavior
- Add `--no-verify` flag to disable all verification
- Add `--verify-pdal` flag for advanced PDAL validation
- Add `--bbox-tolerance` option (default: 10.0 meters)
- Add `--strict-bbox-check` flag to fail on bbox mismatch
- Add validation for verification arguments

### Commit 8: Integrate verification into process pipeline
**Files**: `ahn_cli/process.py`
**Description**: Add verification step after LAZ file creation
- Import verification functions
- Add verification logic after file writing
- Handle verification failures based on CLI flags
- Add logging for verification results
- Pass geojson path to process function

### Commit 9: Add PDAL as optional dependency
**Files**: `pyproject.toml`
**Description**: Add PDAL support for advanced validation
- Add optional dependencies section
- Include pdal>=2.5.0 as optional dependency
- Update development dependencies if needed

### Commit 10: Add integration tests
**Files**: `tests/test_pipeline.py`, `tests/testdata/sample_polygon.geojson`
**Description**: Add end-to-end tests for GeoJSON functionality
- Create sample GeoJSON test file
- Test complete pipeline with GeoJSON input
- Test verification pass/fail scenarios
- Test incompatible header handling
- Test edge cases (empty intersection, complex polygons)

### Commit 11: Update documentation
**Files**: `README.md`, `CLAUDE.md`
**Description**: Document the new GeoJSON functionality
- Add GeoJSON usage examples
- Document verification options
- Update command-line help
- Add troubleshooting section for common issues
- Update CLAUDE.md with architectural changes

## Testing Strategy

### Unit Tests (per commit)
1. **Commit 3**: Test GeoJSON tile selection with various inputs
2. **Commit 6**: Test all verification functions
3. **Commit 7**: Test validation of new CLI arguments

### Integration Tests (Commit 10)
1. End-to-end test with sample GeoJSON
2. Test with incompatible tile headers (should fail gracefully)
3. Test verification failures and recovery
4. Performance test with complex polygons
5. Test CRS transformation edge cases

## Success Criteria
- LAZ files are never corrupted due to header mismatches
- GeoJSON polygons of any shape can be used as input
- Output bounds match input polygon within tolerance
- Clear error messages for all failure cases
- Optional PDAL validation for advanced users
- All tests pass with >90% coverage
- Each commit leaves the codebase in a working state

## Risk Mitigation
- **Header compatibility**: Fixed first to prevent any data corruption
- **Large downloads**: Add warnings when >50 tiles selected
- **CRS issues**: Always transform to EPSG:28992 for consistency
- **PDAL availability**: Gracefully handle when PDAL not installed
- **Performance**: Use spatial indexing for efficient tile selection