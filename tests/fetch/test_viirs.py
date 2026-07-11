"""Tests for the VIIRS GeoTIFF importer.

These build tiny valid GeoTIFFs in-process with rasterio (no network), then
assert metadata extraction, checksum stability, byte-identical copy, the
provenance record, and the reject-invalid-file branch.
"""

from __future__ import annotations

import hashlib
import itertools
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pytest
import rasterio
from rasterio.transform import from_bounds

from ahn_cli.domain import Product
from ahn_cli.fetch.viirs import (
    ViirsImport,
    ViirsImportError,
    ViirsRaster,
    import_viirs,
    inspect_viirs,
)
from ahn_cli.provenance import read_provenance

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

_BOUNDS = (3.0, 50.0, 7.0, 53.0)
_WIDTH = 4
_HEIGHT = 3
_START = datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc)
_FINISH = datetime(2024, 6, 1, 10, 0, 5, tzinfo=timezone.utc)


def _write_geotiff(
    path: Path,
    *,
    crs: str = "EPSG:4326",
    bounds: tuple[float, float, float, float] = _BOUNDS,
    count: int = 1,
    dtype: str = "float32",
) -> None:
    """Write a tiny valid GeoTIFF at ``path`` for use as an import fixture."""
    transform = from_bounds(*bounds, _WIDTH, _HEIGHT)
    pixels: npt.NDArray[np.generic] = np.arange(
        count * _HEIGHT * _WIDTH, dtype=dtype
    ).reshape(count, _HEIGHT, _WIDTH)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=_HEIGHT,
        width=_WIDTH,
        count=count,
        dtype=dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(pixels)


def _fixed_clock(
    values: tuple[datetime, ...],
) -> Callable[[], datetime]:
    """Return a callable yielding ``values`` cyclically, one per call."""
    cursor = itertools.cycle(values)
    return lambda: next(cursor)


def test_viirs_import_error_is_a_value_error() -> None:
    """The typed error subclasses ValueError so callers can catch broadly."""
    assert issubclass(ViirsImportError, ValueError)


def test_inspect_reports_float_raster_metadata(tmp_path: Path) -> None:
    """A single-band float raster yields its CRS, extent, bands and checksum."""
    source = tmp_path / "lights.tif"
    _write_geotiff(source)

    raster = inspect_viirs(source)

    assert raster.crs == "EPSG:4326"
    assert raster.bounds == _BOUNDS
    assert raster.band_count == 1
    assert raster.dtypes == ("float32",)
    assert raster.checksum == hashlib.sha256(source.read_bytes()).hexdigest()


def test_inspect_reports_rgb_raster_metadata(tmp_path: Path) -> None:
    """A three-band uint8 (RGB) raster registers with all three bands."""
    source = tmp_path / "rgb.tif"
    _write_geotiff(source, count=3, dtype="uint8")

    raster = inspect_viirs(source)

    assert raster.band_count == 3
    assert raster.dtypes == ("uint8", "uint8", "uint8")


def test_inspect_checksum_is_deterministic(tmp_path: Path) -> None:
    """Inspecting the same file twice yields an identical checksum."""
    source = tmp_path / "lights.tif"
    _write_geotiff(source)

    first = inspect_viirs(source)
    second = inspect_viirs(source)

    assert first.checksum == second.checksum
    assert first == second


def test_inspect_rejects_a_uniform_placeholder_grid(tmp_path: Path) -> None:
    """A raster whose every pixel is one identical value is refused."""
    source = tmp_path / "flat.tif"
    transform = from_bounds(*_BOUNDS, _WIDTH, _HEIGHT)
    pixels: npt.NDArray[np.float32] = np.full(
        (1, _HEIGHT, _WIDTH), 3.5, dtype=np.float32
    )
    with rasterio.open(
        source,
        "w",
        driver="GTiff",
        height=_HEIGHT,
        width=_WIDTH,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(pixels)

    with pytest.raises(ViirsImportError, match="placeholder grid"):
        inspect_viirs(source)


def _write_pixel_geotiff(path: Path, pixels: npt.NDArray[np.float32]) -> None:
    """Write ``pixels`` (one float32 band) as a valid GeoTIFF at ``path``."""
    transform = from_bounds(*_BOUNDS, _WIDTH, _HEIGHT)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=_HEIGHT,
        width=_WIDTH,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(pixels)


def _write_all_zero_geotiff(path: Path) -> None:
    """Write a raster whose every pixel is exactly zero (unlit terrain)."""
    _write_pixel_geotiff(
        path, np.zeros((1, _HEIGHT, _WIDTH), dtype=np.float32)
    )


def test_inspect_accepts_a_uniform_all_zero_raster(tmp_path: Path) -> None:
    """All-zero radiance is genuine unlit terrain, not a placeholder grid."""
    source = tmp_path / "dark.tif"
    _write_all_zero_geotiff(source)

    raster = inspect_viirs(source)

    assert raster.band_count == 1
    assert raster.checksum == hashlib.sha256(source.read_bytes()).hexdigest()


def test_import_accepts_a_uniform_all_zero_raster(tmp_path: Path) -> None:
    """An all-dark VIIRS raster imports successfully and byte-exact."""
    source = tmp_path / "dark.tif"
    _write_all_zero_geotiff(source)
    site = tmp_path / "delft"

    result = import_viirs(source, site, clock=_fixed_clock((_START, _FINISH)))

    assert result.dest_path == site / "viirs" / "dark.tif"
    assert result.dest_path.read_bytes() == source.read_bytes()


def test_inspect_rejects_an_all_nan_raster(tmp_path: Path) -> None:
    """A raster with no finite pixel at all carries no radiance data."""
    source = tmp_path / "masked.tif"
    _write_pixel_geotiff(
        source, np.full((1, _HEIGHT, _WIDTH), np.nan, dtype=np.float32)
    )

    with pytest.raises(ViirsImportError, match="placeholder grid"):
        inspect_viirs(source)


def test_inspect_accepts_a_nan_border_around_zero_terrain(
    tmp_path: Path,
) -> None:
    """A NaN-masked border around all-zero radiance is genuine dark data."""
    source = tmp_path / "dark_masked.tif"
    pixels: npt.NDArray[np.float32] = np.zeros(
        (1, _HEIGHT, _WIDTH), dtype=np.float32
    )
    pixels[:, 0, :] = np.nan
    _write_pixel_geotiff(source, pixels)

    raster = inspect_viirs(source)

    assert raster.band_count == 1
    assert raster.checksum == hashlib.sha256(source.read_bytes()).hexdigest()


def test_inspect_rejects_a_non_raster_file(tmp_path: Path) -> None:
    """A file that is not a raster is rejected with ViirsImportError."""
    source = tmp_path / "broken.tif"
    source.write_bytes(b"this is not a GeoTIFF")

    with pytest.raises(ViirsImportError):
        inspect_viirs(source)


def test_import_copies_bytes_identically(tmp_path: Path) -> None:
    """The imported file is a byte-for-byte copy under ``<site>/viirs/``."""
    source = tmp_path / "lights.tif"
    _write_geotiff(source)
    site = tmp_path / "delft"

    result = import_viirs(source, site, clock=_fixed_clock((_START, _FINISH)))

    assert result.dest_path == site / "viirs" / "lights.tif"
    assert result.dest_path.read_bytes() == source.read_bytes()


def test_import_writes_a_complete_provenance_sidecar(tmp_path: Path) -> None:
    """The sidecar records product, checksums, extent and band/source keys."""
    source = tmp_path / "lights.tif"
    _write_geotiff(source)
    site = tmp_path / "delft"

    result = import_viirs(source, site, clock=_fixed_clock((_START, _FINISH)))

    assert (
        result.provenance_path
        == site / "viirs" / "lights.tif.provenance.json"
    )
    provenance = read_provenance(result.provenance_path)
    assert provenance == result.provenance
    assert provenance.product is Product.VIIRS
    assert provenance.bbox == _BOUNDS
    assert provenance.input_checksum == result.raster.checksum
    assert provenance.output_checksum == result.raster.checksum
    assert provenance.attribution.strip()
    keys = dict(provenance.request_keys)
    assert keys["source_path"] == str(source)
    assert keys["crs"] == "EPSG:4326"
    assert keys["band_count"] == "1"
    assert keys["band_dtypes"] == "float32"


def test_import_returns_a_frozen_hashable_result(tmp_path: Path) -> None:
    """The import result is a value object: hashable and equal by value."""
    source = tmp_path / "lights.tif"
    _write_geotiff(source)
    site = tmp_path / "delft"

    result = import_viirs(source, site, clock=_fixed_clock((_START, _FINISH)))

    assert isinstance(result, ViirsImport)
    assert isinstance(result.raster, ViirsRaster)
    assert len({result}) == 1


def test_import_uses_the_injected_clock(tmp_path: Path) -> None:
    """The download window is taken from the injected clock, in order."""
    source = tmp_path / "lights.tif"
    _write_geotiff(source)
    site = tmp_path / "delft"

    result = import_viirs(source, site, clock=_fixed_clock((_START, _FINISH)))

    assert result.provenance.download_started_at == _START
    assert result.provenance.download_finished_at == _FINISH


def test_import_default_clock_records_tz_aware_utc(tmp_path: Path) -> None:
    """Without an injected clock, real UTC timestamps bracket the import."""
    source = tmp_path / "lights.tif"
    _write_geotiff(source)
    site = tmp_path / "delft"

    result = import_viirs(source, site)

    started = result.provenance.download_started_at
    finished = result.provenance.download_finished_at
    assert started.tzinfo is not None
    assert finished.tzinfo is not None
    assert finished >= started


def test_import_is_byte_deterministic_for_one_source(tmp_path: Path) -> None:
    """Importing one source into two sites yields identical sidecar bytes."""
    source = tmp_path / "lights.tif"
    _write_geotiff(source)
    site_a = tmp_path / "a"
    site_b = tmp_path / "b"
    clock = _fixed_clock((_START, _FINISH))

    first = import_viirs(source, site_a, clock=clock)
    second = import_viirs(source, site_b, clock=clock)

    assert (
        first.provenance_path.read_bytes()
        == second.provenance_path.read_bytes()
    )
    assert first.dest_path.read_bytes() == second.dest_path.read_bytes()


def test_import_rejects_a_non_raster_file(tmp_path: Path) -> None:
    """Importing a non-raster file raises before anything is copied."""
    source = tmp_path / "broken.tif"
    source.write_bytes(b"this is not a GeoTIFF")
    site = tmp_path / "delft"

    with pytest.raises(ViirsImportError):
        import_viirs(source, site)


def test_import_into_an_existing_viirs_dir_succeeds(tmp_path: Path) -> None:
    """Re-importing into a site whose viirs dir already exists is fine."""
    source = tmp_path / "lights.tif"
    _write_geotiff(source)
    site = tmp_path / "delft"
    import_viirs(source, site, clock=_fixed_clock((_START, _FINISH)))

    result = import_viirs(source, site, clock=_fixed_clock((_START, _FINISH)))

    assert result.dest_path.read_bytes() == source.read_bytes()
