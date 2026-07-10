"""Tests for the GeoTiles.nl fallback distribution source."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from pyproj import Transformer

from ahn_cli.fetch.geotiles_source import (
    GeotilesCatalogError,
    GeotilesSource,
    load_catalog,
)

if TYPE_CHECKING:
    from ahn_cli.domain import BBox
    from ahn_cli.fetch.generation import GenerationSource

_FIXTURES = Path(__file__).parent / "fixtures"
_SAMPLE_CATALOG = _FIXTURES / "ahn_subunit_sample.geojson"
_TO_RD = Transformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=True)


def _rd_bbox(
    minlon: float, minlat: float, maxlon: float, maxlat: float
) -> BBox:
    """Project a WGS84 lon/lat box to an EPSG:28992 AOI for the tests."""
    minx, miny = _TO_RD.transform(minlon, minlat)
    maxx, maxy = _TO_RD.transform(maxlon, maxlat)
    return (minx, miny, maxx, maxy)


_SHARED_EDGE_AOI = _rd_bbox(4.40, 51.99, 4.44, 52.02)
_UNCOVERED_AOI = _rd_bbox(6.40, 52.40, 6.50, 52.50)


def _close(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    """Report whether two coordinate tuples agree within a tight tolerance."""
    return all(abs(x - y) <= 1e-6 for x, y in zip(a, b, strict=True))


def _source() -> GeotilesSource:
    """Return a GeoTiles source reading the small sample catalogue."""
    return GeotilesSource(catalog_path=_SAMPLE_CATALOG)


def _generation_source() -> GenerationSource:
    """Return the newest GeoTiles generation source for resolve() tests."""
    return _source().generation_registry(lambda _url: b"").sources()[0]


def test_load_catalog_reads_polygon_and_multipolygon() -> None:
    """The catalogue reader handles both Polygon and MultiPolygon geometries."""
    catalog = load_catalog(_SAMPLE_CATALOG)

    assert tuple(tile.tile_id for tile in catalog) == (
        "37EN1_01",
        "37EN2_01",
        "02DN1_01",
    )
    assert _close(catalog[0].bbox_wgs84, (4.30, 51.97, 4.42, 52.05))
    assert _close(catalog[2].bbox_wgs84, (5.9139, 53.441, 5.9899, 53.4975))


def _write(tmp_path: Path, body: str) -> Path:
    """Write ``body`` to a catalogue file and return its path."""
    path = tmp_path / "catalog.geojson"
    path.write_text(body)
    return path


def test_load_catalog_rejects_non_object(tmp_path: Path) -> None:
    """A top-level JSON array is not a GeoJSON object."""
    with pytest.raises(GeotilesCatalogError, match="GeoJSON object"):
        load_catalog(_write(tmp_path, "[]"))


def test_load_catalog_requires_features_array(tmp_path: Path) -> None:
    """A document without a features array is rejected."""
    with pytest.raises(GeotilesCatalogError, match="features"):
        load_catalog(_write(tmp_path, '{"type": "FeatureCollection"}'))


def test_load_catalog_rejects_non_object_feature(tmp_path: Path) -> None:
    """A feature that is not an object is rejected."""
    with pytest.raises(
        GeotilesCatalogError, match="feature must be an object"
    ):
        load_catalog(_write(tmp_path, '{"features": [1]}'))


def test_load_catalog_requires_properties(tmp_path: Path) -> None:
    """A feature without a properties object is rejected."""
    with pytest.raises(GeotilesCatalogError, match="properties"):
        load_catalog(_write(tmp_path, '{"features": [{}]}'))


def test_load_catalog_requires_tile_id(tmp_path: Path) -> None:
    """A feature missing its sheet-id property is rejected."""
    body = '{"features": [{"properties": {}, "geometry": {}}]}'
    with pytest.raises(GeotilesCatalogError, match="AHN_subuni"):
        load_catalog(_write(tmp_path, body))


def test_load_catalog_rejects_blank_tile_id(tmp_path: Path) -> None:
    """A blank sheet id is rejected."""
    body = '{"features": [{"properties": {"AHN_subuni": " "}}]}'
    with pytest.raises(GeotilesCatalogError, match="AHN_subuni"):
        load_catalog(_write(tmp_path, body))


def test_load_catalog_requires_geometry(tmp_path: Path) -> None:
    """A feature without a geometry object is rejected."""
    body = '{"features": [{"properties": {"AHN_subuni": "A"}}]}'
    with pytest.raises(GeotilesCatalogError, match="geometry"):
        load_catalog(_write(tmp_path, body))


def test_load_catalog_rejects_geometry_without_coordinates(
    tmp_path: Path,
) -> None:
    """A geometry with no positions has no derivable extent."""
    body = (
        '{"features": [{"properties": {"AHN_subuni": "A"}, '
        '"geometry": {"coordinates": []}}]}'
    )
    with pytest.raises(GeotilesCatalogError, match="no coordinates"):
        load_catalog(_write(tmp_path, body))


def test_load_catalog_rejects_missing_coordinates(tmp_path: Path) -> None:
    """A geometry whose coordinates are absent (not a list) is rejected."""
    body = (
        '{"features": [{"properties": {"AHN_subuni": "A"}, "geometry": {}}]}'
    )
    with pytest.raises(GeotilesCatalogError, match="no coordinates"):
        load_catalog(_write(tmp_path, body))


def test_resolve_selects_covering_sheets_as_geotiles_urls() -> None:
    """Resolving addresses each covering sheet as a GeoTiles .LAZ download."""
    resolved = _source().resolve(
        _generation_source(), _SHARED_EDGE_AOI, lambda _url: b""
    )

    assert [tile.tile_id for tile in resolved.tiles] == [
        "37EN1_01",
        "37EN2_01",
    ]
    assert resolved.tiles[0].download_url.endswith("AHN5_T/37EN1_01.LAZ")
    assert resolved.licence.startswith("https://creativecommons.org")
    assert "GeoTiles" in resolved.attribution


def test_generation_registry_offers_ahn5_and_ahn4() -> None:
    """The GeoTiles registry advertises AHN5 (newest) then AHN4."""
    registry = _source().generation_registry(lambda _url: b"")

    assert registry.tokens() == ("auto", "ahn5", "ahn4")


def test_generation_registry_probe_reflects_catalogue_coverage() -> None:
    """The catalogue probe reports coverage without any network call."""
    registry = _source().generation_registry(lambda _url: b"")
    newest = registry.sources()[0]

    assert newest.probe(_SHARED_EDGE_AOI) is True
    assert newest.probe(_UNCOVERED_AOI) is False
