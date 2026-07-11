"""Tests for the fetch-context acquisition actuation."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pytest
import requests
from pyproj import Transformer

from ahn_cli.domain import Generation, Product
from ahn_cli.fetch import acquisition
from ahn_cli.fetch.acquisition import (
    SITE_SUBDIRS,
    AcquisitionError,
    AcquisitionRequest,
    AreaSelectorKind,
    MalformedGeojsonError,
    acquire,
    aoi_bbox,
    create_site_layout,
    default_http_get,
    source_for,
)
from ahn_cli.fetch.geotiles_source import GeotilesSource
from ahn_cli.fetch.pdok import PdokSource
from ahn_cli.fetch.source import FetchSource, SourceKind
from ahn_cli.provenance import read_provenance

_FIXTURES = Path(__file__).parent / "fixtures"
_ATOM_BYTES = (_FIXTURES / "pdok_ahn_atom.xml").read_bytes()
_SAMPLE_CATALOG = _FIXTURES / "ahn_subunit_sample.geojson"
_TO_RD = Transformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=True)
_FIXED_TIME = datetime(2026, 7, 10, 12, 0, tzinfo=timezone.utc)


def _rd_bbox(
    minlon: float, minlat: float, maxlon: float, maxlat: float
) -> str:
    """Project a WGS84 box to an EPSG:28992 ``--bbox`` string."""
    minx, miny = _TO_RD.transform(minlon, minlat)
    maxx, maxy = _TO_RD.transform(maxlon, maxlat)
    return f"{minx},{miny},{maxx},{maxy}"


_SHARED_EDGE_BBOX = _rd_bbox(4.40, 51.99, 4.44, 52.02)
_UNCOVERED_BBOX = _rd_bbox(6.40, 52.40, 6.50, 52.50)


class _Recorder:
    """An injected HttpGet that serves fixtures and records requested URLs."""

    def __init__(self) -> None:
        self.urls: list[str] = []

    def __call__(self, url: str) -> bytes:
        """Return the ATOM feed for feed URLs, fake bytes for .LAZ URLs."""
        self.urls.append(url)
        if url.endswith(".LAZ"):
            return b"LAZ-BYTES:" + url.rsplit("/", 1)[-1].encode()
        return _ATOM_BYTES

    def laz_urls(self) -> list[str]:
        """Return only the tile-download URLs seen so far."""
        return [url for url in self.urls if url.endswith(".LAZ")]


def _bbox_request(
    site: Path,
    *,
    bbox: str = _SHARED_EDGE_BBOX,
    source: SourceKind = SourceKind.PDOK,
    generation: Generation | None = None,
) -> AcquisitionRequest:
    """Build a bbox-selector acquisition request."""
    return AcquisitionRequest(
        site_dir=site,
        selector=AreaSelectorKind.BBOX,
        area=bbox,
        source=source,
        generation=generation,
    )


def _fixed_now() -> datetime:
    """Return a constant timezone-aware timestamp for determinism."""
    return _FIXED_TIME


# A WGS84 box over the same Delft-edge area the shared RD bbox covers, so a
# GeoJSON polygon over it resolves to the same covering sheets.
_COVERED_WGS84 = (4.40, 51.99, 4.44, 52.02)


def _polygon(
    minlon: float, minlat: float, maxlon: float, maxlat: float
) -> dict[str, object]:
    """Build a closed-ring GeoJSON Polygon geometry over a WGS84 box."""
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [minlon, minlat],
                [maxlon, minlat],
                [maxlon, maxlat],
                [minlon, maxlat],
                [minlon, minlat],
            ]
        ],
    }


def _write_geojson(
    tmp_path: Path, doc: object, name: str = "area.geojson"
) -> Path:
    """Serialise ``doc`` to a GeoJSON file under ``tmp_path`` and return it."""
    path = tmp_path / name
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def _geojson_request(
    site: Path,
    path: Path,
    *,
    source: SourceKind = SourceKind.PDOK,
    generation: Generation | None = None,
) -> AcquisitionRequest:
    """Build a geojson-selector acquisition request pointing at ``path``."""
    return AcquisitionRequest(
        site_dir=site,
        selector=AreaSelectorKind.GEOJSON,
        area=str(path),
        source=source,
        generation=generation,
    )


def _assert_rd_matches(
    bbox: tuple[float, float, float, float],
    minlon: float,
    minlat: float,
    maxlon: float,
    maxlat: float,
) -> None:
    """Assert an EPSG:28992 bbox equals the RD projection of a WGS84 box."""
    exp_minx, exp_miny = _TO_RD.transform(minlon, minlat)
    exp_maxx, exp_maxy = _TO_RD.transform(maxlon, maxlat)
    expected = (exp_minx, exp_miny, exp_maxx, exp_maxy)
    for actual, want in zip(bbox, expected, strict=True):
        assert math.isclose(actual, want, abs_tol=1e-6)


def test_site_subdirs_are_the_three_canonical_products() -> None:
    """The layout is fixed to the ahn/ortho/viirs product subdirectories."""
    assert SITE_SUBDIRS == ("ahn", "ortho", "viirs")


def test_create_site_layout_makes_every_subdir_in_order(
    tmp_path: Path,
) -> None:
    """The layout creates one directory per product, in canonical order."""
    site = tmp_path / "delft"

    created = create_site_layout(site)

    assert created == tuple(site / name for name in SITE_SUBDIRS)
    for subdir in created:
        assert subdir.is_dir()


def test_create_site_layout_is_idempotent(tmp_path: Path) -> None:
    """Re-running on an existing site leaves it intact and does not raise."""
    site = tmp_path / "delft"
    create_site_layout(site)

    created_again = create_site_layout(site)

    assert created_again == tuple(site / name for name in SITE_SUBDIRS)


def test_source_for_maps_kinds_to_their_sources() -> None:
    """The source registry maps each kind to its concrete source."""
    assert isinstance(source_for(SourceKind.PDOK), PdokSource)
    assert isinstance(source_for(SourceKind.GEOTILES), GeotilesSource)


def test_acquisition_request_is_value_typed(tmp_path: Path) -> None:
    """The request is a frozen value object with PDOK/auto defaults."""
    request = _bbox_request(tmp_path)

    assert request.source is SourceKind.PDOK
    assert request.generation is None
    assert len({request, _bbox_request(tmp_path)}) == 1


def test_acquire_downloads_covering_tiles_with_provenance(
    tmp_path: Path,
) -> None:
    """A PDOK bbox fetch writes each covering sheet and its provenance."""
    site = tmp_path / "delft"
    http_get = _Recorder()

    written = acquire(
        _bbox_request(site),
        http_get=http_get,
        now=_fixed_now,
        tool_version="wp6-test",
    )

    assert [path.name for path in written] == [
        "C_37EN1.LAZ",
        "C_37EN2.LAZ",
    ]
    for path in written:
        assert path.read_bytes().startswith(b"LAZ-BYTES:")
    ahn_dir = site / "ahn"
    provenance = read_provenance(ahn_dir / "C_37EN1.provenance.json")
    expected_checksum = hashlib.sha256(
        (ahn_dir / "C_37EN1.LAZ").read_bytes()
    ).hexdigest()
    assert provenance.source_portal == "pdok"
    assert provenance.product is Product.AHN_POINT_CLOUD
    assert provenance.generation == Generation(5)
    assert provenance.licence.startswith("https://creativecommons.org")
    assert provenance.input_checksum == expected_checksum
    assert provenance.output_checksum == expected_checksum
    assert provenance.tool_version == "wp6-test"
    assert ("source", "pdok") in provenance.request_keys
    assert ("tile_id", "C_37EN1") in provenance.request_keys


def test_acquire_is_idempotent_through_the_cache(tmp_path: Path) -> None:
    """A second fetch serves cached tiles with zero tile-download calls."""
    site = tmp_path / "delft"
    http_get = _Recorder()
    request = _bbox_request(site)

    acquire(request, http_get=http_get, now=_fixed_now, tool_version="v")
    first_downloads = sorted(http_get.laz_urls())
    http_get.urls.clear()

    acquire(request, http_get=http_get, now=_fixed_now, tool_version="v")

    assert first_downloads  # the first run did download
    assert http_get.laz_urls() == []  # the second run hit the cache


def test_acquire_honours_an_explicit_generation(tmp_path: Path) -> None:
    """An explicit --ahn ahn4 records AHN4 in provenance, not the newest."""
    site = tmp_path / "delft"

    acquire(
        _bbox_request(site, generation=Generation(4)),
        http_get=_Recorder(),
        now=_fixed_now,
        tool_version="v",
    )

    provenance = read_provenance(site / "ahn" / "C_37EN1.provenance.json")
    assert provenance.generation == Generation(4)


def test_acquire_uses_a_custom_cache_root(tmp_path: Path) -> None:
    """A supplied cache root is used instead of the in-site default."""
    site = tmp_path / "delft"
    cache_root = tmp_path / "shared-cache"

    acquire(
        _bbox_request(site),
        http_get=_Recorder(),
        now=_fixed_now,
        cache_root=cache_root,
        tool_version="v",
    )

    assert (cache_root / "blobs").is_dir()
    assert not (site / ".cache").exists()


def test_acquire_uses_default_clock_and_tool_version(tmp_path: Path) -> None:
    """With neither injected, the real clock and package version are used."""
    site = tmp_path / "delft"

    acquire(_bbox_request(site), http_get=_Recorder())

    provenance = read_provenance(site / "ahn" / "C_37EN1.provenance.json")
    assert provenance.tool_version
    assert provenance.tool_version[0].isdigit()
    assert provenance.download_started_at.tzinfo is not None


def _patch_geotiles_catalog(
    monkeypatch: pytest.MonkeyPatch, catalog_path: Path
) -> None:
    """Point the module's geotiles source at ``catalog_path`` for a test."""
    real_source_for = acquisition.source_for

    def fake_source_for(kind: SourceKind) -> FetchSource:
        if kind is SourceKind.GEOTILES:
            return GeotilesSource(catalog_path=catalog_path)
        return real_source_for(kind)

    monkeypatch.setattr(acquisition, "source_for", fake_source_for)


def test_acquire_fetches_through_the_geotiles_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The geotiles source resolves via the catalogue and downloads tiles."""
    site = tmp_path / "delft"
    _patch_geotiles_catalog(monkeypatch, _SAMPLE_CATALOG)
    http_get = _Recorder()

    written = acquire(
        _bbox_request(site, source=SourceKind.GEOTILES),
        http_get=http_get,
        now=_fixed_now,
        tool_version="v",
    )

    assert [path.name for path in written] == ["37EN1_01.LAZ", "37EN2_01.LAZ"]
    provenance = read_provenance(site / "ahn" / "37EN1_01.provenance.json")
    assert provenance.source_portal == "geotiles"


def test_acquire_rejects_a_malformed_bbox_length(tmp_path: Path) -> None:
    """A bbox without four coordinates is a user-facing acquisition error."""
    request = _bbox_request(tmp_path / "s", bbox="0,0,1")
    with pytest.raises(AcquisitionError, match="four"):
        acquire(request, http_get=_Recorder(), now=_fixed_now)


def test_acquire_rejects_a_non_numeric_bbox(tmp_path: Path) -> None:
    """A bbox with a non-numeric coordinate is rejected."""
    request = _bbox_request(tmp_path / "s", bbox="0,0,1,north")
    with pytest.raises(AcquisitionError, match="non-numeric"):
        acquire(request, http_get=_Recorder(), now=_fixed_now)


def test_acquire_rejects_a_degenerate_bbox(tmp_path: Path) -> None:
    """A zero-area bbox is rejected via the shared validator."""
    request = _bbox_request(tmp_path / "s", bbox="1,1,0,0")
    with pytest.raises(AcquisitionError, match="bbox"):
        acquire(request, http_get=_Recorder(), now=_fixed_now)


def test_acquire_defers_city_selector(tmp_path: Path) -> None:
    """The city selector's AOI derivation is a typed deferral, not silent."""
    site = tmp_path / "delft"
    request = AcquisitionRequest(
        site_dir=site, selector=AreaSelectorKind.CITY, area="delft"
    )

    with pytest.raises(AcquisitionError, match="not wired"):
        acquire(request, http_get=_Recorder(), now=_fixed_now)
    assert (site / "ahn").is_dir()  # layout still created before deferral


def test_malformed_geojson_error_is_an_acquisition_error() -> None:
    """The geojson failure type funnels through the acquisition base error."""
    assert issubclass(MalformedGeojsonError, AcquisitionError)


def test_aoi_bbox_from_a_bare_polygon_geometry(tmp_path: Path) -> None:
    """A top-level Polygon geometry derives its RD extent as the AOI."""
    path = _write_geojson(tmp_path, _polygon(*_COVERED_WGS84))

    bbox = aoi_bbox(_geojson_request(tmp_path / "s", path))

    _assert_rd_matches(bbox, *_COVERED_WGS84)


def test_aoi_bbox_from_a_feature(tmp_path: Path) -> None:
    """A Feature wrapping a Polygon derives the wrapped polygon's extent."""
    feature = {"type": "Feature", "geometry": _polygon(*_COVERED_WGS84)}
    path = _write_geojson(tmp_path, feature)

    bbox = aoi_bbox(_geojson_request(tmp_path / "s", path))

    _assert_rd_matches(bbox, *_COVERED_WGS84)


def test_aoi_bbox_from_a_feature_collection_skips_non_polygons(
    tmp_path: Path,
) -> None:
    """Only polygonal features count; Point/null/non-dict members are skipped."""
    collection = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "geometry": _polygon(*_COVERED_WGS84)},
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [5.0, 52.5]},
            },
            {"type": "Feature", "geometry": None},
            None,
        ],
    }
    path = _write_geojson(tmp_path, collection)

    bbox = aoi_bbox(_geojson_request(tmp_path / "s", path))

    _assert_rd_matches(bbox, *_COVERED_WGS84)


def test_aoi_bbox_from_a_multipolygon(tmp_path: Path) -> None:
    """A MultiPolygon's AOI is the union extent of its member polygons."""
    poly_a = _polygon(4.40, 51.99, 4.42, 52.01)["coordinates"]
    poly_b = _polygon(4.43, 52.00, 4.44, 52.02)["coordinates"]
    geometry = {
        "type": "MultiPolygon",
        "coordinates": [poly_a, poly_b],
    }
    path = _write_geojson(tmp_path, geometry)

    bbox = aoi_bbox(_geojson_request(tmp_path / "s", path))

    _assert_rd_matches(bbox, 4.40, 51.99, 4.44, 52.02)


def test_aoi_bbox_rejects_a_missing_geojson_file(tmp_path: Path) -> None:
    """A path to no file is a user-facing malformed-geojson error."""
    request = _geojson_request(tmp_path / "s", tmp_path / "nope.geojson")
    with pytest.raises(MalformedGeojsonError, match="could not be read"):
        aoi_bbox(request)


def test_aoi_bbox_rejects_invalid_json(tmp_path: Path) -> None:
    """A file that is not valid JSON is a malformed-geojson error."""
    path = tmp_path / "area.geojson"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(MalformedGeojsonError, match="valid JSON"):
        aoi_bbox(_geojson_request(tmp_path / "s", path))


def test_aoi_bbox_rejects_a_document_with_no_polygon(tmp_path: Path) -> None:
    """A doc whose only geometries are non-polygonal has no derivable AOI."""
    collection = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [5.0, 52.5]},
            },
            {"type": "Feature", "geometry": None},
        ],
    }
    path = _write_geojson(tmp_path, collection)
    with pytest.raises(MalformedGeojsonError, match="no polygon"):
        aoi_bbox(_geojson_request(tmp_path / "s", path))


def test_aoi_bbox_rejects_a_non_object_top_level(tmp_path: Path) -> None:
    """Valid JSON that is not a GeoJSON object yields no polygon geometry."""
    path = _write_geojson(tmp_path, [])
    with pytest.raises(MalformedGeojsonError, match="no polygon"):
        aoi_bbox(_geojson_request(tmp_path / "s", path))


def test_aoi_bbox_rejects_a_feature_collection_without_features_list(
    tmp_path: Path,
) -> None:
    """A FeatureCollection whose ``features`` is not a list has no geometry."""
    path = _write_geojson(
        tmp_path, {"type": "FeatureCollection", "features": {}}
    )
    with pytest.raises(MalformedGeojsonError, match="no polygon"):
        aoi_bbox(_geojson_request(tmp_path / "s", path))


def test_aoi_bbox_rejects_a_degenerate_geometry(tmp_path: Path) -> None:
    """A zero-area polygon projects to a degenerate box and is rejected."""
    geometry = {
        "type": "Polygon",
        "coordinates": [[[4.4, 52.0], [4.4, 52.0], [4.4, 52.0], [4.4, 52.0]]],
    }
    path = _write_geojson(tmp_path, geometry)
    with pytest.raises(MalformedGeojsonError, match="bbox"):
        aoi_bbox(_geojson_request(tmp_path / "s", path))


def test_acquire_downloads_covering_tiles_from_a_geojson(
    tmp_path: Path,
) -> None:
    """A geojson polygon feeds the full acquire flow and writes the sheets."""
    site = tmp_path / "delft"
    path = _write_geojson(tmp_path, _polygon(*_COVERED_WGS84))
    http_get = _Recorder()

    written = acquire(
        _geojson_request(site, path),
        http_get=http_get,
        now=_fixed_now,
        tool_version="v",
    )

    assert [p.name for p in written] == ["C_37EN1.LAZ", "C_37EN2.LAZ"]


def test_acquire_reports_when_no_generation_covers_the_aoi(
    tmp_path: Path,
) -> None:
    """An AOI covered by no generation surfaces a clean acquisition error."""
    request = _bbox_request(tmp_path / "s", bbox=_UNCOVERED_BBOX)
    with pytest.raises(AcquisitionError):
        acquire(request, http_get=_Recorder(), now=_fixed_now)


def test_acquire_reports_an_unregistered_generation(tmp_path: Path) -> None:
    """Requesting a generation no source serves is a clean acquisition error."""
    request = _bbox_request(tmp_path / "s", generation=Generation(9))
    with pytest.raises(AcquisitionError):
        acquire(request, http_get=_Recorder(), now=_fixed_now)


def test_acquire_funnels_an_invalid_feed(tmp_path: Path) -> None:
    """A changed/invalid distribution feed surfaces as AcquisitionError."""

    def broken_feed(_url: str) -> bytes:
        return b"<<< not a valid ATOM feed"

    request = _bbox_request(tmp_path / "s")
    with pytest.raises(AcquisitionError):
        acquire(request, http_get=broken_feed, now=_fixed_now)


def test_acquire_funnels_a_download_failure(tmp_path: Path) -> None:
    """A tile-download HTTP failure surfaces as AcquisitionError, not a raw one."""

    def failing_download(url: str) -> bytes:
        if url.endswith(".LAZ"):
            msg = "503 Server Error"
            raise requests.HTTPError(msg)
        return _ATOM_BYTES

    request = _bbox_request(tmp_path / "delft")
    with pytest.raises(AcquisitionError, match="download failed"):
        acquire(request, http_get=failing_download, now=_fixed_now)


def test_default_http_get_returns_body_and_raises_for_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The production getter returns the body and checks the HTTP status."""

    class _Response:
        content = b"payload"
        raised = False

        def raise_for_status(self) -> None:
            type(self).raised = True

    def _fake_get(url: str, *, timeout: int) -> _Response:
        del url, timeout
        return _Response()

    monkeypatch.setattr(requests, "get", _fake_get)

    assert default_http_get("https://x/y") == b"payload"
    assert _Response.raised
