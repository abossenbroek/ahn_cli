"""Tests for the TOP10NL greenhouse fetch (offline, injected HTTP)."""

import json
from collections.abc import Callable, Sequence
from datetime import datetime, timezone

import pytest

from ahn_cli.domain import Product
from ahn_cli.fetch import top10nl
from ahn_cli.fetch.top10nl import (
    KAS_TYPE,
    Top10nlError,
    features_sha256,
    fetch_kas,
    items_url,
    kas_geojson_bytes,
    kas_provenance,
)

_BBOX = (1000.0, 2000.0, 1100.0, 2100.0)
_SQUARE: dict[str, object] = {
    "type": "Polygon",
    "coordinates": [
        [[1010, 2010], [1020, 2010], [1020, 2020], [1010, 2020], [1010, 2010]]
    ],
}


def _feat(
    fid: str,
    kind: str = KAS_TYPE,
    geom: dict[str, object] | None = None,
    date: str | None = "2020-01-01",
) -> dict[str, object]:
    props: dict[str, object] = {"typegebouw": kind}
    if date is not None:
        props["bronactualiteit"] = date
    return {
        "type": "Feature",
        "id": fid,
        "properties": props,
        "geometry": _SQUARE if geom is None else geom,
    }


def _page(features: Sequence[object], nxt: str | None = None) -> bytes:
    links: list[object] = [{"rel": "self", "href": "x"}, "not-a-link"]
    if nxt is not None:
        links.append({"rel": "next", "href": nxt})
    return json.dumps({"features": features, "links": links}).encode()


def _http(pages: dict[str, bytes]) -> Callable[[str], bytes]:
    def get(url: str) -> bytes:
        return pages[url]

    return get


def test_items_url_requests_rd_in_and_out() -> None:
    """The first page asks for EPSG:28992 bbox and output CRS."""
    url = items_url(_BBOX)
    assert "/collections/gebouw_vlak/items?f=json&limit=1000" in url
    assert "bbox=1000.0,2000.0,1100.0,2100.0" in url
    assert url.count("EPSG/0/28992") == 2


def test_fetch_follows_next_and_keeps_only_kas() -> None:
    """Pages are followed; non-kas, non-dict and prop-less entries drop."""
    first = items_url(_BBOX)
    multi: dict[str, object] = {
        "type": "MultiPolygon",
        "coordinates": [_SQUARE["coordinates"]],
    }
    pages = {
        first: _page(
            [_feat("b", date="2024-06-01"), _feat("o", "overig"), 42],
            "p2",
        ),
        "p2": _page(
            [
                _feat("a", geom=multi, date=None),
                {"id": "np", "properties": None},
            ]
        ),
    }
    got = fetch_kas(_BBOX, http_get=_http(pages))
    assert [f["id"] for f in got.features] == ["a", "b"]
    assert got.page_urls == (first, "p2")
    assert got.n_in_box == 5
    assert got.source_dates == ("2024-06-01",)
    assert got.features_sha256 == features_sha256(got.features)


def test_links_absent_ends_paging() -> None:
    """A page without a links list is the last page."""
    raw = json.dumps({"features": [_feat("a")]}).encode()
    got = fetch_kas(_BBOX, http_get=_http({items_url(_BBOX): raw}))
    assert len(got.features) == 1


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        (b"<html>", "not JSON"),
        (b"[]", "no 'features' list"),
        (b'{"features": 3}', "no 'features' list"),
    ],
)
def test_bad_pages_fail(raw: bytes, match: str) -> None:
    """A non-JSON page or one without a features list fails loudly."""
    with pytest.raises(Top10nlError, match=match):
        fetch_kas(_BBOX, http_get=_http({items_url(_BBOX): raw}))


@pytest.mark.parametrize("geom", [{"type": "Point"}, None])
def test_kas_without_polygon_fails(geom: dict[str, object] | None) -> None:
    """A kas feature that is not a polygon is a data error, not a skip."""
    feat = _feat("a")
    feat["geometry"] = geom
    page = _page([feat])
    with pytest.raises(Top10nlError, match="not a polygon"):
        fetch_kas(_BBOX, http_get=_http({items_url(_BBOX): page}))


def test_zero_kas_fails() -> None:
    """An empty selection never passes as 'no greenhouses'."""
    page = _page([_feat("o", "overig")])
    with pytest.raises(Top10nlError, match="no TOP10NL"):
        fetch_kas(_BBOX, http_get=_http({items_url(_BBOX): page}))


def test_page_limit_stops_a_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """A next link that never ends stops at MAX_PAGES."""
    monkeypatch.setattr(top10nl, "MAX_PAGES", 3)

    def get(_url: str) -> bytes:
        return _page([_feat("a")], "again")

    with pytest.raises(Top10nlError, match="exceeded 3 pages"):
        fetch_kas(_BBOX, http_get=get)


def test_degenerate_bbox_fails_before_network() -> None:
    """A bad box raises without calling http_get."""

    def get(_url: str) -> bytes:  # must not be called
        raise AssertionError

    with pytest.raises(Top10nlError, match="minx < maxx"):
        fetch_kas((1.0, 1.0, 0.0, 2.0), http_get=get)


def test_sha_is_order_independent_and_content_sensitive() -> None:
    """The pin ignores page order but sees any property change."""
    a, b = _feat("a"), _feat("b")
    assert features_sha256((a, b)) == features_sha256((b, a))
    assert features_sha256((a, b)) != features_sha256((a, _feat("b", "x")))


def test_geojson_and_provenance() -> None:
    """GeoJSON is deterministic; provenance pins input and dates."""
    page = _page([_feat("a", date=None)])
    got = fetch_kas(_BBOX, http_get=_http({items_url(_BBOX): page}))
    doc = json.loads(kas_geojson_bytes(got))
    assert doc["type"] == "FeatureCollection"
    assert doc["crs"]["properties"]["name"] == "EPSG:28992"
    assert kas_geojson_bytes(got) == kas_geojson_bytes(got)
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    prov = kas_provenance(
        got,
        _BBOX,
        started_at=t0,
        finished_at=t0,
        output_checksum="ff",
        resolution_m=1.0,
    )
    keys = dict(prov.request_keys)
    assert prov.product is Product.FOOTPRINT
    assert prov.input_checksum == got.features_sha256
    assert prov.resolution_tier == "1m"
    assert keys["bronactualiteit_min"] == ""
    assert keys["n_kas"] == "1"
