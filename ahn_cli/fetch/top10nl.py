"""TOP10NL greenhouse footprint fetch from the PDOK OGC API Features.

Fetches the BRT TOP10NL ``gebouw_vlak`` features whose ``typegebouw`` is
``kas, warenhuis`` (greenhouse / glasshouse building polygons) inside an
EPSG:28992 box, following the API's ``next`` links. The polygons are one
classification *cue* for downstream fusion; this module only acquires and
pins them and never transforms pixel or point data.

Pinning: an OGC API page carries a per-request ``timeStamp``, so raw page
bytes are not reproducible. The pin is the SHA-256 of the id-sorted selected
features serialised as compact ``sort_keys`` JSON (:func:`features_sha256`);
the page URLs are recorded alongside. TOP10NL is updated continuously and a
polygon may predate or postdate the AHN flight: each feature keeps its
``bronactualiteit`` (source date), which callers must disclose.

Everything reaching the network is injected (``http_get``), so the fast suite
is offline.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from importlib.metadata import version
from typing import TYPE_CHECKING, cast

from ahn_cli.domain import BBox, Product, Provenance, ensure_valid_bbox
from ahn_cli.fetch.acquisition import AcquisitionError, default_http_get

if TYPE_CHECKING:
    from datetime import datetime

    from ahn_cli.fetch.source import HttpGet

API_BASE = "https://api.pdok.nl/kadaster/brt-top10nl/ogc/v1"
COLLECTION = "gebouw_vlak"
TYPE_PROPERTY = "typegebouw"
KAS_TYPE = "kas, warenhuis"
LICENCE = "https://creativecommons.org/licenses/by/4.0/"
ATTRIBUTION = "Kadaster, BRT TOP10NL (CC BY 4.0), via PDOK"
_RD_CRS_URI = "http://www.opengis.net/def/crs/EPSG/0/28992"
_PAGE_LIMIT = 1000
MAX_PAGES = 100
"""Hard stop on ``next`` links: a larger box is a usage error, not a loop."""
_POLYGON_TYPES = frozenset({"Polygon", "MultiPolygon"})


class Top10nlError(AcquisitionError):
    """Raised when the TOP10NL fetch cannot produce a trustworthy result."""


@dataclass(frozen=True)
class KasFetch:
    """The selected greenhouse features and how they were obtained.

    Contract:
        features: id-sorted GeoJSON feature dicts, geometry in EPSG:28992.
        page_urls: every page URL requested, in order.
        n_in_box: all ``gebouw_vlak`` features returned for the box.
        features_sha256: the canonical content pin (:func:`features_sha256`).
        source_dates: sorted distinct ``bronactualiteit`` values.
    """

    features: tuple[dict[str, object], ...]
    page_urls: tuple[str, ...]
    n_in_box: int
    features_sha256: str
    source_dates: tuple[str, ...]


def items_url(bbox: BBox) -> str:
    """Return the first ``items`` page URL for ``bbox`` (RD in, RD out)."""
    box = ",".join(repr(float(v)) for v in bbox)
    return (
        f"{API_BASE}/collections/{COLLECTION}/items?f=json"
        f"&limit={_PAGE_LIMIT}&bbox={box}"
        f"&bbox-crs={_RD_CRS_URI}&crs={_RD_CRS_URI}"
    )


def features_sha256(features: tuple[dict[str, object], ...]) -> str:
    """Return the canonical SHA-256 of ``features`` (id-sorted, compact)."""
    ordered = sorted(features, key=lambda f: str(f.get("id")))
    blob = json.dumps(ordered, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def _page(raw: bytes, url: str) -> tuple[list[object], str | None]:
    """Decode one page: its feature list and the ``next`` href, if any."""
    try:
        doc = json.loads(raw)
    except ValueError as exc:
        msg = f"TOP10NL page is not JSON: {url}"
        raise Top10nlError(msg) from exc
    if not isinstance(doc, dict) or not isinstance(
        cast("dict[str, object]", doc).get("features"), list
    ):
        msg = f"TOP10NL page has no 'features' list: {url}"
        raise Top10nlError(msg)
    page = cast("dict[str, object]", doc)
    links = page.get("links")
    nxt: str | None = None
    if isinstance(links, list):
        for link in cast("list[object]", links):
            if isinstance(link, dict):
                entry = cast("dict[str, object]", link)
                href = entry.get("href")
                if entry.get("rel") == "next" and isinstance(href, str):
                    nxt = href
    return cast("list[object]", page["features"]), nxt


def _is_kas(feature: dict[str, object]) -> bool:
    """Return whether ``feature`` is a greenhouse with polygon geometry."""
    props = feature.get("properties")
    if not isinstance(props, dict):
        return False
    if cast("dict[str, object]", props).get(TYPE_PROPERTY) != KAS_TYPE:
        return False
    geom = feature.get("geometry")
    if not isinstance(geom, dict) or (
        cast("dict[str, object]", geom).get("type") not in _POLYGON_TYPES
    ):
        msg = f"TOP10NL kas feature {feature.get('id')!r} is not a polygon."
        raise Top10nlError(msg)
    return True


def fetch_kas(
    bbox: BBox, *, http_get: HttpGet = default_http_get
) -> KasFetch:
    """Fetch every TOP10NL greenhouse polygon intersecting ``bbox``.

    Contract:
        - Follows ``next`` links until none remain, at most
          :data:`MAX_PAGES` pages.
        - Keeps features whose ``typegebouw`` equals :data:`KAS_TYPE`.

    Failure modes:
        - :class:`Top10nlError` for a degenerate box, a non-JSON page, a page
          without ``features``, more than :data:`MAX_PAGES` pages, a kas
          feature without polygon geometry, or zero kas features (an empty
          footprint must not pass silently as "no greenhouses").
    """
    try:
        ensure_valid_bbox(bbox)
    except ValueError as exc:
        raise Top10nlError(str(exc)) from exc
    url: str | None = items_url(bbox)
    urls: list[str] = []
    selected: list[dict[str, object]] = []
    n_all = 0
    while url is not None:
        if len(urls) == MAX_PAGES:
            msg = f"TOP10NL fetch exceeded {MAX_PAGES} pages; shrink the box."
            raise Top10nlError(msg)
        urls.append(url)
        feats, url = _page(http_get(url), urls[-1])
        n_all += len(feats)
        selected += [
            cast("dict[str, object]", f)
            for f in feats
            if isinstance(f, dict) and _is_kas(cast("dict[str, object]", f))
        ]
    if not selected:
        msg = f"no TOP10NL '{KAS_TYPE}' polygons in bbox {bbox}."
        raise Top10nlError(msg)
    selected.sort(key=lambda f: str(f.get("id")))
    stamps = (
        cast("dict[str, object]", f["properties"]).get("bronactualiteit")
        for f in selected
    )
    dates = sorted({str(s) for s in stamps if s is not None})
    feats_t = tuple(selected)
    return KasFetch(
        features=feats_t,
        page_urls=tuple(urls),
        n_in_box=n_all,
        features_sha256=features_sha256(feats_t),
        source_dates=tuple(dates),
    )


def kas_geojson_bytes(result: KasFetch) -> bytes:
    """Serialise ``result`` as a deterministic GeoJSON FeatureCollection."""
    doc = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "EPSG:28992"}},
        "features": list(result.features),
    }
    return json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()


def kas_provenance(
    result: KasFetch,
    bbox: BBox,
    *,
    started_at: datetime,
    finished_at: datetime,
    output_checksum: str,
    resolution_m: float,
) -> Provenance:
    """Build the provenance record for a written footprint product.

    Contract:
        - ``input_checksum`` is the canonical feature pin; ``output_checksum``
          is the caller's checksum of the written raster. The page URLs, the
          feature counts and the source-date range are request keys.
    """
    dates = result.source_dates
    return Provenance(
        source_portal="pdok",
        product=Product.FOOTPRINT,
        licence=LICENCE,
        attribution=ATTRIBUTION,
        bbox=bbox,
        download_started_at=started_at,
        download_finished_at=finished_at,
        input_checksum=result.features_sha256,
        output_checksum=output_checksum,
        tool_version=version("ahn_cli"),
        resolution_tier=f"{resolution_m:g}m",
        request_keys=(
            ("api", API_BASE),
            ("collection", COLLECTION),
            ("filter", f"{TYPE_PROPERTY}={KAS_TYPE}"),
            ("n_in_box", str(result.n_in_box)),
            ("n_kas", str(len(result.features))),
            ("bronactualiteit_min", dates[0] if dates else ""),
            ("bronactualiteit_max", dates[-1] if dates else ""),
            ("page_urls", " ".join(result.page_urls)),
        ),
    )
