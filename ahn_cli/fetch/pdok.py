"""The PDOK INSPIRE ATOM distribution source (primary).

PDOK publishes AHN as an INSPIRE *predefined dataset* ATOM service: a dataset
feed carries feed-level ``rights``/``author`` terms and, inside its entries,
one ``<link rel="section">`` per native map sheet ("kaartblad") with the
sheet's download URL and its WGS84 ``bbox`` attribute. This module parses that
feed, resolves the sheets covering an area of interest, and exposes a
:class:`~ahn_cli.fetch.source.FetchSource` whose per-generation coverage probe
is the real "does this feed contain a sheet intersecting the AOI" test.

The parser and fixtures are modelled on the live PDOK AHN feed structure
(``https://service.pdok.nl/rws/ahn/atom/``): feed ``rights`` =
CC0 ``publicdomain/zero/1.0``, ``author`` = RWS, and section links whose
``bbox`` is ``minlon minlat maxlon maxlat`` in EPSG:4326. Per SPIKE-PDAL, PDOK
AHN LAZ is plain (no COPC/EPT), so acquisition downloads whole sheets and clips
locally; this module only resolves and addresses them.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlsplit
from xml.etree.ElementTree import Element, ParseError

from defusedxml.ElementTree import fromstring

from ahn_cli.domain import BBox, Generation
from ahn_cli.fetch.generation import (
    AvailabilityProbe,
    GenerationRegistry,
    GenerationSource,
)
from ahn_cli.fetch.source import (
    HttpGet,
    RemoteTile,
    ResolvedFeed,
    boxes_intersect,
    to_rd,
    to_wgs84,
)

_ATOM_NS = "http://www.w3.org/2005/Atom"
_SECTION_REL = "section"
_BBOX_COORD_COUNT = 4

# The generations PDOK serves as LAZ, newest first, each with its dataset-feed
# ATOM URL and a semantics note. The exact live feed filenames are pinned and
# verified by the nightly portal-contract test (WP14); the fast suite injects a
# feed fetcher, so these URLs are not exercised by the offline gate. Feed base
# duplicated (with citation) from the live PDOK AHN service rather than importing
# the deprecated ``ahn_cli.config``.
_PDOK_ATOM_BASE = "https://service.pdok.nl/rws/ahn/atom/"
_PDOK_FEEDS: tuple[tuple[int, str, str], ...] = (
    (
        5,
        _PDOK_ATOM_BASE + "ahn5_laz.xml",
        "AHN5 point cloud via PDOK ATOM (highest-point-per-cell surface).",
    ),
    (
        4,
        _PDOK_ATOM_BASE + "ahn4_laz.xml",
        "AHN4 point cloud via PDOK ATOM (IDW-mean surface).",
    ),
)


class PdokFeedError(ValueError):
    """Raised when a PDOK ATOM feed cannot be parsed or is missing terms.

    Signals a malformed feed (not XML, no ``rights``/``author``) or a section
    link missing its ``href`` or a well-formed four-number ``bbox`` -- the feed
    cannot then be trusted to address downloads.
    """


@dataclass(frozen=True)
class AtomTile:
    """One PDOK section link: a sheet's id, WGS84 extent, and download URL.

    Contract:
        - ``bbox_wgs84`` is the ``bbox`` attribute as published by PDOK
          (``minlon minlat maxlon maxlat``, EPSG:4326).
    """

    tile_id: str
    bbox_wgs84: BBox
    download_url: str


@dataclass(frozen=True)
class AtomFeed:
    """A parsed PDOK dataset feed: licence terms plus its section tiles.

    Contract:
        - ``licence`` is the feed ``rights`` URL; ``attribution`` the ``author``
          name; both non-blank (enforced by :func:`parse_atom_feed`).
        - ``tiles`` preserves the feed's declaration order.
    """

    licence: str
    attribution: str
    tiles: tuple[AtomTile, ...]


def parse_atom_feed(data: bytes) -> AtomFeed:
    """Parse a PDOK ATOM dataset feed into an :class:`AtomFeed`.

    Contract:
        - Reads the feed-level ``rights`` and ``author/name`` and every
          ``<link rel="section">`` (ignoring ``self``/``up``/``index`` links).
        - Each section link contributes an :class:`AtomTile` whose id is the
          download filename stem and whose extent is its WGS84 ``bbox``.

    Failure modes:
        - :class:`PdokFeedError` if the bytes are not XML, ``rights`` or the
          author name is missing/blank, or a section link lacks a ``href`` or a
          four-number ``bbox``.
    """
    try:
        root = fromstring(data)
    except ParseError as exc:
        msg = "PDOK ATOM feed is not well-formed XML."
        raise PdokFeedError(msg) from exc
    licence = _require_text(root.findtext(f"{{{_ATOM_NS}}}rights"), "rights")
    attribution = _require_text(
        root.findtext(f"{{{_ATOM_NS}}}author/{{{_ATOM_NS}}}name"),
        "author name",
    )
    tiles = tuple(
        _parse_section_link(link)
        for link in root.iter(f"{{{_ATOM_NS}}}link")
        if link.get("rel") == _SECTION_REL
    )
    return AtomFeed(licence=licence, attribution=attribution, tiles=tiles)


def _require_text(value: str | None, field: str) -> str:
    """Return stripped feed text, or raise if it is missing or blank."""
    if value is None or not value.strip():
        msg = f"PDOK ATOM feed is missing a non-blank {field}."
        raise PdokFeedError(msg)
    return value.strip()


def _parse_section_link(element: Element) -> AtomTile:
    """Build an :class:`AtomTile` from one ``rel="section"`` link element."""
    href = element.get("href")
    if href is None or not href.strip():
        msg = "PDOK ATOM section link is missing a href."
        raise PdokFeedError(msg)
    bbox = _parse_bbox_attr(element.get("bbox"))
    tile_id = _tile_id_from_href(href)
    return AtomTile(tile_id=tile_id, bbox_wgs84=bbox, download_url=href)


def _parse_bbox_attr(value: str | None) -> BBox:
    """Parse a ``minlon minlat maxlon maxlat`` attribute into a bbox tuple."""
    if value is None:
        msg = "PDOK ATOM section link is missing a bbox."
        raise PdokFeedError(msg)
    parts = value.split()
    if len(parts) != _BBOX_COORD_COUNT:
        msg = f"PDOK ATOM bbox must have four numbers; got {value!r}."
        raise PdokFeedError(msg)
    try:
        minlon, minlat, maxlon, maxlat = (float(part) for part in parts)
    except ValueError as exc:
        msg = f"PDOK ATOM bbox has a non-numeric coordinate: {value!r}."
        raise PdokFeedError(msg) from exc
    return (minlon, minlat, maxlon, maxlat)


def _tile_id_from_href(href: str) -> str:
    """Derive a sheet id from a download URL (its filename without suffix)."""
    name = urlsplit(href).path.rsplit("/", 1)[-1]
    stem = name.rsplit(".", 1)[0]
    if not stem:
        msg = f"PDOK ATOM section href has no filename: {href!r}."
        raise PdokFeedError(msg)
    return stem


def _covering_tiles(feed: AtomFeed, aoi_rd: BBox) -> tuple[RemoteTile, ...]:
    """Return the feed's tiles whose bbox intersects ``aoi_rd``, ordered by id.

    Intersection is an axis-aligned bounding-box test in WGS84 (the feed's own
    CRS) after projecting the AOI -- a safe superset of the true-geometry cover;
    each selected tile's extent is then projected back to EPSG:28992 for the
    :class:`RemoteTile` the downstream stages expect.
    """
    aoi_wgs84 = to_wgs84(aoi_rd)
    selected = [
        RemoteTile(
            tile_id=tile.tile_id,
            bbox=to_rd(tile.bbox_wgs84),
            download_url=tile.download_url,
        )
        for tile in feed.tiles
        if boxes_intersect(aoi_wgs84, tile.bbox_wgs84)
    ]
    return tuple(sorted(selected, key=lambda tile: tile.tile_id))


class PdokSource:
    """The PDOK INSPIRE ATOM distribution source.

    Implements :class:`~ahn_cli.fetch.source.FetchSource`: it advertises the
    AHN generations PDOK serves and resolves the covering sheets for a chosen
    generation by fetching and parsing that generation's ATOM dataset feed.
    """

    def generation_registry(self, http_get: HttpGet) -> GenerationRegistry:
        """Return a registry of PDOK generations with real coverage probes.

        Each generation's probe fetches and parses that generation's ATOM feed
        and reports whether any section tile intersects the AOI, so ``auto``
        selection picks the newest generation PDOK actually covers.
        """
        registry = GenerationRegistry()
        for number, feed_url, semantics in _PDOK_FEEDS:
            registry.register(
                GenerationSource(
                    generation=Generation(number),
                    base_url=feed_url,
                    probe=self._probe(http_get, feed_url),
                    semantics=semantics,
                )
            )
        return registry

    def _probe(self, http_get: HttpGet, feed_url: str) -> AvailabilityProbe:
        """Return a coverage probe over the feed at ``feed_url``."""

        def probe(aoi: BBox) -> bool:
            feed = parse_atom_feed(http_get(feed_url))
            return len(_covering_tiles(feed, aoi)) > 0

        return probe

    def resolve(
        self,
        generation_source: GenerationSource,
        aoi: BBox,
        http_get: HttpGet,
    ) -> ResolvedFeed:
        """Resolve the PDOK sheets of ``generation_source`` covering ``aoi``.

        Fetches the generation's ATOM feed (``generation_source.base_url``),
        parses it, and returns the intersecting sheets as EPSG:28992
        :class:`RemoteTile` addresses with the feed's licence terms.
        """
        feed = parse_atom_feed(http_get(generation_source.base_url))
        return ResolvedFeed(
            licence=feed.licence,
            attribution=feed.attribution,
            tiles=_covering_tiles(feed, aoi),
        )
