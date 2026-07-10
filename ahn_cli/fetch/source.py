"""Shared value objects for the fetch context's distribution sources.

WP6 makes acquisition *source-aware*: the same area of interest can be served
by more than one distribution portal. This module holds the vocabulary those
sources share -- the closed :class:`SourceKind` set, the :class:`RemoteTile`
address of one downloadable sheet, the :class:`ResolvedFeed` a source returns,
and the :class:`FetchSource` protocol every source implements -- plus the
EPSG:28992 <-> EPSG:4326 helpers both the PDOK and GeoTiles sources need to
reconcile a Dutch-grid AOI with a portal's WGS84 tile index.

It imports no concrete source module, so the source registry in
:mod:`ahn_cli.fetch.acquisition` can depend on this vocabulary without a cycle.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Protocol

from pyproj import Transformer

from ahn_cli.domain import BBox, ensure_valid_bbox

if TYPE_CHECKING:
    from ahn_cli.fetch.generation import GenerationRegistry, GenerationSource

HttpGet = Callable[[str], bytes]
"""An injected HTTP GET: maps an absolute URL to the response body bytes.

Injected everywhere a source reaches the network so the fast test suite passes
a deterministic in-memory fake and never performs real I/O.
"""

_RD_CRS = "EPSG:28992"
_WGS84_CRS = "EPSG:4326"

# Transformers are built once: pyproj instances are reusable and thread-safe,
# and reusing them keeps CRS conversion deterministic and cheap.
_TO_WGS84 = Transformer.from_crs(_RD_CRS, _WGS84_CRS, always_xy=True)
_TO_RD = Transformer.from_crs(_WGS84_CRS, _RD_CRS, always_xy=True)


class UnknownSourceError(LookupError):
    """Raised when a ``--source`` token names no known distribution source.

    Signals that a caller offered a token outside the closed
    :class:`SourceKind` set; the CLI restricts its choices to
    :func:`source_kind_tokens`, so this guards programmatic callers.
    """


class SourceKind(Enum):
    """The closed set of distribution sources acquisition can fetch from.

    Contract:
        - ``PDOK`` is the primary INSPIRE ATOM distribution (native sheets).
        - ``GEOTILES`` is the GeoTiles.nl fallback (1x1.25km re-tiling).
        - ``value`` is the stable ``--source`` token and the portal identifier
          recorded in provenance; branch on the member, never its string.

    Invariants:
        - Immutable and hashable, so a kind is safe as a registry key.
    """

    PDOK = "pdok"
    GEOTILES = "geotiles"


def source_kind_tokens() -> tuple[str, ...]:
    """Return the ``--source`` tokens, PDOK first, derived from the enum.

    Contract:
        - The order is the declaration order of :class:`SourceKind`, so PDOK
          (the default) leads; the CLI uses this for its choice list rather
          than a hardcoded literal.
    """
    return tuple(kind.value for kind in SourceKind)


def resolve_source_token(token: str) -> SourceKind:
    """Map a ``--source`` token to its :class:`SourceKind`.

    Failure modes:
        - :class:`UnknownSourceError` if ``token`` is not a source value.
    """
    try:
        return SourceKind(token)
    except ValueError as exc:
        msg = f"unknown --source token: {token!r}."
        raise UnknownSourceError(msg) from exc


@dataclass(frozen=True)
class RemoteTile:
    """The download address of one distributable source sheet.

    Contract:
        - ``tile_id`` is the portal's sheet identifier (e.g. ``"C_37EN1"``); it
          must be non-blank.
        - ``bbox`` is the sheet's extent as :data:`~ahn_cli.domain.BBox` in
          EPSG:28992 (the Dutch grid every downstream stage expects), even when
          the portal published it in WGS84.
        - ``download_url`` is the absolute URL the bytes are fetched from; it
          must be non-blank.

    Invariants:
        - Immutable and hashable; two tiles are equal iff every field is equal.

    Failure modes:
        - ``ValueError`` if ``tile_id`` or ``download_url`` is blank.
        - ``ValueError`` if ``bbox`` is degenerate (see
          :func:`~ahn_cli.domain.ensure_valid_bbox`).
    """

    tile_id: str
    bbox: BBox
    download_url: str

    def __post_init__(self) -> None:
        """Validate the identifier, extent, and download URL."""
        if not self.tile_id.strip():
            msg = "tile_id must be a non-blank identifier."
            raise ValueError(msg)
        if not self.download_url.strip():
            msg = "download_url must be a non-blank URL."
            raise ValueError(msg)
        ensure_valid_bbox(self.bbox)


@dataclass(frozen=True)
class ResolvedFeed:
    """The tiles covering an AOI plus the licence terms to record for them.

    Contract:
        - ``licence`` and ``attribution`` are the distribution terms captured
          from the source (the ATOM ``rights``/``author`` for PDOK), carried so
          the acquisition records real, per-source provenance.
        - ``tiles`` are the covering :class:`RemoteTile` sheets in a
          deterministic order (ascending ``tile_id``).

    Invariants:
        - Immutable and hashable; equal by field value.
    """

    licence: str
    attribution: str
    tiles: tuple[RemoteTile, ...]


class FetchSource(Protocol):
    """The behaviour every distribution source implements.

    A source knows how to advertise its generations (for ``--ahn`` selection)
    and how to resolve the covering tiles for one chosen generation. Both
    methods receive the injected :data:`HttpGet` so no source reaches the
    network on its own.
    """

    def generation_registry(self, http_get: HttpGet) -> GenerationRegistry:
        """Return this source's generation registry with wired probes.

        Contract:
            - Registers one :class:`~ahn_cli.fetch.generation.GenerationSource`
              per generation the source serves, each carrying a *real* coverage
              probe (backed by ``http_get`` where the probe needs the network),
              so :func:`~ahn_cli.fetch.generation.select_source` can pick the
              newest generation actually covering an AOI.
        """
        ...

    def resolve(
        self,
        generation_source: GenerationSource,
        aoi: BBox,
        http_get: HttpGet,
    ) -> ResolvedFeed:
        """Resolve the sheets of ``generation_source`` covering ``aoi``.

        Contract:
            - Returns a :class:`ResolvedFeed` whose tiles' bounding boxes
              intersect ``aoi`` (EPSG:28992) -- an axis-aligned test, so the
              result is a safe *superset* of the true-geometry cover; any extra
              edge sheet is clipped away in the prep stage. Ordered
              deterministically by ``tile_id``.
        """
        ...


def to_wgs84(bbox_rd: BBox) -> BBox:
    """Convert an EPSG:28992 bbox to an EPSG:4326 ``(minlon, minlat, ...)`` box.

    Contract:
        - ``bbox_rd`` is a valid Dutch-grid box; the result is its extent in
          WGS84 longitude/latitude, deterministic for a given input.
    """
    ensure_valid_bbox(bbox_rd)
    minlon, minlat = _TO_WGS84.transform(bbox_rd[0], bbox_rd[1])
    maxlon, maxlat = _TO_WGS84.transform(bbox_rd[2], bbox_rd[3])
    return (minlon, minlat, maxlon, maxlat)


def to_rd(bbox_wgs84: BBox) -> BBox:
    """Convert an EPSG:4326 ``(minlon, minlat, ...)`` box to EPSG:28992.

    Contract:
        - ``bbox_wgs84`` is a valid longitude/latitude box; the result is its
          extent on the Dutch national grid, deterministic for a given input.
    """
    ensure_valid_bbox(bbox_wgs84)
    minx, miny = _TO_RD.transform(bbox_wgs84[0], bbox_wgs84[1])
    maxx, maxy = _TO_RD.transform(bbox_wgs84[2], bbox_wgs84[3])
    return (minx, miny, maxx, maxy)


def boxes_intersect(a: BBox, b: BBox) -> bool:
    """Report whether two axis-aligned boxes share any area (edges included).

    Contract:
        - Both boxes are ``(minx, miny, maxx, maxy)`` in the same CRS.
        - Returns ``True`` when they overlap or merely touch; the touch-
          inclusive rule keeps a sheet on an AOI's exact edge selected.
    """
    return a[0] <= b[2] and a[2] >= b[0] and a[1] <= b[3] and a[3] >= b[1]
