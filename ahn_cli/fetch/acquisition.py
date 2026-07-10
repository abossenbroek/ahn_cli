"""Fetch-context acquisition: resolve, download-through-cache, record.

The ``fetch`` bounded context turns a validated area of interest into raw,
cached source tiles on disk plus a provenance sidecar per tile. WP6 wires the
real actuation the seam previously deferred:

1. Derive the EPSG:28992 AOI (the ``--bbox`` selector is wired; ``--city`` /
   ``--geojson`` remain a typed deferral).
2. Pick the distribution source (``--source`` -> registry, no stringly switch)
   and, through the source's generation registry, the AHN generation covering
   the AOI (``--ahn``; ``auto`` probes newest-first).
3. Resolve the covering sheets, download each *through the WP4 content cache*
   (a cached sheet costs zero network and zero new bytes), and write it to
   ``data/<site>/ahn/<tile_id>.LAZ``.
4. Record real provenance next to each sheet: source portal, generation,
   licence, checksums, and the request keys that address the fetch.

Everything that reaches the network is injected (``http_get``, ``now``), so the
fast test suite is deterministic and offline.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from importlib.metadata import version
from typing import TYPE_CHECKING

import requests

from ahn_cli.cache import CacheKey, ContentAddressedCache
from ahn_cli.domain import (
    BBox,
    Generation,
    Product,
    Provenance,
    ensure_valid_bbox,
)
from ahn_cli.fetch.generation import (
    GenerationUnavailableError,
    UnknownGenerationError,
    select_source,
)
from ahn_cli.fetch.geotiles_source import GeotilesCatalogError, GeotilesSource
from ahn_cli.fetch.pdok import PdokFeedError, PdokSource
from ahn_cli.fetch.source import FetchSource, HttpGet, SourceKind
from ahn_cli.provenance import write_provenance

if TYPE_CHECKING:
    from pathlib import Path

SITE_SUBDIRS: tuple[str, ...] = ("ahn", "ortho", "viirs")
"""Per-product subdirectories created under every site directory, in order."""

_AHN_SUBDIR = "ahn"
_CACHE_DIRNAME = ".cache"
_BBOX_COORD_COUNT = 4
_HTTP_TIMEOUT_SECONDS = 300

# Expected failures a portal-fetching run surfaces cleanly: a changed/invalid
# distribution feed or catalogue, and a failed HTTP download. They are funnelled
# to AcquisitionError so the CLI reports a tidy message, never a traceback.
# (They are not made subclasses of AcquisitionError because that would require
# pdok/geotiles_source -- imported here -- to import back from this module,
# forming a cycle.)
_FEED_ERRORS: tuple[type[Exception], ...] = (
    PdokFeedError,
    GeotilesCatalogError,
    requests.RequestException,
)
_SELECTION_ERRORS: tuple[type[Exception], ...] = (
    GenerationUnavailableError,
    UnknownGenerationError,
    *_FEED_ERRORS,
)

Clock = Callable[[], datetime]
"""An injected UTC clock, so download timestamps are deterministic in tests."""


class AreaSelectorKind(Enum):
    """Which area-of-interest selector an acquisition request was built from.

    Modelled as an enum so the selected area is a closed, immutable value the
    fetch context records and branches on without a stringly-typed switch.
    """

    CITY = "city"
    BBOX = "bbox"
    GEOJSON = "geojson"


class AcquisitionError(RuntimeError):
    """Base for user-facing acquisition failures the CLI reports cleanly.

    Subclasses distinguish the cause; the CLI catches this base so every
    expected failure becomes a tidy Click error rather than a traceback.
    """


class MalformedBboxError(AcquisitionError):
    """Raised when a ``--bbox`` value is not a valid EPSG:28992 box.

    Signals a wrong coordinate count, a non-numeric coordinate, or a degenerate
    extent (see :func:`~ahn_cli.domain.ensure_valid_bbox`).
    """


class SelectorNotWiredError(AcquisitionError):
    """Raised when AOI derivation for the chosen selector is not wired yet.

    WP6 wires the ``--bbox`` selector; ``--city`` and ``--geojson`` AOI
    derivation (polygon-to-bbox over the bundled boundary data) is a typed
    deferral rather than a silent failure.
    """


@dataclass(frozen=True)
class AcquisitionRequest:
    """A validated intent to acquire source data for one site.

    Contract:
        - ``site_dir`` is the site root beneath which :func:`create_site_layout`
          materialises the product subdirectories and the downloads land.
        - ``selector`` records which area-of-interest kind was chosen; the
          caller guarantees exactly one was given.
        - ``area`` is that selector's raw value (a city name, a bbox string, or
          a GeoJSON path), carried verbatim.
        - ``source`` is the distribution source to fetch from; it defaults to
          :attr:`SourceKind.PDOK`, the primary INSPIRE ATOM distribution.
        - ``generation`` is the requested AHN generation, or ``None`` (the
          default) to request automatic newest-available selection.

    Invariants:
        - Frozen: an immutable, hashable value object, equal by field value.
    """

    site_dir: Path
    selector: AreaSelectorKind
    area: str
    source: SourceKind = SourceKind.PDOK
    generation: Generation | None = None


def create_site_layout(site_dir: Path) -> tuple[Path, ...]:
    """Create ``data/<site>/{ahn,ortho,viirs}/`` and return the subdir paths.

    Contract:
        - Creates ``site_dir`` and each :data:`SITE_SUBDIRS` entry beneath it,
          including any missing parents.
        - Idempotent: pre-existing directories are left intact and no error is
          raised when the layout already exists.
        - Returns the subdirectory paths in :data:`SITE_SUBDIRS` order.
    """
    created: list[Path] = []
    for name in SITE_SUBDIRS:
        subdir = site_dir / name
        subdir.mkdir(parents=True, exist_ok=True)
        created.append(subdir)
    return tuple(created)


def source_for(kind: SourceKind) -> FetchSource:
    """Return the distribution source registered for ``kind``.

    Contract:
        - Registry-driven: the closed :class:`SourceKind` set maps to a source
          instance, so selection never branches on a raw string.
    """
    return _SOURCE_REGISTRY[kind]


_SOURCE_REGISTRY: dict[SourceKind, FetchSource] = {
    SourceKind.PDOK: PdokSource(),
    SourceKind.GEOTILES: GeotilesSource(),
}


def default_http_get(url: str) -> bytes:
    """Fetch ``url`` over HTTPS and return the response body bytes.

    Contract:
        - The production :data:`HttpGet`: raises for a non-2xx status and
          returns the full body. Injected fakes replace it in tests.
    """
    response = requests.get(url, timeout=_HTTP_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.content


def _utcnow() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def acquire(
    request: AcquisitionRequest,
    *,
    http_get: HttpGet = default_http_get,
    now: Clock = _utcnow,
    cache_root: Path | None = None,
    tool_version: str | None = None,
) -> tuple[Path, ...]:
    """Acquire the covering sheets for ``request`` into its site layout.

    Contract:
        - Creates the site layout, derives the AOI, selects the source and
          generation, and downloads each covering sheet *through the content
          cache* into ``<site>/ahn/<tile_id>.LAZ``, writing a
          ``<tile_id>.provenance.json`` sidecar beside each.
        - Returns the written ``.LAZ`` paths, ordered by sheet id.
        - Deterministic: sheet order, checksums, and (with injected ``now`` /
          ``tool_version``) provenance are stable for the same feed and tiles.
        - Idempotent downloads: a sheet already in the cache is not re-fetched
          (``http_get`` is not called for it).

    Failure modes:
        - :class:`AcquisitionError` if the bbox is malformed, the selector is
          not wired, no registered generation covers the AOI, the distribution
          feed/catalogue is invalid, or a tile download fails. Every expected
          failure is funnelled here so the CLI reports it cleanly.
    """
    create_site_layout(request.site_dir)
    aoi = _aoi_bbox(request)
    source = source_for(request.source)
    try:
        registry = source.generation_registry(http_get)
        generation_source = select_source(request.generation, aoi, registry)
        resolved = source.resolve(generation_source, aoi, http_get)
    except _SELECTION_ERRORS as exc:
        raise AcquisitionError(str(exc)) from exc
    generation = generation_source.generation
    version_label = (
        tool_version if tool_version is not None else version("ahn_cli")
    )
    ahn_dir = request.site_dir / _AHN_SUBDIR
    cache = ContentAddressedCache(
        root=cache_root
        if cache_root is not None
        else request.site_dir / _CACHE_DIRNAME
    )
    written: list[Path] = []
    for tile in resolved.tiles:
        key = CacheKey(
            product=Product.AHN_POINT_CLOUD,
            tile_id=tile.tile_id,
            generation=generation,
        )
        started = now()
        try:
            content = cache.get_or_fetch(
                key, _downloader(http_get, tile.download_url)
            )
        except requests.RequestException as exc:
            msg = f"download failed for {tile.download_url}: {exc}"
            raise AcquisitionError(msg) from exc
        finished = now()
        laz_path = ahn_dir / f"{tile.tile_id}.LAZ"
        laz_path.write_bytes(content)
        checksum = hashlib.sha256(content).hexdigest()
        provenance = Provenance(
            source_portal=request.source.value,
            product=Product.AHN_POINT_CLOUD,
            licence=resolved.licence,
            attribution=resolved.attribution,
            bbox=tile.bbox,
            download_started_at=started,
            download_finished_at=finished,
            input_checksum=checksum,
            output_checksum=checksum,
            tool_version=version_label,
            generation=generation,
            request_keys=(
                ("source", request.source.value),
                ("tile_id", tile.tile_id),
                ("url", tile.download_url),
            ),
        )
        write_provenance(
            provenance, ahn_dir / f"{tile.tile_id}.provenance.json"
        )
        written.append(laz_path)
    return tuple(written)


def _downloader(http_get: HttpGet, url: str) -> Callable[[], bytes]:
    """Return a zero-arg fetcher the cache calls only on a miss."""

    def fetch() -> bytes:
        return http_get(url)

    return fetch


def _aoi_bbox(request: AcquisitionRequest) -> BBox:
    """Derive the EPSG:28992 AOI bbox from the request's selector.

    Failure modes:
        - :class:`MalformedBboxError` for a bad ``--bbox`` value.
        - :class:`SelectorNotWiredError` for ``--city`` / ``--geojson`` (their
          AOI derivation is deferred in WP6).
    """
    if request.selector is AreaSelectorKind.BBOX:
        return _parse_bbox(request.area)
    msg = (
        f"AOI derivation for --{request.selector.value} is not wired in WP6; "
        "use --bbox, or wait for the city/geojson AOI derivation."
    )
    raise SelectorNotWiredError(msg)


def _parse_bbox(area: str) -> BBox:
    """Parse a ``minx,miny,maxx,maxy`` EPSG:28992 string into a bbox tuple."""
    parts = area.split(",")
    if len(parts) != _BBOX_COORD_COUNT:
        msg = f"--bbox must have four comma-separated numbers; got {area!r}."
        raise MalformedBboxError(msg)
    try:
        minx, miny, maxx, maxy = (float(part) for part in parts)
    except ValueError as exc:
        msg = f"--bbox has a non-numeric coordinate: {area!r}."
        raise MalformedBboxError(msg) from exc
    bbox: BBox = (minx, miny, maxx, maxy)
    try:
        ensure_valid_bbox(bbox)
    except ValueError as exc:
        raise MalformedBboxError(str(exc)) from exc
    return bbox
