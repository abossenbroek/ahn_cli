"""Fetch-context acquisition: resolve, download-through-cache, record.

The ``fetch`` bounded context turns a validated area of interest into raw,
cached source tiles on disk plus a provenance sidecar per tile. WP6 wires the
real actuation the seam previously deferred:

1. Derive the EPSG:28992 AOI (the ``--bbox`` and ``--geojson`` selectors are
   wired; ``--city`` remains a typed deferral).
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

import concurrent.futures
import hashlib
import io
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from importlib.metadata import version
from pathlib import Path
from typing import cast

import laspy
import requests
from shapely.errors import ShapelyError
from shapely.geometry import shape
from shapely.ops import unary_union

from ahn_cli.cache import CacheKey, ContentAddressedCache
from ahn_cli.domain import (
    BBox,
    Generation,
    Product,
    ProgressCallback,
    Provenance,
    ensure_valid_bbox,
)
from ahn_cli.domain.authenticity import degenerate_cloud
from ahn_cli.fetch.generation import (
    GenerationUnavailableError,
    UnknownGenerationError,
    select_source,
)
from ahn_cli.fetch.geotiles_source import GeotilesCatalogError, GeotilesSource
from ahn_cli.fetch.pdok import PdokFeedError, PdokSource
from ahn_cli.fetch.source import (
    FetchSource,
    HttpGet,
    RemoteTile,
    SourceKind,
    to_rd,
)
from ahn_cli.provenance import write_provenance

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

# Expected failures building a shapely geometry from an untrusted GeoJSON
# geometry dict: a missing/wrongly-shaped ``coordinates`` array (KeyError /
# TypeError / ValueError) or a ring GEOS itself refuses to build, e.g. from a
# non-finite vertex (shapely.errors.ShapelyError). Funnelled to
# MalformedGeojsonError so a malformed file never surfaces as a raw traceback.
_GEOMETRY_ERRORS: tuple[type[Exception], ...] = (
    KeyError,
    TypeError,
    ValueError,
    ShapelyError,
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


class MalformedGeojsonError(AcquisitionError):
    """Raised when a ``--geojson`` file yields no valid EPSG:28992 box.

    Signals an unreadable/missing file, invalid JSON, a document carrying no
    Polygon/MultiPolygon geometry, or a degenerate extent (a zero-area or
    single-point geometry that fails :func:`~ahn_cli.domain.ensure_valid_bbox`).
    """


class SelectorNotWiredError(AcquisitionError):
    """Raised when AOI derivation for the chosen selector is not wired yet.

    WP6 wires the ``--bbox`` and ``--geojson`` selectors; ``--city`` AOI
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


def _no_op_progress(_done: int, _total: int) -> None:
    """Report nothing; the default when the caller supplies no callback."""


def acquire(
    request: AcquisitionRequest,
    *,
    http_get: HttpGet = default_http_get,
    now: Clock = _utcnow,
    cache_root: Path | None = None,
    tool_version: str | None = None,
    progress: ProgressCallback | None = None,
    download_jobs: int = 1,
) -> tuple[Path, ...]:
    """Acquire the covering sheets for ``request`` into its site layout.

    Contract:
        - Creates the site layout, derives the AOI, selects the source and
          generation, and downloads each covering sheet *through the content
          cache* into ``<site>/ahn/<tile_id>.LAZ``, writing a
          ``<tile_id>.provenance.json`` sidecar beside each.
        - Returns the written ``.LAZ`` paths, ordered by sheet id.
        - Deterministic: sheet order, checksums, and (with injected ``now`` /
          ``tool_version``) provenance are stable for the same feed and tiles,
          regardless of ``download_jobs``.
        - Idempotent downloads: a sheet already in the cache is not re-fetched
          (``http_get`` is not called for it).
        - Calls ``progress(tiles_done, total_tiles)`` once per tile (after it
          is written); defaults to a no-op so callers that don't care about
          progress are unaffected.
        - ``download_jobs`` (default ``1``, serial) is the number of tiles
          downloaded concurrently through a ``ThreadPoolExecutor`` when
          greater than 1. Downloads may complete in any order, but every
          other effect -- writes, provenance, and ``progress`` calls -- is
          always emitted in ascending ``tile_id`` order, matching the
          ``download_jobs=1`` sequence exactly.

    Failure modes:
        - :class:`AcquisitionError` if the bbox is malformed, the selector is
          not wired, no registered generation covers the AOI, the distribution
          feed/catalogue is invalid, a tile download fails, or a downloaded
          tile is not a genuine AHN point cloud (unparsable LAZ, no points,
          or every point at one identical position). Every expected failure
          is funnelled here so the CLI reports it cleanly. A tile the
          authenticity gate rejects is evicted from the cache before the
          error is raised, so a retry re-downloads it instead of replaying
          the poisoned bytes. With ``download_jobs=1`` a rejected tile stops
          the run before any later tile is downloaded; with
          ``download_jobs>1`` every tile's download is already in flight, so
          later tiles may already have been fetched (and cached) by the time
          the rejection is raised.
    """
    report = progress if progress is not None else _no_op_progress
    create_site_layout(request.site_dir)
    aoi = aoi_bbox(request)
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

    def emit(
        tile: RemoteTile,
        key: CacheKey,
        content: bytes,
        started: datetime,
        finished: datetime,
    ) -> Path:
        """Verify, write, and record provenance for one downloaded tile."""
        try:
            _verify_ahn_tile(content, tile.tile_id, tile.download_url)
        except AcquisitionError:
            # A rejected download must never stay cached, or every retry
            # would replay the same poisoned bytes; evict the entry so the
            # next run re-downloads the tile.
            cache.discard(key)
            raise
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
        return laz_path

    written: list[Path] = []
    total_tiles = len(resolved.tiles)
    if download_jobs <= 1:
        for tile in resolved.tiles:
            _, key, content, started, finished = _fetch_ahn_tile(
                tile, generation, cache, http_get, now
            )
            written.append(emit(tile, key, content, started, finished))
            report(len(written), total_tiles)
    else:
        for tile, key, content, started, finished in _fetch_ahn_tiles(
            resolved.tiles, generation, cache, http_get, now, download_jobs
        ):
            written.append(emit(tile, key, content, started, finished))
            report(len(written), total_tiles)
    return tuple(written)


def _fetch_ahn_tile(
    tile: RemoteTile,
    generation: Generation,
    cache: ContentAddressedCache,
    http_get: HttpGet,
    now: Clock,
) -> tuple[RemoteTile, CacheKey, bytes, datetime, datetime]:
    """Download (or reuse the cache for) one AHN tile, timestamped.

    Failure modes:
        - :class:`AcquisitionError` if ``http_get`` raises for this tile.
    """
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
    return tile, key, content, started, finished


def _fetch_ahn_tiles(
    tiles: tuple[RemoteTile, ...],
    generation: Generation,
    cache: ContentAddressedCache,
    http_get: HttpGet,
    now: Clock,
    download_jobs: int,
) -> list[tuple[RemoteTile, CacheKey, bytes, datetime, datetime]]:
    """Download every tile through ``download_jobs`` concurrent workers.

    Contract:
        - Returns one result per tile, ordered by ``tile_id`` ascending --
          regardless of the order in which the pool completes them, so
          callers must never rely on as-completed order for the emitted
          sequence.
    """
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=download_jobs
    ) as executor:
        futures = [
            executor.submit(
                _fetch_ahn_tile, tile, generation, cache, http_get, now
            )
            for tile in tiles
        ]
        results = [future.result() for future in futures]
    return sorted(results, key=lambda result: result[0].tile_id)


def _verify_ahn_tile(content: bytes, tile_id: str, url: str) -> None:
    """Hard-verify a downloaded tile is a genuine AHN point cloud.

    Runs before the tile lands in the site layout, so a placeholder or
    corrupted download never becomes an output: the bytes must parse as a
    LAS/LAZ header, and the header must not describe an empty cloud or a
    stack of points all at one identical position.

    Failure modes:
        - :class:`AcquisitionError` if the tile is unparsable or degenerate.
    """
    try:
        with laspy.open(io.BytesIO(content)) as reader:
            header = reader.header
            count = int(header.point_count)
            mins = header.mins
            maxs = header.maxs
    except (OSError, ValueError, laspy.LaspyException) as exc:
        msg = (
            f"tile {tile_id} from {url} is not a readable LAZ point cloud "
            f"({exc}); refusing to store it as AHN data."
        )
        raise AcquisitionError(msg) from exc
    if degenerate_cloud(count, mins, maxs):
        detail = (
            "contains no points"
            if count == 0
            else f"stacks all {count} points at one identical position"
        )
        msg = (
            f"tile {tile_id} from {url} {detail} — that is not genuine "
            "AHN data; refusing to store it."
        )
        raise AcquisitionError(msg)


def _downloader(http_get: HttpGet, url: str) -> Callable[[], bytes]:
    """Return a zero-arg fetcher the cache calls only on a miss."""

    def fetch() -> bytes:
        return http_get(url)

    return fetch


def aoi_bbox(request: AcquisitionRequest) -> BBox:
    """Derive the EPSG:28992 AOI bbox from the request's selector.

    Shared by the AHN point-cloud fetch and the DSM raster fetch so both derive
    the area of interest identically from one request.

    Failure modes:
        - :class:`MalformedBboxError` for a bad ``--bbox`` value.
        - :class:`MalformedGeojsonError` for a bad ``--geojson`` file.
        - :class:`SelectorNotWiredError` for ``--city`` (its AOI derivation is
          deferred in WP6).
    """
    if request.selector is AreaSelectorKind.BBOX:
        return _parse_bbox(request.area)
    if request.selector is AreaSelectorKind.GEOJSON:
        return _geojson_bbox(request.area)
    msg = (
        f"AOI derivation for --{request.selector.value} is not wired in WP6; "
        "use --bbox, or wait for the city AOI derivation."
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


_POLYGON_TYPES = frozenset({"Polygon", "MultiPolygon"})
"""GeoJSON geometry types whose extent defines an area of interest."""


def _geojson_bbox(path: str) -> BBox:
    """Derive an EPSG:28992 bbox from the polygon(s) in a GeoJSON file.

    The GeoJSON is read as WGS84 (EPSG:4326) per RFC 7946: every Polygon and
    MultiPolygon geometry is unioned, and the union's WGS84 bounds are projected
    to the Dutch national grid. Non-polygonal and null members are skipped so a
    mixed document still yields the polygonal AOI; a member typed as a Polygon
    or MultiPolygon but with malformed coordinates is a failure, not a skip.

    Failure modes:
        - :class:`MalformedGeojsonError` if the file is missing/unreadable, is
          not valid JSON, carries no Polygon/MultiPolygon geometry, carries one
          whose coordinates shapely cannot build a geometry from, or has a
          degenerate (zero-area) extent.
    """
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        msg = f"--geojson file could not be read: {path!r} ({exc})."
        raise MalformedGeojsonError(msg) from exc
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as exc:
        msg = f"--geojson is not valid JSON: {path!r} ({exc})."
        raise MalformedGeojsonError(msg) from exc
    geometries = [
        polygon
        for geom in _geometry_objects(doc)
        if (polygon := _polygon_geometry(geom)) is not None
    ]
    if not geometries:
        msg = f"--geojson has no polygon/multipolygon geometry: {path!r}."
        raise MalformedGeojsonError(msg)
    try:
        bounds = unary_union([shape(geom) for geom in geometries]).bounds
    except _GEOMETRY_ERRORS as exc:
        msg = f"--geojson has a malformed geometry: {path!r} ({exc})."
        raise MalformedGeojsonError(msg) from exc
    wgs84: BBox = (bounds[0], bounds[1], bounds[2], bounds[3])
    try:
        return to_rd(wgs84)
    except ValueError as exc:
        raise MalformedGeojsonError(str(exc)) from exc


def _geometry_objects(doc: object) -> list[object]:
    """Return each geometry object embedded in a parsed GeoJSON document.

    Handles the three top-level shapes -- a bare geometry, a ``Feature``, and a
    ``FeatureCollection`` -- returning the raw (possibly null or non-polygonal)
    geometry members for the caller to filter.
    """
    if isinstance(doc, dict):
        document = cast("dict[str, object]", doc)
        kind = document.get("type")
        if kind == "FeatureCollection":
            features = document.get("features")
            if isinstance(features, list):
                feature_list = cast("list[object]", features)
                return [
                    cast("dict[str, object]", feature).get("geometry")
                    for feature in feature_list
                    if isinstance(feature, dict)
                ]
            return []
        if kind == "Feature":
            return [document.get("geometry")]
    return [doc]


def _polygon_geometry(geom: object) -> dict[str, object] | None:
    """Return ``geom`` as a mapping if it is a (Multi)Polygon, else ``None``.

    Skips null and non-polygonal members so a mixed document still yields only
    its polygonal geometries.
    """
    if not isinstance(geom, dict):
        return None
    geometry = cast("dict[str, object]", geom)
    if geometry.get("type") not in _POLYGON_TYPES:
        return None
    return geometry
