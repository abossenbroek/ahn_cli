"""Beeldmateriaal orthophoto fetch: select a vintage/zone, mosaic, clip, record.

WP8 adds the Beeldmateriaal RGB orthophoto product to the ``fetch`` context. It
is a *raster* product keyed by :class:`~ahn_cli.domain.Vintage` (not an AHN
:class:`~ahn_cli.domain.Generation`), so it does not fit the AHN point-cloud
:class:`~ahn_cli.fetch.source.FetchSource` protocol (whose ``resolve`` returns
LAZ sheets). It therefore plugs in through its *own* registry --
:class:`OrthoDatasetRegistry` -- but honours the same DDD rule the AHN
generation registry does: selection consults a preference-ordered registry of
pinned datasets and probes AOI coverage, never a stringly-typed switch on a
resolution string.

Flow:

1. Select the dataset: probe the pinned zones in preference order (5cm before
   8cm) and take the first whose ATOM feed has a sheet covering the AOI.
2. Resolve the covering GeoTIFF sheets from that dataset's feed (reusing the
   shared INSPIRE-ATOM reader) and download each *through the WP4 content
   cache*, so a second fetch costs zero network and zero new bytes.
3. Mosaic the sheets and clip to the AOI with :func:`rasterio.merge.merge`
   (``bounds`` + ``res``) -- never a shelled-out ``gdalbuildvrt`` -- writing
   ``<site>/ortho/ortho.tif``.
4. Record real CC-BY provenance: portal, licence + attribution (from the feed),
   vintage, zone, resolution tier, extent, per-sheet checksums, and the
   mosaic's pixel checksum as the output checksum.

Everything reaching the network is injected (``http_get``, ``now``), so the fast
suite is deterministic and offline. Dataset versions are pinned (never floated
to the newest "Actueel" layer); the pinned feed URLs are verified by the nightly
portal-contract test (WP14), not by this offline gate.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib.metadata import version
from typing import TYPE_CHECKING

import rasterio
import requests
from rasterio.merge import merge

from ahn_cli.cache import CacheKey, ContentAddressedCache
from ahn_cli.domain import (
    BBox,
    Product,
    Provenance,
    Vintage,
    ensure_valid_bbox,
)
from ahn_cli.fetch.acquisition import (
    AcquisitionError,
    AcquisitionRequest,
    Clock,
    aoi_bbox,
    default_http_get,
)
from ahn_cli.fetch.pdok import AtomFeed, PdokFeedError, parse_atom_feed
from ahn_cli.fetch.source import (
    HttpGet,
    RemoteTile,
    boxes_intersect,
    to_rd,
    to_wgs84,
)
from ahn_cli.provenance import write_provenance

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator
    from pathlib import Path

    import numpy as np
    import numpy.typing as npt

_PORTAL = "beeldmateriaal"
_ORTHO_SUBDIR = "ortho"
_TILES_SUBDIR = "tiles"
_MOSAIC_NAME = "ortho.tif"
_CACHE_DIRNAME = ".cache"

# The pinned Beeldmateriaal RGB open-data vintage. Pinned, never floated to the
# newest "Actueel" layer (spec requirement). 2023 is the last confirmed 8cm
# open release; the 5cm zone covers only specific parcels, which is exactly why
# selection probes coverage and falls back to 8cm. Cited from the PDOK/
# Beeldmateriaal open-data announcements (pdok.nl "hoge resolutie luchtfoto
# 2023"; data.overheid.nl Luchtfoto Beeldmateriaal, CC-BY 4.0).
_ORTHO_VINTAGE = Vintage(2023)

# Per-zone ATOM feed URLs for the pinned vintage, preference order first (5cm),
# then the 8cm fallback. Downloads are the source GeoTIFF sheets from the
# Beeldmateriaal open-data portal (never the forbidden PDOK WMS/WMTS GetMap
# crop). The exact live feed filenames are pinned here and verified by the
# nightly portal-contract test (WP14); the offline gate injects a feed fetcher,
# so these URLs are not exercised by it.
_BEELDMATERIAAL_ATOM_BASE = (
    "https://opendata.beeldmateriaal.nl/downloads/rgb/"
)
_ORTHO_ZONES: tuple[tuple[str, float, str, str], ...] = (
    (
        "5cm",
        0.05,
        f"{_BEELDMATERIAAL_ATOM_BASE}2023/5cm/atom.xml",
        "Beeldmateriaal RGB 5cm zone (selected parcels), vintage 2023, CC-BY.",
    ),
    (
        "8cm",
        0.08,
        f"{_BEELDMATERIAAL_ATOM_BASE}2023/8cm/atom.xml",
        "Beeldmateriaal RGB 8cm zone (national), vintage 2023, CC-BY.",
    ),
)

# Recorded as survey facts in provenance, never treated as defects to repair
# (spec rule): the mosaic joins sheets along seamlines, and tall buildings show
# lean ("omvalling") away from nadir. Both are recorded, not corrected.
_SEAMLINE_NOTE = (
    "mosaic seamlines and building lean (omvalling) are recorded survey facts, "
    "not defects corrected here"
)
_MOSAIC_METHOD = "rasterio.merge (method=first) over id-ordered sheets"


class OrthoFeedError(ValueError):
    """Raised when a Beeldmateriaal ATOM feed cannot be parsed.

    Wraps the shared ATOM reader's :class:`~ahn_cli.fetch.pdok.PdokFeedError`
    under an ortho-specific name so the acquisition funnel and callers reason
    about ortho failures without naming the PDOK module. Subclasses
    :class:`ValueError`, so callers may catch either.
    """


class OrthoUnavailableError(RuntimeError):
    """Raised when no pinned ortho zone has a sheet covering the AOI.

    Signals that every registered zone's coverage probe reported the AOI
    uncovered, so no Beeldmateriaal orthophoto can serve the request.
    """


class DuplicateOrthoTierError(ValueError):
    """Raised when registering two datasets at the same resolution tier.

    Signals a registry-assembly programming error: a tier identifies a zone at
    most once, so a duplicate would make preference-ordered selection ambiguous.
    """


@dataclass(frozen=True)
class OrthoDataset:
    """A pinned Beeldmateriaal ortho zone: vintage, tier, and its ATOM feed.

    Contract (fields):
        vintage: The pinned acquisition vintage (never the newest layer).
        zone: The acquisition-zone identifier recorded in provenance.
        resolution_tier: The human tier label (e.g. ``"5cm"``), recorded in
            provenance; never branched on as control flow.
        resolution_m: The ground pixel size in metres, driving the mosaic's
            output resolution; must be positive.
        feed_url: The zone's ATOM dataset feed URL; must be non-blank.
        semantics: A human-readable note on the zone, carried for operator
            context.

    Invariants:
        - Frozen: an immutable value object, equal by field value.

    Failure modes:
        - ``ValueError`` if ``feed_url`` is blank or ``resolution_m`` <= 0.
    """

    vintage: Vintage
    zone: str
    resolution_tier: str
    resolution_m: float
    feed_url: str
    semantics: str

    def __post_init__(self) -> None:
        """Reject a blank feed URL or a non-positive pixel size."""
        if not self.feed_url.strip():
            msg = "feed_url must be a non-blank ATOM feed URL."
            raise ValueError(msg)
        if self.resolution_m <= 0:
            msg = f"resolution_m must be positive; got {self.resolution_m}."
            raise ValueError(msg)


def _empty_dataset_list() -> list[OrthoDataset]:
    """Return an empty ortho-dataset list (a registry's initial state)."""
    return []


@dataclass
class OrthoDatasetRegistry:
    """A preference-ordered registry of pinned ortho zones.

    Contract:
        - Starts empty; :meth:`register` appends a zone. Registration order is
          preference order (5cm before 8cm), which selection honours directly.
        - :meth:`datasets` returns the registered zones in that order, and the
          registry iterates in the same order.

    Invariants:
        - A resolution tier is registered at most once.
    """

    _datasets: list[OrthoDataset] = field(default_factory=_empty_dataset_list)

    def register(self, dataset: OrthoDataset) -> None:
        """Append ``dataset`` to the registry, preserving preference order.

        Failure modes:
            - :class:`DuplicateOrthoTierError` if a dataset with the same
              ``resolution_tier`` is already registered.
        """
        if any(
            existing.resolution_tier == dataset.resolution_tier
            for existing in self._datasets
        ):
            msg = f"resolution tier {dataset.resolution_tier} is already registered."
            raise DuplicateOrthoTierError(msg)
        self._datasets.append(dataset)

    def datasets(self) -> tuple[OrthoDataset, ...]:
        """Return the registered zones in preference order."""
        return tuple(self._datasets)

    def __iter__(self) -> Iterator[OrthoDataset]:
        """Iterate the registered zones in preference order."""
        return iter(self._datasets)


def default_ortho_registry() -> OrthoDatasetRegistry:
    """Return the pinned Beeldmateriaal registry (5cm preferred, 8cm fallback).

    Contract:
        - Registers the pinned :data:`_ORTHO_VINTAGE` zones in preference order:
          5cm first, then 8cm. Deterministic: the same zones every call.
    """
    registry = OrthoDatasetRegistry()
    for tier, resolution_m, feed_url, semantics in _ORTHO_ZONES:
        registry.register(
            OrthoDataset(
                vintage=_ORTHO_VINTAGE,
                zone=f"{_PORTAL}-{_ORTHO_VINTAGE.year}-{tier}",
                resolution_tier=tier,
                resolution_m=resolution_m,
                feed_url=feed_url,
                semantics=semantics,
            )
        )
    return registry


@dataclass(frozen=True)
class ResolvedOrthoFeed:
    """The sheets covering an AOI plus the CC-BY terms to record for them.

    Contract:
        - ``licence`` / ``attribution`` are the feed's ``rights`` / ``author``
          terms (CC-BY for Beeldmateriaal), carried for real provenance.
        - ``tiles`` are the covering sheets in ascending ``tile_id`` order.

    Invariants:
        - Immutable and hashable; equal by field value.
    """

    licence: str
    attribution: str
    tiles: tuple[RemoteTile, ...]


@dataclass(frozen=True)
class OrthoMosaic:
    """The result of mosaicking and clipping the ortho sheets to the AOI.

    Contract (fields):
        bbox: The clipped extent ``(minx, miny, maxx, maxy)`` in EPSG:28992.
        crs: The mosaic CRS rendered as a string (e.g. ``"EPSG:28992"``).
        width / height: The mosaic raster dimensions in pixels.
        resolution_m: The ground pixel size in metres.
        pixel_checksum: A SHA-256 over the mosaic *pixel array* (plus its
            crs/dtype/shape/res header), not the encoded GeoTIFF container, so
            it is immune to GDAL container non-determinism.

    Invariants:
        - Immutable and hashable; equal iff every field is equal.
    """

    bbox: BBox
    crs: str
    width: int
    height: int
    resolution_m: float
    pixel_checksum: str


@dataclass(frozen=True)
class OrthoAcquisition:
    """The result of acquiring one site's orthophoto.

    Contract (fields):
        mosaic_path: The written ``<site>/ortho/ortho.tif``.
        provenance_path: The sidecar written beside the mosaic.
        dataset: The selected pinned zone.
        mosaic: The mosaic metadata and pixel checksum.
        provenance: The provenance record written to ``provenance_path``.
        tile_paths: The downloaded source sheets, in mosaic order.

    Invariants:
        - Immutable and hashable; equal iff every field is equal.
    """

    mosaic_path: Path
    provenance_path: Path
    dataset: OrthoDataset
    mosaic: OrthoMosaic
    provenance: Provenance
    tile_paths: tuple[Path, ...]


def _parse_feed(data: bytes) -> AtomFeed:
    """Parse a Beeldmateriaal ATOM feed, re-raising parse errors as ortho ones.

    Failure modes:
        - :class:`OrthoFeedError` if the bytes are not a well-formed feed with
          the required terms and section links.
    """
    try:
        return parse_atom_feed(data)
    except PdokFeedError as exc:
        raise OrthoFeedError(str(exc)) from exc


def _covering_tiles(feed: AtomFeed, aoi_rd: BBox) -> tuple[RemoteTile, ...]:
    """Return the feed's sheets whose bbox intersects ``aoi_rd``, id-ordered.

    Intersection is an axis-aligned test in the feed's WGS84 CRS (a safe
    superset of the true-geometry cover); each selected sheet's extent is
    projected back to EPSG:28992 for the :class:`RemoteTile` downstream expects.
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


def resolve_ortho_tiles(
    dataset: OrthoDataset,
    aoi: BBox,
    http_get: HttpGet,
) -> ResolvedOrthoFeed:
    """Resolve the sheets of ``dataset`` covering ``aoi`` from its ATOM feed.

    Contract:
        - Fetches and parses ``dataset.feed_url``, returning the covering sheets
          as EPSG:28992 :class:`RemoteTile` addresses plus the feed's CC-BY
          terms. Tiles are ordered by ``tile_id``.

    Failure modes:
        - :class:`OrthoFeedError` if the feed cannot be parsed.
        - Propagates :class:`requests.RequestException` from ``http_get``.
    """
    feed = _parse_feed(http_get(dataset.feed_url))
    return ResolvedOrthoFeed(
        licence=feed.licence,
        attribution=feed.attribution,
        tiles=_covering_tiles(feed, aoi),
    )


def select_ortho_dataset(
    aoi: BBox,
    registry: OrthoDatasetRegistry,
    http_get: HttpGet,
) -> OrthoDataset:
    """Select the pinned ortho zone to fetch, preferring the finest that covers.

    Contract:
        - Probes the registered zones in preference order (5cm before 8cm) and
          returns the first whose feed has a sheet covering ``aoi``.

    Failure modes:
        - ``ValueError`` if ``aoi`` is a degenerate bounding box.
        - :class:`OrthoUnavailableError` if no zone covers ``aoi``.
        - :class:`OrthoFeedError` / :class:`requests.RequestException` from a
          probe's feed fetch/parse.
    """
    ensure_valid_bbox(aoi)
    for dataset in registry.datasets():
        if resolve_ortho_tiles(dataset, aoi, http_get).tiles:
            return dataset
    msg = (
        "no Beeldmateriaal ortho zone covers the requested AOI "
        f"{aoi}; both the 5cm and 8cm zones reported it uncovered."
    )
    raise OrthoUnavailableError(msg)


def _pixel_checksum(
    mosaic: npt.NDArray[np.uint8],
    crs: str,
    resolution_m: float,
) -> str:
    """Return a SHA-256 over the mosaic pixel array and its raster header.

    The digest covers crs, dtype, shape and resolution before the raw pixel
    bytes, so it is a stable content address for the mosaic that does not depend
    on the (non-deterministic) GeoTIFF container encoding.
    """
    header = f"{crs}|{mosaic.dtype}|{mosaic.shape}|{resolution_m}".encode()
    digest = hashlib.sha256(header)
    digest.update(mosaic.tobytes())
    return digest.hexdigest()


def mosaic_and_clip(
    tile_paths: tuple[Path, ...],
    aoi: BBox,
    resolution_m: float,
    out_path: Path,
) -> OrthoMosaic:
    """Mosaic ``tile_paths`` and clip to ``aoi`` at ``resolution_m``, deterministically.

    Contract:
        - Merges the sheets with :func:`rasterio.merge.merge` bounded to ``aoi``
          at ``resolution_m`` -- so the output dimensions are exactly the AOI
          extent divided by the pixel size, and an overlap band is written once
          (no double-counting). Sheets are ordered by filename for a
          deterministic merge; the mosaic is written to ``out_path`` as a
          GeoTIFF and its pixel-array checksum returned.

    Failure modes:
        - Propagates :class:`rasterio.errors.RasterioError` from a bad sheet.
    """
    ordered = sorted(tile_paths, key=lambda path: path.name)
    with rasterio.open(ordered[0]) as first:
        crs = str(first.crs)
        dtype = first.dtypes[0]
    mosaic, transform = merge(
        [str(path) for path in ordered], bounds=aoi, res=resolution_m
    )
    pixels: npt.NDArray[np.uint8] = mosaic
    count = pixels.shape[0]
    height = pixels.shape[1]
    width = pixels.shape[2]
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(pixels)
    return OrthoMosaic(
        bbox=aoi,
        crs=crs,
        width=width,
        height=height,
        resolution_m=resolution_m,
        pixel_checksum=_pixel_checksum(pixels, crs, resolution_m),
    )


def _utcnow() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def _downloader(http_get: HttpGet, url: str) -> Callable[[], bytes]:
    """Return a zero-arg fetcher the cache calls only on a miss."""

    def fetch() -> bytes:
        return http_get(url)

    return fetch


def _combined_checksum(tile_hashes: Iterable[str]) -> str:
    """Return a stable digest over the per-sheet checksums (order-independent)."""
    joined = "\n".join(sorted(tile_hashes))
    return hashlib.sha256(joined.encode()).hexdigest()


def _request_keys(
    dataset: OrthoDataset,
    mosaic: OrthoMosaic,
    tile_checksums: dict[str, str],
) -> tuple[tuple[str, str], ...]:
    """Build the ordered, unique provenance request keys for the mosaic."""
    keys: list[tuple[str, str]] = [
        ("source", _PORTAL),
        ("vintage", str(dataset.vintage.year)),
        ("zone", dataset.zone),
        ("resolution_tier", dataset.resolution_tier),
        ("crs", mosaic.crs),
        ("mosaic_method", _MOSAIC_METHOD),
        ("seamlines_note", _SEAMLINE_NOTE),
    ]
    keys.extend(sorted(tile_checksums.items()))
    return tuple(keys)


def acquire_ortho(
    request: AcquisitionRequest,
    *,
    http_get: HttpGet = default_http_get,
    now: Clock = _utcnow,
    cache_root: Path | None = None,
    tool_version: str | None = None,
    registry: OrthoDatasetRegistry | None = None,
) -> OrthoAcquisition:
    """Acquire the orthophoto for ``request`` into ``<site>/ortho/ortho.tif``.

    Contract:
        - Creates the ortho layout, derives the AOI (shared with AHN), selects
          the pinned zone covering it (5cm preferred, 8cm fallback), downloads
          each covering sheet *through the content cache* into
          ``<site>/ortho/tiles/``, mosaics and clips them to the AOI, and writes
          ``ortho.tif`` plus an ``ortho.tif.provenance.json`` sidecar.
        - Deterministic: sheet order, mosaic pixels, and (with injected ``now``
          / ``tool_version``) provenance bytes are stable for the same feed and
          sheets.
        - Idempotent downloads: a sheet already in the cache is not re-fetched.

    Failure modes:
        - :class:`AcquisitionError` if the bbox is malformed / selector unwired
          (via :func:`aoi_bbox`), no zone covers the AOI, a feed is invalid,
          or a sheet download fails -- every expected failure funnels here so the
          CLI reports it cleanly.
    """
    ortho_dir = request.site_dir / _ORTHO_SUBDIR
    ortho_dir.mkdir(parents=True, exist_ok=True)
    aoi = aoi_bbox(request)
    active_registry = (
        registry if registry is not None else default_ortho_registry()
    )
    try:
        dataset = select_ortho_dataset(aoi, active_registry, http_get)
        resolved = resolve_ortho_tiles(dataset, aoi, http_get)
    except (
        OrthoFeedError,
        OrthoUnavailableError,
        requests.RequestException,
    ) as exc:
        raise AcquisitionError(str(exc)) from exc

    version_label = (
        tool_version if tool_version is not None else version("ahn_cli")
    )
    cache = ContentAddressedCache(
        root=cache_root
        if cache_root is not None
        else request.site_dir / _CACHE_DIRNAME
    )
    tiles_dir = ortho_dir / _TILES_SUBDIR
    tiles_dir.mkdir(parents=True, exist_ok=True)

    started = now()
    tile_paths: list[Path] = []
    tile_checksums: dict[str, str] = {}
    for tile in resolved.tiles:
        key = CacheKey(
            product=Product.ORTHO,
            tile_id=tile.tile_id,
            vintage=dataset.vintage,
        )
        try:
            content = cache.get_or_fetch(
                key, _downloader(http_get, tile.download_url)
            )
        except requests.RequestException as exc:
            msg = f"download failed for {tile.download_url}: {exc}"
            raise AcquisitionError(msg) from exc
        tile_path = tiles_dir / f"{tile.tile_id}.tif"
        tile_path.write_bytes(content)
        tile_paths.append(tile_path)
        tile_checksums[tile.download_url] = hashlib.sha256(
            content
        ).hexdigest()

    mosaic_path = ortho_dir / _MOSAIC_NAME
    mosaic = mosaic_and_clip(
        tuple(tile_paths), aoi, dataset.resolution_m, mosaic_path
    )
    finished = now()

    provenance = Provenance(
        source_portal=_PORTAL,
        product=Product.ORTHO,
        licence=resolved.licence,
        attribution=resolved.attribution,
        bbox=aoi,
        download_started_at=started,
        download_finished_at=finished,
        input_checksum=_combined_checksum(tile_checksums.values()),
        output_checksum=mosaic.pixel_checksum,
        tool_version=version_label,
        vintage=dataset.vintage,
        zone=dataset.zone,
        resolution_tier=dataset.resolution_tier,
        request_keys=_request_keys(dataset, mosaic, tile_checksums),
    )
    provenance_path = ortho_dir / f"{_MOSAIC_NAME}.provenance.json"
    write_provenance(provenance, provenance_path)
    return OrthoAcquisition(
        mosaic_path=mosaic_path,
        provenance_path=provenance_path,
        dataset=dataset,
        mosaic=mosaic,
        provenance=provenance,
        tile_paths=tuple(tile_paths),
    )
