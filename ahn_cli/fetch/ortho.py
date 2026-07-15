"""Beeldmateriaal orthophoto fetch: select a dataset, mosaic, clip, record.

Ortho is a *raster* product keyed by :class:`~ahn_cli.domain.Vintage` (not an
AHN :class:`~ahn_cli.domain.Generation`), so it does not fit the AHN
point-cloud :class:`~ahn_cli.fetch.source.FetchSource` protocol (whose
``resolve`` returns LAZ sheets). It therefore plugs in through its *own*
registry -- :class:`OrthoDatasetRegistry` -- but honours the same DDD rule the
AHN generation registry does: selection consults a preference-ordered registry
of pinned datasets and probes AOI coverage, never a stringly-typed switch.

Flow:

1. Select the dataset: probe the pinned zones in preference order and take
   the first whose GeoJSON tile index has a sheet covering the AOI. Today's
   registry holds a single pinned zone -- Beeldmateriaal Nederland's 2025 HRL
   (high resolution, winter/leaf-off) RGB orthophoto -- distributed as a
   GeoJSON tile index by basisdata.nl's "Dataportaal AHN en Beeldmateriaal".
   (The Beeldmateriaal open-data ATOM feed this module used before is
   retired: ``opendata.beeldmateriaal.nl`` no longer resolves.)
2. Resolve the covering GeoTIFF sheets from that dataset's GeoJSON index --
   already native EPSG:28992, so no WGS84 round-trip is needed -- and
   download each *through the WP4 content cache*, so a second fetch costs
   zero network and zero new bytes. Each download's SHA-256 is verified
   against the index's published digest before it is trusted.
3. Mosaic the sheets and clip to the AOI with :func:`rasterio.merge.merge`
   (``bounds`` + ``res`` derived from the first sheet's actual pixel size,
   since the index publishes no per-tile resolution field) -- never a
   shelled-out ``gdalbuildvrt`` -- writing ``<site>/ortho/ortho.tif``.
4. Record real CC BY 4.0 provenance: portal, licence + attribution (pinned;
   the GeoJSON index carries no rights/author fields the old ATOM feed had),
   vintage, zone, resolution tier, extent, per-sheet checksums, and the
   mosaic's pixel checksum as the output checksum.

Everything reaching the network is injected (``http_get``, ``now``), so the
fast suite is deterministic and offline. The dataset vintage and feed URL are
pinned (never floated to the newest available layer).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, cast

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
from ahn_cli.domain.authenticity import uniform_image
from ahn_cli.fetch.acquisition import (
    AcquisitionError,
    AcquisitionRequest,
    Clock,
    aoi_bbox,
    default_http_get,
)
from ahn_cli.fetch.source import HttpGet, boxes_intersect
from ahn_cli.provenance import write_provenance

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator
    from pathlib import Path

    import numpy as np
    import numpy.typing as npt

    from ahn_cli.domain import ProgressCallback

_PORTAL = "basisdata"
_ORTHO_SUBDIR = "ortho"
_TILES_SUBDIR = "tiles"
_MOSAIC_NAME = "ortho.tif"
_CACHE_DIRNAME = ".cache"
_TIF_SUFFIX = ".tif"
_POSITION_LEN = 2

# The pinned Beeldmateriaal RGB open-data vintage. Pinned, never floated to the
# newest layer (spec requirement). 2025 is the first vintage with full
# nationwide coverage across HRL/LRL x RGB/CIR x stereo/ortho (per
# beeldmateriaal.nl/dataroom's coverage table; 2026 is not yet complete).
_ORTHO_VINTAGE = Vintage(2025)

# The pinned HRL (high resolution, winter/leaf-off) GeoJSON tile index,
# distributed by basisdata.nl's "Dataportaal AHN en Beeldmateriaal" (no formal
# API, but this per-product index is a plain, publicly fetchable JSON
# FeatureCollection in native EPSG:28992). The exact live index URL is pinned
# here; if basisdata.nl relocates it, the fetch fails loudly with a clear HTTP
# error, same as any other pinned distribution URL in this codebase.
_HRL_INDEX_URL = (
    "https://basisdata.nl/hwh-portal/20230609_tmp/links/nationaal/"
    "Nederland/BM_HRL2025O_RGB_TIF.json"
)
_ORTHO_ZONES: tuple[tuple[str, str, str], ...] = (
    (
        "hrl",
        _HRL_INDEX_URL,
        "Beeldmateriaal RGB HRL (winter, leaf-off) orthophoto, vintage 2025, "
        "CC BY 4.0, distributed via basisdata.nl.",
    ),
)

# The GeoJSON tile index carries no rights/author fields (unlike the retired
# ATOM feed), so the CC BY 4.0 terms are pinned here instead.
_ORTHO_LICENCE = "https://creativecommons.org/licenses/by/4.0/"
_ORTHO_ATTRIBUTION = (
    "Beeldmateriaal Nederland, distributed via basisdata.nl (CC BY 4.0)."
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
    """Raised when the basisdata.nl HRL GeoJSON tile index cannot be used.

    Signals either a structurally broken index (not a GeoJSON
    ``FeatureCollection``, or no ``features`` array) or a tile that covers
    the requested AOI but has no published sha256 digest. A malformed or
    incomplete row *elsewhere* in the nationwide index -- outside the AOI --
    is tolerated rather than raised (see :func:`_parse_feature`). Subclasses
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
    """A pinned Beeldmateriaal ortho zone: vintage, tier, and its tile index.

    Contract (fields):
        vintage: The pinned acquisition vintage (never the newest layer).
        zone: The acquisition-zone identifier recorded in provenance.
        resolution_tier: The human tier label (e.g. ``"hrl"``), recorded in
            provenance; never branched on as control flow.
        feed_url: The zone's GeoJSON tile index URL; must be non-blank.
        semantics: A human-readable note on the zone, carried for operator
            context.

    Invariants:
        - Frozen: an immutable value object, equal by field value.

    Failure modes:
        - ``ValueError`` if ``feed_url`` is blank.
    """

    vintage: Vintage
    zone: str
    resolution_tier: str
    feed_url: str
    semantics: str

    def __post_init__(self) -> None:
        """Reject a blank feed URL."""
        if not self.feed_url.strip():
            msg = "feed_url must be a non-blank tile index URL."
            raise ValueError(msg)


def _empty_dataset_list() -> list[OrthoDataset]:
    """Return an empty ortho-dataset list (a registry's initial state)."""
    return []


@dataclass
class OrthoDatasetRegistry:
    """A preference-ordered registry of pinned ortho zones.

    Contract:
        - Starts empty; :meth:`register` appends a zone. Registration order is
          preference order, which selection honours directly.
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
    """Return the pinned basisdata.nl registry (2025 HRL).

    Contract:
        - Registers the pinned :data:`_ORTHO_VINTAGE` HRL zone. Deterministic:
          the same zone every call.
    """
    registry = OrthoDatasetRegistry()
    for tier, feed_url, semantics in _ORTHO_ZONES:
        registry.register(
            OrthoDataset(
                vintage=_ORTHO_VINTAGE,
                zone=f"{_PORTAL}-{_ORTHO_VINTAGE.year}-{tier}",
                resolution_tier=tier,
                feed_url=feed_url,
                semantics=semantics,
            )
        )
    return registry


@dataclass(frozen=True)
class OrthoTile:
    """One HRL tile: its id, EPSG:28992 extent, download URL, and content hash.

    Contract:
        - ``tile_id`` is derived from the tile's filename stem (e.g.
          ``"2025_098000_448000_RGB_JPEG_hrl"``); must be non-blank.
        - ``bbox`` is the tile's extent as :data:`~ahn_cli.domain.BBox`, in the
          index's native EPSG:28992 -- no reprojection needed.
        - ``download_url`` is the tile's absolute HTTPS URL; must be non-blank.
        - ``sha256`` is the index's published digest of the tile's bytes, used
          to verify the download before it is trusted.

    Invariants:
        - Immutable and hashable; two tiles are equal iff every field is equal.

    Failure modes:
        - ``ValueError`` if ``tile_id``, ``download_url``, or ``sha256`` is
          blank.
        - ``ValueError`` if ``bbox`` is degenerate (see
          :func:`~ahn_cli.domain.ensure_valid_bbox`).
    """

    tile_id: str
    bbox: BBox
    download_url: str
    sha256: str

    def __post_init__(self) -> None:
        """Validate the identifier, download URL, digest, and extent."""
        if not self.tile_id.strip():
            msg = "tile_id must be a non-blank identifier."
            raise ValueError(msg)
        if not self.download_url.strip():
            msg = "download_url must be a non-blank URL."
            raise ValueError(msg)
        if not self.sha256.strip():
            msg = "sha256 must be a non-blank digest."
            raise ValueError(msg)
        ensure_valid_bbox(self.bbox)


@dataclass(frozen=True)
class ResolvedOrthoFeed:
    """The sheets covering an AOI plus the CC BY 4.0 terms to record for them.

    Contract:
        - ``licence`` / ``attribution`` are the pinned CC BY 4.0 terms (the
          GeoJSON tile index carries no rights/author fields).
        - ``tiles`` are the covering sheets in ascending ``tile_id`` order.

    Invariants:
        - Immutable and hashable; equal by field value.
    """

    licence: str
    attribution: str
    tiles: tuple[OrthoTile, ...]


@dataclass(frozen=True)
class OrthoMosaic:
    """The result of mosaicking and clipping the ortho sheets to the AOI.

    Contract (fields):
        bbox: The clipped extent ``(minx, miny, maxx, maxy)`` in EPSG:28992.
        crs: The mosaic CRS rendered as a string (e.g. ``"EPSG:28992"``).
        width / height: The mosaic raster dimensions in pixels.
        resolution_m: The ground pixel size in metres, read from the source
            sheets -- the tile index publishes no resolution field.
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


def _iter_ring_positions(node: object) -> Iterator[tuple[float, float]]:
    """Yield every ``(x, y)`` position within nested geometry coordinate lists.

    A ``[x, y]`` leaf (both entries JSON numbers, never booleans) is a
    position; any other list is a coordinate container recursed into. The
    ``isinstance`` guards both classify the node and narrow the numbers so no
    unreachable coercion branch is needed.
    """
    if not isinstance(node, list):
        return
    coordinates = cast("list[object]", node)
    if len(coordinates) >= _POSITION_LEN:
        first = coordinates[0]
        second = coordinates[1]
        if (
            not isinstance(first, bool)
            and isinstance(first, (int, float))
            and not isinstance(second, bool)
            and isinstance(second, (int, float))
        ):
            yield (float(first), float(second))
            return
    for child in coordinates:
        yield from _iter_ring_positions(child)


@dataclass(frozen=True)
class _RawOrthoEntry:
    """One HRL index entry, loosely parsed: extent, URL, and hash if published.

    A module-private intermediate representation. The nationwide index (tens
    of thousands of rows) has occasional anomalous entries far from any given
    AOI -- e.g. a handful of rows nationwide with a ``null`` published
    ``sha256`` -- and those must never block a fetch for an unrelated site.
    Only entries that survive AOI filtering are promoted to a fully-validated
    :class:`OrthoTile` (see :func:`_covering_tiles`), where a missing digest
    becomes a hard failure because that tile is one this fetch actually needs.
    """

    tile_id: str
    bbox: BBox
    download_url: str
    sha256: str | None


def _try_geometry_bounds(geometry: object) -> BBox | None:
    """Compute the EPSG:28992 bbox of a polygon geometry, or ``None`` if unusable."""
    if not isinstance(geometry, dict):
        return None
    geometry_map = cast("dict[str, object]", geometry)
    xs: list[float] = []
    ys: list[float] = []
    for x, y in _iter_ring_positions(geometry_map.get("coordinates")):
        xs.append(x)
        ys.append(y)
    if not xs:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def _parse_feature(feature: object) -> _RawOrthoEntry | None:
    """Loosely parse one GeoJSON feature into a :class:`_RawOrthoEntry`.

    Returns ``None`` for a non-``.tif`` entry (e.g. a ``.tif.aux.xml``
    sidecar, which the index lists alongside each real tile) or for a row
    this fetch cannot place or identify (missing/malformed properties,
    ``file`` URL, or geometry) -- these are tolerated here and simply
    excluded, since the nationwide index occasionally has such rows far from
    any given AOI. A missing ``sha256`` is tolerated too; it is only required
    once an entry survives AOI filtering (see :func:`_covering_tiles`).
    """
    if not isinstance(feature, dict):
        return None
    feature_map = cast("dict[str, object]", feature)
    properties = feature_map.get("properties")
    if not isinstance(properties, dict):
        return None
    property_map = cast("dict[str, object]", properties)
    file_url = property_map.get("file")
    if not isinstance(file_url, str) or not file_url.strip():
        return None
    if not file_url.endswith(_TIF_SUFFIX):
        return None
    bbox = _try_geometry_bounds(feature_map.get("geometry"))
    if bbox is None:
        return None
    sha256 = property_map.get("sha256")
    sha256_value = (
        sha256 if isinstance(sha256, str) and sha256.strip() else None
    )
    tile_id = PurePosixPath(file_url).stem
    return _RawOrthoEntry(
        tile_id=tile_id, bbox=bbox, download_url=file_url, sha256=sha256_value
    )


def _parse_feed(data: bytes) -> tuple[_RawOrthoEntry, ...]:
    """Parse a basisdata.nl HRL GeoJSON tile index into loose entries.

    Failure modes:
        - :class:`OrthoFeedError` if the bytes are not a well-formed
          ``FeatureCollection`` with a ``features`` array -- a structural
          problem with the whole feed, unlike a single malformed row (see
          :func:`_parse_feature`).
    """
    try:
        parsed = json.loads(data)
    except json.JSONDecodeError as exc:
        msg = f"HRL tile index is not valid JSON ({exc})."
        raise OrthoFeedError(msg) from exc
    if not isinstance(parsed, dict):
        msg = "HRL tile index must be a GeoJSON FeatureCollection."
        raise OrthoFeedError(msg)
    document = cast("dict[str, object]", parsed)
    if document.get("type") != "FeatureCollection":
        msg = "HRL tile index must be a GeoJSON FeatureCollection."
        raise OrthoFeedError(msg)
    features = document.get("features")
    if not isinstance(features, list):
        msg = "HRL tile index must have a 'features' array."
        raise OrthoFeedError(msg)
    feature_list = cast("list[object]", features)
    entries = [
        entry
        for feature in feature_list
        if (entry := _parse_feature(feature)) is not None
    ]
    return tuple(sorted(entries, key=lambda entry: entry.tile_id))


def _covering_tiles(
    entries: tuple[_RawOrthoEntry, ...], aoi_rd: BBox
) -> tuple[OrthoTile, ...]:
    """Return the entries whose bbox intersects ``aoi_rd``, as validated tiles.

    Both the index and ``aoi_rd`` are EPSG:28992, so this is a direct
    intersection test -- no WGS84 round-trip needed. Only entries that
    intersect are promoted to a fully-validated :class:`OrthoTile`; this is
    where a missing sha256 becomes a hard failure, scoped to tiles this fetch
    actually needs.

    Failure modes:
        - :class:`OrthoFeedError` if a covering entry has no published sha256.
    """
    covering = [
        entry for entry in entries if boxes_intersect(aoi_rd, entry.bbox)
    ]
    tiles: list[OrthoTile] = []
    for entry in covering:
        if entry.sha256 is None:
            msg = (
                f"HRL tile index entry {entry.download_url!r} covers the "
                "requested AOI but has no published sha256 digest."
            )
            raise OrthoFeedError(msg)
        tiles.append(
            OrthoTile(
                tile_id=entry.tile_id,
                bbox=entry.bbox,
                download_url=entry.download_url,
                sha256=entry.sha256,
            )
        )
    return tuple(sorted(tiles, key=lambda tile: tile.tile_id))


def resolve_ortho_tiles(
    dataset: OrthoDataset,
    aoi: BBox,
    http_get: HttpGet,
) -> ResolvedOrthoFeed:
    """Resolve the sheets of ``dataset`` covering ``aoi`` from its tile index.

    Contract:
        - Fetches and parses ``dataset.feed_url``'s GeoJSON tile index,
          returning the covering sheets as EPSG:28992 :class:`OrthoTile`
          entries plus the pinned CC BY 4.0 terms. Tiles are ordered by
          ``tile_id``.

    Failure modes:
        - :class:`OrthoFeedError` if the index cannot be parsed.
        - Propagates :class:`requests.RequestException` from ``http_get``.
    """
    tiles = _parse_feed(http_get(dataset.feed_url))
    return ResolvedOrthoFeed(
        licence=_ORTHO_LICENCE,
        attribution=_ORTHO_ATTRIBUTION,
        tiles=_covering_tiles(tiles, aoi),
    )


def select_ortho_dataset(
    aoi: BBox,
    registry: OrthoDatasetRegistry,
    http_get: HttpGet,
) -> OrthoDataset:
    """Select the pinned ortho zone to fetch, preferring the first that covers.

    Contract:
        - Probes the registered zones in preference order and returns the
          first whose tile index has a sheet covering ``aoi``.

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
        f"{aoi}; every registered zone reported it uncovered."
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
    out_path: Path,
) -> OrthoMosaic:
    """Mosaic ``tile_paths`` and clip to ``aoi``, deterministically.

    Contract:
        - Reads the ground pixel size off the first (filename-ordered) sheet
          -- the tile index publishes no resolution field -- and merges the
          sheets with :func:`rasterio.merge.merge` bounded to ``aoi`` at that
          resolution, so the output dimensions are exactly the AOI extent
          divided by the pixel size, and an overlap band is written once (no
          double-counting). Sheets are ordered by filename for a
          deterministic merge; the mosaic is written to ``out_path`` as a
          GeoTIFF and its pixel-array checksum returned.

    Failure modes:
        - Propagates :class:`rasterio.errors.RasterioError` from a bad sheet.
        - :class:`AcquisitionError` (the CLI-caught base) if the clipped
          mosaic is one uniform colour across every pixel — placeholder
          imagery, not genuine Beeldmateriaal photography.
    """
    ordered = sorted(tile_paths, key=lambda path: path.name)
    with rasterio.open(ordered[0]) as first:
        crs = str(first.crs)
        dtype = first.dtypes[0]
        res = first.res
    mosaic, transform = merge(
        [str(path) for path in ordered], bounds=aoi, res=res
    )
    pixels: npt.NDArray[np.uint8] = mosaic
    if uniform_image(pixels):
        msg = (
            "orthophoto mosaic is a single uniform colour across every "
            "pixel — that is placeholder imagery, not genuine "
            "Beeldmateriaal photography; refusing to write "
            f"{out_path.name}."
        )
        raise AcquisitionError(msg)
    count = pixels.shape[0]
    height = pixels.shape[1]
    width = pixels.shape[2]
    resolution_m = res[0]
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


def _no_op_progress(_done: int, _total: int) -> None:
    """Report nothing; the default when the caller supplies no callback."""


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
    progress: ProgressCallback | None = None,
) -> OrthoAcquisition:
    """Acquire the orthophoto for ``request`` into ``<site>/ortho/ortho.tif``.

    Contract:
        - Creates the ortho layout, derives the AOI (shared with AHN), selects
          the pinned zone covering it, downloads each covering sheet *through
          the content cache* into ``<site>/ortho/tiles/`` -- verifying each
          download's SHA-256 against the tile index's published digest --
          mosaics and clips them to the AOI, and writes ``ortho.tif`` plus an
          ``ortho.tif.provenance.json`` sidecar.
        - Deterministic: sheet order, mosaic pixels, and (with injected ``now``
          / ``tool_version``) provenance bytes are stable for the same feed and
          sheets.
        - Idempotent downloads: a sheet already in the cache is not re-fetched.
        - Calls ``progress(tiles_done, total_tiles)`` once per downloaded sheet
          (before the mosaic step); defaults to a no-op so callers that don't
          care about progress are unaffected.

    Failure modes:
        - :class:`AcquisitionError` if the bbox is malformed / selector unwired
          (via :func:`aoi_bbox`), no zone covers the AOI, the tile index is
          invalid, a sheet download fails, or a downloaded sheet's SHA-256
          does not match the index's published digest -- every expected
          failure funnels here so the CLI reports it cleanly. A sheet the
          mosaic authenticity gate or the checksum check rejects is evicted
          from the cache before the error is raised, so a retry re-downloads
          it instead of replaying the poisoned bytes.
    """
    report = progress if progress is not None else _no_op_progress
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
    sheet_keys: list[CacheKey] = []
    total_tiles = len(resolved.tiles)
    for tile in resolved.tiles:
        key = CacheKey(
            product=Product.ORTHO,
            tile_id=tile.tile_id,
            vintage=dataset.vintage,
        )
        sheet_keys.append(key)
        try:
            content = cache.get_or_fetch(
                key, _downloader(http_get, tile.download_url)
            )
        except requests.RequestException as exc:
            msg = f"download failed for {tile.download_url}: {exc}"
            raise AcquisitionError(msg) from exc
        digest = hashlib.sha256(content).hexdigest()
        if digest != tile.sha256:
            cache.discard(key)
            msg = (
                f"tile {tile.tile_id} from {tile.download_url} does not "
                f"match the tile index's sha256 digest ({tile.sha256}); "
                "refusing to store it as orthophoto data."
            )
            raise AcquisitionError(msg)
        tile_path = tiles_dir / f"{tile.tile_id}.tif"
        tile_path.write_bytes(content)
        tile_paths.append(tile_path)
        tile_checksums[tile.download_url] = digest
        report(len(tile_paths), total_tiles)

    mosaic_path = ortho_dir / _MOSAIC_NAME
    try:
        mosaic = mosaic_and_clip(tuple(tile_paths), aoi, mosaic_path)
    except AcquisitionError:
        # A uniform (placeholder) mosaic means the sheets it was built from
        # are bad; they must never stay cached, or every retry would replay
        # the same poisoned bytes. Evict every sheet the failed mosaic used
        # so the next run re-downloads them.
        for sheet_key in sheet_keys:
            cache.discard(sheet_key)
        raise
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
