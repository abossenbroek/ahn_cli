"""Fetch a DSM (Digital Surface Model) raster for an AOI and clip it.

The ``fetch`` context can, alongside the AHN point cloud, acquire a *Digital
Surface Model* raster: a PDOK-distributed Cloud-Optimized GeoTIFF (COG) read
through an HTTP-range **windowed** read -- only the AOI window is pulled, never
the whole national raster (per SPIKE-PDAL, COG windowed reads are feasible,
unlike plain LAZ). The window is clipped to the AOI and written to
``data/<site>/dsm.tif`` with a provenance sidecar.

The design plugs into the WP6 seams rather than reinventing them: the covering
COG sheet is resolved from a PDOK INSPIRE ATOM feed (reusing
:func:`~ahn_cli.fetch.pdok.parse_atom_feed`), the download is idempotent through
the WP4 content cache (a second fetch performs zero windowed reads), and every
expected failure funnels through :class:`DsmError`, a subclass of
:class:`~ahn_cli.fetch.acquisition.AcquisitionError`, so the CLI reports a tidy
message rather than a traceback. The DSM source is selected registry-driven
(``SourceKind`` -> source), never a stringly-typed switch.

Glass-roof lidar **voids** (nodata) and **spikes** are *recorded, never
repaired* here: fill-vs-keep is a downstream "look" decision. Nodata is
preserved in the output raster, and the void fraction and spike count are
recorded as a QA note in the provenance ``request_keys``.

Everything reaching the network is injected (``http_get`` for the feed,
``reader`` for the windowed COG read, ``now`` for timestamps), so the fast test
suite is deterministic and offline.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import rasterio
import requests
from rasterio.errors import RasterioIOError
from rasterio.io import MemoryFile
from rasterio.windows import Window, from_bounds

from ahn_cli.cache import CacheKey, ContentAddressedCache
from ahn_cli.domain import BBox, Product, Provenance, Vintage
from ahn_cli.domain.authenticity import flat_surface
from ahn_cli.fetch.acquisition import (
    AcquisitionError,
    AcquisitionRequest,
    aoi_bbox,
    create_site_layout,
    default_http_get,
)
from ahn_cli.fetch.pdok import PdokFeedError, parse_atom_feed
from ahn_cli.fetch.source import (
    HttpGet,
    RemoteTile,
    ResolvedFeed,
    SourceKind,
    boxes_intersect,
    to_rd,
    to_wgs84,
)
from ahn_cli.provenance import write_provenance

if TYPE_CHECKING:
    from pathlib import Path

    from ahn_cli.domain import ProgressCallback

_CACHE_DIRNAME = ".cache"
_DSM_FILENAME = "dsm.tif"
_DSM_CRS = "EPSG:28992"

# The PDOK INSPIRE ATOM feed for the AHN DSM 0.5m COG sheets. Duplicated (with
# citation) from the live PDOK AHN service rather than importing the deprecated
# ``ahn_cli.config``; the fast suite injects the feed fetcher, so this URL is
# exercised only by the nightly portal-contract test (WP14).
_DSM_FEED_URL = "https://service.pdok.nl/rws/ahn/atom/dsm_05m.xml"

# The DSM product's pinned acquisition vintage. Per the domain rule a vintage is
# pinned explicitly, never floated to "Actueel"; this pins the current AHN DSM
# 0.5m release the feed above distributes. It is the cache/provenance temporal
# axis for the raster and is superseded when the live feed advances (tracked by
# the nightly portal-contract test).
_DSM_VINTAGE = Vintage(2023)

# Absolute-height heuristic for flagging DSM spikes (metres, NAP). Values whose
# magnitude exceeds this are physically implausible for a Dutch surface and are
# almost always lidar artefacts (e.g. glass-roof returns). This is a coarse,
# deterministic QA *note* -- the count is recorded, never used to alter pixels.
_SPIKE_ABS_MAX_M = 200.0

# Feed-resolution failures surfaced cleanly: a changed/invalid ATOM feed or a
# failed feed HTTP request, funnelled to DsmError so the CLI stays tidy.
_DSM_FEED_ERRORS: tuple[type[Exception], ...] = (
    PdokFeedError,
    requests.RequestException,
)

WindowedDsmReader = Callable[[str, BBox], bytes]
"""An injected windowed COG reader: maps ``(url, aoi)`` to clipped GeoTIFF bytes.

Injected wherever the DSM fetch reaches a COG so the fast test suite passes a
deterministic reader (or the real one against a local synthetic COG) and never
performs real network I/O.
"""

Clock = Callable[[], datetime]
"""An injected UTC clock, so download timestamps are deterministic in tests."""


class DsmError(AcquisitionError):
    """Raised when a DSM fetch cannot complete for an expected reason.

    Signals a changed/invalid DSM feed, an AOI covered by no or by more than one
    DSM sheet, a non-EPSG:28992 COG, an empty clip window, or a failed windowed
    read. Subclasses :class:`~ahn_cli.fetch.acquisition.AcquisitionError`, so the
    CLI reports every such failure as a tidy error rather than a traceback.
    """


@dataclass(frozen=True)
class DsmStats:
    """The metadata and QA read back from a clipped DSM raster.

    Contract (fields):
        crs: The raster's CRS rendered as a string (always ``"EPSG:28992"``).
        bounds: The clipped extent ``(minx, miny, maxx, maxy)`` in EPSG:28992 --
            the pixel-snapped AOI window, so it need not equal the AOI exactly.
        resolution: The pixel size in metres (the COG's native resolution).
        nodata: The raster's nodata value, or ``None`` if it declares none.
        nodata_fraction: The fraction of pixels equal to ``nodata`` (the void
            fraction); ``0.0`` when the raster declares no nodata.
        spike_count: The number of valid pixels whose magnitude exceeds the
            spike heuristic (:data:`_SPIKE_ABS_MAX_M`).
        width: The clipped raster width in pixels.
        height: The clipped raster height in pixels.

    Invariants:
        - Immutable and hashable; equal iff every field is equal.
    """

    crs: str
    bounds: BBox
    resolution: float
    nodata: float | None
    nodata_fraction: float
    spike_count: int
    width: int
    height: int


@dataclass(frozen=True)
class PdokDsmSource:
    """The PDOK INSPIRE ATOM distribution source for the DSM COG.

    Resolves the DSM sheets covering an AOI from the PDOK DSM ATOM feed. Mirrors
    :class:`~ahn_cli.fetch.pdok.PdokSource`, but yields COG sheets meant for a
    windowed raster read rather than whole-file LAZ downloads.
    """

    feed_url: str = _DSM_FEED_URL

    def resolve(self, aoi: BBox, http_get: HttpGet) -> ResolvedFeed:
        """Resolve the DSM sheets covering ``aoi`` from the ATOM feed.

        Contract:
            - Fetches and parses the DSM ATOM feed, returning the sheets whose
              WGS84 bbox intersects ``aoi`` as EPSG:28992 :class:`RemoteTile`
              addresses, ordered by ``tile_id``, with the feed's licence terms.

        Failure modes:
            - :class:`~ahn_cli.fetch.pdok.PdokFeedError` if the feed is
              malformed (funnelled to :class:`DsmError` by :func:`fetch_dsm`).
        """
        feed = parse_atom_feed(http_get(self.feed_url))
        aoi_wgs84 = to_wgs84(aoi)
        # Mirrors ``pdok._covering_tiles``: an axis-aligned WGS84 intersection
        # test (a safe superset), each hit projected back to the Dutch grid.
        selected = [
            RemoteTile(
                tile_id=tile.tile_id,
                bbox=to_rd(tile.bbox_wgs84),
                download_url=tile.download_url,
            )
            for tile in feed.tiles
            if boxes_intersect(aoi_wgs84, tile.bbox_wgs84)
        ]
        tiles = tuple(sorted(selected, key=lambda tile: tile.tile_id))
        return ResolvedFeed(
            licence=feed.licence,
            attribution=feed.attribution,
            tiles=tiles,
        )


_DSM_SOURCE_REGISTRY: dict[SourceKind, PdokDsmSource] = {
    SourceKind.PDOK: PdokDsmSource(),
}


def dsm_source_for(kind: SourceKind) -> PdokDsmSource:
    """Return the DSM distribution source registered for ``kind``.

    Contract:
        - Registry-driven, mirroring
          :func:`~ahn_cli.fetch.acquisition.source_for`: the closed
          :class:`SourceKind` set maps to a DSM source, so selection never
          branches on a raw string.

    Failure modes:
        - :class:`DsmError` if ``kind`` has no registered DSM source.
    """
    try:
        return _DSM_SOURCE_REGISTRY[kind]
    except KeyError as exc:
        msg = f"no DSM source is registered for --source {kind.value}."
        raise DsmError(msg) from exc


def _utcnow() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def _no_op_progress(_done: int, _total: int) -> None:
    """Report nothing; the default when the caller supplies no callback."""


def read_dsm_window(url: str, aoi: BBox) -> bytes:
    """Windowed-read the DSM COG at ``url``, clipped to ``aoi``.

    Contract:
        - Opens the COG (a local path or an HTTP URL GDAL reads with range
          requests), reads only the pixel window covering ``aoi``, and returns
          the clipped raster as GeoTIFF bytes whose transform/extent are the
          pixel-snapped AOI window. Nodata is preserved, never filled.
        - Deterministic: the same COG and AOI yield byte-identical output.

    Failure modes:
        - :class:`DsmError` if the COG is not in EPSG:28992, the AOI yields an
          empty window on the sheet, or the windowed read fails.
    """
    try:
        with rasterio.open(url) as dataset:
            crs = str(dataset.crs)
            if crs != _DSM_CRS:
                msg = (
                    f"DSM COG at {url} is in {crs}, but a {_DSM_CRS} raster is "
                    "required (no reprojection is performed)."
                )
                raise DsmError(msg)
            window = _aoi_window(aoi, dataset)
            pixels = dataset.read(window=window)
            transform = dataset.window_transform(window)
            return _encode_geotiff(pixels, transform, crs, dataset.nodata)
    except RasterioIOError as exc:
        msg = f"DSM COG windowed read failed for {url}: {exc}"
        raise DsmError(msg) from exc


def _aoi_window(aoi: BBox, dataset: rasterio.DatasetReader) -> Window:
    """Return the pixel window of ``dataset`` covering ``aoi``, grid-snapped.

    The float window from :func:`from_bounds` is rounded to whole pixels and
    intersected with the raster's own extent, so the read never runs off the
    sheet. An AOI that snaps to nothing on the sheet is a :class:`DsmError`.
    """
    raw = from_bounds(aoi[0], aoi[1], aoi[2], aoi[3], dataset.transform)
    col_off = round(raw.col_off)
    row_off = round(raw.row_off)
    start_col = max(0, col_off)
    start_row = max(0, row_off)
    end_col = min(dataset.width, col_off + round(raw.width))
    end_row = min(dataset.height, row_off + round(raw.height))
    if end_col <= start_col or end_row <= start_row:
        msg = f"AOI {aoi} yields an empty DSM window on the covering sheet."
        raise DsmError(msg)
    return Window(
        start_col, start_row, end_col - start_col, end_row - start_row
    )


def _encode_geotiff(
    pixels: npt.NDArray[np.float32],
    transform: rasterio.Affine,
    crs: str,
    nodata: float | None,
) -> bytes:
    """Encode a ``(bands, rows, cols)`` array to in-memory GeoTIFF bytes."""
    count = pixels.shape[0]
    height = pixels.shape[1]
    width = pixels.shape[2]
    with MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            height=height,
            width=width,
            count=count,
            dtype=str(pixels.dtype),
            crs=crs,
            transform=transform,
            nodata=nodata,
        ) as dataset:
            dataset.write(pixels)
        return memfile.read()


def inspect_dsm(content: bytes) -> DsmStats:
    """Read a clipped DSM raster's metadata and QA from its GeoTIFF bytes.

    Contract:
        - Opens ``content`` locally (no network) and returns its CRS, extent,
          resolution, nodata, void fraction, spike count, and pixel dimensions.

    Failure modes:
        - :class:`DsmError` if ``content`` is not a readable raster, or if it
          carries no genuine relief (no valid pixels at all, or a single
          constant elevation across every valid pixel — a placeholder
          surface, not measured AHN DSM data).
    """
    try:
        with MemoryFile(content) as memfile, memfile.open() as dataset:
            crs = str(dataset.crs)
            box = dataset.bounds
            bounds: BBox = (box.left, box.bottom, box.right, box.top)
            resolution = dataset.res[0]
            nodata = dataset.nodata
            width = dataset.width
            height = dataset.height
            pixels = dataset.read(1)
    except RasterioIOError as exc:
        msg = f"clipped DSM raster is not readable: {exc}"
        raise DsmError(msg) from exc
    if flat_surface(pixels, nodata):
        msg = (
            "clipped DSM raster carries no genuine relief (no valid pixels, "
            "or one constant elevation everywhere) — that is a placeholder "
            "surface, not measured AHN DSM data; refusing to store it."
        )
        raise DsmError(msg)
    return DsmStats(
        crs=crs,
        bounds=bounds,
        resolution=resolution,
        nodata=nodata,
        nodata_fraction=_nodata_fraction(pixels, nodata),
        spike_count=_spike_count(pixels, nodata),
        width=width,
        height=height,
    )


def _nodata_fraction(
    pixels: npt.NDArray[np.float32], nodata: float | None
) -> float:
    """Return the fraction of ``pixels`` equal to ``nodata`` (voids)."""
    if nodata is None:
        return 0.0
    voids = int(np.count_nonzero(pixels == nodata))
    return voids / pixels.size


def _spike_count(
    pixels: npt.NDArray[np.float32], nodata: float | None
) -> int:
    """Return the count of valid pixels exceeding the spike heuristic."""
    spikes = np.abs(pixels) > _SPIKE_ABS_MAX_M
    if nodata is not None:
        spikes = spikes & (pixels != nodata)
    return int(np.count_nonzero(spikes))


def fetch_dsm(
    request: AcquisitionRequest,
    *,
    http_get: HttpGet = default_http_get,
    reader: WindowedDsmReader = read_dsm_window,
    now: Clock = _utcnow,
    cache_root: Path | None = None,
    tool_version: str | None = None,
    progress: ProgressCallback | None = None,
) -> Path:
    """Fetch and clip the DSM for ``request``'s AOI to ``<site>/dsm.tif``.

    Contract:
        - Creates the site layout, derives the AOI, resolves the single covering
          DSM sheet, windowed-reads it *through the content cache* (a cached
          window costs zero network -- ``reader`` is not called), writes the
          clipped raster to ``<site>/dsm.tif``, and writes a
          ``dsm.tif.provenance.json`` sidecar recording source, licence, extent,
          resolution, nodata, and the void/spike QA note.
        - Returns the written ``dsm.tif`` path.
        - Deterministic: extent, checksums, and (with injected ``now`` /
          ``tool_version``) provenance are stable for the same feed and sheet.
        - Calls ``progress(0, 1)`` before the fetch and ``progress(1, 1)`` after
          it completes (there is exactly one DSM sheet to resolve); defaults to
          a no-op so callers that don't care about progress are unaffected.

    Failure modes:
        - :class:`~ahn_cli.fetch.acquisition.AcquisitionError` if the bbox is
          malformed or the ``--city`` selector is not wired.
        - :class:`DsmError` if the DSM feed is invalid, no or more than one sheet
          covers the AOI, the COG is not EPSG:28992, or the windowed read fails.
          Every expected failure funnels here so the CLI reports it cleanly.
          A clip the relief gate rejects is evicted from the cache before the
          error is raised, so a retry re-reads the window instead of
          replaying the poisoned bytes.
    """
    report = progress if progress is not None else _no_op_progress
    report(0, 1)
    create_site_layout(request.site_dir)
    aoi = aoi_bbox(request)
    source = dsm_source_for(request.source)
    try:
        resolved = source.resolve(aoi, http_get)
    except _DSM_FEED_ERRORS as exc:
        raise DsmError(str(exc)) from exc
    tile = _single_covering_tile(resolved.tiles, aoi)

    version_label = (
        tool_version if tool_version is not None else version("ahn_cli")
    )
    cache = ContentAddressedCache(
        root=cache_root
        if cache_root is not None
        else request.site_dir / _CACHE_DIRNAME
    )
    key = CacheKey(
        product=Product.DSM,
        tile_id=_clip_id(tile.tile_id, aoi),
        vintage=_DSM_VINTAGE,
    )
    started = now()
    content = cache.get_or_fetch(key, lambda: reader(tile.download_url, aoi))
    finished = now()

    try:
        stats = inspect_dsm(content)
    except DsmError:
        # A rejected clip must never stay cached, or every retry would
        # replay the same poisoned bytes; evict the entry so the next run
        # re-reads the window.
        cache.discard(key)
        raise
    dsm_path = request.site_dir / _DSM_FILENAME
    dsm_path.write_bytes(content)
    checksum = hashlib.sha256(content).hexdigest()
    provenance = Provenance(
        source_portal=request.source.value,
        product=Product.DSM,
        licence=resolved.licence,
        attribution=resolved.attribution,
        bbox=stats.bounds,
        download_started_at=started,
        download_finished_at=finished,
        input_checksum=checksum,
        output_checksum=checksum,
        tool_version=version_label,
        vintage=_DSM_VINTAGE,
        resolution_tier=f"{stats.resolution:.2f}m",
        request_keys=(
            ("source", request.source.value),
            ("tile_id", tile.tile_id),
            ("url", tile.download_url),
            ("crs", stats.crs),
            ("nodata", _format_nodata(stats.nodata)),
            ("qa_nodata_fraction", f"{stats.nodata_fraction:.6f}"),
            ("qa_spike_count", str(stats.spike_count)),
        ),
    )
    write_provenance(
        provenance, request.site_dir / f"{_DSM_FILENAME}.provenance.json"
    )
    report(1, 1)
    return dsm_path


def _single_covering_tile(
    tiles: tuple[RemoteTile, ...], aoi: BBox
) -> RemoteTile:
    """Return the one DSM sheet covering ``aoi``, or fail if not exactly one.

    Windowed DSM clip supports a single covering sheet: multi-sheet mosaicking
    is the orthophoto path's job, not the DSM path's, so a straddling AOI is a
    clean typed error rather than a silent partial clip.
    """
    if not tiles:
        msg = f"no DSM sheet covers the AOI {aoi}."
        raise DsmError(msg)
    if len(tiles) > 1:
        ids = ", ".join(tile.tile_id for tile in tiles)
        msg = (
            f"AOI {aoi} spans {len(tiles)} DSM sheets ({ids}); windowed DSM "
            "clip supports a single covering sheet -- narrow the AOI."
        )
        raise DsmError(msg)
    return next(iter(tiles))


def _clip_id(tile_id: str, aoi: BBox) -> str:
    """Return an AOI-specific cache tile id for the clipped window.

    The cached artefact is the AOI clip of a sheet, not the whole sheet, so the
    AOI extent is folded into the cache tile id: two different AOIs on the same
    sheet address different clips, keeping the content cache correct.
    """
    extent = "_".join(f"{coord:.3f}" for coord in aoi)
    return f"{tile_id}#{extent}"


def _format_nodata(nodata: float | None) -> str:
    """Render the nodata value for the provenance QA note."""
    return "none" if nodata is None else repr(nodata)
