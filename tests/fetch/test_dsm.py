"""Tests for the DSM windowed COG fetch + clip.

These build tiny valid EPSG:28992 GeoTIFFs in-process with rasterio (standing in
for the COG) and synthesise a PDOK DSM ATOM feed pointing at them -- no network.
They assert the windowed read matches the AOI, nodata is preserved, voids/spikes
are recorded, the fetch is idempotent through the cache, provenance is complete,
and every expected failure funnels through the AcquisitionError family.
"""

from __future__ import annotations

import hashlib
import itertools
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pytest
import rasterio
import requests
from pyproj import Transformer
from rasterio.transform import from_bounds

from ahn_cli.domain import BBox, Product, Vintage
from ahn_cli.fetch.acquisition import (
    AcquisitionError,
    AcquisitionRequest,
    AreaSelectorKind,
    MalformedBboxError,
    SelectorNotWiredError,
)
from ahn_cli.fetch.dsm import (
    DsmError,
    DsmStats,
    PdokDsmSource,
    dsm_source_for,
    fetch_dsm,
    inspect_dsm,
    read_dsm_window,
)
from ahn_cli.fetch.source import SourceKind
from ahn_cli.provenance import read_provenance

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

_TO_WGS84 = Transformer.from_crs("EPSG:28992", "EPSG:4326", always_xy=True)

# A 20 m x 20 m COG sheet on the Dutch grid at 0.5 m resolution (40x40 pixels).
_SHEET_RD: BBox = (194000.0, 443000.0, 194020.0, 443020.0)
_RES = 0.5
_SHEET_W = 40
_SHEET_H = 40
_NODATA = -9999.0

# A 10 m x 10 m AOI fully inside the sheet.
_AOI_RD: BBox = (194005.0, 443005.0, 194015.0, 443015.0)
_AOI_STR = "194005.0,443005.0,194015.0,443015.0"

_START = datetime(2026, 7, 10, 9, 0, tzinfo=timezone.utc)
_FINISH = datetime(2026, 7, 10, 9, 0, 3, tzinfo=timezone.utc)


def _fixed_clock() -> Callable[[], datetime]:
    """Return a clock cycling the two fixed timestamps, one per call."""
    cursor = itertools.cycle((_START, _FINISH))
    return lambda: next(cursor)


def _close(actual: float, expected: float, tol: float = 1e-6) -> bool:
    """Return whether two floats agree within ``tol`` (grid-aligned tolerance)."""
    return abs(actual - expected) <= tol


def _wgs84(rd_bbox: BBox) -> BBox:
    """Project an EPSG:28992 box to a WGS84 (minlon, minlat, maxlon, maxlat)."""
    minlon, minlat = _TO_WGS84.transform(rd_bbox[0], rd_bbox[1])
    maxlon, maxlat = _TO_WGS84.transform(rd_bbox[2], rd_bbox[3])
    return (minlon, minlat, maxlon, maxlat)


def _write_cog(
    path: Path,
    *,
    crs: str = "EPSG:28992",
    bounds: BBox = _SHEET_RD,
    width: int = _SHEET_W,
    height: int = _SHEET_H,
    nodata: float | None = _NODATA,
) -> None:
    """Write a tiny valid GeoTIFF standing in for a DSM COG sheet.

    The surface is a constant 10.0 m with a block of nodata voids and a block of
    high-value spikes, so a windowed clip contains both to exercise the QA.
    """
    transform = from_bounds(
        bounds[0], bounds[1], bounds[2], bounds[3], width, height
    )
    pixels: npt.NDArray[np.float32] = np.full(
        (1, height, width), 10.0, dtype=np.float32
    )
    if nodata is not None:
        pixels[0, 10:14, 10:14] = nodata  # a 4x4 void block inside the AOI
    pixels[0, 20:22, 20:22] = 500.0  # a 2x2 spike block inside the AOI
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dataset:
        dataset.write(pixels)


def _dsm_feed(
    *tiles: tuple[str, BBox],
    licence: str = "https://creativecommons.org/licenses/by/4.0/",
    author: str = "RWS",
) -> bytes:
    """Build a PDOK DSM ATOM feed with one section link per ``(href, rd_bbox)``."""
    sections = "".join(
        f'<link rel="section" href="{href}" '
        f'bbox="{box[0]} {box[1]} {box[2]} {box[3]}" />'
        for href, rd in tiles
        for box in (_wgs84(rd),)
    )
    feed = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<feed xmlns="http://www.w3.org/2005/Atom">'
        f"<rights>{licence}</rights>"
        f"<author><name>{author}</name></author>"
        f"<entry>{sections}</entry>"
        "</feed>"
    )
    return feed.encode("utf-8")


def _feed_get(feed_bytes: bytes) -> Callable[[str], bytes]:
    """Return an injected http_get serving ``feed_bytes`` for any URL."""

    def http_get(url: str) -> bytes:
        del url
        return feed_bytes

    return http_get


def _sheet_request(site: Path) -> AcquisitionRequest:
    """Build a bbox-selector request for the standard AOI."""
    return AcquisitionRequest(
        site_dir=site,
        selector=AreaSelectorKind.BBOX,
        area=_AOI_STR,
    )


def _geotiff_bytes(
    tmp_path: Path,
    pixels: npt.NDArray[np.float32],
    *,
    nodata: float | None,
) -> bytes:
    """Write ``pixels`` as a small EPSG:28992 GeoTIFF and return its bytes."""
    height, width = pixels.shape
    path = tmp_path / "probe.tif"
    transform = from_bounds(
        0.0, 0.0, float(width), float(height), width, height
    )
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs="EPSG:28992",
        transform=transform,
        nodata=nodata,
    ) as dataset:
        dataset.write(pixels, 1)
    return path.read_bytes()


# --------------------------------------------------------------------------- #
# Error taxonomy + registry
# --------------------------------------------------------------------------- #


def test_dsm_error_is_an_acquisition_error() -> None:
    """DsmError funnels through AcquisitionError so the CLI stays tidy."""
    assert issubclass(DsmError, AcquisitionError)


def test_dsm_source_for_maps_pdok() -> None:
    """The DSM source registry maps PDOK to its ATOM source, no stringly switch."""
    assert isinstance(dsm_source_for(SourceKind.PDOK), PdokDsmSource)


def test_dsm_source_for_rejects_unregistered_kind() -> None:
    """A source with no DSM entry is a clean DsmError, not a KeyError."""
    with pytest.raises(DsmError, match="no DSM source"):
        dsm_source_for(SourceKind.GEOTILES)


# --------------------------------------------------------------------------- #
# Windowed read
# --------------------------------------------------------------------------- #


def test_read_dsm_window_matches_the_aoi_and_preserves_nodata(
    tmp_path: Path,
) -> None:
    """The clip's extent/transform match the AOI and nodata is not filled."""
    cog = tmp_path / "sheet.tif"
    _write_cog(cog)

    content = read_dsm_window(str(cog), _AOI_RD)

    clip = tmp_path / "clip.tif"
    clip.write_bytes(content)
    with rasterio.open(clip) as dataset:
        assert dataset.nodata == _NODATA
        assert _close(dataset.res[0], _RES)
        bounds = dataset.bounds
        # The 10 m AOI at 0.5 m -> a 20x20 pixel window aligned to the grid.
        assert dataset.width == 20
        assert dataset.height == 20
        assert _close(bounds.left, _AOI_RD[0])
        assert _close(bounds.bottom, _AOI_RD[1])
        assert _close(bounds.right, _AOI_RD[2])
        assert _close(bounds.top, _AOI_RD[3])
        # nodata survives the clip (the void block sits inside the AOI).
        assert np.any(dataset.read(1) == _NODATA)


def test_read_dsm_window_is_byte_deterministic(tmp_path: Path) -> None:
    """The same COG and AOI yield byte-identical clip output."""
    cog = tmp_path / "sheet.tif"
    _write_cog(cog)

    first = read_dsm_window(str(cog), _AOI_RD)
    second = read_dsm_window(str(cog), _AOI_RD)

    assert first == second


def test_read_dsm_window_rejects_non_rd_crs(tmp_path: Path) -> None:
    """A COG that is not EPSG:28992 is a DsmError (no silent reprojection)."""
    cog = tmp_path / "wgs.tif"
    _write_cog(cog, crs="EPSG:4326", bounds=(3.0, 50.0, 3.0002, 50.0002))

    with pytest.raises(DsmError, match="required"):
        read_dsm_window(str(cog), _AOI_RD)


def test_read_dsm_window_rejects_an_empty_window(tmp_path: Path) -> None:
    """An AOI that snaps to nothing on the sheet is a DsmError."""
    cog = tmp_path / "sheet.tif"
    _write_cog(cog)
    # An AOI east of the sheet: valid box, but no overlapping pixels.
    outside: BBox = (194100.0, 443100.0, 194110.0, 443110.0)

    with pytest.raises(DsmError, match="empty DSM window"):
        read_dsm_window(str(cog), outside)


def test_read_dsm_window_funnels_a_read_failure(tmp_path: Path) -> None:
    """A path that is not a raster surfaces as a DsmError, not a traceback."""
    broken = tmp_path / "broken.tif"
    broken.write_bytes(b"not a GeoTIFF")

    with pytest.raises(DsmError, match="windowed read failed"):
        read_dsm_window(str(broken), _AOI_RD)


# --------------------------------------------------------------------------- #
# Inspection / QA
# --------------------------------------------------------------------------- #


def test_inspect_dsm_counts_voids_and_spikes(tmp_path: Path) -> None:
    """Void fraction and spike count are computed over the clipped pixels."""
    pixels: npt.NDArray[np.float32] = np.full((4, 4), 10.0, dtype=np.float32)
    pixels[0, 0] = _NODATA  # one void
    pixels[0, 1] = _NODATA  # two voids
    pixels[3, 3] = 500.0  # one spike (> 200 m)
    content = _geotiff_bytes(tmp_path, pixels, nodata=_NODATA)

    stats = inspect_dsm(content)

    assert isinstance(stats, DsmStats)
    assert stats.crs == "EPSG:28992"
    assert stats.width == 4
    assert stats.height == 4
    assert stats.nodata == _NODATA
    assert _close(stats.nodata_fraction, 2 / 16)
    assert stats.spike_count == 1


def test_inspect_dsm_without_nodata_reports_zero_void_fraction(
    tmp_path: Path,
) -> None:
    """A raster declaring no nodata has void fraction 0 and still counts spikes."""
    pixels: npt.NDArray[np.float32] = np.full((3, 3), 10.0, dtype=np.float32)
    pixels[1, 1] = -400.0  # a negative spike, magnitude > 200 m
    content = _geotiff_bytes(tmp_path, pixels, nodata=None)

    stats = inspect_dsm(content)

    assert stats.nodata is None
    assert stats.nodata_fraction == 0.0
    assert stats.spike_count == 1


def test_inspect_dsm_funnels_a_read_failure() -> None:
    """Non-raster bytes surface as a DsmError from inspection."""
    with pytest.raises(DsmError, match="not readable"):
        inspect_dsm(b"not a GeoTIFF")


# --------------------------------------------------------------------------- #
# Source resolution
# --------------------------------------------------------------------------- #


def test_resolve_returns_covering_sheet_with_licence() -> None:
    """Resolve returns the intersecting sheet as an RD tile with feed terms."""
    feed = _dsm_feed(("https://pdok/dsm/R_37EN1.TIF", _SHEET_RD))

    resolved = PdokDsmSource().resolve(_AOI_RD, _feed_get(feed))

    assert resolved.licence.startswith("https://creativecommons.org")
    assert resolved.attribution == "RWS"
    assert [tile.tile_id for tile in resolved.tiles] == ["R_37EN1"]


def test_resolve_excludes_non_intersecting_sheets() -> None:
    """A sheet whose bbox misses the AOI is not selected."""
    far: BBox = (100000.0, 400000.0, 100020.0, 400020.0)
    feed = _dsm_feed(("https://pdok/dsm/R_FAR.TIF", far))

    resolved = PdokDsmSource().resolve(_AOI_RD, _feed_get(feed))

    assert resolved.tiles == ()


# --------------------------------------------------------------------------- #
# fetch_dsm end to end
# --------------------------------------------------------------------------- #


class _RecordingReader:
    """A windowed reader delegating to the real read, recording each call."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, BBox]] = []

    def __call__(self, url: str, aoi: BBox) -> bytes:
        self.calls.append((url, aoi))
        return read_dsm_window(url, aoi)


def test_fetch_dsm_writes_clip_and_provenance(tmp_path: Path) -> None:
    """A DSM fetch writes <site>/dsm.tif and a complete provenance sidecar."""
    site = tmp_path / "delft"
    cog = tmp_path / "R_37EN1.TIF"
    _write_cog(cog)
    feed = _dsm_feed((str(cog), _SHEET_RD))

    dsm_path = fetch_dsm(
        _sheet_request(site),
        http_get=_feed_get(feed),
        now=_fixed_clock(),
        tool_version="wp7-test",
    )

    assert dsm_path == site / "dsm.tif"
    assert dsm_path.is_file()
    provenance = read_provenance(site / "dsm.tif.provenance.json")
    assert provenance.product is Product.DSM
    assert provenance.source_portal == "pdok"
    assert provenance.licence.startswith("https://creativecommons.org")
    assert provenance.vintage == Vintage(2023)
    assert provenance.resolution_tier == "0.50m"
    assert provenance.tool_version == "wp7-test"
    keys = dict(provenance.request_keys)
    assert keys["crs"] == "EPSG:28992"
    assert keys["tile_id"] == "R_37EN1"
    assert keys["nodata"] == repr(_NODATA)
    # The AOI window contains the 4x4 void block and the 2x2 spike block.
    assert _close(float(keys["qa_nodata_fraction"]), 16 / 400)
    assert int(keys["qa_spike_count"]) == 4
    checksum = hashlib.sha256(dsm_path.read_bytes()).hexdigest()
    assert provenance.input_checksum == checksum
    assert provenance.output_checksum == checksum


def test_fetch_dsm_is_idempotent_through_the_cache(tmp_path: Path) -> None:
    """A second fetch serves the cached clip with zero windowed reads."""
    site = tmp_path / "delft"
    cog = tmp_path / "R_37EN1.TIF"
    _write_cog(cog)
    feed = _dsm_feed((str(cog), _SHEET_RD))
    reader = _RecordingReader()

    first = fetch_dsm(
        _sheet_request(site),
        http_get=_feed_get(feed),
        reader=reader,
        now=_fixed_clock(),
        tool_version="v",
    )
    first_bytes = first.read_bytes()
    assert len(reader.calls) == 1

    second = fetch_dsm(
        _sheet_request(site),
        http_get=_feed_get(feed),
        reader=reader,
        now=_fixed_clock(),
        tool_version="v",
    )

    assert len(reader.calls) == 1  # the second fetch hit the cache
    assert second.read_bytes() == first_bytes


def test_fetch_dsm_uses_default_clock_and_tool_version(
    tmp_path: Path,
) -> None:
    """With neither injected, the real clock and package version are used."""
    site = tmp_path / "delft"
    cog = tmp_path / "R_37EN1.TIF"
    _write_cog(cog)
    feed = _dsm_feed((str(cog), _SHEET_RD))

    fetch_dsm(_sheet_request(site), http_get=_feed_get(feed))

    provenance = read_provenance(site / "dsm.tif.provenance.json")
    assert provenance.tool_version
    assert provenance.tool_version[0].isdigit()
    assert provenance.download_started_at.tzinfo is not None


def test_fetch_dsm_uses_a_custom_cache_root(tmp_path: Path) -> None:
    """A supplied cache root is used instead of the in-site default."""
    site = tmp_path / "delft"
    cache_root = tmp_path / "shared-cache"
    cog = tmp_path / "R_37EN1.TIF"
    _write_cog(cog)
    feed = _dsm_feed((str(cog), _SHEET_RD))

    fetch_dsm(
        _sheet_request(site),
        http_get=_feed_get(feed),
        now=_fixed_clock(),
        tool_version="v",
        cache_root=cache_root,
    )

    assert (cache_root / "blobs").is_dir()
    assert not (site / ".cache").exists()


def test_fetch_dsm_without_nodata_records_none(tmp_path: Path) -> None:
    """A COG declaring no nodata records nodata=none and void fraction 0."""
    site = tmp_path / "delft"
    cog = tmp_path / "R_37EN1.TIF"
    _write_cog(cog, nodata=None)
    feed = _dsm_feed((str(cog), _SHEET_RD))

    fetch_dsm(
        _sheet_request(site),
        http_get=_feed_get(feed),
        now=_fixed_clock(),
        tool_version="v",
    )

    keys = dict(
        read_provenance(site / "dsm.tif.provenance.json").request_keys
    )
    assert keys["nodata"] == "none"
    assert float(keys["qa_nodata_fraction"]) == 0.0


def test_fetch_dsm_reports_no_covering_sheet(tmp_path: Path) -> None:
    """An AOI covered by no DSM sheet is a clean DsmError."""
    far: BBox = (100000.0, 400000.0, 100020.0, 400020.0)
    feed = _dsm_feed(("https://pdok/dsm/R_FAR.TIF", far))

    with pytest.raises(DsmError, match="no DSM sheet"):
        fetch_dsm(
            _sheet_request(tmp_path / "s"),
            http_get=_feed_get(feed),
            now=_fixed_clock(),
        )


def test_fetch_dsm_reports_a_multi_sheet_aoi(tmp_path: Path) -> None:
    """An AOI spanning two DSM sheets is a clean, single-sheet-only DsmError."""
    feed = _dsm_feed(
        ("https://pdok/dsm/R_A.TIF", _SHEET_RD),
        ("https://pdok/dsm/R_B.TIF", _SHEET_RD),
    )

    with pytest.raises(DsmError, match="spans 2 DSM sheets"):
        fetch_dsm(
            _sheet_request(tmp_path / "s"),
            http_get=_feed_get(feed),
            now=_fixed_clock(),
        )


def test_fetch_dsm_funnels_an_invalid_feed(tmp_path: Path) -> None:
    """A changed/invalid DSM feed surfaces as a DsmError."""
    with pytest.raises(DsmError):
        fetch_dsm(
            _sheet_request(tmp_path / "s"),
            http_get=_feed_get(b"<<< not a feed"),
            now=_fixed_clock(),
        )


def test_fetch_dsm_funnels_a_feed_request_failure(tmp_path: Path) -> None:
    """A feed HTTP failure surfaces as a DsmError, not a raw requests error."""

    def failing_get(url: str) -> bytes:
        del url
        msg = "503 Server Error"
        raise requests.HTTPError(msg)

    with pytest.raises(DsmError):
        fetch_dsm(
            _sheet_request(tmp_path / "s"),
            http_get=failing_get,
            now=_fixed_clock(),
        )


def test_fetch_dsm_rejects_a_malformed_bbox(tmp_path: Path) -> None:
    """A malformed --bbox funnels through the acquisition error family."""
    request = AcquisitionRequest(
        site_dir=tmp_path / "s",
        selector=AreaSelectorKind.BBOX,
        area="0,0,1",
    )

    with pytest.raises(MalformedBboxError):
        fetch_dsm(request, http_get=_feed_get(b""), now=_fixed_clock())


def test_fetch_dsm_defers_the_city_selector(tmp_path: Path) -> None:
    """The city selector's AOI derivation is a typed deferral, not silent."""
    request = AcquisitionRequest(
        site_dir=tmp_path / "s",
        selector=AreaSelectorKind.CITY,
        area="delft",
    )

    with pytest.raises(SelectorNotWiredError):
        fetch_dsm(request, http_get=_feed_get(b""), now=_fixed_clock())
