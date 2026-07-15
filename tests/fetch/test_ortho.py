"""Tests for the Beeldmateriaal orthophoto fetcher (WP8).

These build synthetic overlapping RGB GeoTIFF tiles in ``tmp_path`` with rasterio
and a synthetic INSPIRE-ATOM feed in-memory (no network), then assert:

- vintage/zone selection: 5cm preferred, 8cm fallback, none-covered failure;
- the mosaic+clip is overlap-free (no source column contributes twice) and its
  seam is bit-identical to a single-image reference;
- downloads are cache-through (a second fetch performs zero network calls);
- provenance completeness (CC-BY attribution, vintage, zone, resolution tier,
  extent, checksums);
- expected failures funnel to :class:`AcquisitionError`;
- the mosaic pixel array is byte-deterministic across two runs.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from importlib.metadata import version
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pytest
import rasterio
import requests
from rasterio.transform import from_bounds

from ahn_cli.domain import BBox, Product, Vintage
from ahn_cli.fetch.acquisition import (
    AcquisitionError,
    AcquisitionRequest,
    AreaSelectorKind,
    aoi_bbox,
)
from ahn_cli.fetch.ortho import (
    OrthoAcquisition,
    OrthoDataset,
    OrthoDatasetRegistry,
    OrthoFeedError,
    OrthoUnavailableError,
    acquire_ortho,
    default_ortho_registry,
    mosaic_and_clip,
    resolve_ortho_tiles,
    select_ortho_dataset,
)
from ahn_cli.fetch.source import to_wgs84
from ahn_cli.provenance import read_provenance

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

# A small AOI on the Dutch national grid (EPSG:28992), 7m x 3m at 1m/px so the
# fixtures stay tiny while exercising a real overlap band.
_AOI: BBox = (194000.0, 443000.0, 194007.0, 443003.0)
_RES = 1.0
_HEIGHT = 3
_START = datetime(2024, 5, 1, 9, 0, tzinfo=timezone.utc)
_FINISH = datetime(2024, 5, 1, 9, 0, 30, tzinfo=timezone.utc)

_FEED_5CM = "https://opendata.beeldmateriaal.example/2023/5cm/atom.xml"
_FEED_8CM = "https://opendata.beeldmateriaal.example/2023/8cm/atom.xml"
_TILE_BASE = "https://opendata.beeldmateriaal.example/2023/8cm/"


def _write_rgb_tile(path: Path, rd_bbox: BBox, base_col: int) -> None:
    """Write a tiny RGB GeoTIFF whose pixel value encodes its global column.

    Pixel value at column ``j`` is ``base_col + j`` on every band and row, so two
    tiles that share global columns agree there and a mosaic can be checked
    against a single-image reference.
    """
    minx, _miny, maxx, _maxy = rd_bbox
    width = round((maxx - minx) / _RES)
    columns = np.arange(base_col, base_col + width, dtype="uint8")
    band: npt.NDArray[np.uint8] = np.broadcast_to(
        columns, (_HEIGHT, width)
    ).astype("uint8")
    pixels: npt.NDArray[np.uint8] = np.stack([band, band, band])
    transform = from_bounds(minx, _miny, maxx, _maxy, width, _HEIGHT)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=_HEIGHT,
        width=width,
        count=3,
        dtype="uint8",
        crs="EPSG:28992",
        transform=transform,
    ) as dst:
        dst.write(pixels)


def _atom_feed(tiles: tuple[tuple[str, BBox], ...]) -> bytes:
    """Build a minimal INSPIRE-ATOM feed (CC-BY) with one section link per tile.

    Each tile's WGS84 ``bbox`` attribute is derived from its EPSG:28992 extent so
    the covering test intersects a real Dutch-grid AOI.
    """
    links: list[str] = []
    for tile_id, rd_bbox in tiles:
        minlon, minlat, maxlon, maxlat = to_wgs84(rd_bbox)
        href = f"{_TILE_BASE}{tile_id}.tif"
        links.append(
            f'<link rel="section" href="{href}" '
            f'bbox="{minlon} {minlat} {maxlon} {maxlat}"/>'
        )
    body = "".join(links)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<feed xmlns="http://www.w3.org/2005/Atom">'
        "<rights>https://creativecommons.org/licenses/by/4.0/</rights>"
        "<author><name>Beeldmateriaal Nederland (CC BY 4.0)</name></author>"
        f"{body}"
        "</feed>"
    ).encode()


class _RecordingHttp:
    """An injected ``http_get`` over an in-memory URL map that counts calls."""

    def __init__(self, responses: dict[str, bytes]) -> None:
        self._responses = responses
        self.calls: list[str] = []

    def __call__(self, url: str) -> bytes:
        self.calls.append(url)
        if url not in self._responses:
            msg = f"unexpected URL requested: {url}"
            raise AssertionError(msg)
        return self._responses[url]


class _FailingHttp:
    """An injected ``http_get`` that serves the feed but fails tile downloads."""

    def __init__(self, feed_url: str, feed: bytes) -> None:
        self._feed_url = feed_url
        self._feed = feed

    def __call__(self, url: str) -> bytes:
        if url == self._feed_url:
            return self._feed
        msg = "tile server is unreachable"
        raise requests.ConnectionError(msg)


def _fixed_clock(*values: datetime) -> Callable[[], datetime]:
    """Return a callable yielding ``values`` in order, one per call."""
    iterator = iter(values)
    return lambda: next(iterator)


def _two_overlapping_tiles() -> tuple[tuple[str, BBox], ...]:
    """Two edge-overlapping tiles covering the AOI, sharing one global column."""
    tile_a: BBox = (194000.0, 443000.0, 194004.0, 443003.0)
    tile_b: BBox = (194003.0, 443000.0, 194007.0, 443003.0)
    return (("kb_00", tile_a), ("kb_01", tile_b))


def _write_tiles(tiles_dir: Path) -> dict[str, bytes]:
    """Write the two overlapping tiles and return a url->bytes response map."""
    tiles_dir.mkdir(parents=True, exist_ok=True)
    responses: dict[str, bytes] = {}
    for tile_id, rd_bbox in _two_overlapping_tiles():
        base_col = round((rd_bbox[0] - _AOI[0]) / _RES)
        path = tiles_dir / f"{tile_id}.tif"
        _write_rgb_tile(path, rd_bbox, base_col)
        responses[f"{_TILE_BASE}{tile_id}.tif"] = path.read_bytes()
    return responses


def _write_uniform_tiles(tiles_dir: Path) -> dict[str, bytes]:
    """Write single-colour stand-ins for the two tiles (placeholder imagery).

    Same tile ids and URLs as :func:`_write_tiles`, but every pixel carries
    one identical value, so the clipped mosaic trips the authenticity gate.
    """
    tiles_dir.mkdir(parents=True, exist_ok=True)
    responses: dict[str, bytes] = {}
    for tile_id, rd_bbox in _two_overlapping_tiles():
        minx, miny, maxx, maxy = rd_bbox
        width = round((maxx - minx) / _RES)
        band: npt.NDArray[np.uint8] = np.full(
            (_HEIGHT, width), 7, dtype=np.uint8
        )
        pixels: npt.NDArray[np.uint8] = np.stack([band, band, band])
        transform = from_bounds(minx, miny, maxx, maxy, width, _HEIGHT)
        path = tiles_dir / f"{tile_id}.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=_HEIGHT,
            width=width,
            count=3,
            dtype="uint8",
            crs="EPSG:28992",
            transform=transform,
        ) as dst:
            dst.write(pixels)
        responses[f"{_TILE_BASE}{tile_id}.tif"] = path.read_bytes()
    return responses


def _dataset(
    feed_url: str,
    tier: str,
    *,
    vintage: int = 2023,
) -> OrthoDataset:
    """Build a test OrthoDataset pinned to ``feed_url`` at 1m/px (test scale)."""
    return OrthoDataset(
        vintage=Vintage(vintage),
        zone=f"beeldmateriaal-{tier}",
        resolution_tier=tier,
        resolution_m=_RES,
        feed_url=feed_url,
        semantics=f"Beeldmateriaal RGB {tier} orthophoto, {vintage}.",
    )


def _registry(*datasets: OrthoDataset) -> OrthoDatasetRegistry:
    """Assemble an ortho registry from ``datasets`` in preference order."""
    registry = OrthoDatasetRegistry()
    for dataset in datasets:
        registry.register(dataset)
    return registry


def _request(site: Path) -> AcquisitionRequest:
    """Build a bbox acquisition request for the shared AOI into ``site``."""
    minx, miny, maxx, maxy = _AOI
    return AcquisitionRequest(
        site_dir=site,
        selector=AreaSelectorKind.BBOX,
        area=f"{minx},{miny},{maxx},{maxy}",
    )


# --------------------------------------------------------------------------- #
# Value-object and error contracts
# --------------------------------------------------------------------------- #


def test_ortho_feed_error_is_a_value_error() -> None:
    """The feed error subclasses ValueError so callers may catch broadly."""
    assert issubclass(OrthoFeedError, ValueError)


def test_ortho_dataset_rejects_blank_feed_url() -> None:
    """A dataset with a blank feed URL cannot be constructed."""
    with pytest.raises(ValueError, match="feed_url"):
        OrthoDataset(
            vintage=Vintage(2023),
            zone="z",
            resolution_tier="5cm",
            resolution_m=0.05,
            feed_url="   ",
            semantics="s",
        )


def test_ortho_dataset_rejects_non_positive_resolution() -> None:
    """A dataset with a non-positive pixel size cannot be constructed."""
    with pytest.raises(ValueError, match="resolution_m"):
        OrthoDataset(
            vintage=Vintage(2023),
            zone="z",
            resolution_tier="5cm",
            resolution_m=0.0,
            feed_url=_FEED_5CM,
            semantics="s",
        )


def test_registry_rejects_duplicate_resolution_tier() -> None:
    """Registering two datasets at the same tier is a wiring error."""
    registry = _registry(_dataset(_FEED_5CM, "5cm"))
    with pytest.raises(ValueError, match="5cm"):
        registry.register(_dataset(_FEED_8CM, "5cm"))


def test_default_registry_prefers_5cm_then_8cm() -> None:
    """The default registry is pinned, preference-ordered 5cm before 8cm."""
    tiers = [dataset.resolution_tier for dataset in default_ortho_registry()]
    assert tiers == ["5cm", "8cm"]
    for dataset in default_ortho_registry():
        assert dataset.feed_url.strip()
        assert dataset.resolution_m > 0
        assert dataset.zone.strip()


# --------------------------------------------------------------------------- #
# Vintage / zone selection
# --------------------------------------------------------------------------- #


def test_select_prefers_the_5cm_zone_when_it_covers(tmp_path: Path) -> None:
    """When the 5cm feed covers the AOI it is chosen over the 8cm fallback."""
    responses = _write_tiles(tmp_path / "tiles")
    covering = _atom_feed(_two_overlapping_tiles())
    http = _RecordingHttp(
        {**responses, _FEED_5CM: covering, _FEED_8CM: covering}
    )
    registry = _registry(
        _dataset(_FEED_5CM, "5cm"), _dataset(_FEED_8CM, "8cm")
    )

    dataset = select_ortho_dataset(_AOI, registry, http)

    assert dataset.resolution_tier == "5cm"


def test_select_falls_back_to_8cm_when_5cm_uncovered(tmp_path: Path) -> None:
    """An empty 5cm feed makes selection fall back to the 8cm zone."""
    responses = _write_tiles(tmp_path / "tiles")
    empty = _atom_feed(())
    covering = _atom_feed(_two_overlapping_tiles())
    http = _RecordingHttp(
        {**responses, _FEED_5CM: empty, _FEED_8CM: covering}
    )
    registry = _registry(
        _dataset(_FEED_5CM, "5cm"), _dataset(_FEED_8CM, "8cm")
    )

    dataset = select_ortho_dataset(_AOI, registry, http)

    assert dataset.resolution_tier == "8cm"


def test_select_raises_when_no_zone_covers() -> None:
    """When no zone covers the AOI, selection raises OrthoUnavailableError."""
    empty = _atom_feed(())
    http = _RecordingHttp({_FEED_5CM: empty, _FEED_8CM: empty})
    registry = _registry(
        _dataset(_FEED_5CM, "5cm"), _dataset(_FEED_8CM, "8cm")
    )

    with pytest.raises(OrthoUnavailableError):
        select_ortho_dataset(_AOI, registry, http)


def test_select_rejects_a_degenerate_aoi() -> None:
    """A degenerate AOI is rejected before any probing."""
    http = _RecordingHttp({})
    registry = _registry(_dataset(_FEED_5CM, "5cm"))
    with pytest.raises(ValueError, match=r"bbox|min"):
        select_ortho_dataset((1.0, 1.0, 1.0, 2.0), registry, http)


def test_resolve_wraps_a_malformed_feed(tmp_path: Path) -> None:
    """A malformed feed surfaces as the module's OrthoFeedError."""
    _write_tiles(tmp_path / "tiles")
    http = _RecordingHttp({_FEED_5CM: b"not xml at all"})
    dataset = _dataset(_FEED_5CM, "5cm")

    with pytest.raises(OrthoFeedError):
        resolve_ortho_tiles(dataset, _AOI, http)


def test_resolve_returns_cc_by_terms_and_ordered_tiles(
    tmp_path: Path,
) -> None:
    """Resolution reports the feed's CC-BY terms and id-ordered RD tiles."""
    responses = _write_tiles(tmp_path / "tiles")
    covering = _atom_feed(_two_overlapping_tiles())
    http = _RecordingHttp({**responses, _FEED_5CM: covering})
    dataset = _dataset(_FEED_5CM, "5cm")

    resolved = resolve_ortho_tiles(dataset, _AOI, http)

    assert "by/4.0" in resolved.licence
    assert resolved.attribution.strip()
    assert [tile.tile_id for tile in resolved.tiles] == ["kb_00", "kb_01"]


# --------------------------------------------------------------------------- #
# Mosaic + clip
# --------------------------------------------------------------------------- #


def test_mosaic_is_overlap_free_and_seam_matches_reference(
    tmp_path: Path,
) -> None:
    """Overlapping tiles mosaic without double-counting; the seam is exact."""
    tiles_dir = tmp_path / "tiles"
    _write_tiles(tiles_dir)
    tile_paths = tuple(sorted(tiles_dir.glob("*.tif")))
    out = tmp_path / "ortho.tif"

    mosaic = mosaic_and_clip(tile_paths, _AOI, _RES, out)

    reference = tmp_path / "reference.tif"
    _write_rgb_tile(reference, _AOI, base_col=0)
    with rasterio.open(out) as produced, rasterio.open(reference) as expected:
        produced_pixels = produced.read()
        expected_pixels = expected.read()
    width = round((_AOI[2] - _AOI[0]) / _RES)
    # (a) exact-cover dimensions: bbox area / px^2.
    assert mosaic.width == width
    assert mosaic.height == _HEIGHT
    # (b) no column contributes twice: total < sum of per-tile columns (4 + 4).
    per_tile_columns = 4 + 4
    assert width < per_tile_columns
    # (c) seam pixels are bit-identical to a single-image reference.
    assert np.array_equal(produced_pixels, expected_pixels)


def test_mosaic_rejects_a_uniform_single_colour(tmp_path: Path) -> None:
    """A single-colour clipped mosaic is refused as placeholder imagery."""
    path = tmp_path / "uniform.tif"
    width = round((_AOI[2] - _AOI[0]) / _RES)
    band: npt.NDArray[np.uint8] = np.full((_HEIGHT, width), 7, dtype=np.uint8)
    pixels: npt.NDArray[np.uint8] = np.stack([band, band, band])
    transform = from_bounds(
        _AOI[0], _AOI[1], _AOI[2], _AOI[3], width, _HEIGHT
    )
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=_HEIGHT,
        width=width,
        count=3,
        dtype="uint8",
        crs="EPSG:28992",
        transform=transform,
    ) as dst:
        dst.write(pixels)

    with pytest.raises(AcquisitionError, match="uniform colour"):
        mosaic_and_clip((path,), _AOI, _RES, tmp_path / "ortho.tif")


def test_mosaic_pixel_checksum_is_deterministic(tmp_path: Path) -> None:
    """The mosaic pixel checksum is stable across two independent runs."""
    tiles_dir = tmp_path / "tiles"
    _write_tiles(tiles_dir)
    tile_paths = tuple(sorted(tiles_dir.glob("*.tif")))

    first = mosaic_and_clip(tile_paths, _AOI, _RES, tmp_path / "a.tif")
    second = mosaic_and_clip(tile_paths, _AOI, _RES, tmp_path / "b.tif")

    assert first.pixel_checksum == second.pixel_checksum
    assert first == second


# --------------------------------------------------------------------------- #
# End-to-end acquisition
# --------------------------------------------------------------------------- #


def _covering_http(responses: dict[str, bytes]) -> _RecordingHttp:
    """Return a recording http_get serving the tiles plus a 5cm feed."""
    merged = {**responses, _FEED_5CM: _atom_feed(_two_overlapping_tiles())}
    return _RecordingHttp(merged)


def _acquire(
    site: Path,
    http: _RecordingHttp,
    *,
    clock: Callable[[], datetime] | None = None,
) -> OrthoAcquisition:
    """Run acquire_ortho with the 5cm-covering registry and injected I/O."""
    registry = _registry(_dataset(_FEED_5CM, "5cm"))
    return acquire_ortho(
        _request(site),
        http_get=http,
        now=clock or _fixed_clock(_START, _FINISH),
        cache_root=site / ".cache",
        tool_version="wp8-test",
        registry=registry,
    )


def test_acquire_writes_mosaic_and_provenance(tmp_path: Path) -> None:
    """A full run writes ortho.tif and a complete CC-BY provenance sidecar."""
    site = tmp_path / "delft"
    responses = _write_tiles(tmp_path / "tiles")
    result = _acquire(site, _covering_http(responses))

    assert result.mosaic_path == site / "ortho" / "ortho.tif"
    assert result.mosaic_path.exists()

    provenance = read_provenance(result.provenance_path)
    assert provenance == result.provenance
    assert provenance.product is Product.ORTHO
    assert provenance.source_portal == "beeldmateriaal"
    assert "by/4.0" in provenance.licence
    assert provenance.attribution.strip()
    assert provenance.vintage == Vintage(2023)
    assert provenance.zone == "beeldmateriaal-5cm"
    assert provenance.resolution_tier == "5cm"
    assert provenance.bbox == _AOI
    assert provenance.output_checksum == result.mosaic.pixel_checksum
    assert provenance.tool_version == "wp8-test"
    keys = dict(provenance.request_keys)
    assert keys["vintage"] == "2023"
    assert keys["resolution_tier"] == "5cm"
    assert keys[f"{_TILE_BASE}kb_00.tif"]  # per-tile input checksum recorded


def test_acquire_reports_progress_per_tile(tmp_path: Path) -> None:
    """Each downloaded sheet reports (tiles_done, total_tiles) to progress."""
    site = tmp_path / "delft"
    responses = _write_tiles(tmp_path / "tiles")
    calls: list[tuple[int, int]] = []

    acquire_ortho(
        _request(site),
        http_get=_covering_http(responses),
        now=_fixed_clock(_START, _FINISH),
        cache_root=site / ".cache",
        tool_version="wp8-test",
        registry=_registry(_dataset(_FEED_5CM, "5cm")),
        progress=lambda done, total: calls.append((done, total)),
    )

    assert calls == [(1, 2), (2, 2)]


def test_acquire_records_the_input_checksum_over_tiles(
    tmp_path: Path,
) -> None:
    """The input checksum is a stable digest over the per-tile checksums."""
    site = tmp_path / "delft"
    responses = _write_tiles(tmp_path / "tiles")
    result = _acquire(site, _covering_http(responses))

    tile_hashes = sorted(
        hashlib.sha256(data).hexdigest() for data in responses.values()
    )
    expected = hashlib.sha256("\n".join(tile_hashes).encode()).hexdigest()
    assert result.provenance.input_checksum == expected


def test_acquire_second_run_is_cache_through(tmp_path: Path) -> None:
    """A second acquisition performs zero tile downloads (cache hit)."""
    site = tmp_path / "delft"
    responses = _write_tiles(tmp_path / "tiles")

    http_first = _covering_http(responses)
    _acquire(site, http_first)
    tile_calls_first = [c for c in http_first.calls if c.endswith(".tif")]
    assert tile_calls_first  # first run did download the tiles

    http_second = _covering_http(responses)
    _acquire(site, http_second)
    tile_calls_second = [c for c in http_second.calls if c.endswith(".tif")]
    assert tile_calls_second == []


def test_acquire_is_byte_deterministic(tmp_path: Path) -> None:
    """Two runs into distinct sites yield identical provenance sidecar bytes."""
    responses = _write_tiles(tmp_path / "tiles")
    site_a = tmp_path / "a"
    site_b = tmp_path / "b"

    first = _acquire(site_a, _covering_http(responses))
    second = _acquire(site_b, _covering_http(responses))

    assert (
        first.provenance_path.read_bytes()
        == second.provenance_path.read_bytes()
    )


def test_acquire_recovers_after_uniform_sheets_poisoned_the_cache(
    tmp_path: Path,
) -> None:
    """Gate-rejected sheets are evicted; a retry re-downloads and succeeds."""
    site = tmp_path / "delft"
    uniform = _write_uniform_tiles(tmp_path / "uniform")

    with pytest.raises(AcquisitionError, match="uniform colour"):
        _acquire(site, _covering_http(uniform))

    genuine = _write_tiles(tmp_path / "tiles")
    http = _covering_http(genuine)
    result = _acquire(site, http)

    # Both poisoned sheets were re-downloaded, not replayed from the cache.
    tile_calls = [call for call in http.calls if call.endswith(".tif")]
    assert len(tile_calls) == 2
    assert result.mosaic_path.exists()


def test_acquire_funnels_a_download_failure(tmp_path: Path) -> None:
    """A tile download failure is funnelled to AcquisitionError."""
    site = tmp_path / "delft"
    _write_tiles(tmp_path / "tiles")
    covering = _atom_feed(_two_overlapping_tiles())
    registry = _registry(_dataset(_FEED_5CM, "5cm"))

    with pytest.raises(AcquisitionError):
        acquire_ortho(
            _request(site),
            http_get=_FailingHttp(_FEED_5CM, covering),
            now=_fixed_clock(_START, _FINISH),
            cache_root=site / ".cache",
            tool_version="wp8-test",
            registry=registry,
        )


def test_acquire_funnels_no_coverage(tmp_path: Path) -> None:
    """No covering zone funnels to AcquisitionError (not a raw error)."""
    site = tmp_path / "delft"
    empty = _atom_feed(())
    http = _RecordingHttp({_FEED_5CM: empty})
    registry = _registry(_dataset(_FEED_5CM, "5cm"))

    with pytest.raises(AcquisitionError):
        acquire_ortho(
            _request(site),
            http_get=http,
            now=_fixed_clock(_START, _FINISH),
            cache_root=site / ".cache",
            tool_version="wp8-test",
            registry=registry,
        )


def test_acquire_default_registry_funnels_network_failure(
    tmp_path: Path,
) -> None:
    """Omitting the registry uses the pinned default feeds.

    A probe over them that cannot reach the network funnels to AcquisitionError.
    """
    site = tmp_path / "delft"

    def offline(url: str) -> bytes:
        msg = f"no network in test for {url}"
        raise requests.ConnectionError(msg)

    with pytest.raises(AcquisitionError):
        acquire_ortho(
            _request(site),
            http_get=offline,
            now=_fixed_clock(_START, _FINISH),
            cache_root=site / ".cache",
        )


def test_acquire_uses_default_clock_version_and_cache_root(
    tmp_path: Path,
) -> None:
    """Omitting now/tool_version/cache_root falls back to the live defaults.

    Uses live UTC, the package version and the in-site cache, and still produces
    a mosaic and provenance.
    """
    site = tmp_path / "delft"
    responses = _write_tiles(tmp_path / "tiles")
    registry = _registry(_dataset(_FEED_5CM, "5cm"))

    result = acquire_ortho(
        _request(site),
        http_get=_covering_http(responses),
        registry=registry,
    )

    assert result.mosaic_path.exists()
    assert result.provenance.tool_version == version("ahn_cli")
    started = result.provenance.download_started_at
    finished = result.provenance.download_finished_at
    assert started.tzinfo is not None
    assert finished >= started
    assert (site / ".cache").exists()


def test_aoi_bbox_is_shared_with_ortho(tmp_path: Path) -> None:
    """acquire_ortho reuses the shared AOI derivation from acquisition."""
    request = _request(tmp_path / "delft")
    assert aoi_bbox(request) == _AOI
