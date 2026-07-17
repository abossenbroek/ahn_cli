"""Tests for the Beeldmateriaal orthophoto fetcher.

These build synthetic overlapping RGB GeoTIFF tiles in ``tmp_path`` with rasterio
and a synthetic basisdata.nl HRL GeoJSON tile index in-memory (no network), then
assert:

- dataset selection: the pinned HRL zone, coverage-probed, none-covered failure;
- the mosaic+clip is overlap-free (no source column contributes twice) and its
  seam is bit-identical to a single-image reference;
- downloads are cache-through (a second fetch performs zero network calls);
- each download's SHA-256 is verified against the index's published digest;
- provenance completeness (CC BY 4.0 attribution, vintage, zone, resolution
  tier, extent, checksums);
- expected failures funnel to :class:`AcquisitionError`;
- the mosaic pixel array is byte-deterministic across two runs.
"""

from __future__ import annotations

import hashlib
import json
import threading
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
from ahn_cli.fetch import ortho as ortho_module
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
    OrthoTile,
    OrthoUnavailableError,
    acquire_ortho,
    default_ortho_registry,
    mosaic_and_clip,
    resolve_ortho_tiles,
    select_ortho_dataset,
)
from ahn_cli.provenance import read_provenance

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

# A small AOI on the Dutch national grid (EPSG:28992), 7m x 3m at 1m/px so the
# fixtures stay tiny while exercising a real overlap band.
_AOI: BBox = (194000.0, 443000.0, 194007.0, 443003.0)
_RES = 1.0
_HEIGHT = 3
_START = datetime(2024, 5, 1, 9, 0, tzinfo=timezone.utc)
_FINISH = datetime(2024, 5, 1, 9, 0, 30, tzinfo=timezone.utc)

_FEED_HRL = "https://basisdata.nl.example/links/nationaal/Nederland/BM_HRL2025O_RGB_TIF.json"
_FEED_OTHER = "https://basisdata.nl.example/links/nationaal/Nederland/BM_LRL2025O_RGB.json"
_TILE_BASE = "https://fsn1.your-objectstorage.example/hwh-ortho/2025/"


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


def _feature(
    tile_id: str, rd_bbox: BBox, content: bytes, *, suffix: str = ".tif"
) -> dict[str, object]:
    """Build one GeoJSON feature for ``tile_id`` with a real sha256 digest."""
    minx, miny, maxx, maxy = rd_bbox
    ring = [
        [minx, miny],
        [minx, maxy],
        [maxx, maxy],
        [maxx, miny],
        [minx, miny],
    ]
    return {
        "type": "Feature",
        "properties": {
            "file": f"{_TILE_BASE}{tile_id}{suffix}",
            "size": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        },
        "geometry": {"type": "Polygon", "coordinates": [ring]},
    }


def _hrl_index(tiles: tuple[tuple[str, BBox, bytes], ...]) -> bytes:
    """Build a minimal basisdata.nl HRL GeoJSON tile index for ``tiles``.

    Also emits a ``.tif.aux.xml`` sidecar feature per tile, matching the real
    index's shape, to exercise the non-``.tif`` filtering branch.
    """
    features: list[dict[str, object]] = []
    for tile_id, rd_bbox, content in tiles:
        features.append(_feature(tile_id, rd_bbox, content))
        features.append(
            _feature(tile_id, rd_bbox, b"sidecar", suffix=".tif.aux.xml")
        )
    document = {
        "type": "FeatureCollection",
        "name": "BM_HRL2025O_RGB_TIF",
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:EPSG::28992"},
        },
        "features": features,
    }
    return json.dumps(document).encode()


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


class _ScrambledHttp:
    """Serves the feed normally but finishes tile downloads out of order.

    ``2025_kb_00_hrl`` -- the lexicographically smallest tile id, and the one
    the old serial loop (and any naive as-completed implementation) would
    download and write first -- blocks until ``2025_kb_01_hrl`` has already
    completed, proving the emitted order comes from a sort, not from
    pool-completion order.
    """

    def __init__(self, responses: dict[str, bytes]) -> None:
        self._responses = responses
        self._second_done = threading.Event()
        self.completion_order: list[str] = []
        self._lock = threading.Lock()

    def __call__(self, url: str) -> bytes:
        """Return the feed immediately; scramble the two tiles' completion."""
        if url not in self._responses:
            msg = f"unexpected URL requested: {url}"
            raise AssertionError(msg)
        if not url.endswith(".tif"):
            return self._responses[url]
        if "kb_00" in url:
            assert self._second_done.wait(timeout=5), (
                "the second tile never completed"
            )
        content = self._responses[url]
        with self._lock:
            self.completion_order.append(url)
        if "kb_00" not in url:
            self._second_done.set()
        return content


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
    return (("2025_kb_00_hrl", tile_a), ("2025_kb_01_hrl", tile_b))


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
    vintage: int = 2025,
) -> OrthoDataset:
    """Build a test OrthoDataset pinned to ``feed_url``."""
    return OrthoDataset(
        vintage=Vintage(vintage),
        zone=f"basisdata-{vintage}-{tier}",
        resolution_tier=tier,
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
            vintage=Vintage(2025),
            zone="z",
            resolution_tier="hrl",
            feed_url="   ",
            semantics="s",
        )


def test_ortho_tile_rejects_blank_tile_id() -> None:
    """An OrthoTile with a blank tile_id cannot be constructed."""
    with pytest.raises(ValueError, match="tile_id"):
        OrthoTile(
            tile_id="   ",
            bbox=_AOI,
            download_url=f"{_TILE_BASE}a.tif",
            sha256="a" * 64,
        )


def test_ortho_tile_rejects_blank_download_url() -> None:
    """An OrthoTile with a blank download_url cannot be constructed."""
    with pytest.raises(ValueError, match="download_url"):
        OrthoTile(tile_id="t", bbox=_AOI, download_url="   ", sha256="a" * 64)


def test_ortho_tile_rejects_blank_sha256() -> None:
    """An OrthoTile with a blank sha256 cannot be constructed."""
    with pytest.raises(ValueError, match="sha256"):
        OrthoTile(
            tile_id="t",
            bbox=_AOI,
            download_url=f"{_TILE_BASE}a.tif",
            sha256="   ",
        )


def test_registry_rejects_duplicate_resolution_tier() -> None:
    """Registering two datasets at the same tier is a wiring error."""
    registry = _registry(_dataset(_FEED_HRL, "hrl"))
    with pytest.raises(ValueError, match="hrl"):
        registry.register(_dataset(_FEED_OTHER, "hrl"))


def test_default_registry_is_pinned_to_hrl_2025() -> None:
    """The default registry is pinned to the single 2025 HRL zone."""
    tiers = [dataset.resolution_tier for dataset in default_ortho_registry()]
    assert tiers == ["hrl"]
    for dataset in default_ortho_registry():
        assert dataset.feed_url.strip()
        assert dataset.zone.strip()
        assert dataset.vintage == Vintage(2025)


# --------------------------------------------------------------------------- #
# Vintage / zone selection
# --------------------------------------------------------------------------- #


def test_select_prefers_the_hrl_zone_when_it_covers(tmp_path: Path) -> None:
    """When the pinned HRL feed covers the AOI it is chosen."""
    responses = _write_tiles(tmp_path / "tiles")
    covering = _hrl_index(
        tuple(
            (tid, bbox, responses[f"{_TILE_BASE}{tid}.tif"])
            for tid, bbox in _two_overlapping_tiles()
        )
    )
    http = _RecordingHttp({**responses, _FEED_HRL: covering})
    registry = _registry(_dataset(_FEED_HRL, "hrl"))

    dataset = select_ortho_dataset(_AOI, registry, http)

    assert dataset.resolution_tier == "hrl"


def test_select_falls_back_to_second_zone_when_first_uncovered(
    tmp_path: Path,
) -> None:
    """An empty first feed makes selection fall back to the next zone."""
    responses = _write_tiles(tmp_path / "tiles")
    empty = _hrl_index(())
    covering = _hrl_index(
        tuple(
            (tid, bbox, responses[f"{_TILE_BASE}{tid}.tif"])
            for tid, bbox in _two_overlapping_tiles()
        )
    )
    http = _RecordingHttp(
        {**responses, _FEED_HRL: empty, _FEED_OTHER: covering}
    )
    registry = _registry(
        _dataset(_FEED_HRL, "hrl"), _dataset(_FEED_OTHER, "lrl")
    )

    dataset = select_ortho_dataset(_AOI, registry, http)

    assert dataset.resolution_tier == "lrl"


def test_select_raises_when_no_zone_covers() -> None:
    """When no zone covers the AOI, selection raises OrthoUnavailableError."""
    empty = _hrl_index(())
    http = _RecordingHttp({_FEED_HRL: empty})
    registry = _registry(_dataset(_FEED_HRL, "hrl"))

    with pytest.raises(OrthoUnavailableError):
        select_ortho_dataset(_AOI, registry, http)


def test_select_rejects_a_degenerate_aoi() -> None:
    """A degenerate AOI is rejected before any probing."""
    http = _RecordingHttp({})
    registry = _registry(_dataset(_FEED_HRL, "hrl"))
    with pytest.raises(ValueError, match=r"bbox|min"):
        select_ortho_dataset((1.0, 1.0, 1.0, 2.0), registry, http)


def test_resolve_wraps_a_malformed_feed(tmp_path: Path) -> None:
    """A malformed feed surfaces as the module's OrthoFeedError."""
    _write_tiles(tmp_path / "tiles")
    http = _RecordingHttp({_FEED_HRL: b"not json at all"})
    dataset = _dataset(_FEED_HRL, "hrl")

    with pytest.raises(OrthoFeedError):
        resolve_ortho_tiles(dataset, _AOI, http)


def test_resolve_rejects_a_feed_that_is_not_a_feature_collection(
    tmp_path: Path,
) -> None:
    """A well-formed JSON document that isn't a FeatureCollection is rejected."""
    _write_tiles(tmp_path / "tiles")
    http = _RecordingHttp({_FEED_HRL: json.dumps({"type": "Nope"}).encode()})
    dataset = _dataset(_FEED_HRL, "hrl")

    with pytest.raises(OrthoFeedError, match="FeatureCollection"):
        resolve_ortho_tiles(dataset, _AOI, http)


def test_resolve_rejects_a_covering_entry_missing_sha256(
    tmp_path: Path,
) -> None:
    """A .tif entry covering the AOI with no published sha256 is a feed error."""
    _write_tiles(tmp_path / "tiles")
    document = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "file": f"{_TILE_BASE}bad.tif",
                    "size": 1,
                    "sha256": None,
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [194000.0, 443000.0],
                            [194000.0, 443001.0],
                            [194001.0, 443001.0],
                            [194001.0, 443000.0],
                            [194000.0, 443000.0],
                        ]
                    ],
                },
            }
        ],
    }
    http = _RecordingHttp({_FEED_HRL: json.dumps(document).encode()})
    dataset = _dataset(_FEED_HRL, "hrl")

    with pytest.raises(OrthoFeedError, match="sha256"):
        resolve_ortho_tiles(dataset, _AOI, http)


def test_resolve_ignores_a_non_covering_entry_missing_sha256(
    tmp_path: Path,
) -> None:
    """A .tif entry outside the AOI with no sha256 does not block the fetch.

    The real nationwide index has a handful of rows with a null sha256, none
    of which are anywhere near a given site's AOI -- those rows must not
    block fetches for unrelated sites.
    """
    responses = _write_tiles(tmp_path / "tiles")
    covering = tuple(
        (tid, bbox, responses[f"{_TILE_BASE}{tid}.tif"])
        for tid, bbox in _two_overlapping_tiles()
    )
    far_away_no_sha256 = {
        "type": "Feature",
        "properties": {
            "file": f"{_TILE_BASE}far_away.tif",
            "size": 1,
            "sha256": None,
        },
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [500000.0, 500000.0],
                    [500000.0, 501000.0],
                    [501000.0, 501000.0],
                    [501000.0, 500000.0],
                    [500000.0, 500000.0],
                ]
            ],
        },
    }
    document = json.loads(_hrl_index(covering))
    document["features"].append(far_away_no_sha256)
    http = _RecordingHttp(
        {**responses, _FEED_HRL: json.dumps(document).encode()}
    )
    dataset = _dataset(_FEED_HRL, "hrl")

    resolved = resolve_ortho_tiles(dataset, _AOI, http)

    assert [tile.tile_id for tile in resolved.tiles] == [
        "2025_kb_00_hrl",
        "2025_kb_01_hrl",
    ]


def test_resolve_rejects_a_feed_that_is_not_a_json_object() -> None:
    """Well-formed JSON that isn't an object (e.g. an array) is rejected."""
    http = _RecordingHttp({_FEED_HRL: json.dumps([1, 2, 3]).encode()})
    dataset = _dataset(_FEED_HRL, "hrl")

    with pytest.raises(OrthoFeedError, match="FeatureCollection"):
        resolve_ortho_tiles(dataset, _AOI, http)


def test_resolve_rejects_a_feature_collection_missing_features() -> None:
    """A FeatureCollection with no 'features' array is a feed error."""
    document = {"type": "FeatureCollection"}
    http = _RecordingHttp({_FEED_HRL: json.dumps(document).encode()})
    dataset = _dataset(_FEED_HRL, "hrl")

    with pytest.raises(OrthoFeedError, match="features"):
        resolve_ortho_tiles(dataset, _AOI, http)


def test_resolve_skips_a_feature_that_is_not_an_object() -> None:
    """A features entry that is not a JSON object is silently excluded.

    The nationwide index occasionally has anomalous rows; an unusable row is
    tolerated (not raised) unless it turns out to cover the requested AOI.
    """
    document = {"type": "FeatureCollection", "features": ["not-an-object"]}
    http = _RecordingHttp({_FEED_HRL: json.dumps(document).encode()})
    dataset = _dataset(_FEED_HRL, "hrl")

    resolved = resolve_ortho_tiles(dataset, _AOI, http)

    assert resolved.tiles == ()


def test_resolve_skips_a_feature_missing_properties() -> None:
    """A feature with no properties object is silently excluded."""
    document = {
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "geometry": None}],
    }
    http = _RecordingHttp({_FEED_HRL: json.dumps(document).encode()})
    dataset = _dataset(_FEED_HRL, "hrl")

    resolved = resolve_ortho_tiles(dataset, _AOI, http)

    assert resolved.tiles == ()


def test_resolve_skips_a_feature_missing_file() -> None:
    """A feature whose properties has no 'file' entry is silently excluded."""
    document = {
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "properties": {"size": 1}}],
    }
    http = _RecordingHttp({_FEED_HRL: json.dumps(document).encode()})
    dataset = _dataset(_FEED_HRL, "hrl")

    resolved = resolve_ortho_tiles(dataset, _AOI, http)

    assert resolved.tiles == ()


def test_resolve_skips_a_feature_missing_geometry() -> None:
    """A .tif feature with no geometry object is silently excluded.

    Without a geometry, coverage can't be determined, so the entry is
    dropped rather than assumed relevant.
    """
    document = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "file": f"{_TILE_BASE}bad.tif",
                    "sha256": "a" * 64,
                },
            }
        ],
    }
    http = _RecordingHttp({_FEED_HRL: json.dumps(document).encode()})
    dataset = _dataset(_FEED_HRL, "hrl")

    resolved = resolve_ortho_tiles(dataset, _AOI, http)

    assert resolved.tiles == ()


def test_resolve_skips_a_geometry_with_no_ring_coordinates() -> None:
    """A .tif feature whose geometry has no coordinates is silently excluded."""
    document = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "file": f"{_TILE_BASE}bad.tif",
                    "sha256": "a" * 64,
                },
                "geometry": {"type": "Polygon"},
            }
        ],
    }
    http = _RecordingHttp({_FEED_HRL: json.dumps(document).encode()})
    dataset = _dataset(_FEED_HRL, "hrl")

    resolved = resolve_ortho_tiles(dataset, _AOI, http)

    assert resolved.tiles == ()


def test_resolve_returns_cc_by_terms_and_ordered_tiles(
    tmp_path: Path,
) -> None:
    """Resolution reports the pinned CC BY 4.0 terms and id-ordered RD tiles."""
    responses = _write_tiles(tmp_path / "tiles")
    covering = _hrl_index(
        tuple(
            (tid, bbox, responses[f"{_TILE_BASE}{tid}.tif"])
            for tid, bbox in _two_overlapping_tiles()
        )
    )
    http = _RecordingHttp({**responses, _FEED_HRL: covering})
    dataset = _dataset(_FEED_HRL, "hrl")

    resolved = resolve_ortho_tiles(dataset, _AOI, http)

    assert "by/4.0" in resolved.licence
    assert resolved.attribution.strip()
    assert [tile.tile_id for tile in resolved.tiles] == [
        "2025_kb_00_hrl",
        "2025_kb_01_hrl",
    ]


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

    mosaic = mosaic_and_clip(tile_paths, _AOI, out)

    reference = tmp_path / "reference.tif"
    _write_rgb_tile(reference, _AOI, base_col=0)
    with rasterio.open(out) as produced, rasterio.open(reference) as expected:
        produced_pixels = produced.read()
        expected_pixels = expected.read()
    width = round((_AOI[2] - _AOI[0]) / _RES)
    # (a) exact-cover dimensions: bbox area / px^2.
    assert mosaic.width == width
    assert mosaic.height == _HEIGHT
    assert mosaic.resolution_m == _RES
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
        mosaic_and_clip((path,), _AOI, tmp_path / "ortho.tif")


def test_mosaic_pixel_checksum_is_deterministic(tmp_path: Path) -> None:
    """The mosaic pixel checksum is stable across two independent runs."""
    tiles_dir = tmp_path / "tiles"
    _write_tiles(tiles_dir)
    tile_paths = tuple(sorted(tiles_dir.glob("*.tif")))

    first = mosaic_and_clip(tile_paths, _AOI, tmp_path / "a.tif")
    second = mosaic_and_clip(tile_paths, _AOI, tmp_path / "b.tif")

    assert first.pixel_checksum == second.pixel_checksum
    assert first == second


def test_mosaic_handles_jpeg_ycbcr_source_tiles(tmp_path: Path) -> None:
    """A JPEG/YCbCr source mosaics to a valid, uncompressed RGB GeoTIFF.

    The real Beeldmateriaal HRL tiles are ``photometric=YCbCr`` /
    ``compress=JPEG``. ``rasterio.merge`` copies the first sheet's
    ``photometric`` into the destination profile but not its ``compress``,
    and GDAL rejects ``PHOTOMETRIC=YCBCR`` without ``COMPRESS=JPEG`` -- so the
    windowed write fails unless ``mosaic_and_clip`` pins a clean output
    profile. This fixture reproduces that exact failure (the earlier synthetic
    tests used plain RGB GeoTIFFs and never exercised it). JPEG is lossy, so we
    assert a valid, non-uniform, uncompressed RGB result rather than exact
    pixels.
    """
    # 64x64 @ 0.5 m so the tiled JPEG has full 16x16 blocks to encode.
    res = 0.5
    minx, miny = 194000.0, 443000.0
    size = 64
    maxx, maxy = minx + size * res, miny + size * res
    aoi: BBox = (minx, miny, maxx, maxy)
    ramp = np.tile(np.arange(size, dtype=np.uint8), (size, 1))
    pixels: npt.NDArray[np.uint8] = np.stack(
        [
            ramp,
            ((ramp * 2) % 255).astype(np.uint8),
            (255 - ramp).astype(np.uint8),
        ]
    )
    tile = tmp_path / "2025_kb_00_RGB_JPEG_hrl.tif"
    with rasterio.open(
        tile,
        "w",
        driver="GTiff",
        height=size,
        width=size,
        count=3,
        dtype="uint8",
        crs="EPSG:28992",
        transform=from_bounds(minx, miny, maxx, maxy, size, size),
        photometric="YCbCr",
        compress="JPEG",
        tiled=True,
        blockxsize=16,
        blockysize=16,
    ) as dst:
        dst.write(pixels)

    out = tmp_path / "ortho.tif"
    mosaic = mosaic_and_clip((tile,), aoi, out)

    assert mosaic.width == size
    assert mosaic.height == size
    with rasterio.open(out) as produced:
        assert produced.count == 3
        assert produced.dtypes[0] == "uint8"
        # A valid RGB GeoTIFF with lossless DEFLATE -- never a JPEG/YCbCr
        # round-trip (which would be lossy) and never the invalid
        # YCbCr-without-JPEG pairing that GDAL rejects.
        compress = produced.profile.get("compress")
        assert isinstance(compress, str)
        assert compress.lower() == "deflate"
        assert produced.read().std() > 0  # genuine imagery, not uniform


def test_mosaic_uses_the_windowed_merge_write_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """mosaic_and_clip calls rasterio.merge.merge with ``dst_path``.

    That is the windowed-write overload that streams to disk in bounded
    chunks; the old whole-array-returning call (no ``dst_path``) would
    require holding the entire mosaic in RAM. The guard's signature has no
    default for ``dst_path``, so a call missing it would fail with a
    ``TypeError`` before ever reaching the real ``merge``.
    """
    tiles_dir = tmp_path / "tiles"
    _write_tiles(tiles_dir)
    tile_paths = tuple(sorted(tiles_dir.glob("*.tif")))
    real_merge = ortho_module.merge
    calls: list[Path] = []

    def _guarded_merge(
        sources: Sequence[str],
        *,
        bounds: BBox | None = None,
        res: float | tuple[float, float] | None = None,
        dst_path: Path,
        dst_kwds: dict[str, object] | None = None,
    ) -> None:
        calls.append(dst_path)
        return real_merge(
            sources,
            bounds=bounds,
            res=res,
            dst_path=dst_path,
            dst_kwds=dst_kwds,
        )

    monkeypatch.setattr(ortho_module, "merge", _guarded_merge)

    mosaic_and_clip(tile_paths, _AOI, tmp_path / "ortho.tif")

    assert len(calls) == 1


def test_pixel_checksum_matches_a_whole_load_oracle(tmp_path: Path) -> None:
    """The streamed pixel checksum equals an independent whole-array SHA-256.

    Re-derives the checksum from scratch (a from-scratch reimplementation of
    the retired whole-array algorithm: read every band fully, hash the
    crs/dtype/shape/resolution header then the raw pixel bytes) and asserts
    it matches what the streamed :func:`mosaic_and_clip` recorded.
    """
    tiles_dir = tmp_path / "tiles"
    _write_tiles(tiles_dir)
    tile_paths = tuple(sorted(tiles_dir.glob("*.tif")))
    out = tmp_path / "ortho.tif"

    mosaic = mosaic_and_clip(tile_paths, _AOI, out)

    with rasterio.open(out) as dataset:
        pixels = dataset.read()
        crs = str(dataset.crs)
    header = (
        f"{crs}|{pixels.dtype}|{pixels.shape}|{mosaic.resolution_m}"
    ).encode()
    expected = hashlib.sha256(header)
    expected.update(pixels.tobytes())

    assert mosaic.pixel_checksum == expected.hexdigest()
    assert mosaic.crs == crs


# --------------------------------------------------------------------------- #
# Streaming uniformity check (bounded memory, per-band)
# --------------------------------------------------------------------------- #


def test_uniformity_tracker_treats_an_all_void_block_as_uniform() -> None:
    """A block with no finite values contributes no variation."""
    tracker = getattr(ortho_module, "_UniformityTracker")(band_count=1)  # noqa: B009

    tracker.update(0, np.full((2, 2), np.nan, dtype=np.float32))

    assert tracker.is_uniform is True


def test_uniformity_tracker_ignores_a_repeat_of_the_first_value() -> None:
    """A second block equal to the first band value stays uniform."""
    tracker = getattr(ortho_module, "_UniformityTracker")(band_count=1)  # noqa: B009

    tracker.update(0, np.array([[5, 5]], dtype=np.uint8))
    tracker.update(0, np.array([[5, 5]], dtype=np.uint8))

    assert tracker.is_uniform is True


def test_uniformity_tracker_skips_updates_once_a_band_differs() -> None:
    """Once a band is known non-uniform, later updates for it are a no-op."""
    tracker = getattr(ortho_module, "_UniformityTracker")(band_count=1)  # noqa: B009
    tracker.update(0, np.array([[1, 2]], dtype=np.uint8))
    assert tracker.is_uniform is False

    tracker.update(0, np.array([[9, 9]], dtype=np.uint8))  # short-circuited

    assert tracker.is_uniform is False


# --------------------------------------------------------------------------- #
# End-to-end acquisition
# --------------------------------------------------------------------------- #


def _covering_http(responses: dict[str, bytes]) -> _RecordingHttp:
    """Return a recording http_get serving the tiles plus a covering HRL feed."""
    index = _hrl_index(
        tuple(
            (tid, bbox, responses[f"{_TILE_BASE}{tid}.tif"])
            for tid, bbox in _two_overlapping_tiles()
        )
    )
    merged = {**responses, _FEED_HRL: index}
    return _RecordingHttp(merged)


def _acquire(
    site: Path,
    http: Callable[[str], bytes],
    *,
    clock: Callable[[], datetime] | None = None,
    download_jobs: int = 1,
) -> OrthoAcquisition:
    """Run acquire_ortho with the HRL-covering registry and injected I/O."""
    registry = _registry(_dataset(_FEED_HRL, "hrl"))
    return acquire_ortho(
        _request(site),
        http_get=http,
        now=clock or _fixed_clock(_START, _FINISH),
        cache_root=site / ".cache",
        tool_version="ortho-test",
        registry=registry,
        download_jobs=download_jobs,
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
    assert provenance.source_portal == "basisdata"
    assert "by/4.0" in provenance.licence
    assert provenance.attribution.strip()
    assert provenance.vintage == Vintage(2025)
    assert provenance.zone == "basisdata-2025-hrl"
    assert provenance.resolution_tier == "hrl"
    assert provenance.bbox == _AOI
    assert provenance.output_checksum == result.mosaic.pixel_checksum
    assert provenance.tool_version == "ortho-test"
    keys = dict(provenance.request_keys)
    assert keys["vintage"] == "2025"
    assert keys["resolution_tier"] == "hrl"
    assert keys[
        f"{_TILE_BASE}2025_kb_00_hrl.tif"
    ]  # per-tile checksum recorded


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
        tool_version="ortho-test",
        registry=_registry(_dataset(_FEED_HRL, "hrl")),
        progress=lambda done, total: calls.append((done, total)),
    )

    assert calls == [(1, 2), (2, 2)]


def test_acquire_with_jobs_emits_tile_id_order_despite_scrambled_completion(
    tmp_path: Path,
) -> None:
    """download_jobs>1 always writes sheets in tile-id order.

    The pool completes ``2025_kb_01_hrl`` before ``2025_kb_00_hrl`` here, but
    the written sheets (mosaic order) must still come out sorted ascending by
    tile_id -- proving the result is collected and sorted rather than
    emitted in as-completed order.
    """
    site = tmp_path / "delft"
    responses = _write_tiles(tmp_path / "tiles")
    index = _hrl_index(
        tuple(
            (tid, bbox, responses[f"{_TILE_BASE}{tid}.tif"])
            for tid, bbox in _two_overlapping_tiles()
        )
    )
    http = _ScrambledHttp({**responses, _FEED_HRL: index})

    result = _acquire(site, http, download_jobs=4)

    assert "kb_00" not in http.completion_order[0]
    assert [p.name for p in result.tile_paths] == [
        "2025_kb_00_hrl.tif",
        "2025_kb_01_hrl.tif",
    ]


def test_acquire_is_byte_identical_across_job_counts(tmp_path: Path) -> None:
    """download_jobs=1 and download_jobs=8 write byte-identical output."""
    responses = _write_tiles(tmp_path / "tiles")

    site_serial = tmp_path / "serial"
    result_serial = _acquire(
        site_serial, _covering_http(responses), download_jobs=1
    )

    site_parallel = tmp_path / "parallel"
    result_parallel = _acquire(
        site_parallel, _covering_http(responses), download_jobs=8
    )

    assert (
        result_serial.mosaic_path.read_bytes()
        == result_parallel.mosaic_path.read_bytes()
    )
    assert (
        result_serial.provenance_path.read_bytes()
        == result_parallel.provenance_path.read_bytes()
    )
    assert [p.name for p in result_serial.tile_paths] == [
        p.name for p in result_parallel.tile_paths
    ]
    for serial_path, parallel_path in zip(
        result_serial.tile_paths, result_parallel.tile_paths, strict=True
    ):
        assert serial_path.read_bytes() == parallel_path.read_bytes()


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


def test_acquire_rejects_a_tile_whose_bytes_do_not_match_its_sha256(
    tmp_path: Path,
) -> None:
    """A downloaded tile whose bytes mismatch the feed's sha256 is refused."""
    site = tmp_path / "delft"
    responses = _write_tiles(tmp_path / "tiles")
    tile_id, tile_bbox = _two_overlapping_tiles()[0]
    other_id, other_bbox = _two_overlapping_tiles()[1]
    index = _hrl_index(
        (
            (tile_id, tile_bbox, b"not the real bytes"),
            (other_id, other_bbox, responses[f"{_TILE_BASE}{other_id}.tif"]),
        )
    )
    http = _RecordingHttp({**responses, _FEED_HRL: index})
    registry = _registry(_dataset(_FEED_HRL, "hrl"))

    with pytest.raises(AcquisitionError, match="sha256"):
        acquire_ortho(
            _request(site),
            http_get=http,
            now=_fixed_clock(_START, _FINISH),
            cache_root=site / ".cache",
            tool_version="ortho-test",
            registry=registry,
        )


def test_acquire_funnels_a_download_failure(tmp_path: Path) -> None:
    """A tile download failure is funnelled to AcquisitionError."""
    site = tmp_path / "delft"
    responses = _write_tiles(tmp_path / "tiles")
    index = _hrl_index(
        tuple(
            (tid, bbox, responses[f"{_TILE_BASE}{tid}.tif"])
            for tid, bbox in _two_overlapping_tiles()
        )
    )
    registry = _registry(_dataset(_FEED_HRL, "hrl"))

    with pytest.raises(AcquisitionError):
        acquire_ortho(
            _request(site),
            http_get=_FailingHttp(_FEED_HRL, index),
            now=_fixed_clock(_START, _FINISH),
            cache_root=site / ".cache",
            tool_version="ortho-test",
            registry=registry,
        )


def test_acquire_funnels_no_coverage(tmp_path: Path) -> None:
    """No covering zone funnels to AcquisitionError (not a raw error)."""
    site = tmp_path / "delft"
    empty = _hrl_index(())
    http = _RecordingHttp({_FEED_HRL: empty})
    registry = _registry(_dataset(_FEED_HRL, "hrl"))

    with pytest.raises(AcquisitionError):
        acquire_ortho(
            _request(site),
            http_get=http,
            now=_fixed_clock(_START, _FINISH),
            cache_root=site / ".cache",
            tool_version="ortho-test",
            registry=registry,
        )


def test_acquire_default_registry_funnels_network_failure(
    tmp_path: Path,
) -> None:
    """Omitting the registry uses the pinned default feed.

    A probe over it that cannot reach the network funnels to AcquisitionError.
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
    registry = _registry(_dataset(_FEED_HRL, "hrl"))

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
