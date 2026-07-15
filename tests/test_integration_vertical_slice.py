"""End-to-end vertical-slice acceptance test for the 7rad data-acquisition epic.

This is the epic's capstone: it drives a *realistic, fully offline* fetch of
every product into one site directory, then runs the prep transform and the two
raster export commands, and asserts the finished site holds every deliverable
the spec enumerates:

    data/<site>/
      ahn/    ortho/    viirs/          (the fetch product layout)
      dsm.tif                           (WP7 windowed DSM clip)
      ortho/ortho.tif + CC-BY sidecar   (WP8 mosaic)
      viirs/<name>.tif + sidecar        (WP9 import)
      pointcloud.laz / pointcloud.ply   (WP10/13 prep dedup + PLY export)
      positions.exr                     (WP12 DSM -> position map)
      provenance.json                   (WP14 prep lineage, schema-valid)

Every network boundary is injected with an in-memory fake serving synthetic but
*valid* LAZ / GeoTIFF bytes, so the slice never touches the network yet exercises
the real production code paths of each bounded context.
"""

from __future__ import annotations

import hashlib
import io
import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import laspy
import numpy as np
import numpy.typing as npt
import rasterio
from pyproj import Transformer
from rasterio.transform import from_bounds

from ahn_cli.domain import BBox, Vintage
from ahn_cli.fetch.acquisition import (
    AcquisitionRequest,
    AreaSelectorKind,
    acquire,
)
from ahn_cli.fetch.dsm import fetch_dsm, read_dsm_window
from ahn_cli.fetch.ortho import (
    OrthoDataset,
    OrthoDatasetRegistry,
    acquire_ortho,
)
from ahn_cli.fetch.viirs import import_viirs
from ahn_cli.prep.decimate import VoxelThinning
from ahn_cli.prep.positions import export_positions
from ahn_cli.prep.transform import PrepRequest, prepare
from ahn_cli.provenance import read_provenance

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

_TO_WGS84 = Transformer.from_crs("EPSG:28992", "EPSG:4326", always_xy=True)

# A 20 m sheet on the Dutch grid and a 10 m AOI fully inside it.
_SHEET_RD: BBox = (194000.0, 443000.0, 194020.0, 443020.0)
_AOI_RD: BBox = (194005.0, 443005.0, 194015.0, 443015.0)
_AOI_STR = "194005.0,443005.0,194015.0,443015.0"

_T0 = datetime(2026, 7, 10, 9, 0, tzinfo=timezone.utc)
_T1 = datetime(2026, 7, 10, 9, 0, 5, tzinfo=timezone.utc)

_AHN_HREF = "https://pdok.example/ahn/C_37EN1.LAZ"
_DSM_HREF = "https://pdok.example/dsm/R_37EN1.tif"
_ORTHO_FEED_HRL = "https://basisdata.nl.example/links/nationaal/Nederland/BM_HRL2025O_RGB_TIF.json"
_ORTHO_TILE_HREF = "https://fsn1.your-objectstorage.example/hwh-ortho/2025/2025_kb_00_hrl.tif"


def _clock() -> Callable[[], datetime]:
    """Return a two-value fixed clock (start, finish) for deterministic runs."""
    values = iter((_T0, _T1))
    return lambda: next(values)


def _wgs84(rd_bbox: BBox) -> BBox:
    """Project an EPSG:28992 box to a WGS84 (minlon, minlat, maxlon, maxlat)."""
    minlon, minlat = _TO_WGS84.transform(rd_bbox[0], rd_bbox[1])
    maxlon, maxlat = _TO_WGS84.transform(rd_bbox[2], rd_bbox[3])
    return (minlon, minlat, maxlon, maxlat)


def _atom_feed(
    href: str,
    rd_bbox: BBox,
    *,
    licence: str,
    author: str,
) -> bytes:
    """Build a minimal INSPIRE-ATOM feed with one covering section link."""
    minlon, minlat, maxlon, maxlat = _wgs84(rd_bbox)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<feed xmlns="http://www.w3.org/2005/Atom">'
        f"<rights>{licence}</rights>"
        f"<author><name>{author}</name></author>"
        f'<entry><link rel="section" href="{href}" '
        f'bbox="{minlon} {minlat} {maxlon} {maxlat}"/></entry>'
        "</feed>"
    ).encode()


def _ortho_index(href: str, rd_bbox: BBox, content: bytes) -> bytes:
    """Build a minimal basisdata.nl HRL GeoJSON tile index with one feature."""
    minx, miny, maxx, maxy = rd_bbox
    ring = [
        [minx, miny],
        [minx, maxy],
        [maxx, maxy],
        [maxx, miny],
        [minx, miny],
    ]
    document = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "file": href,
                    "size": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                },
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            }
        ],
    }
    return json.dumps(document).encode()


def _ahn_laz_bytes(tmp_path: Path) -> bytes:
    """Build a valid format-6 AHN LAZ with points inside the sheet extent."""
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.offsets = np.array([194000.0, 443000.0, 0.0], dtype=float)
    header.scales = np.array([0.01, 0.01, 0.01], dtype=float)
    las = laspy.LasData(header)
    points = np.array(
        [
            (194008.0, 443008.0, 1.0, 1.0, 2),
            (194010.0, 443010.0, 2.0, 2.0, 6),
            (194012.0, 443012.0, 3.0, 3.0, 2),
        ],
        dtype=float,
    )
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]
    las.gps_time = points[:, 3]
    las.classification = points[:, 4].astype(np.uint8)
    laz_path = tmp_path / "source.laz"
    las.write(str(laz_path))
    return laz_path.read_bytes()


def _dsm_cog_bytes(tmp_path: Path) -> bytes:
    """Build a valid EPSG:28992 DSM COG over the sheet with a nodata void."""
    width = height = 40
    transform = from_bounds(*_SHEET_RD, width, height)
    pixels: npt.NDArray[np.float32] = np.full(
        (1, height, width), 10.0, dtype=np.float32
    )
    pixels[0, 10:12, 10:12] = -9999.0
    pixels[0, 20, 20] = 12.5  # genuine relief inside the AOI window
    cog = tmp_path / "dsm_sheet.tif"
    with rasterio.open(
        cog,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs="EPSG:28992",
        transform=transform,
        nodata=-9999.0,
    ) as dataset:
        dataset.write(pixels)
    return cog.read_bytes()


def _ortho_tile_bytes(tmp_path: Path) -> bytes:
    """Build a valid EPSG:28992 RGB ortho sheet covering the AOI."""
    width = height = 20
    transform = from_bounds(*_SHEET_RD, width, height)
    band: npt.NDArray[np.uint8] = np.arange(width, dtype="uint8")
    plane = np.broadcast_to(band, (height, width)).astype("uint8")
    pixels: npt.NDArray[np.uint8] = np.stack([plane, plane, plane])
    tile = tmp_path / "ortho_sheet.tif"
    with rasterio.open(
        tile,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype="uint8",
        crs="EPSG:28992",
        transform=transform,
    ) as dataset:
        dataset.write(pixels)
    return tile.read_bytes()


def _viirs_source(tmp_path: Path) -> Path:
    """Write a small WGS84 VIIRS-like GeoTIFF to import untouched."""
    path = tmp_path / "viirs_lights.tif"
    transform = from_bounds(4.0, 51.0, 5.0, 52.0, 4, 4)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dataset:
        dataset.write(np.arange(16, dtype="float32").reshape(1, 4, 4))
    return path


def _request(site: Path) -> AcquisitionRequest:
    """Build a bbox acquisition request into ``site``."""
    return AcquisitionRequest(
        site_dir=site,
        selector=AreaSelectorKind.BBOX,
        area=_AOI_STR,
    )


def _ortho_registry() -> OrthoDatasetRegistry:
    """Return a single-zone ortho registry (tiny test mosaic, 1 m/px sheet)."""
    registry = OrthoDatasetRegistry()
    registry.register(
        OrthoDataset(
            vintage=Vintage(2025),
            zone="basisdata-2025-hrl",
            resolution_tier="hrl",
            feed_url=_ORTHO_FEED_HRL,
            semantics="Beeldmateriaal RGB HRL test zone.",
        )
    )
    return registry


def test_vertical_slice_assembles_every_deliverable(tmp_path: Path) -> None:
    """Fetch (offline) -> prep -> exports leaves a fully populated site."""
    site = tmp_path / "delft"
    laz_bytes = _ahn_laz_bytes(tmp_path)
    dsm_bytes = _dsm_cog_bytes(tmp_path)
    ortho_bytes = _ortho_tile_bytes(tmp_path)

    ahn_feed = _atom_feed(
        _AHN_HREF,
        _SHEET_RD,
        licence="https://creativecommons.org/publicdomain/zero/1.0/deed.nl",
        author="Rijkswaterstaat",
    )

    def ahn_http(url: str) -> bytes:
        return laz_bytes if url.endswith(".LAZ") else ahn_feed

    acquire(
        _request(site),
        http_get=ahn_http,
        now=_clock(),
        tool_version="wp14",
    )

    dsm_feed = _atom_feed(
        _DSM_HREF,
        _SHEET_RD,
        licence="https://creativecommons.org/licenses/by/4.0/",
        author="Rijkswaterstaat",
    )
    dsm_cog = tmp_path / "served_dsm.tif"
    dsm_cog.write_bytes(dsm_bytes)

    def dsm_reader(url: str, aoi: BBox) -> bytes:
        del url
        return read_dsm_window(str(dsm_cog), aoi)

    fetch_dsm(
        _request(site),
        http_get=lambda _url: dsm_feed,
        reader=dsm_reader,
        now=_clock(),
        tool_version="wp14",
    )

    ortho_index = _ortho_index(_ORTHO_TILE_HREF, _SHEET_RD, ortho_bytes)

    def ortho_http(url: str) -> bytes:
        return ortho_bytes if url.endswith(".tif") else ortho_index

    acquire_ortho(
        _request(site),
        http_get=ortho_http,
        now=_clock(),
        tool_version="wp14",
        registry=_ortho_registry(),
    )

    import_viirs(_viirs_source(tmp_path), site, clock=_clock())

    prepare(
        PrepRequest(
            data_dir=site,
            include_classes=(2, 6),
            export_points=True,
            thinning=VoxelThinning(grade=1),
        )
    )

    export_positions(site / "dsm.tif", site / "positions.exr")

    # --- the fetch product layout ---------------------------------------- #
    assert (site / "ahn").is_dir()
    assert (site / "ortho").is_dir()
    assert (site / "viirs").is_dir()
    assert list((site / "ahn").glob("*.LAZ"))
    assert (site / "viirs" / "viirs_lights.tif").is_file()

    # --- rasters --------------------------------------------------------- #
    assert (site / "dsm.tif").is_file()
    assert (site / "ortho" / "ortho.tif").is_file()
    exr = (site / "positions.exr").read_bytes()
    assert exr.startswith(b"\x76\x2f\x31\x01")  # OpenEXR magic

    # --- prep point-cloud deliverables ----------------------------------- #
    assert (site / "pointcloud.laz").is_file()
    assert (site / "pointcloud.ply").read_bytes().startswith(b"ply\n")

    # --- provenance: schema-valid site record + CC-BY present ------------ #
    prep_provenance = read_provenance(site / "provenance.json")
    assert prep_provenance.source_portal == "pdok"
    assert dict(prep_provenance.request_keys)["stage"] == "prep"

    ortho_provenance = read_provenance(
        site / "ortho" / "ortho.tif.provenance.json"
    )
    licence_and_credit = (
        ortho_provenance.licence + " " + ortho_provenance.attribution
    ).lower()
    assert (
        "licenses/by" in licence_and_credit or "cc by" in licence_and_credit
    )


def test_ahn_laz_fixture_round_trips(tmp_path: Path) -> None:
    """The synthetic AHN LAZ fixture is a valid, readable format-6 cloud."""
    data = _ahn_laz_bytes(tmp_path)

    with laspy.open(io.BytesIO(data)) as reader:
        cloud = reader.read()

    assert len(cloud.points) == 3
    assert {int(c) for c in cloud.classification} == {2, 6}
