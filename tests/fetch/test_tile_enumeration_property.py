"""Property-based exact-cover check for GeoTiles tile enumeration (WP14).

The epic requires that tile enumeration over a bbox is an *exact cover*: random
areas of interest are never left with a gap, and the returned set is a safe
superset (every returned tile really belongs to the AOI's neighbourhood, per the
documented axis-aligned-in-WGS84 selection).

The substrate is a synthetic regular grid of overlapping WGS84 sheets (mirroring
the real GeoTiles re-tiling) written to a catalogue file, driven through the
*public* :meth:`GeotilesSource.resolve` (so no
private symbol is imported). Each random AOI is a small box whose centre is drawn
inside the grid's interior and whose EPSG:28992 extent is the projection of that
box, so it always lands in covered territory. For every AOI the test asserts:

* completeness -- every corner and the centre of the AOI lies inside the union
  of the returned tiles' extents, so the enumeration leaves no gap; and
* soundness -- every returned tile's extent intersects the AOI in the WGS84 space
  the selection actually operates in, so no unrelated sheet is returned.

Completeness is the load-bearing, non-tautological invariant: a bug that dropped
an edge tile would leave a corner uncovered and fail here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from ahn_cli.fetch.geotiles_source import GeotilesSource
from ahn_cli.fetch.source import boxes_intersect, to_rd, to_wgs84

if TYPE_CHECKING:
    from pathlib import Path

    from ahn_cli.domain import BBox
    from ahn_cli.fetch.generation import GenerationSource

# A regular grid of WGS84 sheets around Delft that *overlap* their neighbours,
# mirroring the real GeoTiles re-tiling (~20-25 m overlap bands -- the very thing
# the dedup stage exists to handle). The overlap keeps the cover gap-free even
# though the shipped ``to_rd`` derives a tile's Dutch-grid bbox from two corners
# (a small skew a touching grid would leave uncovered). The interior (inset from
# every edge by two whole cells) is fully covered, so a small AOI centred there
# always lands inside the returned set.
_GRID_MIN_LON = 4.30
_GRID_MIN_LAT = 51.90
_GRID_STEP = 0.02
_GRID_OVERLAP = 0.003
_GRID_N = 10
_INTERIOR_CELLS = 2

_INTERIOR_MIN_LON = _GRID_MIN_LON + _INTERIOR_CELLS * _GRID_STEP
_INTERIOR_MIN_LAT = _GRID_MIN_LAT + _INTERIOR_CELLS * _GRID_STEP
_INTERIOR_MAX_LON = _GRID_MIN_LON + (_GRID_N - _INTERIOR_CELLS) * _GRID_STEP
_INTERIOR_MAX_LAT = _GRID_MIN_LAT + (_GRID_N - _INTERIOR_CELLS) * _GRID_STEP

# AOI half-extent in degrees (~30 m to ~300 m at this latitude).
_MIN_HALF_DEG = 0.0003
_MAX_HALF_DEG = 0.003


def _cell_polygon(col: int, row: int) -> str:
    """Return the GeoJSON Polygon coordinates for grid cell ``(col, row)``.

    Each cell is grown by :data:`_GRID_OVERLAP` on every side so neighbours
    overlap, exactly as the real GeoTiles sheets do.
    """
    lon0 = _GRID_MIN_LON + col * _GRID_STEP - _GRID_OVERLAP
    lat0 = _GRID_MIN_LAT + row * _GRID_STEP - _GRID_OVERLAP
    lon1 = _GRID_MIN_LON + (col + 1) * _GRID_STEP + _GRID_OVERLAP
    lat1 = _GRID_MIN_LAT + (row + 1) * _GRID_STEP + _GRID_OVERLAP
    return (
        f"[[[{lon0},{lat0}],[{lon1},{lat0}],"
        f"[{lon1},{lat1}],[{lon0},{lat1}],[{lon0},{lat0}]]]"
    )


def _grid_geojson() -> str:
    """Build an ``AHN_subuni`` grid catalogue covering the Delft region."""
    features = [
        (
            '{"type":"Feature",'
            f'"properties":{{"AHN_subuni":"cell_{col}_{row}"}},'
            f'"geometry":{{"type":"Polygon","coordinates":{_cell_polygon(col, row)}}}}}'
        )
        for col in range(_GRID_N)
        for row in range(_GRID_N)
    ]
    return (
        '{"type":"FeatureCollection","features":[' + ",".join(features) + "]}"
    )


@pytest.fixture(scope="module")
def catalog_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Write the synthetic grid catalogue once for the whole module."""
    path = tmp_path_factory.mktemp("grid") / "grid.geojson"
    path.write_text(_grid_geojson())
    return path


def _generation_source(source: GeotilesSource) -> GenerationSource:
    """Return the newest generation source (its base URL addresses tiles)."""
    return source.generation_registry(lambda _url: b"").sources()[0]


def _contains(box: BBox, x: float, y: float) -> bool:
    """Report whether ``(x, y)`` lies within the closed box ``box``."""
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def _probe_points(wgs_box: BBox) -> tuple[tuple[float, float], ...]:
    """Project the AOI's genuine WGS84 corners and centre to EPSG:28992.

    These are real points *inside* the AOI (unlike the axis-aligned RD bbox
    corners, which are virtual projection-overhang points), so each must fall in
    the covering tile the enumeration is required to return.
    """
    minlon, minlat, maxlon, maxlat = wgs_box
    clon = (minlon + maxlon) / 2
    clat = (minlat + maxlat) / 2
    wgs_points = (
        (minlon, minlat),
        (maxlon, minlat),
        (minlon, maxlat),
        (maxlon, maxlat),
        (clon, clat),
    )
    return tuple(
        to_rd((lon, lat, lon + 1e-6, lat + 1e-6))[:2]
        for lon, lat in wgs_points
    )


@settings(
    max_examples=150,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    lon_frac=st.floats(0.0, 1.0),
    lat_frac=st.floats(0.0, 1.0),
    half_lon=st.floats(_MIN_HALF_DEG, _MAX_HALF_DEG),
    half_lat=st.floats(_MIN_HALF_DEG, _MAX_HALF_DEG),
)
def test_enumeration_is_an_exact_cover(
    catalog_path: Path,
    lon_frac: float,
    lat_frac: float,
    half_lon: float,
    half_lat: float,
) -> None:
    """A random interior AOI is covered with no gap by only neighbouring tiles."""
    lon = _INTERIOR_MIN_LON + lon_frac * (
        _INTERIOR_MAX_LON - _INTERIOR_MIN_LON
    )
    lat = _INTERIOR_MIN_LAT + lat_frac * (
        _INTERIOR_MAX_LAT - _INTERIOR_MIN_LAT
    )
    wgs_box: BBox = (
        lon - half_lon,
        lat - half_lat,
        lon + half_lon,
        lat + half_lat,
    )
    aoi = to_rd(wgs_box)

    source = GeotilesSource(catalog_path=catalog_path)
    tiles = source.resolve(
        _generation_source(source), aoi, lambda _url: b""
    ).tiles

    assert tiles, "enumeration must not leave the AOI uncovered"

    # Completeness: no gap -- every genuine interior point is inside some tile.
    for x, y in _probe_points(wgs_box):
        assert any(_contains(tile.bbox, x, y) for tile in tiles), (
            f"point ({x}, {y}) fell in an enumeration gap"
        )

    # Soundness: every returned tile really neighbours the AOI, tested in the
    # WGS84 space the axis-aligned superset selection operates in.
    aoi_wgs84 = to_wgs84(aoi)
    for tile in tiles:
        assert boxes_intersect(aoi_wgs84, to_wgs84(tile.bbox))
