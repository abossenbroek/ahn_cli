"""Tests for the footprint coverage raster (synthetic polygon fixtures)."""

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from ahn_cli.domain import Product
from ahn_cli.fetch.top10nl import KasFetch, features_sha256
from ahn_cli.prep import footprint
from ahn_cli.prep.footprint import (
    FootprintError,
    check_grid,
    cover_fraction,
    grid_shape,
    preflight,
    product_paths,
    write_cover,
    write_footprint_product,
)
from ahn_cli.provenance import read_provenance

_BBOX = (100.0, 200.0, 110.0, 208.0)  # 10 x 8 cells at 1 m


def _rect(x0: float, y0: float, x1: float, y1: float) -> dict[str, object]:
    ring = [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]
    return {"type": "Polygon", "coordinates": [ring]}


def test_grid_shape_ceil_and_exact() -> None:
    """Exact multiples are not padded; a remainder adds one cell."""
    assert grid_shape(_BBOX, 1.0) == (8, 10)
    assert grid_shape((0.0, 0.0, 10.5, 3.0), 1.0) == (3, 11)
    assert grid_shape((0.0, 0.0, 1.0, 1.0), 0.1) == (10, 10)


@pytest.mark.parametrize("res", [0.0, -1.0, float("nan")])
def test_bad_resolution_fails(res: float) -> None:
    """Non-positive or non-finite cells are refused."""
    with pytest.raises(FootprintError, match="positive"):
        grid_shape(_BBOX, res)


def test_bad_bbox_fails() -> None:
    """A degenerate box is refused."""
    with pytest.raises(FootprintError, match="minx < maxx"):
        grid_shape((1.0, 0.0, 0.0, 1.0), 1.0)


def test_check_grid_supersample_and_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Supersample below one and oversize grids fail before allocating."""
    with pytest.raises(FootprintError, match="supersample"):
        check_grid(_BBOX, 1.0, 0)
    monkeypatch.setattr(footprint, "MAX_CELLS", 80 * 16 - 1)
    with pytest.raises(FootprintError, match="exceeds"):
        check_grid(_BBOX, 1.0, 4)


def test_cell_aligned_square_is_exact() -> None:
    """A polygon on cell edges covers whole cells; row 0 is north."""
    cover = cover_fraction([_rect(102, 205, 104, 208)], _BBOX, 1.0)
    assert cover.dtype == np.float32
    assert cover.shape == (8, 10)
    want = np.zeros((8, 10), np.float32)
    want[0:3, 2:4] = 1.0
    np.testing.assert_array_equal(cover, want)


def test_half_cell_edge_and_area() -> None:
    """A half-cell edge gives 0.5; the sum equals the polygon area."""
    cover = cover_fraction([_rect(101.5, 201, 104, 203)], _BBOX, 1.0)
    assert cover[5, 1] == 0.5
    assert cover[5, 2] == 1.0
    assert cover.sum() == 5.0


def test_overlap_counts_once_and_hole_is_empty() -> None:
    """Overlapping polygons do not exceed 1; a hole stays 0."""
    donut = _rect(100, 200, 106, 206)
    ring = [[102, 202], [104, 202], [104, 204], [102, 204], [102, 202]]
    coords = donut["coordinates"]
    assert isinstance(coords, list)
    donut["coordinates"] = [*coords, ring]
    cover = cover_fraction([donut, _rect(100, 200, 102, 203)], _BBOX, 1.0)
    assert cover.max() == 1.0
    assert cover[4:6, 2:4].sum() == 0.0
    assert cover.sum() == 32.0


def test_no_geometry_is_all_zero() -> None:
    """No polygons give an all-zero raster of the right shape."""
    assert cover_fraction([], _BBOX, 2.0).shape == (4, 5)
    assert cover_fraction([], _BBOX, 2.0).sum() == 0.0


def test_write_cover_roundtrip(tmp_path: Path) -> None:
    """The GeoTIFF carries RD New, the grid origin and the same pixels."""
    cover = cover_fraction([_rect(102, 205, 104, 208)], _BBOX, 1.0)
    path = tmp_path / "c.tif"
    sha = write_cover(path, cover, _BBOX, 1.0)
    with rasterio.open(path) as src:
        assert str(src.crs) == "EPSG:28992"
        assert src.transform == from_bounds(100.0, 200.0, 110.0, 208.0, 10, 8)
        np.testing.assert_array_equal(src.read(1), cover)
    assert len(sha) == 64
    with pytest.raises(FootprintError, match="does not match"):
        write_cover(path, cover[:2], _BBOX, 1.0)


def _fetch() -> KasFetch:
    feats: tuple[dict[str, object], ...] = (
        {
            "type": "Feature",
            "id": "a",
            "properties": {"bronactualiteit": "2021-06-01"},
            "geometry": _rect(102, 205, 104, 208),
        },
    )
    return KasFetch(
        features=feats,
        page_urls=("u1",),
        n_in_box=3,
        features_sha256=features_sha256(feats),
        source_dates=("2021-06-01",),
    )


def test_product_writes_and_pins(tmp_path: Path) -> None:
    """GeoJSON, raster and provenance land under footprints/."""
    t0 = datetime(2026, 10, 18, tzinfo=timezone.utc)
    got = write_footprint_product(
        _fetch(), tmp_path, _BBOX, 1.0, started_at=t0, finished_at=t0
    )
    geo, cov, prov = product_paths(tmp_path)
    assert (got.geojson_path, got.cover_path, got.provenance_path) == (
        geo,
        cov,
        prov,
    )
    assert got.covered_m2 == 6.0
    record = read_provenance(prov)
    assert record.product is Product.FOOTPRINT
    assert record.input_checksum == _fetch().features_sha256
    assert record.output_checksum == got.cover_sha256
    assert dict(record.request_keys)["bronactualiteit_max"] == "2021-06-01"
    with pytest.raises(FootprintError, match="refusing to overwrite"):
        preflight(tmp_path, _BBOX, 1.0)
