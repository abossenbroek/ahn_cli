"""Tests for the per-tile :class:`ReconcileStage` halo-kNN adapter.

The load-bearing gate is that a tiled reconcile is **byte-identical** to the
standalone whole-area verb: every interior/edge pixel's kNN set must match the
global run. These tests build a synthetic cloud + ortho, run the real
``reconcile`` verb as the oracle, and assert the stage reproduces it -- and that
the halo *floor* is both necessary (a smaller halo diverges) and sufficient (the
floor is byte-identical).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pytest

from ahn_cli.domain.grid import PixelGrid
from ahn_cli.pipeline.errors import PipelineError
from ahn_cli.pipeline.model import (
    GridTile,
    PointTile,
    TileContext,
    TileKey,
)
from ahn_cli.pipeline.stages.reconcile import (
    OrthoWindow,
    ReconcileStage,
)
from ahn_cli.pipeline.tiling import derive_halo_floor
from ahn_cli.reconcile.method import IdwInterp, LinearInterp
from ahn_cli.reconcile.raster import ReconcileError, load_cloud
from ahn_cli.reconcile.reconcile import ReconcileRequest, reconcile
from ahn_cli.reconcile.writers import OutputFormat
from tests.pipeline.harness import (
    hash_payload,
    make_grid_tile,
    write_synthetic_laz,
    write_synthetic_ortho,
)

if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt

    from ahn_cli.domain import BBox
    from ahn_cli.domain.grid import GeoTransform

_WIDTH = 8
_HEIGHT = 6
_NEIGHBORS = 12
_SPACING_M = 0.3
_AOI: BBox = (0.0, 0.0, float(_WIDTH), float(_HEIGHT))
_HUGE_HALO = 1.0e9


@dataclass
class _FixedWindows:
    """An in-RAM :class:`OrthoWindows` keyed by tile key (a deterministic fake)."""

    windows: dict[TileKey, OrthoWindow]

    def window(self, ctx: TileContext) -> OrthoWindow:
        """Return the pre-built window for ``ctx``'s tile key."""
        return self.windows[ctx.key]


def _whole_grid() -> PixelGrid:
    """Return the whole-area north-up 1 m pixel grid over the AOI."""
    transform: GeoTransform = (1.0, 0.0, 0.0, 0.0, -1.0, float(_HEIGHT))
    return PixelGrid(width=_WIDTH, height=_HEIGHT, transform=transform)


def _build_scene(
    tmp_path: Path, seed: int
) -> tuple[Path, Path, npt.NDArray[np.uint8]]:
    """Write a synthetic cloud LAZ + ortho GeoTIFF; return paths and RGB (HWC).

    The cloud is a jittered regular grid at ``_SPACING_M`` covering a padded
    AOI: uniform spacing means the kNN reach is tightly bounded everywhere (no
    sparse pockets), so the ``sqrt(k) * spacing`` halo floor is reliably
    sufficient, while the continuous jitter keeps every squared distance
    distinct (no tie-break ambiguity).
    """
    rng = np.random.default_rng(seed)
    pad = 1.5
    axis_x = np.arange(-pad, _WIDTH + pad, _SPACING_M)
    axis_y = np.arange(-pad, _HEIGHT + pad, _SPACING_M)
    grid_x, grid_y = np.meshgrid(axis_x, axis_y)
    total = grid_x.size
    jitter = _SPACING_M / 3.0
    x = grid_x.ravel() + rng.uniform(-jitter, jitter, total)
    y = grid_y.ravel() + rng.uniform(-jitter, jitter, total)
    z = rng.uniform(-5.0, 5.0, total)
    gps = rng.uniform(0.0, 1.0, total)
    classification = rng.integers(1, 7, total)
    points = np.column_stack([x, y, z, gps, classification]).astype(
        np.float64
    )
    laz = tmp_path / "cloud.laz"
    write_synthetic_laz(laz, points)

    rgb_chw = rng.integers(0, 256, (3, _HEIGHT, _WIDTH)).astype(np.uint8)
    ortho = tmp_path / "ortho.tif"
    write_synthetic_ortho(ortho, rgb_chw, _AOI)
    rgb_hwc = np.ascontiguousarray(np.transpose(rgb_chw, (1, 2, 0)))
    return laz, ortho, rgb_hwc


def _run_standalone(
    laz: Path,
    ortho: Path,
    out_dir: Path,
    method: IdwInterp | LinearInterp,
) -> npt.NDArray[np.float32]:
    """Run the real reconcile verb and return its ``(H, W, 6)`` PT grid."""
    request = ReconcileRequest(
        ortho_path=ortho,
        cloud_path=laz,
        output_dir=out_dir,
        method=method,
        formats=(OutputFormat.PT,),
    )
    stats = reconcile(request)
    flat = np.fromfile(out_dir / "reconciled.pt", dtype="<f4")
    return flat.reshape(stats.height, stats.width, 6)


def _pixel_window(grid: PixelGrid, bbox: BBox) -> tuple[int, int, int, int]:
    """Return the ``(col0, col1, row0, row1)`` pixel span of ``bbox``."""
    a, _b, c, _d, e, f = grid.transform
    minx, miny, maxx, maxy = bbox
    col0 = round((minx - c) / a)
    col1 = round((maxx - c) / a)
    r_top = round((maxy - f) / e)
    r_bot = round((miny - f) / e)
    row0, row1 = min(r_top, r_bot), max(r_top, r_bot)
    return col0, col1, row0, row1


def _ortho_window(
    grid: PixelGrid, rgb_hwc: npt.NDArray[np.uint8], bbox: BBox
) -> OrthoWindow:
    """Build the pixel-aligned :class:`OrthoWindow` for ``bbox``."""
    col0, col1, row0, row1 = _pixel_window(grid, bbox)
    a, b, c, d, e, f = grid.transform
    sub: GeoTransform = (a, b, a * col0 + c, d, e, e * row0 + f)
    sub_grid = PixelGrid(width=col1 - col0, height=row1 - row0, transform=sub)
    window_rgb = np.ascontiguousarray(rgb_hwc[row0:row1, col0:col1])
    return OrthoWindow(grid=sub_grid, rgb=window_rgb)


def _point_tile(
    coords: npt.NDArray[np.float64],
    classification: npt.NDArray[np.uint8],
    bbox: BBox,
    halo: float,
) -> PointTile:
    """Select the tile's cloud (bbox expanded by ``halo``) as a PointTile."""
    minx, miny, maxx, maxy = bbox
    xs, ys = coords[:, 0], coords[:, 1]
    keep = (
        (xs >= minx - halo)
        & (xs <= maxx + halo)
        & (ys >= miny - halo)
        & (ys <= maxy + halo)
    )
    sub = coords[keep]
    count = int(sub.shape[0])
    return PointTile(
        x=np.ascontiguousarray(sub[:, 0]),
        y=np.ascontiguousarray(sub[:, 1]),
        z=np.ascontiguousarray(sub[:, 2]),
        gps_time=np.zeros(count, dtype=np.float64),
        classification=np.ascontiguousarray(classification[keep]),
    )


def _stage(
    windows: dict[TileKey, OrthoWindow],
    method: IdwInterp | LinearInterp,
) -> ReconcileStage:
    """Build a :class:`ReconcileStage` over an in-RAM ortho-window fake."""
    return ReconcileStage(
        method=method,
        ortho=_FixedWindows(windows),
        neighbors=_NEIGHBORS,
        point_spacing_m=_SPACING_M,
    )


def _ctx(key: TileKey, bbox: BBox, halo: float, workdir: Path) -> TileContext:
    """Build a tile context (the halo is metadata; ``run`` uses the payload)."""
    return TileContext(key=key, bbox=bbox, halo_m=halo, workdir=workdir)


def _idw() -> IdwInterp:
    """Return the default IDW method (power 2, k=12)."""
    return IdwInterp(power=2.0, k=_NEIGHBORS)


def test_single_tile_is_byte_identical_to_standalone(tmp_path: Path) -> None:
    """A single-tile AOI stage output equals the whole-grid reconcile verb."""
    laz, ortho, rgb = _build_scene(tmp_path, seed=0)
    cloud = load_cloud(laz)
    key = TileKey(level=0, tx=0, ty=0)
    grid = _whole_grid()
    windows = {key: _ortho_window(grid, rgb, _AOI)}
    stage = _stage(windows, _idw())
    ctx = _ctx(key, _AOI, stage.halo_m(), tmp_path)
    tile = _point_tile(cloud.coords, cloud.classification, _AOI, _HUGE_HALO)

    result = stage.run(tile, ctx)

    oracle = _run_standalone(laz, ortho, tmp_path / "out", _idw())
    assert result.heights.shape == (_HEIGHT, _WIDTH)
    assert result.heights.tobytes() == oracle[:, :, 2].tobytes()
    assert np.array_equal(result.red, oracle[:, :, 3].astype(np.uint8))
    assert np.array_equal(result.green, oracle[:, :, 4].astype(np.uint8))
    assert np.array_equal(result.blue, oracle[:, :, 5].astype(np.uint8))


def _tile_bboxes() -> list[tuple[TileKey, BBox]]:
    """Return the 2x2 pixel-aligned partition of the AOI."""
    xs = (0.0, 4.0, 8.0)
    ys = (0.0, 3.0, 6.0)
    tiles: list[tuple[TileKey, BBox]] = []
    for ty in range(2):
        for tx in range(2):
            key = TileKey(level=0, tx=tx, ty=ty)
            bbox: BBox = (xs[tx], ys[ty], xs[tx + 1], ys[ty + 1])
            tiles.append((key, bbox))
    return tiles


def _stitch_tiles(
    grid: PixelGrid,
    tiles: list[tuple[TileKey, BBox]],
    results: dict[TileKey, GridTile],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.uint8]]:
    """Place each tile's grid at its pixel offset into a whole-area mosaic."""
    heights = np.zeros((_HEIGHT, _WIDTH), dtype=np.float32)
    colour = np.zeros((_HEIGHT, _WIDTH, 3), dtype=np.uint8)
    for key, bbox in tiles:
        col0, col1, row0, row1 = _pixel_window(grid, bbox)
        result = results[key]
        heights[row0:row1, col0:col1] = result.heights
        colour[row0:row1, col0:col1, 0] = result.red
        colour[row0:row1, col0:col1, 1] = result.green
        colour[row0:row1, col0:col1, 2] = result.blue
    return heights, colour


def _run_tiled(
    tmp_path: Path,
    cloud_coords: npt.NDArray[np.float64],
    classification: npt.NDArray[np.uint8],
    rgb: npt.NDArray[np.uint8],
    halo: float,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.uint8]]:
    """Run the 2x2 tiling at a given source ``halo`` and stitch the mosaic."""
    grid = _whole_grid()
    tiles = _tile_bboxes()
    windows = {key: _ortho_window(grid, rgb, bbox) for key, bbox in tiles}
    stage = _stage(windows, _idw())
    results: dict[TileKey, GridTile] = {}
    for key, bbox in tiles:
        ctx = _ctx(key, bbox, stage.halo_m(), tmp_path)
        tile = _point_tile(cloud_coords, classification, bbox, halo)
        results[key] = stage.run(tile, ctx)
    return _stitch_tiles(grid, tiles, results)


def test_tiled_with_halo_matches_whole_area(tmp_path: Path) -> None:
    """A 2x2 tiled run with the floor halo equals the whole-AOI reconcile."""
    laz, ortho, rgb = _build_scene(tmp_path, seed=1)
    cloud = load_cloud(laz)
    stage = _stage({}, _idw())

    heights, colour = _run_tiled(
        tmp_path,
        cloud.coords,
        cloud.classification,
        rgb,
        stage.halo_m(),
    )

    oracle = _run_standalone(laz, ortho, tmp_path / "out", _idw())
    assert heights.tobytes() == oracle[:, :, 2].tobytes()
    assert np.array_equal(colour[:, :, 0], oracle[:, :, 3].astype(np.uint8))
    assert np.array_equal(colour[:, :, 1], oracle[:, :, 4].astype(np.uint8))
    assert np.array_equal(colour[:, :, 2], oracle[:, :, 5].astype(np.uint8))


def test_halo_floor_is_necessary_and_sufficient(tmp_path: Path) -> None:
    """Below the floor at least one edge estimate diverges; at it, identical."""
    laz, ortho, rgb = _build_scene(tmp_path, seed=2)
    cloud = load_cloud(laz)
    oracle = _run_standalone(laz, ortho, tmp_path / "out", _idw())
    oracle_z = np.ascontiguousarray(oracle[:, :, 2])

    at_floor, _colour = _run_tiled(
        tmp_path,
        cloud.coords,
        cloud.classification,
        rgb,
        _stage({}, _idw()).halo_m(),
    )
    assert at_floor.tobytes() == oracle_z.tobytes()

    starved, _c2 = _run_tiled(
        tmp_path, cloud.coords, cloud.classification, rgb, 0.0
    )
    assert not np.array_equal(starved, oracle_z)


def test_run_is_deterministic_across_runs(tmp_path: Path) -> None:
    """Two runs over identical inputs produce a byte-identical GridTile.

    Worker-count invariance is inherited from ``neighbors.query_knn``
    (``workers=-1`` with the lexsort tie-break), proven in that module.
    """
    laz, _ortho, rgb = _build_scene(tmp_path, seed=3)
    cloud = load_cloud(laz)
    key = TileKey(level=0, tx=0, ty=0)
    grid = _whole_grid()
    windows = {key: _ortho_window(grid, rgb, _AOI)}
    stage = _stage(windows, _idw())
    ctx = _ctx(key, _AOI, stage.halo_m(), tmp_path)
    tile = _point_tile(cloud.coords, cloud.classification, _AOI, _HUGE_HALO)

    first = stage.run(tile, ctx)
    second = stage.run(tile, ctx)

    assert hash_payload(first) == hash_payload(second)


def test_empty_source_is_a_void_error(tmp_path: Path) -> None:
    """An empty source cloud yields all-void cells, a hard ReconcileError."""
    _laz, _ortho, rgb = _build_scene(tmp_path, seed=4)
    key = TileKey(level=0, tx=0, ty=0)
    grid = _whole_grid()
    windows = {key: _ortho_window(grid, rgb, _AOI)}
    stage = _stage(windows, _idw())
    ctx = _ctx(key, _AOI, stage.halo_m(), tmp_path)
    empty = PointTile(
        x=np.zeros(0, dtype=np.float64),
        y=np.zeros(0, dtype=np.float64),
        z=np.zeros(0, dtype=np.float64),
        gps_time=np.zeros(0, dtype=np.float64),
        classification=np.zeros(0, dtype=np.uint8),
    )

    with pytest.raises(ReconcileError, match="no genuine elevation"):
        stage.run(empty, ctx)


def test_non_point_tile_is_a_pipeline_error(tmp_path: Path) -> None:
    """A stage fed a non-PointTile payload raises a PipelineError."""
    stage = _stage({}, _idw())
    ctx = _ctx(TileKey(level=0, tx=0, ty=0), _AOI, 0.0, tmp_path)

    with pytest.raises(PipelineError, match="expected a PointTile"):
        stage.run(make_grid_tile(), ctx)


def test_halo_m_matches_the_derived_floor() -> None:
    """The stage halo is the tiling floor from neighbours and spacing."""
    stage = _stage({}, _idw())
    expected = derive_halo_floor(
        neighbors=_NEIGHBORS, point_spacing_m=_SPACING_M, margin=1.5
    )
    assert stage.halo_m() == expected
    assert stage.halo_m() > 0.0


def test_halo_m_honours_a_custom_margin() -> None:
    """A larger margin widens the halo floor proportionally."""
    stage = ReconcileStage(
        method=_idw(),
        ortho=_FixedWindows({}),
        neighbors=_NEIGHBORS,
        point_spacing_m=_SPACING_M,
        margin=3.0,
    )
    expected = derive_halo_floor(
        neighbors=_NEIGHBORS, point_spacing_m=_SPACING_M, margin=3.0
    )
    assert stage.halo_m() == expected


def test_ortho_window_rejects_a_shape_mismatch() -> None:
    """An RGB plane not matching the grid shape is refused."""
    grid = _whole_grid()
    bad = np.zeros((_HEIGHT + 1, _WIDTH, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="ortho window rgb"):
        OrthoWindow(grid=grid, rgb=bad)
