"""Tests for the sink-driven tiling plan and ``halo: auto`` resolution."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest

from ahn_cli.pipeline.machine import MachineFacts
from ahn_cli.pipeline.tiling import (
    GridTilePlanner,
    HaloDecision,
    TilePlan,
    derive_halo_floor,
    plan_tiles,
    resolve_halo,
)

if TYPE_CHECKING:
    from pathlib import Path

    from ahn_cli.domain import BBox
    from ahn_cli.pipeline.model import TileContext

_FACTS = MachineFacts(cache_line_bytes=128, page_bytes=4096)
_AOI: BBox = (0.0, 0.0, 100.0, 80.0)


def _grid_tiles(
    workdir: Path,
    *,
    tile_size: float = 40.0,
    free_ram: int = 64 * 1024**3,
    per_tile_bytes: int = 1024**3,
    cpu_count: int = 8,
) -> TilePlan:
    return plan_tiles(
        GridTilePlanner(tile_size_m=tile_size),
        aoi_bbox=_AOI,
        halo_floor_m=5.0,
        free_ram_bytes=free_ram,
        machine_facts=_FACTS,
        per_tile_bytes=per_tile_bytes,
        workdir=workdir,
        cpu_count=cpu_count,
    )


# --- derive_halo_floor -----------------------------------------------------


def test_halo_floor_grows_with_neighbours() -> None:
    """A larger neighbour count needs a wider correctness floor."""
    assert derive_halo_floor(
        neighbors=24, point_spacing_m=0.5
    ) > derive_halo_floor(neighbors=12, point_spacing_m=0.5)


def test_halo_floor_grows_with_spacing() -> None:
    """Sparser points (wider spacing) need a wider floor."""
    assert derive_halo_floor(
        neighbors=12, point_spacing_m=1.0
    ) > derive_halo_floor(neighbors=12, point_spacing_m=0.5)


def test_halo_floor_value_is_reach_times_margin() -> None:
    """The floor is the kNN reach (sqrt(k) rings * spacing) times the margin."""
    got = derive_halo_floor(neighbors=16, point_spacing_m=0.5, margin=2.0)
    assert math.isclose(got, math.sqrt(16) * 0.5 * 2.0)


@pytest.mark.parametrize(
    ("neighbors", "spacing", "margin"),
    [(0, 0.5, 1.5), (12, 0.0, 1.5), (12, 0.5, 0.5), (12, math.inf, 1.5)],
)
def test_halo_floor_rejects_bad_inputs(
    neighbors: int, spacing: float, margin: float
) -> None:
    """Zero neighbours, non-positive spacing, or a shrinking margin are errors."""
    with pytest.raises(ValueError, match="floor"):
        derive_halo_floor(
            neighbors=neighbors, point_spacing_m=spacing, margin=margin
        )


# --- resolve_halo ----------------------------------------------------------


def test_resolve_halo_floor_does_not_fit() -> None:
    """When one floor tile exceeds the budget, run serially at the floor."""
    decision = resolve_halo(
        floor_m=5.0,
        free_ram_bytes=1000,
        per_tile_bytes=10_000,
        cpu_count=8,
    )
    assert decision == HaloDecision(halo_m=5.0, concurrency=1)


def test_resolve_halo_grows_and_parallelises_with_ram() -> None:
    """A generous budget grows the halo above the floor and adds workers."""
    decision = resolve_halo(
        floor_m=5.0,
        free_ram_bytes=100 * 1024**3,
        per_tile_bytes=1024**3,
        cpu_count=4,
    )
    assert decision.halo_m > 5.0
    assert decision.concurrency == 4


def test_resolve_halo_concurrency_capped_by_budget() -> None:
    """Concurrency never exceeds how many tiles the safe budget holds."""
    decision = resolve_halo(
        floor_m=5.0,
        free_ram_bytes=4000,
        per_tile_bytes=1000,
        cpu_count=64,
        safe_fraction=0.5,
    )
    # budget = 2000, holds 2 floor tiles.
    assert decision.concurrency == 2


def test_resolve_halo_growth_is_capped() -> None:
    """The halo never grows past the documented cap of the floor."""
    decision = resolve_halo(
        floor_m=3.0,
        free_ram_bytes=10**18,
        per_tile_bytes=1,
        cpu_count=1,
    )
    assert math.isclose(decision.halo_m, 3.0 * 4.0)


def test_resolve_halo_is_monotone_in_ram() -> None:
    """More free RAM never shrinks the halo or the concurrency."""
    prev = resolve_halo(
        floor_m=2.0, free_ram_bytes=1, per_tile_bytes=1000, cpu_count=16
    )
    for exponent in range(4, 20):
        cur = resolve_halo(
            floor_m=2.0,
            free_ram_bytes=10**exponent,
            per_tile_bytes=1000,
            cpu_count=16,
        )
        assert cur.halo_m >= prev.halo_m
        assert cur.concurrency >= prev.concurrency
        prev = cur


def test_resolve_halo_zero_floor_stays_zero() -> None:
    """A stage needing no halo keeps a zero halo however much RAM there is."""
    decision = resolve_halo(
        floor_m=0.0,
        free_ram_bytes=10**15,
        per_tile_bytes=1024,
        cpu_count=8,
    )
    assert decision.halo_m == 0.0


@pytest.mark.parametrize(
    ("floor_m", "free_ram", "per_tile", "cpu", "fraction"),
    [
        (-1.0, 100, 10, 1, 0.6),
        (math.inf, 100, 10, 1, 0.6),
        (5.0, 0, 10, 1, 0.6),
        (5.0, 100, 0, 1, 0.6),
        (5.0, 100, 10, 0, 0.6),
        (5.0, 100, 10, 1, 0.0),
        (5.0, 100, 10, 1, 1.5),
    ],
)
def test_resolve_halo_rejects_bad_inputs(
    floor_m: float, free_ram: int, per_tile: int, cpu: int, fraction: float
) -> None:
    """Every out-of-domain sizing input is a hard error."""
    with pytest.raises(ValueError, match="must be"):
        resolve_halo(
            floor_m=floor_m,
            free_ram_bytes=free_ram,
            per_tile_bytes=per_tile,
            cpu_count=cpu,
            safe_fraction=fraction,
        )


def test_halo_decision_rejects_bad_fields() -> None:
    """The decision value object guards its own invariants."""
    with pytest.raises(ValueError, match="halo"):
        HaloDecision(halo_m=-1.0, concurrency=1)
    with pytest.raises(ValueError, match="concurrency"):
        HaloDecision(halo_m=1.0, concurrency=0)


# --- GridTilePlanner + plan_tiles -----------------------------------------


def test_grid_planner_covers_every_pixel_once(tmp_path: Path) -> None:
    """The grid's tiles partition the AOI: union covers it, interiors disjoint."""
    plan = _grid_tiles(tmp_path)
    tiles = plan.tiles
    # Union of tile extents equals the AOI extent.
    assert min(t.bbox[0] for t in tiles) == _AOI[0]
    assert min(t.bbox[1] for t in tiles) == _AOI[1]
    assert max(t.bbox[2] for t in tiles) == _AOI[2]
    assert max(t.bbox[3] for t in tiles) == _AOI[3]
    # Every sample point of the AOI lands in exactly one tile.
    for px in range(1, 100):
        for py in range(1, 80):
            hits = [
                t
                for t in tiles
                if t.bbox[0] <= px < t.bbox[2] and t.bbox[1] <= py < t.bbox[3]
            ]
            assert len(hits) == 1


def test_grid_planner_single_tile_when_aoi_fits(tmp_path: Path) -> None:
    """An AOI smaller than the tile size yields one tile covering it exactly."""
    plan = _grid_tiles(tmp_path, tile_size=1000.0)
    assert len(plan.tiles) == 1
    assert plan.tiles[0].bbox == _AOI


def test_grid_planner_last_tile_is_clipped(tmp_path: Path) -> None:
    """A non-dividing AOI clips its final row/column to the AOI edge."""
    plan = plan_tiles(
        GridTilePlanner(tile_size_m=30.0),
        aoi_bbox=_AOI,
        halo_floor_m=1.0,
        free_ram_bytes=64 * 1024**3,
        machine_facts=_FACTS,
        per_tile_bytes=1024**3,
        workdir=tmp_path,
        cpu_count=4,
    )
    # 100/30 -> 4 columns, last clipped at x=100 (90..100).
    last = max(plan.tiles, key=lambda t: (t.bbox[1], t.bbox[0]))
    assert last.bbox[2] == 100.0
    assert last.bbox[3] == 80.0


def test_grid_planner_rejects_bad_tile_size(tmp_path: Path) -> None:
    """A non-positive tile size cannot tile anything."""
    with pytest.raises(ValueError, match="tile_size"):
        GridTilePlanner(tile_size_m=0.0).plan(
            aoi_bbox=_AOI, halo_m=1.0, workdir=tmp_path
        )


def test_grid_planner_rejects_degenerate_aoi(tmp_path: Path) -> None:
    """A degenerate AOI is rejected by the shared bbox validator."""
    with pytest.raises(ValueError, match="bbox"):
        GridTilePlanner(tile_size_m=10.0).plan(
            aoi_bbox=(10.0, 0.0, 0.0, 10.0), halo_m=1.0, workdir=tmp_path
        )


def test_plan_is_a_pure_function(tmp_path: Path) -> None:
    """Identical inputs produce an identical plan (tiles, halo, concurrency)."""
    first = _grid_tiles(tmp_path)
    second = _grid_tiles(tmp_path)
    assert first == second


def test_plan_stamps_resolved_halo_on_every_tile(tmp_path: Path) -> None:
    """The RAM-resolved halo is written onto each tile context."""
    plan = _grid_tiles(tmp_path)
    assert plan.halo_m >= 5.0
    assert all(t.halo_m == plan.halo_m for t in plan.tiles)


def test_two_budgets_share_grid_but_differ_in_halo(tmp_path: Path) -> None:
    """Two RAM budgets keep the output grid but change halo and concurrency."""
    tiny = _grid_tiles(
        tmp_path, free_ram=2000, per_tile_bytes=1000, cpu_count=8
    )
    huge = _grid_tiles(
        tmp_path,
        free_ram=200 * 1024**3,
        per_tile_bytes=1024**3,
        cpu_count=8,
    )
    keys_tiny = [t.key for t in tiny.tiles]
    keys_huge = [t.key for t in huge.tiles]
    bboxes_tiny = [t.bbox for t in tiny.tiles]
    bboxes_huge = [t.bbox for t in huge.tiles]
    assert keys_tiny == keys_huge
    assert bboxes_tiny == bboxes_huge
    assert huge.concurrency > tiny.concurrency
    assert huge.halo_m > tiny.halo_m


def test_plan_rounds_per_tile_bytes_to_a_page(tmp_path: Path) -> None:
    """The byte estimate is page-aligned before sizing (uses machine facts)."""
    # per_tile_bytes just over one page -> rounds to two pages, which with a
    # tight budget yields a single-tile serial plan.
    plan = plan_tiles(
        GridTilePlanner(tile_size_m=40.0),
        aoi_bbox=_AOI,
        halo_floor_m=5.0,
        free_ram_bytes=int((2 * 4096) / 0.6) - 1,
        machine_facts=_FACTS,
        per_tile_bytes=4097,
        workdir=tmp_path,
        cpu_count=8,
    )
    assert plan.concurrency == 1


def _tile_keys(tiles: tuple[TileContext, ...]) -> list[tuple[int, int, int]]:
    return [(t.key.tx, t.key.ty, t.key.tz) for t in tiles]


def test_grid_planner_orders_tiles_row_major(tmp_path: Path) -> None:
    """Tiles come back in a deterministic row-major order."""
    plan = _grid_tiles(tmp_path)
    assert _tile_keys(plan.tiles) == sorted(
        _tile_keys(plan.tiles), key=lambda k: (k[1], k[0])
    )
