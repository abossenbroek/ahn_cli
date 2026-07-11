"""Tests for the copc-context cube/build planning (Netherlands-shaped data)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from ahn_cli.copc.octree import (
    BUCKET_LEVEL_CAP,
    BuildPlan,
    plan_build,
    rebalance_bucket_level,
)

# The Moerkapelle shape from the bug report: ~3.76 km x 3.08 km x 58.8 m.
_MINS = (98874.936, 448127.496, -8.569)
_MAXS = (102636.936, 451205.996, 50.266)
_GRID = 2**BUCKET_LEVEL_CAP


def _plan(count: int = 46_000_000) -> BuildPlan:
    return plan_build(_MINS, _MAXS, count)


def test_anchor_sits_on_whole_metres_strictly_below_data() -> None:
    """The cube anchor is whole metres, at least 1 m below every data min."""
    plan = _plan()
    for anchor, low in zip(plan.anchor_m, _MINS, strict=False):
        assert isinstance(anchor, int)
        assert anchor <= low - 1.0
        assert anchor > low - 3.0  # but not needlessly far below


def test_anchor_handles_below_sea_level() -> None:
    """A below-NAP Z minimum yields a negative whole-metre anchor."""
    plan = _plan()
    assert plan.anchor_m[2] <= -10  # -8.569 floored, then 1 m pad
    assert plan.anchor_m[2] < 0


def test_cube_side_covers_padded_extent_and_is_bucket_aligned() -> None:
    """The cube side covers the padded extent, in bucket-aligned whole metres."""
    plan = _plan()
    widest = max(high - low for low, high in zip(_MINS, _MAXS, strict=False))
    assert plan.side_m >= widest + 2.0
    assert plan.side_m % (2**plan.bucket_level) == 0


def test_flat_terrain_forces_horizontal_cube_side() -> None:
    """Dutch-shaped data: the cube side comes from XY, dwarfing the Z range."""
    plan = _plan()
    z_range = _MAXS[2] - _MINS[2]
    assert plan.side_m > 60 * z_range  # ~3764 m vs ~58.8 m


def test_bucket_level_bounds_points_per_bucket() -> None:
    """The bucket level is the smallest k with count/4^k under the target."""
    plan = plan_build(
        _MINS, _MAXS, 46_000_000, target_bucket_points=4_000_000
    )
    assert 46_000_000 / 4**plan.bucket_level <= 4_000_000
    assert 46_000_000 / 4 ** (plan.bucket_level - 1) > 4_000_000


def test_small_cloud_uses_single_bucket() -> None:
    """A cloud under the target needs no spatial split (k == 0)."""
    plan = plan_build(_MINS, _MAXS, 100_000)
    assert plan.bucket_level == 0


def test_bucket_level_is_capped() -> None:
    """Absurd point counts cannot demand an unbounded bucket grid."""
    plan = plan_build(_MINS, _MAXS, 10**15)
    assert plan.bucket_level <= 8


def test_max_depth_reaches_native_cell_sampling() -> None:
    """Max depth is the shallowest level whose sampling grid hits 0.5 m."""
    plan = _plan()
    leaf_side = plan.side_m / 2**plan.max_depth
    assert leaf_side <= plan.sample_grid * 0.5
    assert plan.side_m / 2 ** (plan.max_depth - 1) > plan.sample_grid * 0.5


def test_max_depth_never_above_bucket_level() -> None:
    """Leaves are never shallower than the bucket level itself."""
    tiny = plan_build((0.0, 0.0, 0.0), (10.0, 10.0, 2.0), 50_000_000)
    assert tiny.max_depth >= tiny.bucket_level


def test_derived_unit_quantities_are_exact_integers() -> None:
    """Side, bucket edge and voxel edge are exact integer scale units."""
    plan = _plan()
    assert plan.units_per_m == 1000
    assert plan.side_units == plan.side_m * 1000
    assert plan.side_units % 2**plan.bucket_level == 0
    assert plan.bucket_units == plan.side_units // 2**plan.bucket_level
    assert plan.voxel_units == 500


def test_offsets_are_exact_doubles_of_the_anchor() -> None:
    """LAS offsets are the anchor metres as (exactly representable) doubles."""
    plan = _plan()
    assert plan.offsets == tuple(float(a) for a in plan.anchor_m)


def test_degenerate_single_point_cloud_still_plans() -> None:
    """A single point yields a tiny but valid padded cube."""
    plan = plan_build((1.0, 2.0, 3.0), (1.0, 2.0, 3.0), 1)
    assert plan.side_m >= 2
    assert plan.bucket_level == 0
    assert plan.max_depth >= 0


def test_empty_cloud_is_rejected() -> None:
    """Planning for zero points is a caller error."""
    with pytest.raises(ValueError, match="empty"):
        plan_build(_MINS, _MAXS, 0)


def test_multi_bucket_side_aligns_to_the_cap_grid() -> None:
    """Multi-bucket sides are cap-aligned so the bucket level can be raised."""
    plan = _plan()
    assert plan.bucket_level > 0
    assert plan.side_m % 2**BUCKET_LEVEL_CAP == 0


def test_rebalance_keeps_a_uniform_fill_unchanged() -> None:
    """A uniformly filled cube already meets the target: same plan object."""
    plan = _plan()
    uniform = np.full((_GRID, _GRID), 46_000_000 // _GRID**2, dtype=np.int64)
    assert rebalance_bucket_level(plan, uniform, 4_000_000) is plan


def test_rebalance_raises_to_the_smallest_fitting_level() -> None:
    """A diagonal concentration deepens buckets just enough to fit."""
    plan = _plan()
    per_cell = 46_000_000 // _GRID
    counts = np.zeros((_GRID, _GRID), dtype=np.int64)
    np.fill_diagonal(counts, per_cell)
    got = rebalance_bucket_level(plan, counts, 4_000_000)
    # A level-k column aggregates 2**(cap - k) diagonal cells: level 4 is
    # the smallest whose peak fits the target, level 3 still overshoots.
    assert 2 ** (BUCKET_LEVEL_CAP - 4) * per_cell <= 4_000_000
    assert 2 ** (BUCKET_LEVEL_CAP - 3) * per_cell > 4_000_000
    assert got == replace(plan, bucket_level=4)


def test_rebalance_caps_at_the_bucket_level_cap() -> None:
    """A single overloaded column even at the cap: cap wins, overflow stays."""
    plan = _plan()
    counts = np.zeros((_GRID, _GRID), dtype=np.int64)
    counts[7, 11] = 10**9
    got = rebalance_bucket_level(plan, counts, 4_000_000)
    assert got.bucket_level == BUCKET_LEVEL_CAP
    assert got.max_depth == BUCKET_LEVEL_CAP  # lifted with the bucket level
    assert (got.scale, got.anchor_m, got.side_m, got.sample_grid) == (
        plan.scale,
        plan.anchor_m,
        plan.side_m,
        plan.sample_grid,
    )
