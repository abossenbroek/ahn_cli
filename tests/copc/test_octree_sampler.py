"""Tests for validator-exact node bounds and the LOD descent sampler."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ahn_cli.copc.octree import (
    BuildPlan,
    LodSampler,
    NodeKey,
    node_bounds,
)


def _plan(
    side_m: int = 64,
    bucket_level: int = 1,
    max_depth: int = 2,
    sample_grid: int = 4,
) -> BuildPlan:
    return BuildPlan(
        scale=0.001,
        anchor_m=(-1, -1, -10),
        side_m=side_m,
        bucket_level=bucket_level,
        max_depth=max_depth,
        sample_grid=sample_grid,
        units_per_m=1000,
        voxel_units=500,
    )


def _decode(
    plan: BuildPlan, quantized: npt.NDArray[np.int64]
) -> npt.NDArray[np.float64]:
    offsets = np.asarray(plan.offsets)
    return quantized.astype(np.float64) * plan.scale + offsets


def test_root_bounds_are_the_cube() -> None:
    """The root node's bounds are exactly the cube corners."""
    plan = _plan()
    low_x, low_y, low_z, high_x, high_y, high_z = node_bounds(
        plan, NodeKey(0, 0, 0, 0)
    )
    assert (low_x, low_y, low_z) == (-1.0, -1.0, -10.0)
    assert (high_x, high_y, high_z) == (63.0, 63.0, 54.0)


def test_child_bounds_split_at_the_double_midpoint() -> None:
    """Level-1 children split the cube at the float64 midpoint per axis."""
    plan = _plan()
    lower = node_bounds(plan, NodeKey(1, 0, 0, 0))
    upper = node_bounds(plan, NodeKey(1, 1, 1, 1))
    mid_x = (-1.0 + 63.0) / 2.0
    mid_z = (-10.0 + 54.0) / 2.0
    assert lower[3] == mid_x
    assert upper[0] == mid_x
    assert lower[5] == mid_z
    assert upper[2] == mid_z


def test_deep_key_bounds_nest_inside_their_parent() -> None:
    """A depth-2 node's bounds sit inside its depth-1 parent's bounds."""
    plan = _plan()
    parent = node_bounds(plan, NodeKey(1, 1, 0, 0))
    child = node_bounds(plan, NodeKey(2, 2, 1, 0))
    assert parent[0] <= child[0]
    assert parent[1] <= child[1]
    assert parent[2] <= child[2]
    assert child[3] <= parent[3]
    assert child[4] <= parent[4]
    assert child[5] <= parent[5]


def test_every_sampled_point_is_inside_its_node_bounds() -> None:
    """Assignment by descent keeps every point inside its node (inclusive)."""
    plan = _plan()
    rng = np.random.default_rng(7)
    quantized = np.column_stack(
        [
            rng.integers(0, plan.side_units, 4000),
            rng.integers(0, plan.side_units, 4000),
            rng.integers(0, 60_000, 4000),  # flat: Z pinned near the floor
        ]
    ).astype(np.int64)
    coords = _decode(plan, quantized)
    sampler = LodSampler(plan)
    assigned = sampler.sample(coords)
    for key, indices in assigned.items():
        low_x, low_y, low_z, high_x, high_y, high_z = node_bounds(plan, key)
        picked = coords[indices]
        assert bool(np.all(picked[:, 0] >= low_x))
        assert bool(np.all(picked[:, 1] >= low_y))
        assert bool(np.all(picked[:, 2] >= low_z))
        assert bool(np.all(picked[:, 0] <= high_x))
        assert bool(np.all(picked[:, 1] <= high_y))
        assert bool(np.all(picked[:, 2] <= high_z))


def test_assignment_partitions_all_points() -> None:
    """Every input point lands in exactly one node."""
    plan = _plan()
    rng = np.random.default_rng(11)
    quantized = rng.integers(0, plan.side_units, (2500, 3)).astype(np.int64)
    sampler = LodSampler(plan)
    assigned = sampler.sample(_decode(plan, quantized))
    gathered = np.concatenate(list(assigned.values()))
    assert gathered.shape == (2500,)
    assert np.array_equal(np.sort(gathered), np.arange(2500))


def test_point_on_cube_floor_is_contained() -> None:
    """A point exactly on the cube's Z-minimum face stays in-bounds.

    This is the geometry that broke PDAL on Dutch terrain: everything pinned
    at the Z floor of a horizontally-huge cube.
    """
    plan = _plan()
    quantized = np.asarray(
        [[0, 0, 0], [plan.side_units - 1, plan.side_units - 1, 0]],
        dtype=np.int64,
    )
    coords = _decode(plan, quantized)
    sampler = LodSampler(plan)
    assigned = sampler.sample(coords)
    for key, indices in assigned.items():
        bounds = node_bounds(plan, key)
        for row in coords[indices]:
            assert bounds[2] <= row[2] <= bounds[5]


def test_every_populated_node_has_a_populated_parent() -> None:
    """LOD sampling fills top-down: no orphan deep nodes."""
    plan = _plan(max_depth=3)
    rng = np.random.default_rng(3)
    quantized = rng.integers(0, plan.side_units, (5000, 3)).astype(np.int64)
    sampler = LodSampler(plan)
    assigned = sampler.sample(_decode(plan, quantized))
    populated = set(assigned)
    for key in populated:
        if key.level == 0:
            continue
        parent = NodeKey(key.level - 1, key.x // 2, key.y // 2, key.z // 2)
        assert parent in populated


def test_shallow_occupancy_persists_across_columns() -> None:
    """A second column cannot reclaim shallow grid cells already taken."""
    plan = _plan(bucket_level=1, max_depth=2)
    point = np.asarray([[1000, 1000, 1000]], dtype=np.int64)
    coords = _decode(plan, point)
    sampler = LodSampler(plan)
    first = sampler.sample(coords)
    second = sampler.sample(coords)
    first_key = next(iter(first))
    second_key = next(iter(second))
    assert first_key.level == 0  # first point claims the root cell
    assert second_key.level > 0  # the same cell is now taken


def test_max_depth_accepts_unconditionally() -> None:
    """Leaf level takes every remaining point, even in one grid cell."""
    plan = _plan(bucket_level=0, max_depth=1, sample_grid=1)
    # Ten points in the same corner: grid 1x1x1 -> one acceptance per level,
    # the rest all land in the single leaf reached by descent.
    quantized = np.asarray([[i, i, i] for i in range(10)], dtype=np.int64)
    sampler = LodSampler(plan)
    assigned = sampler.sample(_decode(plan, quantized))
    assert sum(len(v) for v in assigned.values()) == 10
    leaf_counts = {k: len(v) for k, v in assigned.items() if k.level == 1}
    assert sum(leaf_counts.values()) == 9


def test_sampling_is_deterministic() -> None:
    """Identical input yields identical node assignment."""
    plan = _plan()
    rng = np.random.default_rng(5)
    quantized = rng.integers(0, plan.side_units, (1000, 3)).astype(np.int64)
    coords = _decode(plan, quantized)
    one = LodSampler(plan).sample(coords)
    two = LodSampler(plan).sample(coords)
    assert set(one) == set(two)
    for key, indices in one.items():
        assert np.array_equal(indices, two[key])
