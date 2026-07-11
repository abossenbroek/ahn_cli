"""Copc-context octree geometry: cube fit and build planning.

COPC requires a *cubic* octree volume. Dutch elevation data is the worst case
for that constraint: kilometres of horizontal extent against a few dozen
metres of Z, often dipping below NAP zero — so the cube side is forced by XY
and every point lives in a thin slab pinned near the cube's Z floor. The bug
in ``docs/bugs/2026-07-11-pdal-copc-xyz-bounds-flat-terrain.md`` was born at
exactly that floor: PDAL declares cube and header bounds through two float64
paths that disagree by an epsilon there.

This module removes the problem *structurally* instead of chasing epsilons:

- The cube anchor sits on **whole metres**, at least 1 m below/left of every
  data minimum, and the side covers the padded extent — so no point can sit on
  an outer cube face, and the anchor metres are exactly representable doubles
  (negative, below-sea-level anchors included).
- All derived quantities (side, bucket edge, 0.5 m voxel edge) are exact
  integers in output scale units; whole-metre bucket edges keep the 0.5 m
  dedup voxels bucket-aligned, so per-bucket dedup equals global dedup.

The plan is computed from the input header alone, before any point streams,
which lets pass 1 scatter points into their level-``bucket_level`` XY columns
in a single pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import ceil, floor
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

_ANCHOR_PAD_M = 1  # whole metres between the cube faces and the data
BUCKET_LEVEL_CAP = 8  # at most (2^8)^2 = 65536 scatter buckets


class CopcError(Exception):
    """Raised when a COPC export cannot be completed."""


@dataclass(frozen=True)
class BuildPlan:
    """The geometry every build stage shares, fixed before streaming starts.

    Contract:
        - ``anchor_m`` is the cube's min corner in whole metres (exact
          doubles); ``side_m`` the cube side in whole metres, a multiple of
          ``2**bucket_level`` so bucket edges are whole metres too.
          Multi-bucket plans align it to ``2**BUCKET_LEVEL_CAP``, so
          ``bucket_level`` can be raised in place (occupancy rebalance)
          without breaking that alignment.
        - ``bucket_level`` is the octree level whose XY footprints are the
          pass-1 scatter buckets; ``max_depth`` the deepest octree level.
        - ``units_per_m`` and ``voxel_units`` express the output quantization
          (e.g. 1000 units/m and 500-unit voxels for scale 0.001, cell 0.5 m).
    """

    scale: float
    anchor_m: tuple[int, int, int]
    side_m: int
    bucket_level: int
    max_depth: int
    sample_grid: int
    units_per_m: int
    voxel_units: int

    @property
    def offsets(self) -> tuple[float, float, float]:
        """LAS offsets: the anchor metres as exactly-representable doubles."""
        x, y, z = self.anchor_m
        return (float(x), float(y), float(z))

    @property
    def side_units(self) -> int:
        """Cube side in integer scale units."""
        return self.side_m * self.units_per_m

    @property
    def bucket_units(self) -> int:
        """Scatter-bucket (level-``bucket_level`` node) edge in scale units."""
        return self.side_units // 2**self.bucket_level


def plan_build(
    mins: tuple[float, float, float],
    maxs: tuple[float, float, float],
    count: int,
    *,
    scale: float = 0.001,
    target_bucket_points: int = 4_000_000,
    sample_grid: int = 128,
    native_cell_m: float = 0.5,
) -> BuildPlan:
    """Fit the octree cube and streaming layout to a cloud's declared bounds.

    Contract:
        - ``mins``/``maxs`` are the input header's world bounds; ``count`` the
          declared point count (must be positive — an empty cloud has no
          geometry to plan).
        - The returned cube contains every coordinate in
          ``[mins - 1 m, maxs + 1 m]``; ``bucket_level`` is the smallest level
          keeping ``count / 4**level`` at or under ``target_bucket_points``
          (capped); ``max_depth`` the shallowest level whose per-node sampling
          grid resolves ``native_cell_m``, never below ``bucket_level``.

    Invariants:
        - Pure arithmetic on the header: deterministic, no I/O.
        - ``bucket_level`` assumes the cloud fills the cube's XY extent
          uniformly; :func:`rebalance_bucket_level` raises it afterwards when
          a measured occupancy histogram says the fill is concentrated.
        - The per-leaf point budget is density-driven: leaves span
          ``(sample_grid / 2, sample_grid] * native_cell_m`` of world per
          side — (32, 64] m at the defaults — which is healthy at AHN's
          10-20 pts/m² regime; extreme densities are a known tuning lever
          (``sample_grid``/``native_cell_m``), deliberately not handled here.
    """
    if count <= 0:
        msg = "cannot plan a COPC build for an empty cloud"
        raise ValueError(msg)

    anchor = tuple(floor(low) - _ANCHOR_PAD_M for low in mins)
    needed = max(
        high - low + _ANCHOR_PAD_M
        for low, high in zip(anchor, maxs, strict=False)
    )

    bucket_level = 0
    while (
        count > target_bucket_points * 4**bucket_level
        and bucket_level < BUCKET_LEVEL_CAP
    ):
        bucket_level += 1

    # Multi-bucket layouts may later be deepened by the occupancy-aware
    # rebalance (up to the cap), so align the side to the cap's grid: every
    # candidate level then keeps whole-metre, exactly nested bucket columns.
    side_multiple = 2 ** (BUCKET_LEVEL_CAP if bucket_level > 0 else 0)
    side_m = side_multiple * ceil(needed / side_multiple)

    max_depth = bucket_level
    while side_m / 2**max_depth > sample_grid * native_cell_m:
        max_depth += 1

    units_per_m = round(1 / scale)
    return BuildPlan(
        scale=scale,
        anchor_m=(anchor[0], anchor[1], anchor[2]),
        side_m=side_m,
        bucket_level=bucket_level,
        max_depth=max_depth,
        sample_grid=sample_grid,
        units_per_m=units_per_m,
        voxel_units=round(native_cell_m * units_per_m),
    )


def rebalance_bucket_level(
    plan: BuildPlan,
    column_counts: npt.NDArray[np.int64],
    target_bucket_points: int,
) -> BuildPlan:
    """Deepen a plan's bucket level to fit the *measured* XY occupancy.

    :func:`plan_build` sizes ``bucket_level`` assuming uniform XY fill; an
    irregular AOI (a thin diagonal polygon: canals, dikes, rail corridors)
    concentrates the cloud in a few columns, so the busiest scatter bucket
    can overshoot the per-bucket target severalfold and break the bounded
    pass-2 working set. Given the cap-level column histogram from a cheap
    streaming pre-scan, pick the smallest level at or above
    ``plan.bucket_level`` whose busiest aggregated column holds at most
    ``target_bucket_points`` points.

    Contract:
        - ``column_counts`` is the ``(2**BUCKET_LEVEL_CAP,) * 2`` int64
          per-column histogram (axis 0 the X column, axis 1 the Y column),
          measured with pass-1 scatter's exact quantization/column math;
          ``plan.side_m`` must be a multiple of ``2**BUCKET_LEVEL_CAP``
          (guaranteed by :func:`plan_build` whenever ``bucket_level > 0``)
          so candidate-level columns nest exactly.
        - Returns ``plan`` itself (the identical object) when its own peak
          already fits; otherwise a copy with ``bucket_level`` raised to the
          smallest fitting level and ``max_depth`` kept consistent
          (``max(plan.max_depth, bucket_level)``); every other field is
          preserved exactly.
        - The cap bounds hierarchy growth: if even level
          ``BUCKET_LEVEL_CAP`` overshoots the target, that level is used
          and the overflow is accepted.

    Invariants:
        - Pure arithmetic on the histogram: deterministic, no I/O.
    """
    level = plan.bucket_level
    while (
        level < BUCKET_LEVEL_CAP
        and _peak_column_points(column_counts, level) > target_bucket_points
    ):
        level += 1
    if level == plan.bucket_level:
        return plan
    return replace(
        plan, bucket_level=level, max_depth=max(plan.max_depth, level)
    )


def _peak_column_points(
    column_counts: npt.NDArray[np.int64], level: int
) -> int:
    """Aggregate the cap-level histogram to ``level``, return its peak count."""
    dim = 2**level
    block = 2 ** (BUCKET_LEVEL_CAP - level)
    aggregated = column_counts.reshape(dim, block, dim, block).sum(
        axis=(1, 3)
    )
    return int(aggregated.max())


class NodeKey(NamedTuple):
    """An octree node address: depth level plus per-axis cell indices."""

    level: int
    x: int
    y: int
    z: int


def cube_bounds(
    plan: BuildPlan,
) -> tuple[float, float, float, float, float, float]:
    """Return the octree cube as ``(min_x, min_y, min_z, max_x, max_y, max_z)``.

    The corners are whole metres, so both are exactly representable doubles
    and every reader deriving node bounds from them starts from the same bits.
    """
    ax, ay, az = plan.offsets
    side = float(plan.side_m)
    return (ax, ay, az, ax + side, ay + side, az + side)


def node_bounds(
    plan: BuildPlan, key: NodeKey
) -> tuple[float, float, float, float, float, float]:
    """Compute a node's bounds by float64 midpoint halving from the cube.

    This replicates, bit for bit, how COPC readers (copc.js inside
    ``copc-validator`` included) subdivide the cube: at each level the parent
    interval is split at ``min + (max - min) / 2`` — copc.js's exact
    ``Bounds.mid`` expression, which is *not* always the same double as
    ``(min + max) / 2`` — and the half is picked from the key's bit
    (``Bounds.stepTo``). Assignment via :class:`LodSampler` descends with the
    same midpoints, so a point can never fall outside the bounds a validator
    recomputes for its node (its comparison is inclusive on both ends).
    """
    min_x, min_y, min_z, max_x, max_y, max_z = cube_bounds(plan)
    for shift in range(key.level - 1, -1, -1):
        mid_x = min_x + (max_x - min_x) / 2.0
        mid_y = min_y + (max_y - min_y) / 2.0
        mid_z = min_z + (max_z - min_z) / 2.0
        if (key.x >> shift) & 1:
            min_x = mid_x
        else:
            max_x = mid_x
        if (key.y >> shift) & 1:
            min_y = mid_y
        else:
            max_y = mid_y
        if (key.z >> shift) & 1:
            min_z = mid_z
        else:
            max_z = mid_z
    return (min_x, min_y, min_z, max_x, max_y, max_z)


@dataclass
class _Descent:
    """Mutable per-point descent state while sampling one point batch."""

    indices: npt.NDArray[np.int64]
    coords: npt.NDArray[np.float64]
    low: npt.NDArray[np.float64]
    high: npt.NDArray[np.float64]
    cells: npt.NDArray[np.int64]

    def keep(self, mask: npt.NDArray[np.bool_]) -> None:
        """Drop accepted points, keeping only ``mask`` rows."""
        self.indices = self.indices[mask]
        self.coords = self.coords[mask]
        self.low = self.low[mask]
        self.high = self.high[mask]
        self.cells = self.cells[mask]

    def step_down(self) -> None:
        """Descend every remaining point into its child node (copc.js math)."""
        mid = self.low + (self.high - self.low) / 2.0
        upper = self.coords >= mid
        self.low = np.where(upper, mid, self.low)
        self.high = np.where(upper, self.high, mid)
        self.cells = self.cells * 2 + upper.astype(np.int64)


@dataclass
class LodSampler:
    """Top-down LOD sampler: assign each point to exactly one octree node.

    Every node samples its volume on a ``sample_grid``-cubed occupancy grid;
    a point is accepted at the shallowest level whose grid cell is still
    free, and pushed down otherwise. At ``max_depth`` everything remaining is
    accepted (the plan sizes that level to resolve the native 0.5 m cell).
    Grid occupancy for levels above ``bucket_level`` persists across calls,
    because those nodes span multiple scatter buckets; deeper nodes belong to
    exactly one bucket, so their occupancy is per call.

    Determinism: points are visited in input order; earlier points win cells.
    """

    plan: BuildPlan
    _shallow_taken: dict[NodeKey, set[int]] = field(
        default_factory=dict[NodeKey, set[int]]
    )

    def sample(
        self, coords: npt.NDArray[np.float64]
    ) -> dict[NodeKey, npt.NDArray[np.int64]]:
        """Assign ``(n, 3)`` decoded world coords to octree nodes.

        Returns a mapping from node key to the ascending input indices that
        node keeps. The union over all nodes is exactly ``arange(n)``.
        """
        n = coords.shape[0]
        cube = cube_bounds(self.plan)
        state = _Descent(
            indices=np.arange(n, dtype=np.int64),
            coords=coords.astype(np.float64),
            low=np.tile(np.asarray(cube[:3]), (n, 1)),
            high=np.tile(np.asarray(cube[3:]), (n, 1)),
            cells=np.zeros((n, 3), dtype=np.int64),
        )
        assigned: dict[NodeKey, npt.NDArray[np.int64]] = {}
        local_taken: dict[NodeKey, set[int]] = {}
        for level in range(self.plan.max_depth + 1):
            if state.indices.shape[0] == 0:
                break
            taken = (
                self._shallow_taken
                if level < self.plan.bucket_level
                else local_taken
            )
            accepted = self._accept_level(state, level, taken)
            for key, rows in accepted.items():
                assigned[key] = np.sort(state.indices[rows])
            kept = np.ones(state.indices.shape[0], dtype=np.bool_)
            for rows in accepted.values():
                kept[rows] = False
            state.keep(kept)
            if level < self.plan.max_depth:
                state.step_down()
        return assigned

    def _accept_level(
        self,
        state: _Descent,
        level: int,
        taken: dict[NodeKey, set[int]],
    ) -> dict[NodeKey, npt.NDArray[np.int64]]:
        """Pick, per node at this level, the points this level keeps."""
        grid = self.plan.sample_grid
        cell_side = (float(self.plan.side_m) / 2**level) / grid
        cell = (state.coords - state.low) / cell_side
        cell_ids = np.clip(cell.astype(np.int64), 0, grid - 1)
        flat_cell = (cell_ids[:, 0] * grid + cell_ids[:, 1]) * grid + (
            cell_ids[:, 2]
        )
        order = np.lexsort(
            (
                np.arange(state.cells.shape[0]),
                state.cells[:, 2],
                state.cells[:, 1],
                state.cells[:, 0],
            )
        )
        node_sorted = state.cells[order]
        is_start: npt.NDArray[np.bool_] = np.ones(
            int(order.shape[0]), dtype=np.bool_
        )
        is_start[1:] = np.any(node_sorted[1:] != node_sorted[:-1], axis=1)
        starts = np.flatnonzero(is_start)
        stops = np.concatenate((starts[1:], [order.shape[0]]))
        accepted: dict[NodeKey, npt.NDArray[np.int64]] = {}
        at_leaf = level == self.plan.max_depth
        for begin, end in zip(starts, stops, strict=True):
            rows = order[begin:end]
            key = NodeKey(
                level,
                int(node_sorted[begin, 0]),
                int(node_sorted[begin, 1]),
                int(node_sorted[begin, 2]),
            )
            if at_leaf:
                accepted[key] = rows
                continue
            free = taken.setdefault(key, set())
            picked = _first_free_cells(flat_cell[rows], free)
            if picked.shape[0] > 0:
                accepted[key] = rows[picked]
        return accepted


def _first_free_cells(
    cells: npt.NDArray[np.int64], taken: set[int]
) -> npt.NDArray[np.int64]:
    """Return positions of the first point in each not-yet-taken cell."""
    _, first = np.unique(cells, return_index=True)
    picked = [
        position
        for position in sorted(first.tolist())
        if int(cells[position]) not in taken
    ]
    for position in picked:
        taken.add(int(cells[position]))
    return np.asarray(picked, dtype=np.int64)
