"""Sink-driven tiling plan and RAM-adaptive ``halo: auto`` resolution.

The executor drives one output tile at a time end-to-end. This module decides
**which** tiles exist and **how wide** their source halo is, keeping the two
concerns cleanly split:

* The output **grid** is a sink concern: a :class:`TilePlanner` (injected by the
  caller -- a quadtree for a ``tiles3d`` sink, an AHN-sheet grid for a cloud
  sink) turns an area of interest plus a resolved halo into an ordered sequence
  of :class:`~ahn_cli.pipeline.model.TileContext`. The grid is a pure function
  of the area of interest, **never** of RAM, so the deliverable's tile set is
  identical whatever the machine.
* The **halo** and cross-tile **concurrency** are the only RAM-adaptive knobs
  (:func:`resolve_halo`): a hard correctness floor (:func:`derive_halo_floor`,
  from the reconcile neighbour count and the local point spacing) is never
  undercut, and above it the halo grows toward a safe fraction of free RAM while
  concurrency rises with the budget. Because a correct stage's output depends
  only on the tile plus *at least* the floor halo, growing the halo or changing
  the worker count is a pure performance knob -- the bytes on disk do not move.

:func:`plan_tiles` composes the two into a single deterministic function of
``(aoi, floor, free_ram, machine_facts, per_tile_bytes, cpu_count)``: same
inputs, same plan.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ahn_cli.domain import ensure_valid_bbox
from ahn_cli.pipeline.model import TileContext, TileKey

if TYPE_CHECKING:
    from pathlib import Path

    from ahn_cli.domain import BBox
    from ahn_cli.pipeline.machine import MachineFacts

__all__ = [
    "GridTilePlanner",
    "HaloDecision",
    "TilePlan",
    "TilePlanner",
    "derive_halo_floor",
    "plan_tiles",
    "resolve_halo",
]

_SAFE_RAM_FRACTION = 0.6
"""Default share of free RAM the working set may occupy (a safety margin)."""

_MAX_HALO_GROWTH = 4.0
"""Cap on how far above the floor a generous RAM budget grows the halo."""

_DEFAULT_FLOOR_MARGIN = 1.5
"""Default safety multiple on the raw kNN reach (sparse/clustered areas)."""


@runtime_checkable
class TilePlanner(Protocol):
    """The sink's output-grid abstraction.

    Contract:
        - :meth:`plan` is a **pure function** of ``(aoi_bbox, halo_m, workdir)``:
          the same inputs always yield the same ordered tuple of
          :class:`~ahn_cli.pipeline.model.TileContext`.
        - The grid (tile keys and extents) depends only on ``aoi_bbox``; the
          resolved ``halo_m`` is stamped onto every tile but never changes which
          tiles exist, so the deliverable's tile set is RAM-independent.
    """

    def plan(
        self, *, aoi_bbox: BBox, halo_m: float, workdir: Path
    ) -> tuple[TileContext, ...]:
        """Return the ordered output tiles covering ``aoi_bbox``."""
        ...


@dataclass(frozen=True)
class HaloDecision:
    """The RAM-adaptive halo and cross-tile concurrency for a run.

    Contract:
        - ``halo_m`` is the resolved source halo in metres, always at least the
          correctness floor and finite.
        - ``concurrency`` is the number of tiles processed at once, at least 1.

    Invariants:
        - Frozen value object, equal by field value.

    Failure modes:
        - :class:`ValueError` if ``halo_m`` is non-finite/negative or
          ``concurrency`` is below one.
    """

    halo_m: float
    concurrency: int

    def __post_init__(self) -> None:
        """Reject an invalid halo or concurrency."""
        if not math.isfinite(self.halo_m) or self.halo_m < 0.0:
            msg = (
                f"halo_m must be finite and non-negative; got {self.halo_m}."
            )
            raise ValueError(msg)
        if self.concurrency < 1:
            msg = f"concurrency must be >= 1; got {self.concurrency}."
            raise ValueError(msg)


@dataclass(frozen=True)
class TilePlan:
    """A resolved plan: the ordered tiles plus the RAM-adaptive knobs.

    Contract:
        - ``tiles`` is the sink's ordered output grid, each context carrying the
          resolved ``halo_m``.
        - ``halo_m`` / ``concurrency`` are the :class:`HaloDecision` the sizing
          produced.

    Invariants:
        - Frozen value object, equal by field value; two runs with identical
          inputs compare equal.
    """

    tiles: tuple[TileContext, ...]
    halo_m: float
    concurrency: int


def derive_halo_floor(
    *,
    neighbors: int,
    point_spacing_m: float,
    margin: float = _DEFAULT_FLOOR_MARGIN,
) -> float:
    """Return the correctness floor for the source halo, in metres.

    Contract:
        - The floor is ``sqrt(neighbors) * point_spacing_m * margin``: to see the
          ``k`` nearest neighbours a global run would, a tile-edge pixel must
          reach roughly ``sqrt(k)`` rings of points at the local spacing, times a
          safety multiple for sparse or clustered areas.
        - Monotone non-decreasing in both ``neighbors`` and ``point_spacing_m``.

    Failure modes:
        - :class:`ValueError` if ``neighbors < 1``, ``point_spacing_m`` is not a
          finite positive length, or ``margin < 1`` (a floor may not shrink the
          reach).
    """
    if neighbors < 1:
        msg = f"halo floor needs at least one neighbour; got {neighbors}."
        raise ValueError(msg)
    if not math.isfinite(point_spacing_m) or point_spacing_m <= 0.0:
        msg = (
            "halo floor needs a finite positive point spacing; "
            f"got {point_spacing_m}."
        )
        raise ValueError(msg)
    if margin < 1.0:
        msg = f"halo floor margin must be >= 1; got {margin}."
        raise ValueError(msg)
    return math.sqrt(neighbors) * point_spacing_m * margin


def resolve_halo(
    *,
    floor_m: float,
    free_ram_bytes: int,
    per_tile_bytes: int,
    cpu_count: int,
    safe_fraction: float = _SAFE_RAM_FRACTION,
) -> HaloDecision:
    """Size the halo and concurrency to a safe fraction of free RAM.

    Contract:
        - The working set targets ``free_ram_bytes * safe_fraction`` bytes.
        - ``concurrency`` is how many floor-sized tiles fit in that budget,
          clamped to ``[1, cpu_count]`` -- so even when one floor tile exceeds
          the budget the run still proceeds serially (the "shrink the tile,
          lower concurrency" regime), never below the floor.
        - The halo grows above ``floor_m`` by ``sqrt(budget / per_tile_bytes)``,
          clamped to ``[1, _MAX_HALO_GROWTH]``; a zero floor stays zero.
        - Both ``halo_m`` and ``concurrency`` are monotone non-decreasing in
          ``free_ram_bytes`` -- more RAM never shrinks the working set. Neither
          changes any tile's output, since the halo never dips below the floor.

    Failure modes:
        - :class:`ValueError` on a non-finite/negative floor, a non-positive
          RAM or per-tile estimate, ``cpu_count < 1``, or a ``safe_fraction``
          outside ``(0, 1]``.
    """
    if not math.isfinite(floor_m) or floor_m < 0.0:
        msg = f"halo floor must be finite and non-negative; got {floor_m}."
        raise ValueError(msg)
    if free_ram_bytes <= 0:
        msg = f"free RAM estimate must be positive; got {free_ram_bytes}."
        raise ValueError(msg)
    if per_tile_bytes <= 0:
        msg = (
            f"per-tile byte estimate must be positive; got {per_tile_bytes}."
        )
        raise ValueError(msg)
    if cpu_count < 1:
        msg = f"cpu count must be >= 1; got {cpu_count}."
        raise ValueError(msg)
    if not 0.0 < safe_fraction <= 1.0:
        msg = f"safe fraction must be in (0, 1]; got {safe_fraction}."
        raise ValueError(msg)
    budget = free_ram_bytes * safe_fraction
    concurrency = max(1, min(cpu_count, int(budget // per_tile_bytes)))
    headroom = budget / per_tile_bytes
    growth = min(max(math.sqrt(headroom), 1.0), _MAX_HALO_GROWTH)
    return HaloDecision(halo_m=floor_m * growth, concurrency=concurrency)


def _round_up(value: int, multiple: int) -> int:
    """Round ``value`` up to the next ``multiple`` (both positive)."""
    return ((value + multiple - 1) // multiple) * multiple


def plan_tiles(
    planner: TilePlanner,
    *,
    aoi_bbox: BBox,
    halo_floor_m: float,
    free_ram_bytes: int,
    machine_facts: MachineFacts,
    per_tile_bytes: int,
    workdir: Path,
    cpu_count: int,
    safe_fraction: float = _SAFE_RAM_FRACTION,
) -> TilePlan:
    """Resolve the halo/concurrency and lay out the sink's tiles.

    Contract:
        - Page-aligns ``per_tile_bytes`` to ``machine_facts.page_bytes`` (the
          working set is mmap-page granular), resolves the
          :class:`HaloDecision`, then asks ``planner`` for the grid at the
          resolved halo.
        - A **pure function** of its inputs: identical arguments produce an
          identical :class:`TilePlan`.

    Failure modes:
        - propagates :class:`ValueError` from :func:`resolve_halo` and from the
          planner (e.g. a degenerate area of interest).
    """
    aligned = _round_up(per_tile_bytes, machine_facts.page_bytes)
    decision = resolve_halo(
        floor_m=halo_floor_m,
        free_ram_bytes=free_ram_bytes,
        per_tile_bytes=aligned,
        cpu_count=cpu_count,
        safe_fraction=safe_fraction,
    )
    tiles = planner.plan(
        aoi_bbox=aoi_bbox, halo_m=decision.halo_m, workdir=workdir
    )
    return TilePlan(
        tiles=tiles,
        halo_m=decision.halo_m,
        concurrency=decision.concurrency,
    )


@dataclass(frozen=True)
class GridTilePlanner:
    """A rectangular fixed-size grid planner (a sink-agnostic reference).

    Contract:
        - ``tile_size_m`` is the side length in metres of every full tile.
        - :meth:`plan` covers ``aoi_bbox`` with a row-major grid of tiles, each
          ``tile_size_m`` wide except the last row/column, which is clipped to
          the area of interest -- so the tiles **partition** the area (their
          union is the whole box and their interiors are disjoint).
        - ``level`` labels every tile (a single-level grid by default).

    Invariants:
        - Frozen value object; :meth:`plan` is a pure function of its arguments.

    Failure modes:
        - :class:`ValueError` if ``tile_size_m`` is not positive or the area of
          interest is degenerate.
    """

    tile_size_m: float
    level: int = 0

    def plan(
        self, *, aoi_bbox: BBox, halo_m: float, workdir: Path
    ) -> tuple[TileContext, ...]:
        """Return the row-major grid of tiles covering ``aoi_bbox``."""
        if not math.isfinite(self.tile_size_m) or self.tile_size_m <= 0.0:
            msg = f"tile_size_m must be positive; got {self.tile_size_m}."
            raise ValueError(msg)
        ensure_valid_bbox(aoi_bbox)
        minx, miny, maxx, maxy = aoi_bbox
        size = self.tile_size_m
        n_cols = math.ceil((maxx - minx) / size)
        n_rows = math.ceil((maxy - miny) / size)
        tiles: list[TileContext] = []
        for ty in range(n_rows):
            y0 = miny + ty * size
            y1 = min(miny + (ty + 1) * size, maxy)
            for tx in range(n_cols):
                x0 = minx + tx * size
                x1 = min(minx + (tx + 1) * size, maxx)
                tiles.append(
                    TileContext(
                        key=TileKey(level=self.level, tx=tx, ty=ty),
                        bbox=(x0, y0, x1, y1),
                        halo_m=halo_m,
                        workdir=workdir,
                    )
                )
        return tuple(tiles)
