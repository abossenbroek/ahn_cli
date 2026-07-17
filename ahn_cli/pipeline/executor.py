"""The tile-streaming executor: drive each tile through the stages, resumably.

:func:`run_pipeline` is the pipeline context's entry point. It senses the
machine through an **injected** probe, resolves the RAM-adaptive halo and
cross-tile concurrency (:mod:`.tiling`), lays out the sink's output grid, then
drives every tile end-to-end through a :class:`~ahn_cli.pipeline.model.Stage`
chain in RAM (``PointTile -> GridTile -> EncodedTile``), persisting each finished
tile's blobs and an atomic commit marker (:mod:`.manifest`). Nothing between
stages touches disk; only one tile's payload is resident at a time (or a handful
under the cross-tile pool), so peak memory is flat as the tile count grows.

Determinism is load-bearing: RAM, the CPU count and the cross-tile pool are all
injected, tiles are processed in the planner's order, and the aggregate
``manifest.json`` is sorted -- so a run is byte-identical across machines, worker
counts and interrupt/resume histories. A re-run recomputes only tiles whose
input hash changed (or whose marker is missing/corrupt); a kill at any stage or
commit boundary resumes to the same bytes with no partial or temp survivor,
because a tile's blobs are rewritten wholesale on reprocess and the marker's
:func:`os.replace` is the sole commit point.

The ``source`` seam supplies each tile's initial payload plus a cheap content
hash of its inputs (in the real pipeline, the cache-backed fetch stage); the
executor never serializes bulk point/grid data to decide freshness.
"""

from __future__ import annotations

import hashlib
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ahn_cli.pipeline.errors import PipelineError
from ahn_cli.pipeline.machine import free_ram_bytes, machine_facts
from ahn_cli.pipeline.manifest import TileStore
from ahn_cli.pipeline.model import EncodedTile, TileContext, TilePayload
from ahn_cli.pipeline.tiling import plan_tiles

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence
    from pathlib import Path

    from typing_extensions import Self

    from ahn_cli.domain import BBox
    from ahn_cli.pipeline.machine import MachineFacts, SystemProbe
    from ahn_cli.pipeline.model import Stage, TileKey
    from ahn_cli.pipeline.tiling import TilePlanner

__all__ = [
    "PipelineResult",
    "PoolFactory",
    "SourceTile",
    "TileSource",
    "run_pipeline",
]

_DEFAULT_PER_TILE_BYTES = 1024**3
"""Fallback per-tile working-set estimate (1 GiB) when a caller gives none."""

_HASH_FIELD_LEN = 8
"""Bytes of the big-endian length prefix framing each hashed input field."""


@dataclass(frozen=True)
class SourceTile:
    """A tile's initial payload plus a cheap content hash of its inputs.

    Contract:
        - ``payload`` is the first :class:`~ahn_cli.pipeline.model.TilePayload`
          fed to the stage chain (the fetched/cached source for the tile).
        - ``content_hash`` is a cheap, deterministic hash of the tile's inputs
          used as the resume freshness key -- it changes iff the underlying
          source data changes, so a stale tile reprocesses.

    Invariants:
        - Frozen value object.

    Failure modes:
        - :class:`ValueError` if ``content_hash`` is blank.
    """

    payload: TilePayload
    content_hash: str

    def __post_init__(self) -> None:
        """Reject a blank content hash."""
        if not self.content_hash.strip():
            msg = "source tile content hash must be non-blank."
            raise ValueError(msg)


@runtime_checkable
class TileSource(Protocol):
    """Supplies a tile's initial payload and input content hash.

    In the real pipeline this is the cache-backed fetch stage; in tests it is a
    deterministic fake. The executor never materializes bulk data to check
    freshness -- it trusts :attr:`SourceTile.content_hash`.
    """

    def load(self, ctx: TileContext) -> SourceTile:
        """Return the initial payload and input hash for ``ctx``."""
        ...


class _Pool(Protocol):
    """The minimal ordered-``map`` pool interface the executor relies on."""

    def __enter__(self) -> Self: ...

    def __exit__(self, *exc: object) -> object: ...

    def map(
        self, fn: Callable[[TileContext], bool], items: Sequence[TileContext]
    ) -> Iterable[bool]: ...


class PoolFactory(Protocol):
    """Builds the cross-tile worker pool used when concurrency exceeds one.

    When no factory is injected the executor uses a
    :class:`~concurrent.futures.ProcessPoolExecutor` directly; tests inject a
    factory to run the parallel path in-process. The returned object is a
    context manager exposing an **ordered** ``map`` (results in submission
    order), so the executor's output is deterministic whatever the pool.
    """

    def __call__(self, *, max_workers: int) -> _Pool:
        """Return a context-managed pool with ``max_workers`` workers."""
        ...


@dataclass(frozen=True)
class PipelineResult:
    """The outcome of a pipeline run.

    Contract:
        - ``out_dir`` / ``manifest_path`` locate the deliverable and its index.
        - ``tile_count`` is the plan size; ``processed`` is how many tiles were
          (re)computed this run and ``skipped`` how many were already committed
          with a matching input hash (``processed + skipped == tile_count``).

    Invariants:
        - Frozen value object, equal by field value.
    """

    out_dir: Path
    manifest_path: Path
    tile_count: int
    processed: int
    skipped: int


def _resolve_cpu(cpu_count: int | None) -> int:
    """Return an explicit CPU count, or fall back to ``os.cpu_count`` / 1."""
    if cpu_count is not None:
        return cpu_count
    return os.cpu_count() or 1


def _sense(probe: SystemProbe | None) -> tuple[int, MachineFacts]:
    """Read free RAM and machine geometry through ``probe`` (or the real system)."""
    if probe is None:
        return free_ram_bytes(), machine_facts()
    return free_ram_bytes(probe=probe), machine_facts(probe=probe)


def _input_hash(signature: str, ctx: TileContext, content_hash: str) -> str:
    """Hash a tile's identity, extent and input content into a freshness key.

    Deliberately excludes ``halo_m``: the halo only ever grows above the
    correctness floor, which never changes a tile's output, so a RAM-driven
    halo change must not invalidate an otherwise-fresh tile.
    """
    key = ctx.key
    parts = (
        signature,
        f"{key.level},{key.tx},{key.ty},{key.tz}",
        repr(ctx.bbox),
        content_hash,
    )
    digest = hashlib.sha256()
    for part in parts:
        encoded = part.encode("utf-8")
        digest.update(len(encoded).to_bytes(_HASH_FIELD_LEN, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _require_encoded(payload: TilePayload, key: TileKey) -> EncodedTile:
    """Return ``payload`` as an :class:`EncodedTile`, or raise for a sink error."""
    if not isinstance(payload, EncodedTile):
        msg = (
            f"tile {key} did not reduce to an EncodedTile; the final stage "
            f"produced {type(payload).__name__}. The last stage must be a sink."
        )
        raise PipelineError(msg)
    return payload


def _ensure_unique_keys(tiles: Sequence[TileContext]) -> None:
    """Reject a plan with a repeated tile key (an ambiguous output layout)."""
    seen: set[TileKey] = set()
    for ctx in tiles:
        if ctx.key in seen:
            msg = f"tiling plan has a duplicate tile key {ctx.key}."
            raise PipelineError(msg)
        seen.add(ctx.key)


def _fire(fault: Callable[[str], None] | None, point: str) -> None:
    """Invoke the fault hook at ``point`` if one is installed."""
    if fault is not None:
        fault(point)


@dataclass(frozen=True)
class _TileWorker:
    """Processes one tile: source -> skip-check -> stages -> two-phase commit."""

    stages: Sequence[Stage]
    source: TileSource
    signature: str
    store: TileStore

    def process(
        self,
        ctx: TileContext,
        fault: Callable[[str], None] | None = None,
    ) -> bool:
        """Run ``ctx`` through the chain, committing it. Return True if computed.

        Returns ``False`` (a skip) when the tile is already committed with a
        matching input hash. ``fault``, when given, is called at each stage and
        commit boundary; a raising hook simulates a kill at that point.
        """
        src = self.source.load(ctx)
        input_hash = _input_hash(self.signature, ctx, src.content_hash)
        if self.store.is_done(ctx.key, input_hash):
            return False
        payload: TilePayload = src.payload
        for index, stage in enumerate(self.stages):
            payload = stage.run(payload, ctx)
            _fire(fault, f"after-stage-{index}")
        encoded = _require_encoded(payload, ctx.key)
        self.store.write_blobs(encoded)
        _fire(fault, "after-blobs")
        self.store.commit(ctx.key, input_hash, encoded)
        return True


def _run_sequential(
    worker: _TileWorker,
    tiles: Sequence[TileContext],
    fault: Callable[[str], None] | None,
) -> int:
    """Process ``tiles`` in order in-process; return the (re)computed count."""
    return sum(worker.process(ctx, fault) for ctx in tiles)


def _run_parallel(
    worker: _TileWorker,
    tiles: Sequence[TileContext],
    concurrency: int,
    pool_factory: PoolFactory,
) -> int:
    """Process ``tiles`` across an injected pool; return the (re)computed count.

    Each tile writes into its own directory and commits its own marker, so the
    tiles are independent and safe to process concurrently; the ordered ``map``
    keeps the run deterministic.
    """
    with pool_factory(max_workers=concurrency) as pool:
        outcomes = list(pool.map(worker.process, tuple(tiles)))
    return sum(1 for done in outcomes if done)


def _run_process_pool(
    worker: _TileWorker, tiles: Sequence[TileContext], concurrency: int
) -> int:
    """Process ``tiles`` across a default process pool; the (re)computed count.

    The independent, per-tile commits make the tiles safe to fan out; the
    ordered ``map`` keeps the aggregate index deterministic.
    """
    with ProcessPoolExecutor(max_workers=concurrency) as pool:
        outcomes = list(pool.map(worker.process, tuple(tiles)))
    return sum(1 for done in outcomes if done)


def run_pipeline(  # noqa: PLR0913 -- one keyword per injected seam; a bag object would only hide the contract
    *,
    planner: TilePlanner,
    aoi_bbox: BBox,
    stages: Sequence[Stage],
    source: TileSource,
    signature: str,
    out_dir: Path,
    workdir: Path,
    halo_floor_m: float,
    per_tile_bytes: int = _DEFAULT_PER_TILE_BYTES,
    probe: SystemProbe | None = None,
    cpu_count: int | None = None,
    pool_factory: PoolFactory | None = None,
    fault: Callable[[str], None] | None = None,
) -> PipelineResult:
    """Run the fused pipeline over ``aoi_bbox``, tile by tile, resumably.

    Contract:
        - Senses free RAM and machine geometry through ``probe`` (never live),
          resolves the halo/concurrency, and lays out ``planner``'s grid.
        - Drives each tile through ``stages`` in RAM; the final payload must be
          an :class:`EncodedTile` (the last stage is a sink). Persists each
          finished tile's blobs plus an atomic marker, then writes the aggregate
          ``manifest.json``.
        - Resumable and crash-safe: a re-run skips tiles committed with a
          matching input hash and recomputes the rest; a kill at any boundary
          resumes byte-identically with no partial/temp survivor.
        - Deterministic: identical inputs and source content produce identical
          on-disk bytes regardless of the RAM-chosen halo, concurrency or pool.

    ``probe`` (default the real system), ``cpu_count`` (default
    :func:`os.cpu_count`) and ``pool_factory`` (default a
    :class:`~concurrent.futures.ProcessPoolExecutor`) are the injected seams for
    deterministic tests. ``fault`` is a test-only hook fired at every stage and
    commit boundary (a raising hook simulates a kill); it forces the serial path.

    Failure modes:
        - :class:`PipelineError` if the plan is empty, has duplicate keys, or a
          tile fails to reduce to an :class:`EncodedTile`.
        - propagates :class:`ValueError` from sizing/planning.
    """
    cpu = _resolve_cpu(cpu_count)
    free_ram, facts = _sense(probe)
    plan = plan_tiles(
        planner,
        aoi_bbox=aoi_bbox,
        halo_floor_m=halo_floor_m,
        free_ram_bytes=free_ram,
        machine_facts=facts,
        per_tile_bytes=per_tile_bytes,
        workdir=workdir,
        cpu_count=cpu,
    )
    tiles = plan.tiles
    if not tiles:
        msg = (
            f"tiling plan produced no tiles for area of interest {aoi_bbox}."
        )
        raise PipelineError(msg)
    _ensure_unique_keys(tiles)
    out_dir.mkdir(parents=True, exist_ok=True)
    workdir.mkdir(parents=True, exist_ok=True)
    store = TileStore(out_dir)
    worker = _TileWorker(
        stages=stages, source=source, signature=signature, store=store
    )
    if plan.concurrency > 1 and fault is None:
        processed = (
            _run_process_pool(worker, tiles, plan.concurrency)
            if pool_factory is None
            else _run_parallel(worker, tiles, plan.concurrency, pool_factory)
        )
    else:
        processed = _run_sequential(worker, tiles, fault)
    manifest_path = store.write_manifest([ctx.key for ctx in tiles])
    return PipelineResult(
        out_dir=out_dir,
        manifest_path=manifest_path,
        tile_count=len(tiles),
        processed=processed,
        skipped=len(tiles) - processed,
    )
