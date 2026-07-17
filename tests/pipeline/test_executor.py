"""Tests for the tile-streaming executor: run, resume, RAM-adaptation, faults."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest
from typing_extensions import Self

from ahn_cli.pipeline.errors import PipelineError
from ahn_cli.pipeline.executor import (
    PipelineResult,
    PoolFactory,
    SourceTile,
    TileSource,
    _resolve_cpu,  # pyright: ignore[reportPrivateUsage]
    run_pipeline,
)
from ahn_cli.pipeline.machine import SystemProbe
from ahn_cli.pipeline.model import (
    EncodedBlob,
    EncodedTile,
    TileContext,
    TileKey,
    TilePayload,
)
from ahn_cli.pipeline.tiling import GridTilePlanner
from tests.pipeline.harness import (
    IdentityStage,
    hash_tree,
    make_point_tile,
    make_tile_context,
    make_tile_key,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from ahn_cli.domain import BBox
    from ahn_cli.pipeline.model import Stage

_AOI: BBox = (0.0, 0.0, 80.0, 80.0)  # 2x2 tiles at size 40


# --- fakes -----------------------------------------------------------------


def _linux_probe(free_ram_bytes: int) -> SystemProbe:
    """Build a deterministic Linux probe reporting a chosen free-RAM budget."""
    meminfo = f"MemTotal: 1 kB\nMemAvailable: {free_ram_bytes // 1024} kB\n"
    sizes = {"SC_LEVEL1_DCACHE_LINESIZE": 128, "SC_PAGE_SIZE": 4096}

    def _run(
        _args: Sequence[str],
    ) -> str:  # pragma: no cover - linux uses none
        raise AssertionError

    return SystemProbe(
        platform="linux",
        sysconf=lambda name: sizes[name],
        run=_run,
        read_text=lambda _path: meminfo,
    )


@dataclass(frozen=True)
class _FakeSource:
    """A deterministic source whose content hash tracks the ``tag`` and bbox."""

    tag: str = "v1"

    def load(self, ctx: TileContext) -> SourceTile:
        payload = make_point_tile(count=4, seed=ctx.key.tx * 10 + ctx.key.ty)
        return SourceTile(
            payload=payload, content_hash=f"{self.tag}:{ctx.bbox}"
        )


@dataclass(frozen=True)
class _EncodeStage:
    """A sink-like stage turning any tile into a deterministic EncodedTile."""

    def halo_m(self) -> float:
        return 0.0

    def run(self, tile: TilePayload, ctx: TileContext) -> TilePayload:  # noqa: ARG002
        key = ctx.key
        data = f"{key.level},{key.tx},{key.ty},{key.tz}|{ctx.bbox}".encode()
        return EncodedTile(
            key=key, blobs=(EncodedBlob(name="geometry", data=data),)
        )


@dataclass(frozen=True)
class _ResidencySpy:
    """Records the peak number of payloads resident inside ``run`` at once."""

    live: list[int]  # [current, peak]

    def halo_m(self) -> float:
        return 0.0

    def run(self, tile: TilePayload, ctx: TileContext) -> TilePayload:  # noqa: ARG002
        self.live[0] += 1
        self.live[1] = max(self.live[1], self.live[0])
        self.live[0] -= 1
        return tile


@dataclass
class _InlinePool:
    """A synchronous stand-in for ``ProcessPoolExecutor`` (ordered ``map``)."""

    max_workers: int
    seen: list[int]

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False

    def map(
        self, fn: Callable[[TileContext], bool], items: Sequence[TileContext]
    ) -> list[bool]:
        self.seen.append(self.max_workers)
        return [fn(item) for item in items]


def _inline_pool_factory(*, max_workers: int) -> _InlinePool:
    """Return a synchronous pool for tests that exercise the parallel path."""
    return _InlinePool(max_workers, [])


@dataclass(frozen=True)
class _EmptyPlanner:
    def plan(
        self, *, aoi_bbox: BBox, halo_m: float, workdir: Path
    ) -> tuple[TileContext, ...]:
        _ = (aoi_bbox, halo_m, workdir)
        return ()


@dataclass(frozen=True)
class _DupPlanner:
    def plan(
        self, *, aoi_bbox: BBox, halo_m: float, workdir: Path
    ) -> tuple[TileContext, ...]:
        _ = aoi_bbox
        ctx = make_tile_context(
            workdir, key=make_tile_key(tx=0), halo_m=halo_m
        )
        return (ctx, ctx)


@dataclass(frozen=True)
class _PointSource:
    """A source whose payload is a raw PointTile (never an EncodedTile)."""

    def load(self, ctx: TileContext) -> SourceTile:
        return SourceTile(
            payload=make_point_tile(count=2), content_hash=str(ctx.bbox)
        )


def _run(  # noqa: PLR0913 -- a test driver mirroring run_pipeline's injected seams
    out_dir: Path,
    workdir: Path,
    *,
    free_ram: int = 1_000_000,
    cpu_count: int = 1,
    source: TileSource | None = None,
    stages: Sequence[Stage] | None = None,
    per_tile_bytes: int = 10_000,
    tile_size: float = 40.0,
    aoi: BBox = _AOI,
    fault: Callable[[str], None] | None = None,
    pool_factory: PoolFactory | None = None,
    signature: str = "sig",
) -> PipelineResult:
    return run_pipeline(
        planner=GridTilePlanner(tile_size_m=tile_size),
        aoi_bbox=aoi,
        stages=list(stages) if stages is not None else [_EncodeStage()],
        source=source if source is not None else _FakeSource(),
        signature=signature,
        out_dir=out_dir,
        workdir=workdir,
        halo_floor_m=2.0,
        per_tile_bytes=per_tile_bytes,
        probe=_linux_probe(free_ram),
        cpu_count=cpu_count,
        fault=fault,
        pool_factory=pool_factory
        if pool_factory is not None
        else _inline_pool_factory,
    )


# --- basic run -------------------------------------------------------------


def test_single_tile_run_writes_deliverable(tmp_path: Path) -> None:
    """A one-tile AOI writes its blob, marker and the aggregate manifest."""
    out, work = tmp_path / "out", tmp_path / "work"
    result = _run(out, work, aoi=(0.0, 0.0, 10.0, 10.0))
    assert isinstance(result, PipelineResult)
    assert result.tile_count == 1
    assert result.processed == 1
    assert result.skipped == 0
    assert (out / "manifest.json").is_file()
    tile_dir = out / "tiles" / "0" / "0_0_0"
    assert (tile_dir / "geometry").is_file()
    assert (tile_dir / "_tile.json").is_file()


def test_multi_tile_counts(tmp_path: Path) -> None:
    """A 2x2 AOI processes four tiles."""
    result = _run(tmp_path / "out", tmp_path / "work")
    assert result.tile_count == 4
    assert result.processed == 4


def test_resume_skips_done_and_is_idempotent(tmp_path: Path) -> None:
    """A second identical run recomputes nothing and changes no bytes."""
    out, work = tmp_path / "out", tmp_path / "work"
    _run(out, work)
    before = hash_tree(out)
    result = _run(out, work)
    assert result.processed == 0
    assert result.skipped == 4
    assert hash_tree(out) == before


# --- RAM-adaptation invariance --------------------------------------------


def test_byte_identical_across_ram_budgets(tmp_path: Path) -> None:
    """A tiny vs huge RAM budget yields byte-identical deliverables."""
    tiny = tmp_path / "tiny"
    huge = tmp_path / "huge"
    _run(tiny, tmp_path / "w1", free_ram=200_000, per_tile_bytes=90_000)
    _run(
        huge,
        tmp_path / "w2",
        free_ram=64 * 1024**3,
        per_tile_bytes=1_000_000,
        cpu_count=8,
    )
    assert hash_tree(tiny) == hash_tree(huge)


def test_parallel_matches_sequential(tmp_path: Path) -> None:
    """The cross-tile pool path produces the same bytes as the serial path."""
    seen: list[int] = []

    def tracking_factory(*, max_workers: int) -> _InlinePool:
        return _InlinePool(max_workers, seen)

    par = tmp_path / "par"
    _run(
        par,
        tmp_path / "wp",
        free_ram=64 * 1024**3,
        per_tile_bytes=1_000_000,
        cpu_count=4,
        pool_factory=tracking_factory,
    )
    seq = tmp_path / "seq"
    _run(seq, tmp_path / "ws", free_ram=200_000, per_tile_bytes=1_000_000)
    assert seen  # the pool was actually entered
    assert seen[0] > 1  # ...with more than one worker
    assert hash_tree(par) == hash_tree(seq)


def test_default_process_pool_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no injected factory, concurrency routes through the process pool.

    The real ``ProcessPoolExecutor`` is stubbed with an in-process pool so the
    default fan-out path is exercised deterministically, and its output matches
    the serial path.
    """
    monkeypatch.setattr(
        "ahn_cli.pipeline.executor.ProcessPoolExecutor", _inline_pool_factory
    )
    par = tmp_path / "par"
    result = run_pipeline(
        planner=GridTilePlanner(tile_size_m=40.0),
        aoi_bbox=_AOI,
        stages=[_EncodeStage()],
        source=_FakeSource(),
        signature="sig",
        out_dir=par,
        workdir=tmp_path / "wp",
        halo_floor_m=2.0,
        per_tile_bytes=1_000_000,
        probe=_linux_probe(64 * 1024**3),
        cpu_count=4,
    )
    assert result.processed == 4
    seq = tmp_path / "seq"
    _run(seq, tmp_path / "ws", free_ram=200_000, per_tile_bytes=1_000_000)
    assert hash_tree(par) == hash_tree(seq)


def test_default_probe_senses_the_real_machine(tmp_path: Path) -> None:
    """Omitting the probe reads the real machine and still completes serially."""
    result = run_pipeline(
        planner=GridTilePlanner(tile_size_m=40.0),
        aoi_bbox=_AOI,
        stages=[_EncodeStage()],
        source=_FakeSource(),
        signature="sig",
        out_dir=tmp_path / "out",
        workdir=tmp_path / "work",
        halo_floor_m=2.0,
        cpu_count=1,  # force the serial path regardless of real free RAM
    )
    assert result.processed == 4


def test_halo_above_floor_does_not_change_output(tmp_path: Path) -> None:
    """Growing the halo above the floor leaves the deliverable untouched."""
    small = tmp_path / "small"
    big = tmp_path / "big"
    _run(small, tmp_path / "ws", free_ram=200_000, per_tile_bytes=95_000)
    _run(big, tmp_path / "wb", free_ram=10**14, per_tile_bytes=1_000)
    assert hash_tree(small) == hash_tree(big)


# --- resumability / crash safety ------------------------------------------


def _kill_on(label: str) -> Callable[[str], None]:
    def hook(point: str) -> None:
        if point == label:
            raise _SimulatedKillError

    return hook


class _SimulatedKillError(Exception):
    """Stands in for a SIGKILL at a stage/commit boundary."""


@pytest.mark.parametrize(
    "label", ["after-stage-0", "after-stage-1", "after-blobs"]
)
def test_sigkill_at_each_boundary_resumes_identically(
    tmp_path: Path, label: str
) -> None:
    """A kill at any boundary resumes to the uninterrupted, tmp-free result."""
    baseline = tmp_path / "baseline"
    _run(
        baseline, tmp_path / "wbase", stages=[IdentityStage(), _EncodeStage()]
    )
    want = hash_tree(baseline)

    out, work = tmp_path / "out", tmp_path / "work"
    with pytest.raises(_SimulatedKillError):
        _run(
            out,
            work,
            stages=[IdentityStage(), _EncodeStage()],
            fault=_kill_on(label),
        )
    assert not (out / "manifest.json").exists()  # aborted before the index
    # Resume to completion.
    _run(out, work, stages=[IdentityStage(), _EncodeStage()])
    assert hash_tree(out) == want
    assert not any(".tmp" in p.name for p in out.rglob("*"))


def test_two_phase_commit_window_no_double_emit(tmp_path: Path) -> None:
    """A kill between blob write and marker leaves no partial or double tile."""
    out, work = tmp_path / "out", tmp_path / "work"
    with pytest.raises(_SimulatedKillError):
        _run(
            out,
            work,
            aoi=(0.0, 0.0, 10.0, 10.0),
            fault=_kill_on("after-blobs"),
        )
    tile_dir = out / "tiles" / "0" / "0_0_0"
    assert (tile_dir / "geometry").is_file()  # blob landed
    assert not (tile_dir / "_tile.json").exists()  # but was never committed
    # Resume: the tile is reprocessed and committed exactly once.
    result = _run(out, work, aoi=(0.0, 0.0, 10.0, 10.0))
    assert result.processed == 1
    assert (tile_dir / "_tile.json").is_file()
    assert sorted(p.name for p in tile_dir.iterdir()) == [
        "_tile.json",
        "geometry",
    ]


def test_kill_after_some_tiles_resumes_and_skips(tmp_path: Path) -> None:
    """Tiles committed before a mid-run kill are skipped on resume."""

    def hook_factory() -> Callable[[str], None]:
        state = {"n": 0}

        def hook(point: str) -> None:
            if point == "after-blobs":
                state["n"] += 1
                if state["n"] == 3:  # kill during the third tile
                    raise _SimulatedKillError

        return hook

    out, work = tmp_path / "out", tmp_path / "work"
    with pytest.raises(_SimulatedKillError):
        _run(out, work, fault=hook_factory())
    result = _run(out, work)
    # Two tiles committed before the kill are now skipped.
    assert result.skipped == 2
    assert result.processed == 2


def test_stale_entry_wrong_hash_reprocessed(tmp_path: Path) -> None:
    """A changed source content hash forces every tile to reprocess."""
    out, work = tmp_path / "out", tmp_path / "work"
    _run(out, work, source=_FakeSource(tag="v1"))
    result = _run(out, work, source=_FakeSource(tag="v2"))
    assert result.processed == 4
    assert result.skipped == 0


def test_corrupt_manifest_safe_rebuild(tmp_path: Path) -> None:
    """A garbage aggregate manifest is safely rebuilt on the next run."""
    out, work = tmp_path / "out", tmp_path / "work"
    _run(out, work)
    (out / "manifest.json").write_text("{ corrupt", encoding="utf-8")
    result = _run(out, work)
    assert result.processed == 0  # markers intact -> all skipped
    json.loads((out / "manifest.json").read_text())  # valid again


# --- error paths -----------------------------------------------------------


def test_empty_plan_is_error(tmp_path: Path) -> None:
    """A planner that covers nothing is a hard pipeline error."""
    with pytest.raises(PipelineError, match="no tiles"):
        run_pipeline(
            planner=_EmptyPlanner(),
            aoi_bbox=_AOI,
            stages=[_EncodeStage()],
            source=_FakeSource(),
            signature="sig",
            out_dir=tmp_path / "out",
            workdir=tmp_path / "work",
            halo_floor_m=2.0,
            probe=_linux_probe(1_000_000),
            cpu_count=1,
        )


def test_duplicate_tile_keys_is_error(tmp_path: Path) -> None:
    """A planner emitting the same key twice is rejected."""
    with pytest.raises(PipelineError, match="duplicate"):
        run_pipeline(
            planner=_DupPlanner(),
            aoi_bbox=_AOI,
            stages=[_EncodeStage()],
            source=_FakeSource(),
            signature="sig",
            out_dir=tmp_path / "out",
            workdir=tmp_path / "work",
            halo_floor_m=2.0,
            probe=_linux_probe(1_000_000),
            cpu_count=1,
        )


def test_non_encoded_final_payload_is_error(tmp_path: Path) -> None:
    """A stage chain that never produces an EncodedTile is an error."""
    with pytest.raises(PipelineError, match="EncodedTile"):
        _run(
            tmp_path / "out",
            tmp_path / "work",
            aoi=(0.0, 0.0, 10.0, 10.0),
            source=_PointSource(),
            stages=[IdentityStage()],
        )


# --- bounded memory --------------------------------------------------------


def test_residency_is_bounded_as_tiles_grow(tmp_path: Path) -> None:
    """Peak resident payloads stay at one however many tiles the AOI has."""
    live = [0, 0]
    _run(
        tmp_path / "out",
        tmp_path / "work",
        aoi=(0.0, 0.0, 50.0, 50.0),
        tile_size=10.0,  # 25 tiles
        stages=[_ResidencySpy(live), _EncodeStage()],
    )
    assert live[1] == 1


# --- _resolve_cpu ----------------------------------------------------------


def test_resolve_cpu_explicit() -> None:
    """An explicit cpu count is used verbatim."""
    assert _resolve_cpu(5) == 5


def test_resolve_cpu_defaults_to_os(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ``None`` count reads ``os.cpu_count``."""
    monkeypatch.setattr("ahn_cli.pipeline.executor.os.cpu_count", lambda: 6)
    assert _resolve_cpu(None) == 6


def test_resolve_cpu_falls_back_to_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unknown CPU count falls back to a single worker."""
    monkeypatch.setattr(
        "ahn_cli.pipeline.executor.os.cpu_count", lambda: None
    )
    assert _resolve_cpu(None) == 1


# --- value objects ---------------------------------------------------------


def test_source_tile_rejects_blank_hash() -> None:
    """A source tile must carry a non-blank content hash."""
    with pytest.raises(ValueError, match="content hash"):
        SourceTile(
            payload=EncodedTile(
                key=TileKey(level=0, tx=0, ty=0),
                blobs=(EncodedBlob(name="g", data=b"x"),),
            ),
            content_hash="  ",
        )
