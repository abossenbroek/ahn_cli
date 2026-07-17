"""Build-level tests for the parallel per-tile encode (workstream W10).

These drive the real :func:`build_tiles3d` and prove the parallel encode is
byte-identical to the serial reference across all four profiles, deterministic
across worker counts, bounded in memory during the streaming write, and that
the free-disk floor and crash-safe swap still hold with a pool in the loop.
"""

from __future__ import annotations

import hashlib
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Generic, TypeVar

import pytest

from ahn_cli.prep.spill import DiskFloorError
from ahn_cli.tiles3d import build as build_module
from ahn_cli.tiles3d import pack as pack_module
from ahn_cli.tiles3d.build import build_tiles3d
from ahn_cli.tiles3d.emit import compute_build
from ahn_cli.tiles3d.errors import Tiles3dError
from ahn_cli.tiles3d.parallel import default_window
from ahn_cli.tiles3d.profile import Profile
from ahn_cli.tiles3d.quadtree import plan_quadtree
from ahn_cli.tiles3d.sources import load_terrain
from tests.tiles3d.conftest import (
    grid_for_ortho,
    make_ortho,
    synth_rgb,
    write_exr,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from typing_extensions import Self

_K = TypeVar("_K")
_V = TypeVar("_V")

_PROFILES = [Profile.STRICT, Profile.GAME, Profile.HEIGHTFIELD, Profile.SPLAT]


def _make_pair(
    tmp_path: Path, width: int, height: int, seed: int
) -> tuple[Path, Path]:
    """Write a matching ortho GeoTIFF + reconcile EXR pair to ``tmp_path``."""
    rgb = synth_rgb(width, height, seed)
    ortho = make_ortho(tmp_path / "ortho.tif", rgb)
    heights = write_exr(tmp_path / "r.exr", grid_for_ortho(rgb))
    return ortho, heights


def _digests(root: Path) -> dict[str, str]:
    """Map each file under ``root`` to the sha256 of its bytes."""
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize("profile", _PROFILES)
def test_parallel_build_byte_identical_to_serial(
    tmp_path: Path, profile: Profile
) -> None:
    """A parallel build (workers=4) byte-equals the serial (workers=1) one.

    Covers ``tiles.hfp`` + all sidecars for the packed profiles and every glb
    + ``tileset.json`` for strict. The build's own verifier byte-rebuilds
    serially and would reject any divergence, so a green build is itself proof;
    the digest compare pins it explicitly.
    """
    ortho, heights = _make_pair(tmp_path, 12, 12, seed=41)
    serial = tmp_path / "serial"
    parallel = tmp_path / "parallel"
    build_tiles3d(
        ortho, heights, serial, tile_pixels=4, profile=profile, workers=1
    )
    build_tiles3d(
        ortho, heights, parallel, tile_pixels=4, profile=profile, workers=4
    )
    assert _digests(serial) == _digests(parallel)


@pytest.mark.parametrize("profile", _PROFILES)
def test_build_is_deterministic_across_worker_counts(
    tmp_path: Path, profile: Profile
) -> None:
    """workers=1, 2 and 5 all produce byte-identical output."""
    ortho, heights = _make_pair(tmp_path, 12, 12, seed=42)
    outs: list[dict[str, str]] = []
    for index, workers in enumerate((1, 2, 5)):
        out = tmp_path / f"w{index}"
        build_tiles3d(
            ortho,
            heights,
            out,
            tile_pixels=4,
            profile=profile,
            workers=workers,
        )
        outs.append(_digests(out))
    assert outs[0] == outs[1] == outs[2]


class _CountingFuture(Generic[_V]):
    """Wraps a future, decrementing the pool's outstanding count on consume."""

    def __init__(self, future: Future[_V], pool: _CountingPool) -> None:
        self._future = future
        self._pool = pool
        self._consumed = False

    def result(self) -> _V:
        """Return the result and free this slot exactly once."""
        value = self._future.result()
        if not self._consumed:
            self._consumed = True
            self._pool.release()
        return value


class _CountingPool:
    """A thread pool that tracks the peak outstanding (resident) futures."""

    def __init__(self, *, max_workers: int) -> None:
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.Lock()
        self.outstanding = 0
        self.peak = 0

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> bool:
        self._pool.shutdown()
        return False

    def submit(
        self, fn: Callable[[_K], _V], key: _K, /
    ) -> _CountingFuture[_V]:
        """Submit ``fn(key)`` and count it as outstanding until consumed."""
        with self._lock:
            self.outstanding += 1
            self.peak = max(self.peak, self.outstanding)
        return _CountingFuture(self._pool.submit(fn, key), self)

    def release(self) -> None:
        """Mark one outstanding future as consumed."""
        with self._lock:
            self.outstanding -= 1


def test_build_write_path_is_bounded(tmp_path: Path) -> None:
    """The streaming write keeps at most ``window`` encodes resident.

    An instrumented pool records the peak outstanding futures during the real
    packed build; it must never exceed the driver's window, and must be well
    below the tile count — proof the writer streams rather than buffering every
    tile. (An unbounded "encode all then write" would push the peak to the
    tile count.)
    """
    ortho, heights = _make_pair(tmp_path, 16, 16, seed=43)
    created: list[_CountingPool] = []

    def factory(*, max_workers: int) -> _CountingPool:
        pool = _CountingPool(max_workers=max_workers)
        created.append(pool)
        return pool

    result = build_tiles3d(
        ortho,
        heights,
        tmp_path / "out",
        tile_pixels=4,
        profile=Profile.GAME,
        workers=2,
        pool_factory=factory,
    )

    assert len(created) == 1  # the build write; verify rebuilds serially
    pool = created[0]
    assert 0 < pool.peak <= default_window(2)
    assert pool.peak < result.tile_count


@pytest.mark.parametrize("profile", [Profile.STRICT, Profile.GAME])
def test_disk_floor_breach_is_typed_and_leaves_prior_intact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    profile: Profile,
) -> None:
    """A floor breach raises Tiles3dError and restores the prior deliverable.

    A good build is written first; a rebuild whose every blob write breaches
    the floor must raise the tiles3d typed error and leave the previously
    verified deliverable byte-identical (the crash-safe swap restores it).
    """
    ortho, heights = _make_pair(tmp_path, 12, 12, seed=44)
    out = tmp_path / "out"
    build_tiles3d(ortho, heights, out, tile_pixels=4, profile=profile)
    before = _digests(out)

    def boom(*_args: object, **_kwargs: object) -> None:
        msg = "synthetic floor breach"
        raise DiskFloorError(msg)

    monkeypatch.setattr(pack_module, "ensure_free_disk", boom)
    with pytest.raises(Tiles3dError, match="synthetic floor breach"):
        build_tiles3d(ortho, heights, out, tile_pixels=4, profile=profile)

    assert _digests(out) == before


def test_disk_floor_breach_on_first_build_leaves_no_deliverable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A floor breach on a fresh build leaves no partial deliverable behind."""
    ortho, heights = _make_pair(tmp_path, 8, 8, seed=45)
    out = tmp_path / "out"

    def boom(*_args: object, **_kwargs: object) -> None:
        msg = "no space"
        raise DiskFloorError(msg)

    monkeypatch.setattr(pack_module, "ensure_free_disk", boom)
    with pytest.raises(Tiles3dError, match="no space"):
        build_tiles3d(
            ortho, heights, out, tile_pixels=4, profile=Profile.GAME
        )

    assert not (out / "tiles.hfp").exists()
    assert not (out / "tileset.json").exists()


class _FaultyPool:
    """A pool whose Nth submission's future raises (a mid-build kill)."""

    def __init__(self, *, max_workers: int, trip_on: int) -> None:
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.Lock()
        self._count = 0
        self._trip_on = trip_on

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> bool:
        self._pool.shutdown()
        return False

    def submit(self, fn: Callable[[_K], _V], key: _K, /) -> Future[_V]:
        """Return a failed future on the trip submission, else the real one."""
        with self._lock:
            self._count += 1
            trip = self._count == self._trip_on
        if trip:
            future: Future[_V] = Future()
            future.set_exception(RuntimeError("synthetic encode fault"))
            return future
        return self._pool.submit(fn, key)


def test_worker_fault_mid_build_leaves_prior_intact(tmp_path: Path) -> None:
    """An encode raising in a worker restores the prior verified deliverable.

    Fault injection through the pool seam: one tile's future raises partway
    through the rebuild. The exception propagates out of the parallel driver,
    and the two-phase swap's finally-block moves the held previous deliverable
    back into place.
    """
    ortho, heights = _make_pair(tmp_path, 12, 12, seed=46)
    out = tmp_path / "out"
    build_tiles3d(ortho, heights, out, tile_pixels=4, profile=Profile.GAME)
    before = _digests(out)

    def factory(*, max_workers: int) -> _FaultyPool:
        return _FaultyPool(max_workers=max_workers, trip_on=3)

    with pytest.raises(RuntimeError, match="synthetic encode fault"):
        build_tiles3d(
            ortho,
            heights,
            out,
            tile_pixels=4,
            profile=Profile.GAME,
            workers=4,
            pool_factory=factory,
        )

    assert _digests(out) == before


def test_build_module_exposes_no_glbs_dict(tmp_path: Path) -> None:
    """The strict build plan holds no encoded blobs (lazy, bounded).

    Regression lock on the refactor: ``ComputedBuild`` carries a lazy
    ``blob_source`` and per-tile order/uris, not a materialised glb dict.
    """
    ortho, heights = _make_pair(tmp_path, 8, 8, seed=47)
    terrain = load_terrain(ortho, heights)
    tree = plan_quadtree(terrain.width, terrain.height, 4)
    computed = compute_build(terrain, tree, encoder=Profile.STRICT.encoder())
    assert not hasattr(computed, "glbs")
    assert len(computed.order) == tree.tile_count
    assert set(computed.uri_of) == set(computed.order)
    content, texture = computed.blob_source(computed.order[0])
    assert content
    assert texture is None
    # build_module is imported to assert the write helper is reachable.
    assert hasattr(build_module, "_write_strict")
