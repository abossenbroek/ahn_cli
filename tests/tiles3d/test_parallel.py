"""Unit tests for the bounded-window parallel encode driver.

These prove the three load-bearing properties of :func:`ordered_encode`
directly, without a full build: order preservation whatever the workers do,
the serial ``workers=1`` reference, and the bounded in-flight window (with a
teeth test showing an unbounded window would hold everything resident).
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future
from typing import TYPE_CHECKING, TypeVar

import pytest

from ahn_cli.tiles3d.parallel import (
    default_window,
    default_workers,
    ordered_encode,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from typing_extensions import Self

_V = TypeVar("_V")


class _SyncPool:
    """A pool that runs each submission inline, so timing is deterministic.

    ``submit`` executes ``fn`` immediately and returns an already-completed
    future. This makes the driver's window the *only* thing bounding how many
    results are resident, so a test can assert an exact peak.
    """

    def __init__(self, *, max_workers: int) -> None:
        self.max_workers = max_workers

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def submit(self, fn: Callable[..., _V], key: object, /) -> Future[_V]:
        """Run ``fn(key)`` now and return a completed future."""
        future: Future[_V] = Future()
        future.set_result(fn(key))
        return future


def _sync_factory(*, max_workers: int) -> _SyncPool:
    return _SyncPool(max_workers=max_workers)


def test_default_workers_is_positive() -> None:
    """The default worker count is at least one."""
    assert default_workers() >= 1


def test_default_window_is_twice_workers_min_two() -> None:
    """The default window is ``2 * workers`` with a floor of two."""
    assert default_window(1) == 2
    assert default_window(4) == 8


def test_workers_one_runs_inline_in_order() -> None:
    """``workers=1`` calls ``encode`` in a plain loop, in key order."""
    calls: list[int] = []

    def encode(key: int) -> int:
        calls.append(key)
        return key * 10

    out = list(ordered_encode([1, 2, 3], encode, workers=1, window=5))

    assert out == [10, 20, 30]
    assert calls == [1, 2, 3]


def test_parallel_preserves_key_order() -> None:
    """Results come back in key order however the workers interleave.

    Earlier keys sleep longer, so they finish *after* later keys — yet the
    driver still yields them in ascending key order.
    """
    count = 24

    def encode(key: int) -> int:
        time.sleep(0.003 * ((count - key) % 4))
        return key * key

    out = list(ordered_encode(range(count), encode, workers=4, window=8))

    assert out == [key * key for key in range(count)]


def test_parallel_bounds_resident_results() -> None:
    """At most ``window`` encoded results are ever resident at once.

    The sync pool completes each submission immediately, so the peak resident
    count equals the driver's window — never the tile count.
    """
    window = 4
    count = 100
    resident = 0
    peak = 0

    def encode(key: int) -> int:
        nonlocal resident, peak
        resident += 1
        peak = max(peak, resident)
        return key

    out: list[int] = []
    for value in ordered_encode(
        range(count),
        encode,
        workers=2,
        window=window,
        pool_factory=_sync_factory,
    ):
        resident -= 1  # the writer consumes (frees) this result
        out.append(value)

    assert out == list(range(count))
    assert peak == window
    assert peak < count


def test_unbounded_window_holds_everything_resident() -> None:
    """Teeth for the bounded test: a window >= tile count resides everything.

    With the window widened to the tile count, the driver primes every encode
    up front and the peak resident count rises to the full count — exactly the
    unbounded "encode all then write" failure the bounded window prevents.
    """
    count = 100
    resident = 0
    peak = 0

    def encode(key: int) -> int:
        nonlocal resident, peak
        resident += 1
        peak = max(peak, resident)
        return key

    for _ in ordered_encode(
        range(count),
        encode,
        workers=2,
        window=count,
        pool_factory=_sync_factory,
    ):
        resident -= 1

    assert peak == count


def test_default_pool_is_used_when_no_factory_injected() -> None:
    """With no injected factory the driver fans out on a real thread pool."""
    seen: set[int] = set()
    lock = threading.Lock()

    def encode(key: int) -> int:
        with lock:
            seen.add(threading.get_ident())
        return key

    out = list(ordered_encode(range(32), encode, workers=4, window=8))

    assert out == list(range(32))
    # A real pool ran encodes off the calling thread (more than one thread id).
    assert len(seen) > 1


def test_injected_factory_is_used() -> None:
    """An injected ``pool_factory`` receives the worker count and is used."""
    built: list[int] = []

    def factory(*, max_workers: int) -> _SyncPool:
        built.append(max_workers)
        return _SyncPool(max_workers=max_workers)

    out = list(
        ordered_encode(
            range(6),
            lambda k: k + 1,
            workers=3,
            window=4,
            pool_factory=factory,
        )
    )

    assert out == [1, 2, 3, 4, 5, 6]
    assert built == [3]


def test_rejects_non_positive_workers() -> None:
    """``workers < 1`` is a ValueError (raised on iteration)."""
    with pytest.raises(ValueError, match="workers must be >= 1"):
        list(ordered_encode([1], lambda k: k, workers=0, window=1))


def test_rejects_non_positive_window() -> None:
    """``window < 1`` is a ValueError (raised on iteration)."""
    with pytest.raises(ValueError, match="window must be >= 1"):
        list(ordered_encode([1], lambda k: k, workers=2, window=0))
