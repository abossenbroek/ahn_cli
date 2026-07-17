"""Bounded-window parallel encode driver for the tiles3d build.

A real 2x2 km Westland run showed the standalone ``tiles3d`` build is
single-threaded: ~47 min for 21,845 splat tiles, because the pack writer
(:func:`ahn_cli.tiles3d.pack.write_pack`) drives one CPU-bound
``blob_source(key)`` encode per tile inline on the calling thread while the
writer's own work is pure I/O. Per-tile encoding is the cost that matters
(gaussian build + zstd for splat; quantize/meshopt/JPEG for game;
``.hf``/JPEG for heightfield; float32 glTF + PNG for strict), and every tile
is an independent, deterministic pure function of the terrain, so the encodes
fan out trivially.

:func:`ordered_encode` is that fan-out. It encodes up to ``window`` tiles
concurrently across an injectable worker pool but yields the results in the
**exact** order of ``keys``, so the writer keeps streaming to disk in
canonical order with the output bytes unchanged. Two properties hold at once:
parallel encode *and* bounded, on-disk streaming — at most ``window`` encoded
blobs are ever resident (a small multiple of the worker count), independent of
the tile count. It never "encodes everything into RAM, then writes".

**Thread pool, not process pool.** The heavy per-tile work is numpy array
math plus C-extension codecs (zstd, JPEG via Pillow, pyproj), all of which
release the GIL, so a :class:`~concurrent.futures.ThreadPoolExecutor`
parallelises the hot sections without pickling anything. A process pool would
have to pickle the (large) terrain grid to every worker and pickle the
``blob_source`` closure — which is not picklable at all — for no benefit over
threads on GIL-releasing codecs. The thread pool shares the one in-memory
terrain grid and keeps the encode a pure function, so byte-identity and
determinism are automatic: the results are consumed in ``keys`` order
regardless of the order the workers finish, so ``workers=1`` (the serial
reference, run inline with no pool) and ``workers=N`` produce identical bytes.
"""

from __future__ import annotations

import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Protocol, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from typing_extensions import Self

__all__ = [
    "Pool",
    "PoolFactory",
    "Task",
    "default_window",
    "default_workers",
    "ordered_encode",
]

_K = TypeVar("_K")
_V = TypeVar("_V")
_V_co = TypeVar("_V_co", covariant=True)


class Task(Protocol[_V_co]):
    """A submitted encode's result handle — the driver only needs ``result``.

    :class:`~concurrent.futures.Future` satisfies this; a test's instrumented
    handle need only expose ``result``.
    """

    def result(self) -> _V_co:
        """Return the encoded result, blocking until it is ready."""
        ...


class Pool(Protocol):
    """The minimal ``submit``-and-context-manage pool the driver relies on.

    Satisfied by :class:`~concurrent.futures.ThreadPoolExecutor`; tests inject
    a synchronous or instrumented stand-in to make the bounded-window
    behaviour deterministic.
    """

    def __enter__(self) -> Self:
        """Enter the pool's context, returning the pool."""
        ...

    def __exit__(self, *exc: object) -> object:
        """Exit the pool's context, shutting the workers down."""
        ...

    def submit(self, fn: Callable[[_K], _V], key: _K, /) -> Task[_V]:
        """Schedule ``fn(key)`` and return a handle for its result."""
        ...


class PoolFactory(Protocol):
    """Builds the worker pool the driver fans encodes out across.

    When no factory is injected the driver uses a
    :class:`~concurrent.futures.ThreadPoolExecutor`; tests inject a factory to
    drive the parallel path deterministically in-process.
    """

    def __call__(self, *, max_workers: int) -> Pool:
        """Return a context-managed pool with ``max_workers`` workers."""
        ...


def default_workers() -> int:
    """Return the default worker count: the machine's CPU count (>= 1)."""
    return os.cpu_count() or 1


def default_window(workers: int) -> int:
    """Return the default in-flight window: ``2 * workers`` (min 2).

    A small multiple of the worker count keeps every worker fed with a tile to
    start next while still bounding the resident encoded blobs to ``window``.
    """
    return max(2, 2 * workers)


def _default_pool(*, max_workers: int) -> ThreadPoolExecutor:
    """Build the default thread pool (see the module docstring for why)."""
    return ThreadPoolExecutor(max_workers=max_workers)


def ordered_encode(
    keys: Sequence[_K],
    encode: Callable[[_K], _V],
    *,
    workers: int,
    window: int,
    pool_factory: PoolFactory | None = None,
) -> Iterator[_V]:
    """Encode ``keys`` concurrently, yielding results in ``keys`` order.

    Contract:
        - Yields ``encode(key)`` for every key in ``keys``, strictly in
          ``keys`` order, whatever order the workers finish.
        - ``workers == 1`` runs inline with no pool — the serial reference
          path, byte-identical to calling ``encode`` in a plain loop.
        - ``workers >= 2`` fans the encodes across ``pool_factory`` (default a
          :class:`~concurrent.futures.ThreadPoolExecutor`), keeping at most
          ``window`` encodes in flight and at most ``window`` results resident
          at once — independent of ``len(keys)``.

    Failure modes:
        - :class:`ValueError` if ``workers < 1`` or ``window < 1``.
    """
    if workers < 1:
        msg = f"workers must be >= 1; got {workers}."
        raise ValueError(msg)
    if window < 1:
        msg = f"window must be >= 1; got {window}."
        raise ValueError(msg)
    if workers == 1:
        for key in keys:
            yield encode(key)
        return
    factory = _default_pool if pool_factory is None else pool_factory
    with factory(max_workers=workers) as pool:
        pending: deque[Task[_V]] = deque()
        source = iter(keys)
        for key in _take(source, window):
            pending.append(pool.submit(encode, key))
        while pending:
            yield pending.popleft().result()
            for key in _take(source, 1):
                pending.append(pool.submit(encode, key))


def _take(source: Iterator[_K], count: int) -> list[_K]:
    """Return up to ``count`` items from ``source`` (fewer when exhausted)."""
    taken: list[_K] = []
    for _ in range(count):
        try:
            taken.append(next(source))
        except StopIteration:  # noqa: PERF203 -- exhaustion is the loop's exit
            break
    return taken
