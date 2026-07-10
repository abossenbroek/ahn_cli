"""Apple-silicon Metal backend: the kNN primitive as a real ``metal_kernel``.

IDW and kriging spend nearly all their time in the k-nearest-neighbour search
(``O(q * n)`` distance work). :class:`MlxBackend` runs that search as a
hand-written ``mlx.fast.metal_kernel`` on the GPU, returning the same
``(sq_dist, idx)`` contract as the numpy reference so the interpolation
algorithms are unchanged.

**Determinism.** The GPU searches in float32, which on dense data can flip which
points sit at the k-boundary; :func:`_refine` re-ranks ``k + margin`` candidates
in float64 host-side, so the neighbour set matches the numpy reference (exact but
for genuinely equidistant ties) and the returned distances are
``numpy.allclose`` to it -- not byte-identical. The numpy backend remains the CLI
default and the byte-deterministic path; ``--backend mlx`` opts into the
accelerator.

**Performance.** The raw float32 kNN is a brute-force ``O(q * n)`` scan -- fast
per-query on the GPU, but it does not out-scale scipy's indexed ``cKDTree`` once
the float64 refinement is included, and cannot handle a full ortho tile
(``q * n`` is astronomical there). It is a correct, real ``metal_kernel``; a
binned/grid kernel is the route to a robust at-scale speedup (a documented
follow-up). Prefer the numpy default unless benchmarking shows a win for the
workload at hand.

**Coverage.** Every branch (empty source, empty target) is host-side Python; the
GPU kernel is straight-line per thread. The ``mlx.core`` handle *and*
``importlib.import_module`` are injectable, so a numpy-backed fake satisfying the
narrow :class:`MlxModule` surface exercises every host line -- and both the
"mlx present / absent" paths of :func:`select_backend` -- on machines without
``mlx`` (the Linux CI runners). The real kernel's numerical correctness is proven
by the Apple-only equivalence test, which is skipped where ``mlx`` is absent.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np
import numpy.typing as npt

from ahn_cli.reconcile.backend import InterpBackend, NumpyBackend

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

_THREADGROUP = 256
"""Threads per Metal threadgroup; the grid is one thread per target cell."""

_REFINE_MARGIN = 16
"""Extra float32 candidates fetched per query so the exact float64 refinement
below re-selects the *true* k nearest. The GPU's float32 distances can flip which
points sit at the k-boundary on dense data (swinging an interpolated elevation by
metres at surface discontinuities); fetching ``k + margin`` candidates and
re-ranking them in float64 host-side recovers the numpy reference's neighbour set
(exact but for genuine equidistant ties), so ``--backend mlx`` stays
``allclose``-faithful, not merely close."""

_KNN_SOURCE = """
    uint qi = thread_position_in_grid.x;
    int total = dims[0];
    int n = dims[1];
    if ((int)qi >= total) {{ return; }}
    float bx = tx[qi];
    float by = ty[qi];
    float best_sq[{k}];
    int best_idx[{k}];
    for (int t = 0; t < {k}; t++) {{
        best_sq[t] = 1e30f;
        best_idx[t] = -1;
    }}
    for (int j = 0; j < n; j++) {{
        float dx = sx[j] - bx;
        float dy = sy[j] - by;
        float d = dx * dx + dy * dy;
        if (d < best_sq[{k} - 1]) {{
            int p = {k} - 1;
            while (p > 0 && best_sq[p - 1] > d) {{
                best_sq[p] = best_sq[p - 1];
                best_idx[p] = best_idx[p - 1];
                p--;
            }}
            best_sq[p] = d;
            best_idx[p] = j;
        }}
    }}
    for (int t = 0; t < {k}; t++) {{
        out_idx[qi * {k} + t] = best_idx[t];
    }}
"""
"""The candidate-kNN kernel *body* (mlx synthesises the signature from the
input/output names). It returns the ``{k}`` nearest *indices* only -- exact
distances are recomputed in float64 host-side. ``{k}`` is substituted with the
candidate count so the per-thread best arrays are fixed-size; mlx caches the
compiled kernel per distinct source."""


def _refine(
    target_xy: npt.NDArray[np.float64],
    source_xy: npt.NDArray[np.float64],
    candidate_idx: npt.NDArray[np.intp],
    k: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.intp]]:
    """Re-rank the GPU candidates in float64 and take the true ``k`` nearest.

    Recomputes exact squared distances to the candidate indices, then returns
    the ``k`` smallest per row ordered by ascending distance with source-index
    tie-break -- matching :func:`ahn_cli.reconcile.neighbors.knn` (exact but for
    genuinely equidistant boundary ties).
    """
    candidate_points = source_xy[candidate_idx]
    diff = candidate_points - target_xy[:, np.newaxis, :]
    sq = (diff * diff).sum(axis=2)
    order = np.lexsort((candidate_idx, sq), axis=1)[:, :k]
    return (
        np.take_along_axis(sq, order, axis=1),
        np.take_along_axis(candidate_idx, order, axis=1),
    )


class MlxArray(Protocol):
    """The narrow ``mlx.core`` array surface this backend relies on."""

    def __array__(self) -> npt.NDArray[np.generic]:
        """Return a numpy view of the array (the device-to-host bridge)."""
        ...


class MlxKernel(Protocol):
    """A compiled ``metal_kernel`` callable."""

    def __call__(
        self,
        *,
        inputs: Sequence[MlxArray],
        grid: tuple[int, int, int],
        threadgroup: tuple[int, int, int],
        output_shapes: Sequence[tuple[int, ...]],
        output_dtypes: Sequence[object],
    ) -> tuple[MlxArray, ...]:
        """Launch the kernel, returning its output device arrays."""
        ...


class MlxFast(Protocol):
    """The ``mlx.core.fast`` surface: just ``metal_kernel``."""

    def metal_kernel(
        self,
        *,
        name: str,
        input_names: Sequence[str],
        output_names: Sequence[str],
        source: str,
    ) -> MlxKernel:
        """Jit-compile a custom Metal kernel from a source body string."""
        ...


class MlxModule(Protocol):
    """The narrow ``mlx.core`` module surface :class:`MlxBackend` relies on."""

    @property
    def fast(self) -> MlxFast:
        """The ``mlx.core.fast`` submodule holding ``metal_kernel``."""
        ...

    @property
    def float32(self) -> object:
        """The float32 dtype sentinel (an ``output_dtypes`` value)."""
        ...

    @property
    def int32(self) -> object:
        """The int32 dtype sentinel (an ``output_dtypes`` value)."""
        ...

    def array(self, values: object) -> MlxArray:
        """Build a device array from array-like ``values``."""
        ...

    def eval(self, *arrays: MlxArray) -> None:
        """Force evaluation of the given device arrays."""
        ...


class MlxBackend:
    """kNN backend routing through an injected ``mlx.core`` handle.

    The handle is stored, never imported here, so this class imports and
    type-checks without ``mlx`` installed. Every numerical op goes through the
    narrow :class:`MlxModule` surface, so a numpy-backed fake covers every host
    line.
    """

    def __init__(self, mx: MlxModule) -> None:
        """Store the injected ``mlx.core``-like handle."""
        self._mx = mx

    @property
    def name(self) -> str:
        """Identify this backend as ``"mlx"``."""
        return "mlx"

    def knn(
        self,
        target_xy: npt.NDArray[np.float64],
        source_xy: npt.NDArray[np.float64],
        k: int,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.intp]]:
        """Return the ``k`` nearest source points of each target, on the GPU.

        Mirrors :func:`ahn_cli.reconcile.neighbors.knn`: ``(sq_dist, idx)``
        shaped ``(q, min(k, n))`` with each row ordered by ascending squared
        distance and source-index tie-break. The GPU finds ``k + margin`` float32
        candidates; :func:`_refine` re-ranks them in float64, so the neighbour
        set matches the numpy reference (exact but for genuine ties) and the
        returned distances are ``allclose`` to it.
        """
        q = target_xy.shape[0]
        n = source_xy.shape[0]
        k_eff = min(k, n)
        if k_eff == 0:
            return (
                np.empty((q, 0), dtype=np.float64),
                np.empty((q, 0), dtype=np.intp),
            )
        if q == 0:
            return (
                np.empty((0, k_eff), dtype=np.float64),
                np.empty((0, k_eff), dtype=np.intp),
            )
        candidate_k = min(n, k_eff + _REFINE_MARGIN)
        candidate_idx = self._gpu_candidates(
            target_xy, source_xy, candidate_k
        )
        return _refine(target_xy, source_xy, candidate_idx, k_eff)

    def _gpu_candidates(
        self,
        target_xy: npt.NDArray[np.float64],
        source_xy: npt.NDArray[np.float64],
        candidate_k: int,
    ) -> npt.NDArray[np.intp]:
        """Return the ``candidate_k`` nearest source indices per target, on GPU."""
        mx = self._mx
        q = target_xy.shape[0]
        n = source_xy.shape[0]
        kernel = mx.fast.metal_kernel(
            name=f"reconcile_knn_{candidate_k}",
            input_names=["tx", "ty", "sx", "sy", "dims"],
            output_names=["out_idx"],
            source=_KNN_SOURCE.format(k=candidate_k),
        )
        inputs = [
            mx.array(np.ascontiguousarray(target_xy[:, 0], dtype=np.float32)),
            mx.array(np.ascontiguousarray(target_xy[:, 1], dtype=np.float32)),
            mx.array(np.ascontiguousarray(source_xy[:, 0], dtype=np.float32)),
            mx.array(np.ascontiguousarray(source_xy[:, 1], dtype=np.float32)),
            mx.array(np.array([q, n], dtype=np.int32)),
        ]
        outputs = kernel(
            inputs=inputs,
            grid=(q, 1, 1),
            threadgroup=(min(q, _THREADGROUP), 1, 1),
            output_shapes=[(q * candidate_k,)],
            output_dtypes=[mx.int32],
        )
        out_idx = outputs[0]
        mx.eval(out_idx)
        return np.asarray(out_idx).astype(np.intp).reshape(q, candidate_k)


def select_backend(
    *,
    prefer_gpu: bool = True,
    import_module: Callable[[str], object] = importlib.import_module,
) -> InterpBackend:
    """Return the Metal backend when available, else the numpy reference.

    Contract:
        - ``prefer_gpu`` (default ``True``) tries to import ``mlx.core`` and
          returns an :class:`MlxBackend` around it; ``False`` forces the numpy
          reference without importing anything.
        - Falls back to :class:`~ahn_cli.reconcile.backend.NumpyBackend`
          whenever ``mlx`` cannot be imported (e.g. the Linux CI runners).

    ``import_module`` is injectable so both the "mlx present" and "mlx absent"
    branches are exercisable without ``mlx`` installed.
    """
    if prefer_gpu:
        try:
            module = import_module("mlx.core")
        except ImportError:
            module = None
        if module is not None:
            return MlxBackend(cast("MlxModule", module))
    return NumpyBackend()
