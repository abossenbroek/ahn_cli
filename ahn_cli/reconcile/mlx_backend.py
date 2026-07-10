"""Apple-silicon Metal backend: the kNN primitive as a real ``metal_kernel``.

IDW and kriging spend nearly all their time in the k-nearest-neighbour search
(``O(q * n)`` distance work). :class:`MlxBackend` runs that search as a
hand-written ``mlx.fast.metal_kernel`` on the GPU, returning the same
``(sq_dist, idx)`` contract as the numpy reference so the interpolation
algorithms are unchanged.

**Determinism.** The kernel computes distances in float32 (Metal), so its
squared distances differ from the float64 numpy reference in the last ULPs: the
backend is ``numpy.allclose``-equivalent, not byte-identical. The numpy backend
remains the CLI default and the byte-deterministic path; ``--backend mlx`` opts
into the accelerator. Within the kernel, ties are broken by ascending source
index (the insertion uses a strict ``>`` while scanning sources in order), so on
tie-free inputs the neighbour *sets* match the reference exactly.

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
        out_sq[qi * {k} + t] = best_sq[t];
        out_idx[qi * {k} + t] = best_idx[t];
    }}
"""
"""The kNN kernel *body* (mlx synthesises the signature from the input/output
names). ``{k}`` is substituted with the neighbour count so the per-thread best
arrays are fixed-size; mlx caches the compiled kernel per distinct source."""


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
    ) -> tuple[MlxArray, MlxArray]:
        """Launch the kernel, returning the ``(out_sq, out_idx)`` device arrays."""
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
        distance and source-index tie-break. Squared distances are computed in
        float32, so they are ``allclose``- rather than byte-equal to the numpy
        reference.
        """
        mx = self._mx
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

        kernel = mx.fast.metal_kernel(
            name=f"reconcile_knn_{k_eff}",
            input_names=["tx", "ty", "sx", "sy", "dims"],
            output_names=["out_sq", "out_idx"],
            source=_KNN_SOURCE.format(k=k_eff),
        )
        inputs = [
            mx.array(np.ascontiguousarray(target_xy[:, 0], dtype=np.float32)),
            mx.array(np.ascontiguousarray(target_xy[:, 1], dtype=np.float32)),
            mx.array(np.ascontiguousarray(source_xy[:, 0], dtype=np.float32)),
            mx.array(np.ascontiguousarray(source_xy[:, 1], dtype=np.float32)),
            mx.array(np.array([q, n], dtype=np.int32)),
        ]
        out_sq, out_idx = kernel(
            inputs=inputs,
            grid=(q, 1, 1),
            threadgroup=(min(q, _THREADGROUP), 1, 1),
            output_shapes=[(q * k_eff,), (q * k_eff,)],
            output_dtypes=[mx.float32, mx.int32],
        )
        mx.eval(out_sq, out_idx)
        sq = np.asarray(out_sq).astype(np.float64).reshape(q, k_eff)
        idx = np.asarray(out_idx).astype(np.intp).reshape(q, k_eff)
        return sq, idx


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
