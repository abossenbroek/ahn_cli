"""Host-side coverage for the Metal backend via a numpy-backed ``mlx`` fake.

These tests never import ``mlx``: a numpy fake satisfies the narrow
:class:`~ahn_cli.reconcile.mlx_backend.MlxModule` surface, so every host line of
:class:`MlxBackend` and both branches of :func:`select_backend` run on the Linux
CI. The *real* kernel's numerical correctness is proven separately by the
Apple-only equivalence test.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ahn_cli.reconcile.backend import NumpyBackend
from ahn_cli.reconcile.mlx_backend import (
    MlxArray,
    MlxBackend,
    MlxModule,
    select_backend,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


class _FakeArray:
    """A numpy-backed stand-in for an ``mlx.core`` array."""

    def __init__(self, data: npt.NDArray[np.generic]) -> None:
        self._data = data

    def __array__(self) -> npt.NDArray[np.generic]:
        return self._data


class _FakeKernel:
    """A numpy reimplementation of the candidate-kNN kernel's computation."""

    def __call__(
        self,
        *,
        inputs: Sequence[MlxArray],
        grid: tuple[int, int, int],
        threadgroup: tuple[int, int, int],
        output_shapes: Sequence[tuple[int, ...]],
        output_dtypes: Sequence[object],
    ) -> tuple[MlxArray, ...]:
        del grid, threadgroup, output_dtypes
        tx, ty, sx, sy, dims = (np.asarray(item) for item in inputs)
        q, n = int(dims[0]), int(dims[1])
        k = output_shapes[0][0] // q
        target = np.stack([tx, ty], axis=1).astype(np.float32)
        source = np.stack([sx, sy], axis=1).astype(np.float32)
        diff = target[:, None, :] - source[None, :, :]
        sq = (diff * diff).sum(axis=2)
        # Stable argsort matches the kernel's ascending, index-tie-break order.
        order = np.argsort(sq, axis=1, kind="stable")[:, :k]
        out_idx = order.reshape(q * k).astype(np.int32)
        del n
        return (_FakeArray(out_idx),)


class _FakeFast:
    """The ``mlx.core.fast`` stand-in exposing ``metal_kernel``."""

    def metal_kernel(
        self,
        *,
        name: str,
        input_names: Sequence[str],
        output_names: Sequence[str],
        source: str,
    ) -> _FakeKernel:
        del name, input_names, output_names, source
        return _FakeKernel()


class _FakeMlx:
    """A numpy-backed stand-in for the ``mlx.core`` module."""

    @property
    def fast(self) -> _FakeFast:
        return _FakeFast()

    @property
    def float32(self) -> object:
        return np.float32

    @property
    def int32(self) -> object:
        return np.int32

    def array(self, values: object) -> _FakeArray:
        return _FakeArray(np.asarray(values))

    def eval(self, *arrays: MlxArray) -> None:
        del arrays


def _fake_module() -> MlxModule:
    return _FakeMlx()


class TestMlxBackendHost:
    """The MlxBackend host orchestration, exercised through the fake."""

    def test_name(self) -> None:
        """The Metal backend identifies itself as ``"mlx"``."""
        assert MlxBackend(_fake_module()).name == "mlx"

    def test_matches_numpy_backend_on_tie_free_input(self) -> None:
        """On well-separated points the fake kNN matches the numpy reference."""
        rng = np.random.default_rng(7)
        source = rng.random((120, 2)) * 50.0
        target = rng.random((30, 2)) * 50.0
        mlx_sq, mlx_idx = MlxBackend(_fake_module()).knn(target, source, 6)
        ref_sq, ref_idx = NumpyBackend().knn(target, source, 6)
        assert np.array_equal(mlx_idx, ref_idx)
        assert np.allclose(mlx_sq, ref_sq, atol=1e-3)

    def test_empty_source_returns_zero_width(self) -> None:
        """No source points yields a (q, 0) result without launching a kernel."""
        sq, idx = MlxBackend(_fake_module()).knn(
            np.array([[0.0, 0.0], [1.0, 1.0]]), np.empty((0, 2)), 4
        )
        assert sq.shape == (2, 0)
        assert idx.shape == (2, 0)

    def test_empty_target_returns_zero_rows(self) -> None:
        """No targets yields a (0, k') result without launching a kernel."""
        sq, idx = MlxBackend(_fake_module()).knn(
            np.empty((0, 2)), np.array([[0.0, 0.0], [1.0, 0.0]]), 2
        )
        assert sq.shape == (0, 2)
        assert idx.shape == (0, 2)


class TestSelectBackend:
    """Backend selection across the mlx-present / mlx-absent branches."""

    def test_prefers_cpu_when_gpu_not_requested(self) -> None:
        """prefer_gpu=False returns the numpy reference without importing."""
        backend = select_backend(prefer_gpu=False)
        assert backend.name == "cpu"

    def test_returns_mlx_when_import_succeeds(self) -> None:
        """A successful ``mlx.core`` import yields the Metal backend."""
        backend = select_backend(
            prefer_gpu=True, import_module=lambda _name: _FakeMlx()
        )
        assert backend.name == "mlx"

    def test_falls_back_to_cpu_when_import_fails(self) -> None:
        """A failed ``mlx.core`` import falls back to the numpy reference."""

        def _raise(_name: str) -> object:
            raise ImportError(_name)

        backend = select_backend(prefer_gpu=True, import_module=_raise)
        assert backend.name == "cpu"
