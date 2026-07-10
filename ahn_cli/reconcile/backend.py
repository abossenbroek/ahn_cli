"""Interpolation backends: the numpy reference and its narrow primitive surface.

The interpolation algorithms (:mod:`ahn_cli.reconcile.interpolate`) are written
once against the :class:`InterpBackend` protocol. A backend supplies a single
data-parallel primitive -- the k-nearest-neighbour search that IDW and kriging
share -- so the algorithms, and every branch in them, stay backend-independent.

:class:`NumpyBackend` is the pure numpy/scipy reference and the correctness
source of truth; it is the CLI default, so the shipped output is byte-identical
across runs and machines. The optional Apple-silicon :class:`MlxBackend` (see
:mod:`ahn_cli.reconcile.mlx_backend`) accelerates the same kNN on the GPU and is
``numpy.allclose``-equivalent, not byte-identical -- GPU float reductions differ
from CPU in the last ULPs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from ahn_cli.reconcile.neighbors import knn

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt


class InterpBackend(Protocol):
    """The data-parallel primitive an interpolation backend must provide."""

    @property
    def name(self) -> str:
        """A short backend identifier (e.g. ``"cpu"`` or ``"mlx"``)."""
        ...

    def knn(
        self,
        target_xy: npt.NDArray[np.float64],
        source_xy: npt.NDArray[np.float64],
        k: int,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.intp]]:
        """Return the ``k`` nearest source points of each target.

        Contract:
            - Mirrors :func:`ahn_cli.reconcile.neighbors.knn` exactly: returns
              ``(sq_dist, idx)`` shaped ``(q, min(k, n))``, each row ordered by
              ascending squared distance with source-index tie-break.
        """
        ...


class NumpyBackend:
    """Pure numpy/scipy reference backend; the correctness source of truth."""

    @property
    def name(self) -> str:
        """Identify this backend as ``"cpu"``."""
        return "cpu"

    def knn(
        self,
        target_xy: npt.NDArray[np.float64],
        source_xy: npt.NDArray[np.float64],
        k: int,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.intp]]:
        """Delegate to the scipy ``cKDTree`` reference kNN."""
        return knn(target_xy, source_xy, k)
