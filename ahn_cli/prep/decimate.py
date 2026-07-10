"""Prep-context graded thinning: voxel-grid + Poisson-disk decimation.

Skeleton for the WP11 red step: every public symbol is declared so the tests
import and run, but the numerical logic and validation are intentionally
missing, so the assertions fail. The green step fills the bodies in.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Protocol

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from collections.abc import Callable

GRADE_MIN = 0
GRADE_MAX = 9
DEFAULT_SEED = 0


class ThinMethod(Enum):
    """The available graded-thinning methods."""

    VOXEL = "voxel"
    POISSON = "poisson"


def voxel_size_for_grade(grade: int) -> float:
    """Return the voxel edge length for a grade (unimplemented)."""
    _ = grade
    return 0.0


@dataclass(frozen=True)
class VoxelThinning:
    """A voxel-grid thinning request (validation unimplemented)."""

    grade: int


@dataclass(frozen=True)
class PoissonThinning:
    """A Poisson-disk thinning request (validation unimplemented)."""

    radius: float
    seed: int = DEFAULT_SEED


Thinning = VoxelThinning | PoissonThinning


class MlxArray(Protocol):
    """The narrow mlx array surface."""

    def __getitem__(self, key: object) -> MlxArray:
        """Slice the array."""
        ...

    def __array__(self) -> npt.NDArray[np.generic]:
        """Return a numpy view."""
        ...


class MlxModule(Protocol):
    """The narrow mlx module surface."""

    def array(self, values: object) -> MlxArray:
        """Build a device array."""
        ...


class DecimationBackend(Protocol):
    """The numerical primitives a decimation backend must provide."""

    @property
    def name(self) -> str:
        """A short backend identifier."""
        ...

    def voxel_first_indices(
        self, group_ids: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.intp]:
        """Min original index per distinct group id."""
        ...

    def squared_distances(
        self,
        point: npt.NDArray[np.float64],
        others: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Row-wise squared distances."""
        ...


class NumpyBackend:
    """Pure-numpy reference backend (unimplemented)."""

    @property
    def name(self) -> str:
        """Identify this backend."""
        return "unset"

    def voxel_first_indices(
        self, group_ids: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.intp]:
        """Unimplemented."""
        _ = group_ids
        return np.empty(0, dtype=np.intp)

    def squared_distances(
        self,
        point: npt.NDArray[np.float64],
        others: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Unimplemented."""
        _ = (point, others)
        return np.empty(0, dtype=np.float64)


class MlxBackend:
    """MLX accelerator backend (unimplemented)."""

    def __init__(self, mx: MlxModule) -> None:
        """Store the handle."""
        self._mx = mx

    @property
    def name(self) -> str:
        """Identify this backend."""
        return "unset"

    def voxel_first_indices(
        self, group_ids: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.intp]:
        """Unimplemented."""
        _ = group_ids
        return np.empty(0, dtype=np.intp)

    def squared_distances(
        self,
        point: npt.NDArray[np.float64],
        others: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Unimplemented."""
        _ = (point, others)
        return np.empty(0, dtype=np.float64)


def select_backend(
    *,
    prefer_gpu: bool = True,
    import_module: Callable[[str], object] | None = None,
) -> DecimationBackend:
    """Return a backend (unimplemented)."""
    _ = (prefer_gpu, import_module)
    return NumpyBackend()


def decimate_voxel(
    coords: npt.NDArray[np.float64],
    grade: int,
    *,
    backend: DecimationBackend,
) -> npt.NDArray[np.intp]:
    """Voxel-grid thinning (unimplemented)."""
    _ = (coords, grade, backend)
    return np.empty(0, dtype=np.intp)


def decimate_poisson(
    coords: npt.NDArray[np.float64],
    radius: float,
    *,
    seed: int = DEFAULT_SEED,
    backend: DecimationBackend,
) -> npt.NDArray[np.intp]:
    """Poisson-disk thinning (unimplemented)."""
    _ = (coords, radius, seed, backend)
    return np.empty(0, dtype=np.intp)


def thin(
    coords: npt.NDArray[np.float64],
    spec: Thinning,
    *,
    backend: DecimationBackend,
) -> npt.NDArray[np.intp]:
    """Dispatch a thinning request (unimplemented)."""
    _ = (coords, spec, backend)
    return np.empty(0, dtype=np.intp)
