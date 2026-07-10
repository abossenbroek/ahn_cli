"""Prep-context graded thinning: voxel-grid + Poisson-disk decimation.

This transform is *additive* to the legacy nth-point ``decimate`` step in
``ahn_cli.manipulator.ptc_handler`` (which is left untouched): it offers two
spatially-aware thinning methods that the legacy step cannot.

* **Voxel-grid**, graded 0-9. Grade 0 keeps every point; grades 1-9 map to a
  strictly increasing voxel edge length (see :data:`_VOXEL_SIZES`), so density
  strictly decreases with grade. Within each occupied voxel exactly one point is
  kept -- deterministically the one with the smallest original index -- giving a
  spatially uniform thinning that the nth-point step does not.
* **Poisson-disk**. Greedy dart-throwing over a seeded point permutation: a
  point is accepted only if no already-accepted point lies within ``radius``.
  The result satisfies the Poisson-disk minimum-distance property (every kept
  pair is at least ``radius`` apart) and is deterministic given the seed.

Determinism (a load-bearing guardrail): every public entry point returns the
kept original indices sorted ascending, so identical input and parameters yield
byte-identical output regardless of backend.

Backends
--------
The numerical core runs through an injectable :class:`DecimationBackend`. The
default :class:`NumpyBackend` is a pure-numpy reference and the correctness
source of truth. :class:`MlxBackend` is an optional Apple-silicon accelerator
that routes the same computation through an injected ``mlx.core``-like handle
(see :class:`MlxModule`); it never imports ``mlx`` at module scope, so this
module imports -- and is fully tested -- on machines without ``mlx`` (e.g. the
Linux CI runners). Backend selection (:func:`select_backend`) is likewise
injectable, so both the "mlx present" and "mlx absent" paths are exercisable
without ``mlx`` installed.

The mlx handle is used through the narrow :class:`MlxModule`/:class:`MlxArray`
surface only, so a numpy-backed fake satisfying that surface exercises every
adapter line off-device. ``mlx.fast.metal_kernel`` is deliberately *not* used:
the plain ``mlx.core`` op surface keeps both methods exactly equivalent to the
CPU reference and keeps the fake minimal; a hand-written Metal kernel is a
future perf optimization, not a correctness requirement.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from collections.abc import Callable

GRADE_MIN = 0
"""Smallest voxel grade: identity (every point kept)."""

GRADE_MAX = 9
"""Largest voxel grade: coarsest thinning."""

DEFAULT_SEED = 0
"""Default Poisson-disk RNG seed. Fixed (never wall-clock) so runs reproduce."""

_VOXEL_SIZES: tuple[float, ...] = (
    0.0,  # grade 0: identity, size unused
    0.25,
    0.5,
    1.0,
    2.0,
    4.0,
    8.0,
    16.0,
    32.0,
    64.0,
)
"""Voxel edge length (metres, EPSG:28992) per grade.

Indexed by grade 0-9. Entry 0 is a placeholder -- grade 0 is the identity and
never quantizes. Entries 1-9 are strictly increasing, so a higher grade always
yields a coarser grid and thus strictly fewer (or equal) points.
"""


class ThinMethod(Enum):
    """The available graded-thinning methods (the ``--thin-method`` tokens)."""

    VOXEL = "voxel"
    POISSON = "poisson"


def voxel_size_for_grade(grade: int) -> float:
    """Return the voxel edge length in metres for a voxel ``grade``.

    Contract:
        - ``grade`` must be an integer in ``[GRADE_MIN, GRADE_MAX]``.
        - Grade 0 returns ``0.0`` (the identity; callers must not quantize).
        - Grades 1-9 return strictly increasing positive edge lengths.

    Failure modes:
        - :class:`ValueError` if ``grade`` is outside ``[0, 9]``.
    """
    if not GRADE_MIN <= grade <= GRADE_MAX:
        msg = (
            f"voxel grade must be in [{GRADE_MIN}, {GRADE_MAX}]; got {grade}."
        )
        raise ValueError(msg)
    return _VOXEL_SIZES[grade]


@dataclass(frozen=True)
class VoxelThinning:
    """A validated request to voxel-grid thin at a given grade.

    Contract:
        - ``grade`` is an integer in ``[GRADE_MIN, GRADE_MAX]``; higher is
          coarser (grade 0 keeps everything).

    Invariants:
        - Frozen value object, equal by field value.

    Failure modes:
        - :class:`ValueError` if ``grade`` is out of range.
    """

    grade: int

    def __post_init__(self) -> None:
        """Reject a grade outside ``[GRADE_MIN, GRADE_MAX]``."""
        if not GRADE_MIN <= self.grade <= GRADE_MAX:
            msg = (
                f"voxel grade must be in [{GRADE_MIN}, {GRADE_MAX}]; "
                f"got {self.grade}."
            )
            raise ValueError(msg)


@dataclass(frozen=True)
class PoissonThinning:
    """A validated request to Poisson-disk thin at a given radius.

    Contract:
        - ``radius`` is the minimum spacing in metres; must be finite and
          strictly positive.
        - ``seed`` is the RNG seed making the greedy sampling deterministic;
          defaults to :data:`DEFAULT_SEED`.

    Invariants:
        - Frozen value object, equal by field value.

    Failure modes:
        - :class:`ValueError` if ``radius`` is not finite and positive.
    """

    radius: float
    seed: int = DEFAULT_SEED

    def __post_init__(self) -> None:
        """Reject a non-finite or non-positive ``radius``."""
        if not np.isfinite(self.radius) or self.radius <= 0.0:
            msg = f"poisson radius must be finite and positive; got {self.radius}."
            raise ValueError(msg)


Thinning = VoxelThinning | PoissonThinning
"""A validated thinning request: either voxel-grid or Poisson-disk."""


class MlxArray(Protocol):
    """The narrow ``mlx.core`` array surface :class:`MlxBackend` relies on.

    Only slicing and the numpy-conversion hook are declared -- every arithmetic
    op is routed through :class:`MlxModule` functions instead of operators, so a
    numpy-backed fake can satisfy this surface exactly. ``mlx`` arrays support
    all of these at runtime.
    """

    def __getitem__(self, key: object) -> MlxArray:
        """Slice or index the array, returning another array."""
        ...

    def __array__(self) -> npt.NDArray[np.generic]:
        """Return a numpy view of the array (the device-to-host bridge)."""
        ...


class MlxModule(Protocol):
    """The narrow ``mlx.core`` module surface :class:`MlxBackend` relies on.

    Declaring only the used symbols keeps the injectable handle honest and the
    numpy-backed test fake minimal. Every symbol is a real ``mlx.core`` function
    at runtime.
    """

    def array(self, values: object) -> MlxArray:
        """Build a device array from array-like ``values``."""
        ...

    def arange(self, size: int) -> MlxArray:
        """Return ``0..size-1`` as a device array."""
        ...

    def argsort(self, values: MlxArray) -> MlxArray:
        """Return the indices that sort ``values`` ascending."""
        ...

    def take(self, values: MlxArray, indices: MlxArray) -> MlxArray:
        """Gather ``values`` at ``indices``."""
        ...

    def multiply(self, left: MlxArray, right: MlxArray | int) -> MlxArray:
        """Return the elementwise product."""
        ...

    def add(self, left: MlxArray, right: MlxArray) -> MlxArray:
        """Return the elementwise sum."""
        ...

    def subtract(self, left: MlxArray, right: MlxArray) -> MlxArray:
        """Return the elementwise difference ``left - right``."""
        ...

    def not_equal(self, left: MlxArray, right: MlxArray) -> MlxArray:
        """Return the elementwise ``left != right`` mask."""
        ...

    def sum(self, values: MlxArray, axis: int) -> MlxArray:
        """Reduce ``values`` by summation along ``axis``."""
        ...


class DecimationBackend(Protocol):
    """The numerical primitives a decimation backend must provide.

    The thinning algorithms are written once against this surface; a backend
    supplies only the two data-parallel kernels. Contracts are exact so the CPU
    reference and any accelerator produce identical voxel results.
    """

    @property
    def name(self) -> str:
        """A short backend identifier (e.g. ``"cpu"`` or ``"mlx"``)."""
        ...

    def voxel_first_indices(
        self, group_ids: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.intp]:
        """Return, per distinct group id, the minimum original index.

        Contract:
            - ``group_ids`` labels each point with its voxel's dense group id.
            - Returns the original index of the first (smallest-index) point in
              each distinct group, sorted ascending. Empty in, empty out.
        """
        ...

    def squared_distances(
        self,
        point: npt.NDArray[np.float64],
        others: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Return the squared Euclidean distance from ``point`` to each row.

        Contract:
            - ``point`` is shape ``(3,)``; ``others`` is shape ``(m, 3)``.
            - Returns shape ``(m,)`` squared distances; empty ``others`` yields
              an empty result.
        """
        ...


class NumpyBackend:
    """Pure-numpy reference backend; the correctness source of truth."""

    @property
    def name(self) -> str:
        """Identify this backend as ``"cpu"``."""
        return "cpu"

    def voxel_first_indices(
        self, group_ids: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.intp]:
        """Min original index per distinct group id, via ``numpy.unique``."""
        if group_ids.size == 0:
            return np.empty(0, dtype=np.intp)
        _, first_occurrence = np.unique(group_ids, return_index=True)
        return np.sort(first_occurrence).astype(np.intp)

    def squared_distances(
        self,
        point: npt.NDArray[np.float64],
        others: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Row-wise squared Euclidean distance, computed with numpy."""
        delta = others - point
        return np.einsum("ij,ij->i", delta, delta)


class MlxBackend:
    """Apple-silicon accelerator routing through an injected ``mlx.core`` handle.

    The handle is stored, never imported here, so this class is importable and
    exercisable without ``mlx`` installed. Every numerical op goes through the
    :class:`MlxModule` surface, so a numpy-backed fake covers every line.
    """

    def __init__(self, mx: MlxModule) -> None:
        """Store the injected ``mlx.core``-like handle."""
        self._mx = mx

    @property
    def name(self) -> str:
        """Identify this backend as ``"mlx"``."""
        return "mlx"

    def voxel_first_indices(
        self, group_ids: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.intp]:
        """Min original index per group id, via on-device argsort + diff.

        A composite key ``group_id * n + original_index`` is argsorted so each
        group's points land contiguously in ascending-index order; the first of
        each group (an on-device adjacent ``not_equal`` on the sorted group ids)
        is therefore its minimum-index member. Mirrors
        :meth:`NumpyBackend.voxel_first_indices` exactly.
        """
        mx = self._mx
        n = group_ids.shape[0]
        if n == 0:
            return np.empty(0, dtype=np.intp)
        groups = mx.array(group_ids)
        composite = mx.add(mx.multiply(groups, n), mx.arange(n))
        order = mx.argsort(composite)
        sorted_groups = mx.take(groups, order)
        starts = mx.not_equal(sorted_groups[1:], sorted_groups[:-1])
        order_np = np.asarray(order).astype(np.intp)
        starts_np = np.asarray(starts).astype(bool)
        keep = np.empty(n, dtype=bool)
        keep[0] = True
        keep[1:] = starts_np
        return np.sort(order_np[keep]).astype(np.intp)

    def squared_distances(
        self,
        point: npt.NDArray[np.float64],
        others: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Row-wise squared Euclidean distance, computed on-device."""
        mx = self._mx
        delta = mx.subtract(mx.array(others), mx.array(point))
        squared = mx.sum(mx.multiply(delta, delta), axis=1)
        return np.asarray(squared).astype(np.float64)


def select_backend(
    *,
    prefer_gpu: bool = True,
    import_module: Callable[[str], object] = importlib.import_module,
) -> DecimationBackend:
    """Return the accelerated backend when available, else the CPU reference.

    Contract:
        - ``prefer_gpu`` (default ``True``) tries to import ``mlx.core`` and
          returns an :class:`MlxBackend` around it; ``False`` forces the CPU
          reference without importing anything.
        - Falls back to :class:`NumpyBackend` whenever ``mlx`` cannot be
          imported (e.g. on the Linux CI runners).

    ``import_module`` is injectable so both the "mlx present" and "mlx absent"
    branches are exercisable without ``mlx`` installed -- ``mlx.core`` is never
    imported at module scope.
    """
    if prefer_gpu:
        try:
            module = import_module("mlx.core")
        except ImportError:
            module = None
        if module is not None:
            return MlxBackend(cast("MlxModule", module))
    return NumpyBackend()


def _voxel_group_ids(
    coords: npt.NDArray[np.float64], voxel_size: float
) -> npt.NDArray[np.int64]:
    """Assign each point a dense voxel group id (backend-agnostic quantization).

    Coordinates are shifted to a per-cloud origin, floored onto the voxel grid,
    and their integer triples rank-compressed to dense ``[0, k)`` ids via
    ``numpy.unique``. Rank compression keeps the id well below the point count,
    so the accelerator's ``group_id * n`` composite key never overflows int64
    for realistically bounded tiles.
    """
    origin = coords.min(axis=0)
    cells = np.floor((coords - origin) / voxel_size).astype(np.int64)
    # ``np.unique(..., return_inverse=True, axis=0)`` types the inverse as a
    # partially unknown array under numpy 2.2.6's stubs but as ``intp`` under
    # 2.3.2. Cast to the ``int64`` this function returns: on 2.2.6 it launders
    # the unknown to a concrete dtype, and on 2.3.2 ``int64`` differs from the
    # inferred ``intp`` so the cast is not flagged unnecessary. (On every target
    # platform ``intp`` is 64-bit, so this is a no-op at runtime.)
    inverse = cast(
        "npt.NDArray[np.int64]",
        np.unique(cells, axis=0, return_inverse=True)[1],
    )
    return inverse.reshape(-1)


def decimate_voxel(
    coords: npt.NDArray[np.float64],
    grade: int,
    *,
    backend: DecimationBackend,
) -> npt.NDArray[np.intp]:
    """Return the kept indices of a voxel-grid thinning at ``grade``.

    Contract:
        - ``coords`` is an ``(n, 3)`` float array of world coordinates.
        - ``grade`` is in ``[GRADE_MIN, GRADE_MAX]``; grade 0 keeps every point.
        - Returns the original indices to keep, sorted ascending: one point per
          occupied voxel (the smallest original index within it).

    Invariants:
        - Deterministic and backend-independent: identical ``coords``/``grade``
          yield the identical index set on any backend.

    Failure modes:
        - :class:`ValueError` if ``grade`` is out of range.
    """
    size = voxel_size_for_grade(grade)
    n = coords.shape[0]
    if grade == GRADE_MIN or n == 0:
        return np.arange(n, dtype=np.intp)
    group_ids = _voxel_group_ids(coords, size)
    return backend.voxel_first_indices(group_ids)


def decimate_poisson(
    coords: npt.NDArray[np.float64],
    radius: float,
    *,
    seed: int = DEFAULT_SEED,
    backend: DecimationBackend,
) -> npt.NDArray[np.intp]:
    """Return the kept indices of a Poisson-disk thinning at ``radius``.

    Contract:
        - ``coords`` is an ``(n, 3)`` float array of world coordinates.
        - ``radius`` is the minimum spacing (metres); must be finite, positive.
        - ``seed`` seeds the point-visitation permutation (deterministic).
        - Returns the kept original indices, sorted ascending. Every kept pair
          is at least ``radius`` apart (the Poisson-disk property).

    Invariants:
        - Deterministic: identical ``coords``/``radius``/``seed`` yield the
          identical index set.

    Failure modes:
        - :class:`ValueError` if ``radius`` is not finite and positive.
    """
    if not np.isfinite(radius) or radius <= 0.0:
        msg = f"poisson radius must be finite and positive; got {radius}."
        raise ValueError(msg)
    n = coords.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.intp)

    origin = coords.min(axis=0)
    radius_sq = radius * radius
    # A uniform grid with cell = radius: any point within radius of a candidate
    # lies in the candidate's cell or one of the 26 neighbours, so only those
    # cells are searched.
    cells = np.floor((coords - origin) / radius).astype(np.int64)
    accepted_by_cell: dict[tuple[int, int, int], list[int]] = {}
    accepted: list[int] = []

    rng = np.random.default_rng(seed)
    for idx in rng.permutation(n):
        point_idx = int(idx)
        cell = (
            int(cells[point_idx, 0]),
            int(cells[point_idx, 1]),
            int(cells[point_idx, 2]),
        )
        neighbours = _neighbour_indices(cell, accepted_by_cell)
        if _is_isolated(coords, point_idx, neighbours, radius_sq, backend):
            accepted.append(point_idx)
            accepted_by_cell.setdefault(cell, []).append(point_idx)
    return np.sort(np.asarray(accepted, dtype=np.intp))


def _neighbour_indices(
    cell: tuple[int, int, int],
    accepted_by_cell: dict[tuple[int, int, int], list[int]],
) -> list[int]:
    """Collect accepted point indices in ``cell`` and its 26 neighbours."""
    found: list[int] = []
    cx, cy, cz = cell
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                bucket = accepted_by_cell.get((cx + dx, cy + dy, cz + dz))
                if bucket is not None:
                    found.extend(bucket)
    return found


def _is_isolated(
    coords: npt.NDArray[np.float64],
    point_idx: int,
    neighbours: list[int],
    radius_sq: float,
    backend: DecimationBackend,
) -> bool:
    """Return whether ``point_idx`` is at least ``radius`` from every neighbour."""
    if not neighbours:
        return True
    distances = backend.squared_distances(
        coords[point_idx], coords[neighbours]
    )
    return bool(np.all(distances >= radius_sq))


def thin(
    coords: npt.NDArray[np.float64],
    spec: Thinning,
    *,
    backend: DecimationBackend,
) -> npt.NDArray[np.intp]:
    """Dispatch a validated :data:`Thinning` request to its algorithm.

    Contract:
        - Returns the kept original indices, sorted ascending, per the method's
          own contract (:func:`decimate_voxel` / :func:`decimate_poisson`).

    Typed dispatch on the request variant -- no stringly-typed method switch.
    """
    if isinstance(spec, VoxelThinning):
        return decimate_voxel(coords, spec.grade, backend=backend)
    return decimate_poisson(
        coords, spec.radius, seed=spec.seed, backend=backend
    )
