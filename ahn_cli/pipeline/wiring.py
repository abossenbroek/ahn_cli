"""Translate parsed :mod:`~ahn_cli.pipeline.spec` value objects into stages.

The spec favours long, self-explanatory keys (``voxel_size_m``, ``idw:
{power, neighbors}``); the stage adapters take the verbs' own value objects
(:class:`~ahn_cli.prep.decimate.Thinning`, an explicit neighbour count). These
pure helpers reconcile the two representations explicitly -- notably the
``voxel_size_m`` -> discrete grade resolution, which errors on a size that has
no exact grade rather than silently rounding.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ahn_cli.pipeline.errors import PipelineError
from ahn_cli.prep.decimate import (
    GRADE_MAX,
    GRADE_MIN,
    PoissonThinning,
    ThinMethod,
    VoxelThinning,
    voxel_size_for_grade,
)
from ahn_cli.reconcile.method import IdwInterp, KrigingInterp

if TYPE_CHECKING:
    from ahn_cli.pipeline.spec import ThinStage as ThinStageSpec
    from ahn_cli.prep.decimate import Thinning
    from ahn_cli.reconcile.method import InterpMethod

__all__ = [
    "DEFAULT_LINEAR_NEIGHBORS",
    "grade_for_voxel_size",
    "neighbors_for",
    "thinning_for",
]

DEFAULT_LINEAR_NEIGHBORS = 6
"""Halo-floor neighbour count for linear interpolation.

:class:`~ahn_cli.reconcile.method.LinearInterp` carries no ``k`` (it is a
Delaunay barycentric interpolant), but the halo correctness floor still needs a
neighbour count: a barycentric estimate at a target uses the three vertices of
its enclosing triangle, whose reach a small, explicit neighbour count bounds
with margin. Six -- twice the triangle's three vertices -- is that conservative
count.
"""

_GRADE_BY_SIZE: dict[float, int] = {
    voxel_size_for_grade(grade): grade
    for grade in range(GRADE_MIN, GRADE_MAX + 1)
}
"""Reverse of :func:`~ahn_cli.prep.decimate.voxel_size_for_grade` (exact sizes)."""


def grade_for_voxel_size(voxel_size_m: float) -> int:
    """Return the discrete voxel grade whose edge length is ``voxel_size_m``.

    Contract:
        - The spec stores a metres edge length; the ``prep`` voxel thinner is
          graded (:data:`~ahn_cli.prep.decimate._VOXEL_SIZES`). Only an exact
          match resolves -- ``0.0`` -> grade 0 (identity), ``0.25`` -> grade 1,
          ``1.0`` -> grade 3, and so on.

    Failure modes:
        - :class:`~ahn_cli.pipeline.errors.PipelineError` if ``voxel_size_m`` is
          not one of the discrete grade edge lengths (the gap is reconciled
          explicitly, never silently rounded).
    """
    grade = _GRADE_BY_SIZE.get(voxel_size_m)
    if grade is None:
        choices = sorted(_GRADE_BY_SIZE)
        msg = (
            f"voxel_size_m {voxel_size_m} has no exact prep grade; choose one "
            f"of {choices} metres (or use voxel_grade in the spec)."
        )
        raise PipelineError(msg)
    return grade


def thinning_for(stage: ThinStageSpec) -> Thinning:
    """Return the ``prep`` thinning request for a spec ``thin`` stage.

    Voxel thinning resolves ``voxel_size_m`` to a discrete grade
    (:func:`grade_for_voxel_size`); Poisson thinning passes ``radius_m`` and
    ``seed`` straight through.

    Failure modes:
        - :class:`~ahn_cli.pipeline.errors.PipelineError` via
          :func:`grade_for_voxel_size` on a voxel size with no exact grade.
    """
    if stage.method is ThinMethod.VOXEL:
        # The spec's voxel branch always carries a resolved voxel_size_m.
        assert stage.voxel_size_m is not None  # noqa: S101 -- spec invariant
        return VoxelThinning(grade=grade_for_voxel_size(stage.voxel_size_m))
    # The spec's poisson branch always carries a positive radius_m.
    assert stage.radius_m is not None  # noqa: S101 -- spec invariant
    return PoissonThinning(radius=stage.radius_m, seed=stage.seed)


def neighbors_for(method: InterpMethod) -> int:
    """Return the halo-floor neighbour count for a reconcile ``method``.

    IDW and kriging carry their own ``k``; linear has none, so it uses
    :data:`DEFAULT_LINEAR_NEIGHBORS`.
    """
    if isinstance(method, (IdwInterp, KrigingInterp)):
        return method.k
    return DEFAULT_LINEAR_NEIGHBORS
