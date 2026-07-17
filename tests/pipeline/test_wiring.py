"""Tests for the spec -> stage translation helpers."""

from __future__ import annotations

import pytest

from ahn_cli.pipeline.errors import PipelineError
from ahn_cli.pipeline.spec import ThinStage
from ahn_cli.pipeline.wiring import (
    DEFAULT_LINEAR_NEIGHBORS,
    grade_for_voxel_size,
    neighbors_for,
    thinning_for,
)
from ahn_cli.prep.decimate import (
    DEFAULT_SEED,
    PoissonThinning,
    ThinMethod,
    VoxelThinning,
)
from ahn_cli.reconcile.method import (
    IdwInterp,
    KrigingInterp,
    LinearInterp,
    Variogram,
    VariogramModel,
)


@pytest.mark.parametrize(
    ("size", "grade"),
    [(0.0, 0), (0.25, 1), (0.5, 2), (1.0, 3), (64.0, 9)],
)
def test_grade_for_voxel_size_exact(size: float, grade: int) -> None:
    """Each discrete edge length resolves to its grade."""
    assert grade_for_voxel_size(size) == grade


def test_grade_for_voxel_size_rejects_off_grid() -> None:
    """A size that is not a discrete grade is a hard error, never rounded."""
    with pytest.raises(PipelineError, match="no exact prep grade"):
        grade_for_voxel_size(0.3)


def test_thinning_for_voxel() -> None:
    """A voxel thin spec maps to a graded VoxelThinning."""
    stage = ThinStage(
        method=ThinMethod.VOXEL, voxel_size_m=1.0, radius_m=None, seed=0
    )
    assert thinning_for(stage) == VoxelThinning(grade=3)


def test_thinning_for_poisson() -> None:
    """A poisson thin spec passes radius and seed straight through."""
    stage = ThinStage(
        method=ThinMethod.POISSON, voxel_size_m=None, radius_m=1.5, seed=7
    )
    assert thinning_for(stage) == PoissonThinning(radius=1.5, seed=7)


def test_thinning_for_voxel_default_seed_is_irrelevant() -> None:
    """Voxel thinning ignores the seed (a grade is deterministic)."""
    stage = ThinStage(
        method=ThinMethod.VOXEL,
        voxel_size_m=0.0,
        radius_m=None,
        seed=DEFAULT_SEED,
    )
    assert thinning_for(stage) == VoxelThinning(grade=0)


def test_neighbors_for_idw() -> None:
    """IDW carries its own neighbour count."""
    assert neighbors_for(IdwInterp(power=2.0, k=9)) == 9


def test_neighbors_for_kriging() -> None:
    """Kriging carries its own neighbour count."""
    variogram = Variogram(
        model=VariogramModel.SPHERICAL, nugget=0.0, sill=1.0, vrange=50.0
    )
    assert neighbors_for(KrigingInterp(variogram=variogram, k=20)) == 20


def test_neighbors_for_linear_uses_default() -> None:
    """Linear has no k, so the conservative default floor count is used."""
    assert neighbors_for(LinearInterp()) == DEFAULT_LINEAR_NEIGHBORS
