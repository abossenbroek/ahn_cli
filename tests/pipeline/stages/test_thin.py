"""Tests for :class:`ahn_cli.pipeline.stages.thin.ThinStage` (WP: pipeline W5).

The load-bearing correctness gate is byte-identity to standalone ``prep``
thinning over the same points: voxel-grid thinning through
:func:`~ahn_cli.prep.voxel_stream.stream_voxel_thin` (the file-based oracle
``prep.transform`` itself routes every voxel request through) and
Poisson-disk thinning through :func:`~ahn_cli.prep.decimate.thin` with the
CPU reference backend (the backend ``prep.transform`` hardcodes). Every
byte-identity test below re-derives its "want" value from that oracle
directly -- never from :mod:`ahn_cli.pipeline.stages.thin` -- so a
regression in the stage's plumbing cannot silently pass by comparing itself
to itself.

Coverage strategy for the Apple-silicon accelerator on Linux CI (no
``mlx``): mirrors ``tests/prep/test_decimate.py`` -- a numpy-backed fake
``mlx.core`` handle (:class:`FakeMx`) exercises :class:`MlxBackend` off
device, proving the voxel-selection primitive :mod:`ahn_cli.prep.decimate`
exposes (and any future in-memory acceleration of this stage would depend
on, per the design doc's ``ahn_cli.prep.decimate`` docstring) stays
CPU/MLX-equivalent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import laspy
import numpy as np
import numpy.typing as npt
import pytest

from ahn_cli.pipeline.model import PointTile, Stage
from ahn_cli.pipeline.stages.thin import ThinStage
from ahn_cli.prep.decimate import (
    GRADE_MAX,
    GRADE_MIN,
    MlxArray,
    MlxBackend,
    NumpyBackend,
    PoissonThinning,
    VoxelThinning,
    decimate_voxel,
)
from ahn_cli.prep.decimate import thin as decimate_thin
from ahn_cli.prep.voxel_stream import stream_voxel_thin
from tests.pipeline.harness import (
    IdentityStage,
    hash_payload,
    make_grid_tile,
    make_point_tile,
    make_tile_context,
    run_stages,
    write_synthetic_laz,
)

if TYPE_CHECKING:
    from pathlib import Path

# --------------------------------------------------------------------------- #
# A numpy-backed fake ``mlx.core`` handle (mirrors tests/prep/test_decimate.py).
# --------------------------------------------------------------------------- #


def _np(values: MlxArray) -> npt.NDArray[np.generic]:
    """Return the numpy view of a fake mlx array."""
    return np.asarray(values)


class FakeArray:
    """A numpy-wrapping stand-in for an ``mlx`` array."""

    def __init__(self, values: npt.ArrayLike) -> None:
        """Wrap ``values`` as a numpy array."""
        self.value: npt.NDArray[np.generic] = np.asarray(values)

    def __getitem__(self, key: object) -> FakeArray:
        """Slice the wrapped array (only slices are used by the adapter)."""
        return FakeArray(self.value[cast("slice", key)])

    def __array__(
        self, dtype: npt.DTypeLike = None
    ) -> npt.NDArray[np.generic]:
        """Expose the numpy view (the fake's device-to-host bridge)."""
        return np.asarray(self.value, dtype=dtype)


class FakeMx:
    """A numpy-backed fake of the ``mlx.core`` module function surface."""

    def array(self, values: object) -> FakeArray:
        """Wrap array-like ``values``."""
        return FakeArray(cast("npt.ArrayLike", values))

    def arange(self, size: int) -> FakeArray:
        """Return ``0..size-1``."""
        return FakeArray(np.arange(size))

    def argsort(self, values: MlxArray) -> FakeArray:
        """Return a stable ascending argsort of ``values``."""
        return FakeArray(np.argsort(_np(values), kind="stable"))

    def take(self, values: MlxArray, indices: MlxArray) -> FakeArray:
        """Gather ``values`` at ``indices``."""
        return FakeArray(np.take(_np(values), _np(indices).astype(np.intp)))

    def multiply(self, left: MlxArray, right: MlxArray | int) -> FakeArray:
        """Return the elementwise product."""
        other = right if isinstance(right, int) else _np(right)
        return FakeArray(np.multiply(_np(left), other))

    def add(self, left: MlxArray, right: MlxArray) -> FakeArray:
        """Return the elementwise sum."""
        return FakeArray(np.add(_np(left), _np(right)))

    def subtract(self, left: MlxArray, right: MlxArray) -> FakeArray:
        """Return the elementwise difference."""
        return FakeArray(np.subtract(_np(left), _np(right)))

    def not_equal(self, left: MlxArray, right: MlxArray) -> FakeArray:
        """Return the elementwise inequality mask."""
        return FakeArray(np.not_equal(_np(left), _np(right)))

    def sum(self, values: MlxArray, axis: int) -> FakeArray:
        """Reduce ``values`` by summation along ``axis``."""
        return FakeArray(np.sum(_np(values), axis=axis))


def _fake_mlx_backend() -> MlxBackend:
    """Return an :class:`MlxBackend` wired to the numpy-backed fake handle."""
    return MlxBackend(FakeMx())


# --------------------------------------------------------------------------- #
# An independent (non-module) LAZ -> PointTile reader, so the byte-identity
# tests compare the stage against the oracle rather than against itself.
# --------------------------------------------------------------------------- #


def _read_point_tile(path: Path) -> PointTile:
    """Read a LAZ file into a :class:`PointTile`, independent of the stage module."""
    with laspy.open(str(path)) as reader:
        las = reader.read()
    rgb = None
    if "red" in las.point_format.dimension_names:
        rgb = np.ascontiguousarray(
            np.column_stack(
                [
                    np.asarray(las.red),
                    np.asarray(las.green),
                    np.asarray(las.blue),
                ]
            ).astype(np.uint16)
        )
    return PointTile(
        x=np.ascontiguousarray(np.asarray(las.x, dtype=np.float64)),
        y=np.ascontiguousarray(np.asarray(las.y, dtype=np.float64)),
        z=np.ascontiguousarray(np.asarray(las.z, dtype=np.float64)),
        gps_time=np.ascontiguousarray(
            np.asarray(las.gps_time, dtype=np.float64)
        ),
        classification=np.ascontiguousarray(
            np.asarray(las.classification, dtype=np.uint8)
        ),
        rgb=rgb,
    )


def _synthetic_points(n: int = 200, seed: int = 3) -> npt.NDArray[np.float64]:
    """Return an ``(n, 5)`` array of ``x, y, z, gps_time, classification``.

    Coordinates are rounded to the centimetre grid before being handed to
    :func:`~tests.pipeline.harness.write_synthetic_laz` (default ``scale``
    ``0.01``), so the synthetic LAZ's stored coordinates exactly equal the
    values generated here -- the precondition the voxel byte-identity tests
    rely on.
    """
    rng = np.random.default_rng(seed)
    x = np.round(rng.uniform(0.0, 6.0, n), 2)
    y = np.round(rng.uniform(0.0, 6.0, n), 2)
    z = np.round(rng.uniform(-2.0, 2.0, n), 2)
    gps_time = rng.uniform(0.0, 100.0, n)
    classification = rng.integers(0, 10, n).astype(np.float64)
    return np.column_stack([x, y, z, gps_time, classification])


# --------------------------------------------------------------------------- #
# Protocol conformance / structural behaviour.
# --------------------------------------------------------------------------- #


def test_thin_stage_satisfies_stage_protocol() -> None:
    """A :class:`ThinStage` is a valid :class:`Stage`."""
    assert isinstance(ThinStage(thinning=None), Stage)


@pytest.mark.parametrize(
    "thinning",
    [None, VoxelThinning(grade=2), PoissonThinning(radius=1.0)],
)
def test_halo_m_is_always_zero(
    thinning: VoxelThinning | PoissonThinning | None,
) -> None:
    """Thinning is tile-local: the stage never requests a source halo."""
    assert ThinStage(thinning=thinning).halo_m() == 0.0


def test_run_rejects_a_non_point_tile_payload(tmp_path: Path) -> None:
    """``run()`` type-guards its payload against the wrong stage-chain wiring."""
    stage = ThinStage(thinning=None)
    ctx = make_tile_context(tmp_path)
    with pytest.raises(TypeError, match="PointTile"):
        stage.run(make_grid_tile(), ctx)


def test_thin_stage_composes_through_run_stages(tmp_path: Path) -> None:
    """A :class:`ThinStage` runs correctly inside a fused stage chain."""
    tile = make_point_tile(count=6, seed=1)
    ctx = make_tile_context(tmp_path)
    stage = ThinStage(thinning=None)
    got = run_stages(tile, ctx, [IdentityStage(), stage, IdentityStage()])
    assert got is tile


# --------------------------------------------------------------------------- #
# No thinning: class-filter-only (or pure identity).
# --------------------------------------------------------------------------- #


def test_no_thinning_no_classes_returns_the_same_tile(tmp_path: Path) -> None:
    """With no thinning and no class filter, the tile passes through unchanged."""
    tile = make_point_tile(count=5, seed=2)
    ctx = make_tile_context(tmp_path)
    got = ThinStage(thinning=None).run(tile, ctx)
    assert got is tile


def test_no_thinning_include_only_filters_by_class(tmp_path: Path) -> None:
    """Class-filter-only (include side) matches a direct boolean mask."""
    tile = make_point_tile(count=40, seed=4)
    ctx = make_tile_context(tmp_path)
    got = ThinStage(thinning=None, include_classes=(1, 3)).run(tile, ctx)
    assert isinstance(got, PointTile)
    keep = np.isin(tile.classification, (1, 3))
    assert np.array_equal(got.x, tile.x[keep])
    assert np.array_equal(got.classification, tile.classification[keep])


def test_no_thinning_exclude_only_filters_by_class(tmp_path: Path) -> None:
    """Class-filter-only (exclude side) matches a direct boolean mask."""
    tile = make_point_tile(count=40, seed=5)
    ctx = make_tile_context(tmp_path)
    got = ThinStage(thinning=None, exclude_classes=(0, 2)).run(tile, ctx)
    assert isinstance(got, PointTile)
    keep = ~np.isin(tile.classification, (0, 2))
    assert np.array_equal(got.x, tile.x[keep])
    assert np.array_equal(got.classification, tile.classification[keep])


def test_no_thinning_include_and_exclude_both_apply(tmp_path: Path) -> None:
    """Include and exclude combine (AND) exactly like ``prep.transform``."""
    tile = make_point_tile(count=60, seed=6)
    ctx = make_tile_context(tmp_path)
    got = ThinStage(
        thinning=None, include_classes=(0, 1, 2, 3), exclude_classes=(1,)
    ).run(tile, ctx)
    assert isinstance(got, PointTile)
    keep = np.isin(tile.classification, (0, 1, 2, 3)) & ~np.isin(
        tile.classification, (1,)
    )
    assert np.array_equal(got.x, tile.x[keep])
    assert np.array_equal(got.classification, tile.classification[keep])


# --------------------------------------------------------------------------- #
# Voxel-grid thinning: byte-identity to ``stream_voxel_thin``.
# --------------------------------------------------------------------------- #


def test_voxel_thin_byte_identical_to_stream_voxel_thin(
    tmp_path: Path,
) -> None:
    """Voxel thinning matches the ``stream_voxel_thin`` oracle exactly."""
    points = _synthetic_points(n=300, seed=11)
    source = tmp_path / "source.laz"
    write_synthetic_laz(source, points)
    tile = _read_point_tile(source)

    ctx = make_tile_context(tmp_path)
    stage = ThinStage(
        thinning=VoxelThinning(grade=3),
        include_classes=(1, 2, 3, 4, 5, 6),
        exclude_classes=(2,),
    )
    got = stage.run(tile, ctx)
    assert isinstance(got, PointTile)

    oracle_out = tmp_path / "oracle_out.laz"
    stream_voxel_thin(
        source,
        oracle_out,
        3,
        (1, 2, 3, 4, 5, 6),
        (2,),
        workdir=tmp_path / "oracle_work",
    )
    want = _read_point_tile(oracle_out)

    assert hash_payload(got) == hash_payload(want)
    # A meaningful comparison, not a vacuous empty-vs-empty one.
    assert got.x.shape[0] > 0
    assert got.x.shape[0] < tile.x.shape[0]


@pytest.mark.parametrize("grade", range(GRADE_MIN, GRADE_MAX + 1))
def test_voxel_thin_matches_oracle_across_every_grade(
    tmp_path: Path, grade: int
) -> None:
    """Byte-identity holds at every voxel grade, including the coarsest."""
    points = _synthetic_points(n=150, seed=grade + 100)
    source = tmp_path / "source.laz"
    write_synthetic_laz(source, points)
    tile = _read_point_tile(source)

    ctx = make_tile_context(tmp_path)
    got = ThinStage(thinning=VoxelThinning(grade=grade)).run(tile, ctx)

    oracle_out = tmp_path / "oracle_out.laz"
    stream_voxel_thin(
        source, oracle_out, grade, (), (), workdir=tmp_path / "oracle_work"
    )
    want = _read_point_tile(oracle_out)

    assert hash_payload(got) == hash_payload(want)


def test_voxel_grade_zero_is_class_filter_only(tmp_path: Path) -> None:
    """Grade 0 is the documented class-filter-only path (still streamed)."""
    points = _synthetic_points(n=80, seed=21)
    source = tmp_path / "source.laz"
    write_synthetic_laz(source, points)
    tile = _read_point_tile(source)

    ctx = make_tile_context(tmp_path)
    got = ThinStage(
        thinning=VoxelThinning(grade=0), include_classes=(1, 2, 3)
    ).run(tile, ctx)
    assert isinstance(got, PointTile)

    keep = np.isin(tile.classification, (1, 2, 3))
    assert got.x.shape[0] == int(keep.sum())
    assert np.array_equal(got.x, tile.x[keep])


def test_voxel_thin_preserves_rgb_and_attributes(tmp_path: Path) -> None:
    """RGB, GPS time, and classification survive the scratch-LAZ round trip."""
    tile = make_point_tile(count=15, seed=8, with_rgb=True)
    ctx = make_tile_context(tmp_path)
    got = ThinStage(thinning=VoxelThinning(grade=0)).run(tile, ctx)
    assert isinstance(got, PointTile)

    assert got.rgb is not None
    assert tile.rgb is not None
    assert np.array_equal(got.rgb, tile.rgb)
    assert np.array_equal(got.classification, tile.classification)
    assert np.allclose(got.gps_time, tile.gps_time)
    assert np.allclose(got.x, tile.x, atol=0.01)
    assert np.allclose(got.y, tile.y, atol=0.01)
    assert np.allclose(got.z, tile.z, atol=0.01)


def test_voxel_thin_handles_an_empty_tile(tmp_path: Path) -> None:
    """An empty tile thins to an empty tile, without error."""
    tile = make_point_tile(count=0)
    ctx = make_tile_context(tmp_path)
    got = ThinStage(thinning=VoxelThinning(grade=4)).run(tile, ctx)
    assert isinstance(got, PointTile)
    assert got.x.shape[0] == 0


def test_voxel_thin_is_deterministic_across_runs(tmp_path: Path) -> None:
    """Identical input and parameters yield byte-identical output every time."""
    points = _synthetic_points(n=120, seed=31)
    source = tmp_path / "source.laz"
    write_synthetic_laz(source, points)
    tile = _read_point_tile(source)
    ctx = make_tile_context(tmp_path)
    stage = ThinStage(thinning=VoxelThinning(grade=2))

    first = stage.run(tile, ctx)
    second = stage.run(tile, ctx)
    assert hash_payload(first) == hash_payload(second)


# --------------------------------------------------------------------------- #
# Poisson-disk thinning: byte-identity to ``decimate.thin``.
# --------------------------------------------------------------------------- #


def test_poisson_thin_byte_identical_to_decimate_thin(tmp_path: Path) -> None:
    """Poisson thinning matches the in-memory ``decimate.thin`` oracle exactly."""
    tile = make_point_tile(count=80, seed=9)
    ctx = make_tile_context(tmp_path)
    include = (0, 1, 2, 3, 4)
    request = PoissonThinning(radius=1.5, seed=7)
    got = ThinStage(thinning=request, include_classes=include).run(tile, ctx)
    assert isinstance(got, PointTile)

    keep = np.flatnonzero(np.isin(tile.classification, include))
    coords = np.column_stack([tile.x[keep], tile.y[keep], tile.z[keep]])
    survivors_local = decimate_thin(coords, request, backend=NumpyBackend())
    survivors = keep[survivors_local]

    assert got.x.shape[0] > 0
    assert np.array_equal(got.x, tile.x[survivors])
    assert np.array_equal(got.y, tile.y[survivors])
    assert np.array_equal(got.z, tile.z[survivors])
    assert np.array_equal(got.gps_time, tile.gps_time[survivors])
    assert np.array_equal(got.classification, tile.classification[survivors])


def test_poisson_thin_preserves_rgb(tmp_path: Path) -> None:
    """Poisson thinning threads RGB through the survivor selection."""
    tile = make_point_tile(count=50, seed=13, with_rgb=True)
    ctx = make_tile_context(tmp_path)
    request = PoissonThinning(radius=2.0, seed=3)
    got = ThinStage(thinning=request).run(tile, ctx)
    assert isinstance(got, PointTile)

    coords = np.column_stack([tile.x, tile.y, tile.z])
    survivors = decimate_thin(coords, request, backend=NumpyBackend())

    assert got.rgb is not None
    assert tile.rgb is not None
    assert np.array_equal(got.rgb, tile.rgb[survivors])


def test_poisson_thin_is_deterministic_with_the_same_seed(
    tmp_path: Path,
) -> None:
    """Identical input, radius, and seed yield byte-identical output every time."""
    tile = make_point_tile(count=90, seed=14)
    ctx = make_tile_context(tmp_path)
    stage = ThinStage(thinning=PoissonThinning(radius=1.0, seed=42))

    first = stage.run(tile, ctx)
    second = stage.run(tile, ctx)
    assert hash_payload(first) == hash_payload(second)


def test_poisson_thin_seed_changes_the_output(tmp_path: Path) -> None:
    """A different seed yields a different (still deterministic) survivor set."""
    tile = make_point_tile(count=90, seed=15)
    ctx = make_tile_context(tmp_path)

    got_a = ThinStage(thinning=PoissonThinning(radius=1.0, seed=1)).run(
        tile, ctx
    )
    got_b = ThinStage(thinning=PoissonThinning(radius=1.0, seed=2)).run(
        tile, ctx
    )
    assert hash_payload(got_a) != hash_payload(got_b)


# --------------------------------------------------------------------------- #
# MLX == CPU backend contract (guards the decimate.py primitive this stage's
# voxel oracle -- and any future in-memory acceleration of it -- relies on).
# Mirrors tests/prep/test_decimate.py's coverage strategy: the fake handle
# runs the accelerator's wiring off-device, with no real ``mlx`` required.
# --------------------------------------------------------------------------- #


def test_mlx_and_cpu_voxel_backends_select_identical_points() -> None:
    """CPU and fake-MLX voxel selections are byte-identical across grades."""
    rng = np.random.default_rng(17)
    coords = rng.uniform(0.0, 5.0, size=(250, 3)).astype(np.float64)
    for grade in range(GRADE_MIN, GRADE_MAX + 1):
        cpu = decimate_voxel(coords, grade, backend=NumpyBackend())
        mlx = decimate_voxel(coords, grade, backend=_fake_mlx_backend())
        assert cpu.tolist() == mlx.tolist()
