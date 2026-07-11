"""Tests for the copc-context pass-1 streaming scatter."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from ahn_cli.copc.octree import BuildPlan, CopcError
from ahn_cli.copc.scatter import RECORD_DTYPE, scatter_cloud

if TYPE_CHECKING:
    from pathlib import Path

    from tests.copc.conftest import WriteLaz


def _plan(bucket_level: int = 1) -> BuildPlan:
    """Build a 4 m cube plan at (-1, -1, -2): 2x2 buckets of 2 m."""
    return BuildPlan(
        scale=0.001,
        anchor_m=(-1, -1, -2),
        side_m=4,
        bucket_level=bucket_level,
        max_depth=bucket_level,
        sample_grid=128,
        units_per_m=1000,
        voxel_units=500,
    )


def test_points_land_in_their_buckets_and_roundtrip(
    write_laz: WriteLaz, tmp_path: Path
) -> None:
    """Each point lands in its level-k XY column with exact quantized coords."""
    cloud = write_laz(
        [(0.0, 0.0, 0.0), (1.5, 0.5, 0.1), (0.2, 1.9, -0.4)],
        rgb=[(1, 2, 3), (4, 5, 6), (7, 8, 9)],
    )
    result = scatter_cloud(cloud, _plan(), tmp_path / "work")
    assert set(result.bucket_paths) == {(0, 0), (1, 0), (0, 1)}
    lower_left = np.fromfile(result.bucket_paths[(0, 0)], dtype=RECORD_DTYPE)
    assert lower_left.shape == (1,)
    assert (
        lower_left["x"][0],
        lower_left["y"][0],
        lower_left["z"][0],
    ) == (1000, 1000, 2000)
    assert (
        lower_left["red"][0],
        lower_left["green"][0],
        lower_left["blue"][0],
    ) == (1, 2, 3)
    assert result.count == 3


def test_quantized_bounds_track_the_extremes(
    write_laz: WriteLaz, tmp_path: Path
) -> None:
    """Reported integer bounds are the min/max of the quantized coords."""
    cloud = write_laz([(0.0, 0.0, -0.4), (1.5, 1.9, 0.1)])
    result = scatter_cloud(cloud, _plan(), tmp_path / "work")
    assert result.quantized_mins == (1000, 1000, 1600)
    assert result.quantized_maxs == (2500, 2900, 2100)


def test_point_outside_planned_cube_is_an_error(
    write_laz: WriteLaz, tmp_path: Path
) -> None:
    """A point beyond the planned cube fails fast instead of corrupting."""
    cloud = write_laz([(0.0, 0.0, 0.0), (5.0, 0.0, 0.0)])
    with pytest.raises(CopcError, match="outside the planned cube"):
        scatter_cloud(cloud, _plan(), tmp_path / "work")


def test_dimension_flags_follow_the_point_format(
    write_laz: WriteLaz, tmp_path: Path
) -> None:
    """RGB/GPS presence flags reflect the input's point format."""
    legacy_rgb = write_laz([(0.0, 0.0, 0.0)], point_format=2, name="a.laz")
    modern_bare = write_laz([(0.0, 0.0, 0.0)], point_format=6, name="b.laz")
    modern_rgb = write_laz([(0.0, 0.0, 0.0)], point_format=7, name="c.laz")
    got_legacy = scatter_cloud(legacy_rgb, _plan(), tmp_path / "w1")
    got_bare = scatter_cloud(modern_bare, _plan(), tmp_path / "w2")
    got_rgb = scatter_cloud(modern_rgb, _plan(), tmp_path / "w3")
    assert (got_legacy.has_rgb, got_legacy.has_gps) == (True, False)
    assert (got_bare.has_rgb, got_bare.has_gps) == (False, True)
    assert (got_rgb.has_rgb, got_rgb.has_gps) == (True, True)


def test_legacy_scan_angle_rank_converts_to_modern_units(
    write_laz: WriteLaz, tmp_path: Path
) -> None:
    """PDRF-2 scan_angle_rank degrees become 0.006-degree scan_angle units."""
    cloud = write_laz(
        [(0.0, 0.0, 0.0), (1.0, 1.0, 0.0)],
        point_format=2,
        scan_angle_rank=[-90, 3],
    )
    result = scatter_cloud(cloud, _plan(bucket_level=0), tmp_path / "work")
    records = np.fromfile(result.bucket_paths[(0, 0)], dtype=RECORD_DTYPE)
    assert records["scan_angle"].tolist() == [-15000, 500]


def test_zero_returns_are_lifted_to_valid_values(
    write_laz: WriteLaz, tmp_path: Path
) -> None:
    """Synthetic clouds with 0 return fields become 1/1 (LAS-valid)."""
    cloud = write_laz([(0.0, 0.0, 0.0)], returns=([0], [0]))
    result = scatter_cloud(cloud, _plan(bucket_level=0), tmp_path / "work")
    records = np.fromfile(result.bucket_paths[(0, 0)], dtype=RECORD_DTYPE)
    assert records["return_number"].tolist() == [1]
    assert records["number_of_returns"].tolist() == [1]


def test_number_of_returns_never_below_return_number(
    write_laz: WriteLaz, tmp_path: Path
) -> None:
    """An inconsistent nr < rn pair is lifted to nr == rn."""
    cloud = write_laz([(0.0, 0.0, 0.0)], returns=([3], [1]))
    result = scatter_cloud(cloud, _plan(bucket_level=0), tmp_path / "work")
    records = np.fromfile(result.bucket_paths[(0, 0)], dtype=RECORD_DTYPE)
    assert records["return_number"].tolist() == [3]
    assert records["number_of_returns"].tolist() == [3]


def test_scatter_starts_from_an_empty_bucket_directory(
    write_laz: WriteLaz, tmp_path: Path
) -> None:
    """A leftover record file from an earlier run is removed, not appended to."""
    work = tmp_path / "work"
    work.mkdir()
    np.zeros(5, dtype=RECORD_DTYPE).tofile(work / "bucket_000_000.bin")
    cloud = write_laz([(0.0, 0.0, 0.0)])
    result = scatter_cloud(cloud, _plan(), work)
    assert result.count == 1
    records = np.fromfile(result.bucket_paths[(0, 0)], dtype=RECORD_DTYPE)
    assert records.shape == (1,)


def test_chunked_scatter_matches_single_pass(
    write_laz: WriteLaz, tmp_path: Path
) -> None:
    """Tiny chunks produce byte-identical bucket files (streaming-safe)."""
    coords = [(x * 0.3, y * 0.3, 0.0) for x in range(6) for y in range(6)]
    cloud = write_laz(coords)
    one = scatter_cloud(
        cloud, _plan(), tmp_path / "one", chunk_points=2_000_000
    )
    many = scatter_cloud(cloud, _plan(), tmp_path / "many", chunk_points=2)
    assert set(one.bucket_paths) == set(many.bucket_paths)
    for key, path in one.bucket_paths.items():
        assert path.read_bytes() == many.bucket_paths[key].read_bytes()
    assert one.quantized_mins == many.quantized_mins
    assert one.quantized_maxs == many.quantized_maxs


def test_progress_reports_monotonically_to_completion(
    write_laz: WriteLaz, tmp_path: Path
) -> None:
    """The progress callback sees monotone (done, total) up to completion."""
    coords = [(x * 0.3, 0.0, 0.0) for x in range(7)]
    cloud = write_laz(coords)
    seen: list[tuple[int, int]] = []
    scatter_cloud(
        cloud,
        _plan(),
        tmp_path / "work",
        chunk_points=3,
        progress=lambda done, total: seen.append((done, total)),
    )
    assert seen == [(3, 7), (6, 7), (7, 7)]


def test_unreadable_cloud_is_a_copc_error(tmp_path: Path) -> None:
    """A missing input file surfaces as the context's typed error."""
    with pytest.raises(CopcError, match="not readable"):
        scatter_cloud(tmp_path / "absent.laz", _plan(), tmp_path / "work")


def test_gps_time_is_carried_through(
    write_laz: WriteLaz, tmp_path: Path
) -> None:
    """GPS time survives the packed record round-trip bit-exactly."""
    cloud = write_laz(
        [(0.0, 0.0, 0.0), (1.0, 1.0, 0.0)],
        gps_time=[123456.789, -1.5],
    )
    result = scatter_cloud(cloud, _plan(bucket_level=0), tmp_path / "work")
    records = np.fromfile(result.bucket_paths[(0, 0)], dtype=RECORD_DTYPE)
    assert records["gps_time"].tolist() == [123456.789, -1.5]


def test_rgb_max_is_tracked(write_laz: WriteLaz, tmp_path: Path) -> None:
    """The largest RGB channel value is reported (drives 8-bit widening)."""
    cloud = write_laz(
        [(0.0, 0.0, 0.0), (1.0, 1.0, 0.0)],
        rgb=[(10, 20, 30), (200, 90, 40)],
    )
    result = scatter_cloud(cloud, _plan(), tmp_path / "work")
    assert result.rgb_max == 200


def test_rgb_max_is_zero_without_rgb(
    write_laz: WriteLaz, tmp_path: Path
) -> None:
    """A PDRF-6 input (no RGB dims) reports rgb_max == 0."""
    cloud = write_laz([(0.0, 0.0, 0.0)], point_format=6)
    result = scatter_cloud(cloud, _plan(), tmp_path / "work")
    assert result.rgb_max == 0
