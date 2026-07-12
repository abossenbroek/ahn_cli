"""Tests for the copc-context build orchestrator (end-to-end, offline)."""

from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn

import copclib
import laspy
import numpy as np
import pytest
from laspy.copc import CopcReader

from ahn_cli.copc.build import build_copc
from ahn_cli.copc.octree import CopcError
from ahn_cli.copc.scatter import RECORD_DTYPE
from ahn_cli.copc.writer import BARE_POINT_FORMAT, RGB_POINT_FORMAT

if TYPE_CHECKING:
    from pathlib import Path

    from tests.copc.conftest import WriteLaz


def _grid_coords(step: float = 0.6) -> list[tuple[float, float, float]]:
    """Return a 6x6 grid, one point per 0.5 m voxel, Z below NAP."""
    return [
        (x * step, y * step, -0.8 + 0.01 * (x + y))
        for x in range(6)
        for y in range(6)
    ]


def test_end_to_end_build_readable_and_exact(
    write_laz: WriteLaz, tmp_path: Path
) -> None:
    """A built file reopens as COPC with bit-exact header/point bounds."""
    cloud = write_laz(
        _grid_coords(), rgb=[(300, 400, 500)] * 36, gps_time=[7.5] * 36
    )
    out = tmp_path / "site.copc.laz"
    result = build_copc(cloud, out)
    assert result.written_points == 36  # distinct voxels: nothing dropped
    assert result.point_format_id == RGB_POINT_FORMAT
    with laspy.open(str(out)) as reader:
        header = reader.header
        las = reader.read()
    assert header.point_count == 36
    assert float(np.asarray(las.z).min()) == header.mins[2]
    assert float(np.asarray(las.x).max()) == header.maxs[0]
    # openable through laspy's dedicated COPC reader (hierarchy intact)
    with CopcReader.open(str(out)) as copc_reader:
        assert copc_reader.header.point_count == 36


def test_duplicates_collapse_to_native_coarseness(
    write_laz: WriteLaz, tmp_path: Path
) -> None:
    """Points sharing a 0.5 m voxel collapse to one survivor."""
    coords = [(0.1, 0.1, 0.0), (0.2, 0.2, 0.1), (3.0, 3.0, 0.0)]
    cloud = write_laz(coords, rgb=[(300, 300, 300)] * 3)
    out = tmp_path / "dedup.copc.laz"
    result = build_copc(cloud, out)
    assert result.input_points == 3
    assert result.written_points == 2


def test_eight_bit_rgb_is_widened(
    write_laz: WriteLaz, tmp_path: Path
) -> None:
    """8-bit-looking RGB is widened to 16-bit (validator rgbi heuristic)."""
    cloud = write_laz([(0.0, 0.0, 0.0)], rgb=[(200, 100, 50)])
    out = tmp_path / "wide.copc.laz"
    build_copc(cloud, out)
    with laspy.open(str(out)) as reader:
        las = reader.read()
    assert int(np.asarray(las.red)[0]) == 200 * 257
    assert int(np.asarray(las.blue)[0]) == 50 * 257


def test_true_sixteen_bit_rgb_passes_through(
    write_laz: WriteLaz, tmp_path: Path
) -> None:
    """Genuine 16-bit RGB is written untouched."""
    cloud = write_laz([(0.0, 0.0, 0.0)], rgb=[(1000, 2000, 3000)])
    out = tmp_path / "as16.copc.laz"
    build_copc(cloud, out)
    with laspy.open(str(out)) as reader:
        las = reader.read()
    assert int(np.asarray(las.red)[0]) == 1000


def test_black_rgb_input_becomes_bare_format(
    write_laz: WriteLaz, tmp_path: Path
) -> None:
    """All-zero RGB would warn in the validator: drop to PDRF 6 instead."""
    cloud = write_laz([(0.0, 0.0, 0.0)], rgb=[(0, 0, 0)])
    out = tmp_path / "black.copc.laz"
    result = build_copc(cloud, out)
    assert result.point_format_id == BARE_POINT_FORMAT


def test_rgbless_input_becomes_bare_format(
    write_laz: WriteLaz, tmp_path: Path
) -> None:
    """A PDRF-6 input stays PDRF 6 (no fabricated colour)."""
    cloud = write_laz([(0.0, 0.0, 0.0)], point_format=6)
    out = tmp_path / "bare.copc.laz"
    result = build_copc(cloud, out)
    assert result.point_format_id == BARE_POINT_FORMAT


def test_multi_bucket_build_preserves_every_voxel(
    write_laz: WriteLaz, tmp_path: Path
) -> None:
    """A forced multi-bucket layout still writes every voxel's survivor."""
    cloud = write_laz(_grid_coords(), rgb=[(300, 400, 500)] * 36)
    out = tmp_path / "multi.copc.laz"
    result = build_copc(cloud, out, target_bucket_points=4)
    assert result.plan.bucket_level >= 1
    assert result.written_points == 36
    with CopcReader.open(str(out)) as copc_reader:
        assert copc_reader.header.point_count == 36


def test_progress_covers_both_passes(
    write_laz: WriteLaz, tmp_path: Path
) -> None:
    """Progress runs monotonically to (2n, 2n) across scatter and build."""
    cloud = write_laz(_grid_coords(), rgb=[(300, 400, 500)] * 36)
    seen: list[tuple[int, int]] = []
    build_copc(
        cloud,
        tmp_path / "p.copc.laz",
        progress=lambda done, total: seen.append((done, total)),
    )
    assert seen[-1] == (72, 72)
    dones = [done for done, _ in seen]
    assert dones == sorted(dones)


def test_explicit_workdir_is_used_and_drained(
    write_laz: WriteLaz, tmp_path: Path
) -> None:
    """A caller-provided workdir hosts the buckets, all consumed by the end."""
    cloud = write_laz(_grid_coords(), rgb=[(300, 400, 500)] * 36)
    workdir = tmp_path / "scratch"
    build_copc(cloud, tmp_path / "w.copc.laz", workdir=workdir)
    assert list((workdir / "buckets").glob("*.bin")) == []


def test_stale_buckets_from_an_aborted_run_never_leak(
    write_laz: WriteLaz, tmp_path: Path
) -> None:
    """Records an aborted earlier run left in a persistent workdir are discarded.

    Regression: bucket files used to be appended to, so a build reusing the
    workdir of a run that died mid-build decoded the leftover records under
    its own offsets — fabricated points far outside the new cloud's extent.
    """
    workdir = tmp_path / "scratch"
    buckets = workdir / "buckets"
    buckets.mkdir(parents=True)
    stale = np.zeros(179, dtype=RECORD_DTYPE)
    stale["x"] = 900_000_000  # decodes kilometres outside the new cube
    stale["y"] = 900_000_000
    stale["z"] = 900_000_000
    stale.tofile(buckets / "bucket_000_000.bin")

    cloud = write_laz(_grid_coords())
    out = tmp_path / "fresh.copc.laz"
    result = build_copc(cloud, out, workdir=workdir)

    assert result.written_points == 36
    with laspy.open(str(out)) as reader:
        assert reader.header.point_count == 36
        las = reader.read()
    coords = np.column_stack([las.x, las.y, las.z])
    lows = np.array([0.0, 0.0, -0.8])
    highs = np.array([3.0, 3.0, -0.7])
    assert bool(np.all(coords >= lows))
    assert bool(np.all(coords <= highs))


def test_failed_build_clears_its_bucket_scratch(
    write_laz: WriteLaz, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed build removes its buckets directory, not just the output."""
    cloud = write_laz(_grid_coords())
    workdir = tmp_path / "scratch"
    out = tmp_path / "aborted.copc.laz"

    def exploding_add_node(
        self: copclib.FileWriter,
        key: copclib.VoxelKey,
        uncompressed_data: copclib.VectorChar,
    ) -> None:
        del self, key, uncompressed_data
        msg = "native writer exploded"
        raise RuntimeError(msg)

    monkeypatch.setattr(copclib.FileWriter, "AddNode", exploding_add_node)

    with pytest.raises(CopcError, match="failed to write node"):
        build_copc(cloud, out, workdir=workdir)
    assert not out.exists()
    assert not (workdir / "buckets").exists()


def test_empty_cloud_is_a_copc_error(tmp_path: Path) -> None:
    """A zero-point input cannot be planned."""
    header = laspy.LasHeader(version="1.4", point_format=7)
    empty = tmp_path / "empty.laz"
    laspy.LasData(header).write(str(empty))
    with pytest.raises(CopcError, match="empty"):
        build_copc(empty, tmp_path / "out.copc.laz")


def test_stacked_identical_points_are_a_copc_error(
    write_laz: WriteLaz, tmp_path: Path
) -> None:
    """Two or more points at one identical XYZ are fabricated, not AHN."""
    cloud = write_laz([(1.0, 1.0, 1.0), (1.0, 1.0, 1.0)])
    with pytest.raises(CopcError, match="identical position"):
        build_copc(cloud, tmp_path / "out.copc.laz")


def test_missing_cloud_is_a_copc_error(tmp_path: Path) -> None:
    """A missing input surfaces as the context's typed error."""
    with pytest.raises(CopcError, match="not readable"):
        build_copc(tmp_path / "absent.laz", tmp_path / "out.copc.laz")


def test_failed_node_write_removes_the_partial_output(
    write_laz: WriteLaz, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A copclib write failure is a CopcError and leaves no partial file."""
    cloud = write_laz(_grid_coords(), rgb=[(300, 400, 500)] * 36)
    out = tmp_path / "partial.copc.laz"

    def exploding_add_node(
        self: copclib.FileWriter,
        key: copclib.VoxelKey,
        uncompressed_data: copclib.VectorChar,
    ) -> None:
        del self, key, uncompressed_data
        msg = "native writer exploded"
        raise RuntimeError(msg)

    monkeypatch.setattr(copclib.FileWriter, "AddNode", exploding_add_node)

    with pytest.raises(CopcError, match="failed to write node"):
        build_copc(cloud, out)
    assert not out.exists()


def test_failed_close_removes_the_partial_output(
    write_laz: WriteLaz, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A copclib close failure is a CopcError and leaves no partial file."""
    cloud = write_laz(_grid_coords(), rgb=[(300, 400, 500)] * 36)
    out = tmp_path / "unsealed.copc.laz"

    def exploding_close(self: copclib.FileWriter) -> None:
        del self
        msg = "native close exploded"
        raise RuntimeError(msg)

    monkeypatch.setattr(copclib.FileWriter, "Close", exploding_close)

    with pytest.raises(CopcError, match="failed to close"):
        build_copc(cloud, out)
    assert not out.exists()


def _diagonal_coords(n: int = 40) -> list[tuple[float, float, float]]:
    """Return a thin diagonal strip: the worst case for uniform bucketing."""
    return [(i * 0.6, i * 0.6, 0.01 * i) for i in range(n)]


def test_diagonal_strip_triggers_occupancy_rebalance(
    write_laz: WriteLaz, tmp_path: Path
) -> None:
    """A thin diagonal AOI deepens the bucket level and still builds green."""
    cloud = write_laz(_diagonal_coords(), rgb=[(300, 400, 500)] * 40)
    out = tmp_path / "diag.copc.laz"
    result = build_copc(cloud, out, target_bucket_points=4)
    # The uniform estimate for 40 points at target 4 is level 2; the
    # measured diagonal occupancy forces a deeper bucket level.
    assert result.plan.bucket_level > 2
    assert result.plan.max_depth >= result.plan.bucket_level
    assert result.written_points == 40
    with CopcReader.open(str(out)) as copc_reader:
        assert copc_reader.header.point_count == 40


def test_rebalanced_build_is_deterministic(
    write_laz: WriteLaz, tmp_path: Path
) -> None:
    """Two pre-scanned builds of the same diagonal input are byte-identical."""
    cloud = write_laz(
        _diagonal_coords(), gps_time=[float(i) for i in range(40)]
    )
    one = tmp_path / "one.copc.laz"
    two = tmp_path / "two.copc.laz"
    build_copc(cloud, one, target_bucket_points=4)
    build_copc(cloud, two, target_bucket_points=4)
    assert one.read_bytes() == two.read_bytes()


def test_unreadable_cloud_during_prescan_is_a_copc_error(
    write_laz: WriteLaz, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cloud that turns unreadable after planning fails as a CopcError."""
    cloud = write_laz(_diagonal_coords())

    def exploding_chunks(self: laspy.LasReader, points: int) -> NoReturn:
        del self, points
        msg = "stream went away"
        raise laspy.LaspyException(msg)

    monkeypatch.setattr(laspy.LasReader, "chunk_iterator", exploding_chunks)
    # target 4 -> multi-bucket plan -> the pre-scan reads first and fails.
    with pytest.raises(CopcError, match="not readable"):
        build_copc(cloud, tmp_path / "out.copc.laz", target_bucket_points=4)


def test_build_is_deterministic(write_laz: WriteLaz, tmp_path: Path) -> None:
    """Two builds of the same input are byte-identical."""
    cloud = write_laz(
        _grid_coords(), rgb=[(300, 400, 500)] * 36, gps_time=[3.0] * 36
    )
    one = tmp_path / "one.copc.laz"
    two = tmp_path / "two.copc.laz"
    build_copc(cloud, one)
    build_copc(cloud, two)
    assert one.read_bytes() == two.read_bytes()
