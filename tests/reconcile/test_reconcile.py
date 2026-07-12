"""Tests for the reconcile orchestration end to end (synthetic fixtures)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import laspy
import numpy as np
import pytest

from ahn_cli.reconcile.method import IdwInterp, LinearInterp
from ahn_cli.reconcile.reconcile import (
    ReconcileError,
    ReconcileRequest,
    _verify_source_coords,  # pyright: ignore[reportPrivateUsage]
    reconcile,
)
from ahn_cli.reconcile.writers import OutputFormat

if TYPE_CHECKING:
    import numpy.typing as npt

_ALL_FORMATS = tuple(OutputFormat)


def _request(
    ortho: Path,
    cloud: Path,
    out: Path,
    formats: tuple[OutputFormat, ...] = _ALL_FORMATS,
) -> ReconcileRequest:
    return ReconcileRequest(
        ortho_path=ortho,
        cloud_path=cloud,
        output_dir=out,
        method=IdwInterp(k=8),
        formats=formats,
    )


def test_reconcile_writes_all_formats(
    ortho_path: Path, cloud_path: Path, tmp_path: Path
) -> None:
    """A full run writes one file per format and reports the grid + points."""
    stats = reconcile(_request(ortho_path, cloud_path, tmp_path / "out"))
    assert (stats.width, stats.height) == (6, 6)
    assert stats.valid_points == 36
    assert len(stats.outputs) == 4
    for path in stats.outputs:
        assert path.exists()


def test_reconcile_single_format(
    ortho_path: Path, cloud_path: Path, tmp_path: Path
) -> None:
    """Requesting one format writes exactly that file."""
    stats = reconcile(
        _request(ortho_path, cloud_path, tmp_path / "out", (OutputFormat.PT,))
    )
    assert [path.name for path in stats.outputs] == ["reconciled.pt"]


def test_reconcile_is_deterministic(
    ortho_path: Path, cloud_path: Path, tmp_path: Path
) -> None:
    """Two runs on the numpy backend produce byte-identical outputs."""
    first = reconcile(_request(ortho_path, cloud_path, tmp_path / "a"))
    second = reconcile(_request(ortho_path, cloud_path, tmp_path / "b"))
    for left, right in zip(first.outputs, second.outputs, strict=True):
        assert left.read_bytes() == right.read_bytes()


def test_reconcile_blocked_equals_whole_grid(
    ortho_path: Path,
    cloud_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Streaming in tiny row-blocks equals a whole-grid run.

    The default block schedule fits the 6x6 fixture in one block; forcing one
    row per block exercises the multi-block loop and proves the block schedule
    does not change the result (bytes for the uncompressed formats; point
    read-back for the chunk-compressed LAZ).
    """
    whole = reconcile(_request(ortho_path, cloud_path, tmp_path / "whole"))
    monkeypatch.setattr(
        "ahn_cli.reconcile.reconcile._BLOCK_CELLS", 6
    )  # width 6 -> one row per block
    blocked = reconcile(
        _request(ortho_path, cloud_path, tmp_path / "blocked")
    )
    assert blocked.valid_points == whole.valid_points
    for whole_path, blocked_path in zip(
        whole.outputs, blocked.outputs, strict=True
    ):
        if whole_path.suffix == ".laz":
            with laspy.open(str(whole_path)) as reader:
                one = reader.read()
            with laspy.open(str(blocked_path)) as reader:
                many = reader.read()
            assert np.array_equal(
                np.c_[one.x, one.y, one.z], np.c_[many.x, many.y, many.z]
            )
        else:
            assert whole_path.read_bytes() == blocked_path.read_bytes()


def test_reconcile_reports_progress_across_blocks(
    ortho_path: Path,
    cloud_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The progress callback fires once per block with (rows-done, total)."""
    monkeypatch.setattr(
        "ahn_cli.reconcile.reconcile._BLOCK_CELLS", 6
    )  # width 6 -> one row per block -> six calls
    calls: list[tuple[int, int]] = []

    reconcile(
        _request(ortho_path, cloud_path, tmp_path / "out"),
        progress=lambda done, total: calls.append((done, total)),
    )

    assert calls == [(1, 6), (2, 6), (3, 6), (4, 6), (5, 6), (6, 6)]


def test_reconcile_without_a_progress_callback_does_not_raise(
    ortho_path: Path, cloud_path: Path, tmp_path: Path
) -> None:
    """Omitting ``progress`` (the default) runs exactly as before."""
    stats = reconcile(_request(ortho_path, cloud_path, tmp_path / "out"))
    assert stats.valid_points == 36


def test_reconcile_missing_ortho_raises(
    cloud_path: Path, tmp_path: Path
) -> None:
    """A missing orthophoto surfaces as ReconcileError."""
    with pytest.raises(ReconcileError):
        reconcile(
            _request(tmp_path / "absent.tif", cloud_path, tmp_path / "out")
        )


def test_reconcile_missing_cloud_raises(
    ortho_path: Path, tmp_path: Path
) -> None:
    """A missing point cloud surfaces as ReconcileError."""
    with pytest.raises(ReconcileError):
        reconcile(
            _request(ortho_path, tmp_path / "absent.laz", tmp_path / "out")
        )


def _write_cloud(path: Path, xyz: npt.NDArray[np.float64]) -> Path:
    """Write ``xyz`` rows as a small point-format-2 LAZ at ``path``."""
    header = laspy.LasHeader(point_format=2)
    header.offsets = np.floor(xyz.min(axis=0))
    header.scales = [0.001, 0.001, 0.001]
    las = laspy.LasData(header)
    las.x = xyz[:, 0]
    las.y = xyz[:, 1]
    las.z = xyz[:, 2]
    las.write(str(path))
    return path


def test_reconcile_rejects_a_filter_that_empties_the_cloud(
    ortho_path: Path, cloud_path: Path, tmp_path: Path
) -> None:
    """A class filter leaving no points is a typed refusal, not infill."""
    request = ReconcileRequest(
        ortho_path=ortho_path,
        cloud_path=cloud_path,
        output_dir=tmp_path / "out",
        method=IdwInterp(k=8),
        formats=(OutputFormat.PT,),
        include_classes=(26,),  # the fixture cloud holds only class 0
    )

    with pytest.raises(ReconcileError, match="no points left"):
        reconcile(request)


def test_reconcile_rejects_a_single_point_cloud(
    ortho_path: Path, tmp_path: Path
) -> None:
    """One point has no XY extent to interpolate a grid from."""
    cloud = _write_cloud(
        tmp_path / "one.laz", np.array([[101.0, 101.0, 3.0]])
    )

    with pytest.raises(ReconcileError, match="single XY position"):
        reconcile(_request(ortho_path, cloud, tmp_path / "out"))


def test_reconcile_rejects_a_cloud_collapsed_to_one_xy(
    ortho_path: Path, tmp_path: Path
) -> None:
    """Coincident-XY returns collapse in cleaning, leaving no XY extent."""
    xyz = np.array(
        [[101.0, 101.0, 1.0], [101.0, 101.0, 2.0], [101.0, 101.0, 3.0]]
    )
    cloud = _write_cloud(tmp_path / "stack.laz", xyz)

    with pytest.raises(ReconcileError, match="single XY position"):
        reconcile(_request(ortho_path, cloud, tmp_path / "out"))


def test_verify_source_coords_rejects_identical_xy_points() -> None:
    """Two or more points all sharing one XY are refused directly.

    The public pipeline always XY-dedupes before this check, so the
    several-points-one-XY arm is exercised on the helper itself.
    """
    coords = np.array([[101.0, 101.0, 1.0], [101.0, 101.0, 2.0]])

    with pytest.raises(ReconcileError, match="single XY position"):
        _verify_source_coords(coords, Path("cloud.laz"))


def test_reconcile_rejects_a_cloud_west_of_the_ortho(
    ortho_path: Path, tmp_path: Path
) -> None:
    """A cloud entirely west of the ortho extent shares no ground with it."""
    xyz = np.array([[50.0, 101.0, 1.0], [51.0, 102.0, 2.0]])
    cloud = _write_cloud(tmp_path / "west.laz", xyz)

    with pytest.raises(ReconcileError, match="does not cover"):
        reconcile(_request(ortho_path, cloud, tmp_path / "out"))


def test_reconcile_rejects_a_cloud_south_of_the_ortho(
    ortho_path: Path, tmp_path: Path
) -> None:
    """A cloud overlapping in X but entirely south in Y is still disjoint."""
    xyz = np.array([[101.0, 50.0, 1.0], [102.0, 51.0, 2.0]])
    cloud = _write_cloud(tmp_path / "south.laz", xyz)

    with pytest.raises(ReconcileError, match="does not cover"):
        reconcile(_request(ortho_path, cloud, tmp_path / "out"))


def test_reconcile_rejects_a_cloud_covering_only_half_the_grid(
    ortho_path: Path, tmp_path: Path
) -> None:
    """Partial overlap is refused: every pixel centre must be covered.

    A cloud spanning only the west half of the grid used to pass the old
    bbox-overlap test; the uncovered east pixels would then be estimated
    from unrelated points — fabricated data — so it is now a refusal.
    """
    xyz = np.array(
        [
            [100.0, 100.0, 1.0],
            [101.4, 103.0, 2.0],
            [100.7, 101.5, 1.5],
        ]
    )
    cloud = _write_cloud(tmp_path / "half.laz", xyz)

    with pytest.raises(ReconcileError, match="east"):
        reconcile(_request(ortho_path, cloud, tmp_path / "out"))


def test_reconcile_refuses_void_estimates_and_leaves_no_outputs(
    ortho_path: Path, tmp_path: Path
) -> None:
    """Any pixel without a genuine estimate aborts and removes outputs.

    A diagonal-line cloud covers the pixel-centre bbox on every side (so
    the coverage gate passes) but its convex hull is a segment, so the
    linear method cannot estimate any pixel. That is missing data — a
    hard error — and no partial file (nor a PLY payload temp) may
    survive.
    """
    diag = np.linspace(100.25, 102.75, 7)
    cloud = _write_cloud(
        tmp_path / "diag.laz", np.c_[diag, diag, np.ones_like(diag)]
    )
    request = ReconcileRequest(
        ortho_path=ortho_path,
        cloud_path=cloud,
        output_dir=tmp_path / "out",
        method=LinearInterp(),
        formats=(OutputFormat.LAZ, OutputFormat.PLY),
    )

    with pytest.raises(ReconcileError, match="no genuine elevation estimate"):
        reconcile(request)

    assert list((tmp_path / "out").iterdir()) == []


def test_reconcile_accepts_a_cloud_covering_exactly_the_pixel_centres(
    ortho_path: Path, tmp_path: Path
) -> None:
    """A cloud whose bbox equals the pixel-centre extent is sufficient.

    Pixel centres span [100.25, 102.75] on both axes for the 6x6/0.5 m
    fixture; a cloud reaching exactly that far covers every target.
    """
    xyz = np.array(
        [
            [100.25, 100.25, 1.0],
            [102.75, 100.25, 1.2],
            [100.25, 102.75, 1.4],
            [102.75, 102.75, 1.6],
            [101.5, 101.5, 1.3],
        ]
    )
    cloud = _write_cloud(tmp_path / "exact.laz", xyz)

    stats = reconcile(
        _request(ortho_path, cloud, tmp_path / "out", (OutputFormat.PT,))
    )

    assert stats.valid_points == 36
