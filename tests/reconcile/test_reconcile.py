"""Tests for the reconcile orchestration end to end (synthetic fixtures)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import laspy
import numpy as np
import pytest

from ahn_cli.reconcile.method import IdwInterp
from ahn_cli.reconcile.reconcile import (
    ReconcileError,
    ReconcileRequest,
    reconcile,
)
from ahn_cli.reconcile.writers import OutputFormat

if TYPE_CHECKING:
    from pathlib import Path

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
