"""Tests for the reconcile orchestration end to end (synthetic fixtures)."""

from __future__ import annotations

from typing import TYPE_CHECKING

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
