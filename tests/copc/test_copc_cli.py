"""Tests for the ``copc`` CLI command (thin adapter over build_copc)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import laspy
from click.testing import CliRunner

from ahn_cli.cli.app import cli

if TYPE_CHECKING:
    from pathlib import Path

    from tests.copc.conftest import WriteLaz


def test_copc_command_builds_a_readable_file(
    write_laz: WriteLaz, tmp_path: Path
) -> None:
    """The happy path writes the COPC and reports the point accounting."""
    cloud = write_laz(
        [(x * 0.6, y * 0.6, 0.5) for x in range(4) for y in range(4)],
        rgb=[(300, 400, 500)] * 16,
    )
    out = tmp_path / "site.copc.laz"
    result = CliRunner().invoke(
        cli, ["copc", "--cloud", str(cloud), "--out", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert "16 pts -> 16 written" in result.output
    with laspy.open(str(out)) as reader:
        assert reader.header.point_count == 16


def test_copc_command_accepts_a_workdir(
    write_laz: WriteLaz, tmp_path: Path
) -> None:
    """An explicit --workdir hosts (and drains) the scatter buckets."""
    cloud = write_laz([(0.0, 0.0, 0.0)], rgb=[(300, 300, 300)])
    workdir = tmp_path / "scratch"
    result = CliRunner().invoke(
        cli,
        [
            "copc",
            "--cloud",
            str(cloud),
            "--out",
            str(tmp_path / "w.copc.laz"),
            "--workdir",
            str(workdir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (workdir / "buckets").is_dir()


def test_copc_error_becomes_a_click_error(tmp_path: Path) -> None:
    """A context CopcError surfaces as a clean CLI failure, not a trace."""
    header = laspy.LasHeader(version="1.4", point_format=7)
    empty = tmp_path / "empty.laz"
    laspy.LasData(header).write(str(empty))
    result = CliRunner().invoke(
        cli,
        [
            "copc",
            "--cloud",
            str(empty),
            "--out",
            str(tmp_path / "out.copc.laz"),
        ],
    )
    assert result.exit_code == 1
    assert "empty" in result.output


def test_missing_cloud_fails_argument_validation(tmp_path: Path) -> None:
    """A nonexistent --cloud is rejected by Click itself (exit code 2)."""
    result = CliRunner().invoke(
        cli,
        [
            "copc",
            "--cloud",
            str(tmp_path / "absent.laz"),
            "--out",
            str(tmp_path / "out.copc.laz"),
        ],
    )
    assert result.exit_code == 2
