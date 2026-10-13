"""Tests for the ``export-ply`` verb: explicit class filter, thinning, sidecar."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest
from click.testing import CliRunner

from ahn_cli.cli import cli
from ahn_cli.provenance.product import read_product_sidecar, sidecar_path
from tests.prep.ply_helpers import make_cloud, read_ply, write_cloud

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def cloud_path(tmp_path: Path) -> Path:
    """Write a 300-point synthetic cloud and return its path."""
    path = tmp_path / "in.laz"
    write_cloud(path, make_cloud(300))
    return path


def _run(*args: str) -> Any:  # noqa: ANN401 -- click's Result is untyped
    """Invoke ``ahn_cli export-ply`` with the progress bar off."""
    return CliRunner().invoke(
        cli, ["export-ply", "--no-progress", *args], catch_exceptions=False
    )


def _filters(out: Path) -> dict[str, Any]:
    """Return the sidecar's filters block."""
    record = read_product_sidecar(sidecar_path(out))
    return cast("dict[str, Any]", record["filters"])


def test_help_states_the_default_class_filter() -> None:
    """The legacy 2,6 default is visible in --help, not implicit."""
    result = CliRunner().invoke(cli, ["export-ply", "--help"])

    assert result.exit_code == 0
    assert "Default 2,6" in " ".join(result.output.split())
    assert "--all-classes" in result.output


def test_default_keeps_classes_2_and_6_and_records_the_default(
    cloud_path: Path, tmp_path: Path
) -> None:
    """With no class option the filter is 2,6 and the sidecar says 'default'."""
    out = tmp_path / "out.ply"

    result = _run("--cloud", str(cloud_path), "--out", str(out))

    assert result.exit_code == 0, result.output
    assert "sidecar out.ply.provenance.json" in result.output
    filters = _filters(out)
    assert filters["class_include"] == [2, 6]
    assert filters["class_origin"] == "default"
    classes = set(read_ply(out).vertex["classification"].tolist())
    assert classes <= {2, 6}
    assert classes


def test_all_classes_keeps_everything_and_records_it(
    cloud_path: Path, tmp_path: Path
) -> None:
    """--all-classes applies no class filter."""
    out = tmp_path / "out.ply"

    result = _run(
        "--cloud", str(cloud_path), "--out", str(out), "--all-classes"
    )

    assert result.exit_code == 0, result.output
    filters = _filters(out)
    assert filters["class_include"] is None
    assert filters["class_origin"] == "all"
    assert len(read_ply(out).vertex) == 300


def test_explicit_classes_are_recorded_as_explicit(
    cloud_path: Path, tmp_path: Path
) -> None:
    """--classes 1,9 keeps those and is recorded as an explicit choice."""
    out = tmp_path / "out.ply"

    result = _run(
        "--cloud", str(cloud_path), "--out", str(out), "--classes", "1,9"
    )

    assert result.exit_code == 0, result.output
    assert _filters(out)["class_origin"] == "explicit"
    assert set(read_ply(out).vertex["classification"].tolist()) <= {1, 9}


def test_spacing_and_coords_and_bbox_reach_the_sidecar(
    cloud_path: Path, tmp_path: Path
) -> None:
    """--spacing, --bbox and --coords are applied and recorded."""
    out = tmp_path / "out.ply"

    result = _run(
        "--cloud",
        str(cloud_path),
        "--out",
        str(out),
        "--all-classes",
        "--spacing",
        "5",
        "--seed",
        "2",
        "--bbox",
        "194000,443000,194060,443060",
        "--coords",
        "float32-centred",
    )

    assert result.exit_code == 0, result.output
    filters = _filters(out)
    assert filters["thinning"]["spacing_m"] == 5.0
    assert filters["thinning"]["seed"] == 2
    assert filters["bbox"] == [194000.0, 443000.0, 194060.0, 443060.0]
    assert read_ply(out).vertex["x"].dtype.str == "<f4"


def test_share_is_available_and_recorded(
    cloud_path: Path, tmp_path: Path
) -> None:
    """The legacy random share still works and is recorded as such."""
    out = tmp_path / "out.ply"

    result = _run(
        "--cloud", str(cloud_path), "--out", str(out), "--all-classes",
        "--share", "0.1",
    )  # fmt: skip

    assert result.exit_code == 0, result.output
    assert _filters(out)["thinning"]["mode"] == "share"
    assert len(read_ply(out).vertex) == 30


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--classes", "2", "--all-classes"], "mutually exclusive"),
        (["--spacing", "1", "--share", "0.5"], "mutually exclusive"),
        (["--share", "2"], "share"),
        (["--spacing", "0"], "radius"),
        (["--bbox", "1,2,3"], "--bbox"),
        (["--bbox", "5,5,1,9"], "--bbox"),
        (["--classes", "two"], "comma-separated"),
    ],
)
def test_bad_options_are_usage_errors(
    cloud_path: Path, tmp_path: Path, extra: list[str], message: str
) -> None:
    """Contradictory or malformed options exit 2 with a clear message."""
    out = tmp_path / "out.ply"

    result = CliRunner().invoke(
        cli,
        ["export-ply", "--cloud", str(cloud_path), "--out", str(out), *extra],
    )

    assert result.exit_code == 2
    assert message in result.output
    assert not out.exists()


def test_an_empty_selection_is_a_clean_error(
    cloud_path: Path, tmp_path: Path
) -> None:
    """A filter that keeps nothing exits 1 with a message, no traceback."""
    out = tmp_path / "out.ply"

    result = CliRunner().invoke(
        cli,
        ["export-ply", "--no-progress", "--cloud", str(cloud_path),
         "--out", str(out), "--classes", "18"],
    )  # fmt: skip

    assert result.exit_code == 1
    assert "no points survived" in result.output
    assert not sidecar_path(out).exists()


def test_progress_bar_path_runs(cloud_path: Path, tmp_path: Path) -> None:
    """With the default progress bar the verb still succeeds."""
    out = tmp_path / "out.ply"

    result = CliRunner().invoke(
        cli,
        ["export-ply", "--cloud", str(cloud_path), "--out", str(out)],
    )

    assert result.exit_code == 0, result.output
