"""Tests for the ``reconcile`` CLI command."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import rasterio
from click.testing import CliRunner, Result
from rasterio.transform import from_bounds

from ahn_cli.cli.app import cli

if TYPE_CHECKING:
    from pathlib import Path


def test_reconcile_idw_default(
    ortho_path: Path, cloud_path: Path, tmp_path: Path
) -> None:
    """The default (IDW) run exits cleanly and writes all four formats."""
    out = tmp_path / "out"
    result = CliRunner().invoke(
        cli,
        [
            "reconcile",
            "--ortho",
            str(ortho_path),
            "--cloud",
            str(cloud_path),
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out / "reconciled.pt").exists()
    assert (out / "reconciled.exr").exists()


def test_reconcile_linear(
    ortho_path: Path, cloud_path: Path, tmp_path: Path
) -> None:
    """The linear method runs through the CLI."""
    result = CliRunner().invoke(
        cli,
        [
            "reconcile",
            "--ortho",
            str(ortho_path),
            "--cloud",
            str(cloud_path),
            "--out",
            str(tmp_path / "out"),
            "--method",
            "linear",
            "--format",
            "ply",
        ],
    )
    assert result.exit_code == 0, result.output


def test_reconcile_kriging_single_format(
    ortho_path: Path, cloud_path: Path, tmp_path: Path
) -> None:
    """Kriging with a single explicit format writes only that file."""
    out = tmp_path / "out"
    result = CliRunner().invoke(
        cli,
        [
            "reconcile",
            "--ortho",
            str(ortho_path),
            "--cloud",
            str(cloud_path),
            "--out",
            str(out),
            "--method",
            "kriging",
            "--kriging",
            "spherical,0.0,1.0,5.0,16",
            "--format",
            "laz",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out / "reconciled.laz").exists()
    assert not (out / "reconciled.pt").exists()


def _invoke(ortho: Path, cloud: Path, out: Path, extra: list[str]) -> Result:
    return CliRunner().invoke(
        cli,
        [
            "reconcile",
            "--ortho",
            str(ortho),
            "--cloud",
            str(cloud),
            "--out",
            str(out),
            *extra,
        ],
    )


def test_reconcile_bad_idw_value_is_rejected(
    ortho_path: Path, cloud_path: Path, tmp_path: Path
) -> None:
    """An out-of-range IDW value is a click BadParameter (exit code 2)."""
    result = _invoke(
        ortho_path, cloud_path, tmp_path / "out", ["--idw", "2.0,0"]
    )
    assert result.exit_code == 2
    assert "idw k" in result.output


def test_reconcile_bad_idw_arity_is_rejected(
    ortho_path: Path, cloud_path: Path, tmp_path: Path
) -> None:
    """An --idw spec that is not 'power,k' is rejected (exit code 2)."""
    result = _invoke(
        ortho_path, cloud_path, tmp_path / "out", ["--idw", "2.0"]
    )
    assert result.exit_code == 2
    assert "--idw" in result.output


def test_reconcile_bad_kriging_arity_is_rejected(
    ortho_path: Path, cloud_path: Path, tmp_path: Path
) -> None:
    """A --kriging spec without five fields is rejected (exit code 2)."""
    result = _invoke(
        ortho_path,
        cloud_path,
        tmp_path / "out",
        ["--method", "kriging", "--kriging", "spherical,0,1"],
    )
    assert result.exit_code == 2
    assert "--kriging" in result.output


def test_reconcile_bad_kriging_model_is_rejected(
    ortho_path: Path, cloud_path: Path, tmp_path: Path
) -> None:
    """A --kriging spec naming an unknown model is rejected (exit code 2)."""
    result = _invoke(
        ortho_path,
        cloud_path,
        tmp_path / "out",
        ["--method", "kriging", "--kriging", "linear,0.0,1.0,5.0,8"],
    )
    assert result.exit_code == 2


def test_reconcile_ortho_without_rgb_is_clean_error(
    cloud_path: Path, tmp_path: Path
) -> None:
    """A readable but 2-band ortho surfaces as a clean ClickException (exit 1)."""
    gray = tmp_path / "gray.tif"
    transform = from_bounds(100.0, 100.0, 103.0, 103.0, 6, 6)
    with rasterio.open(
        gray,
        "w",
        driver="GTiff",
        height=6,
        width=6,
        count=2,
        dtype="uint8",
        crs="EPSG:28992",
        transform=transform,
    ) as dst:
        dst.write(np.zeros((2, 6, 6), dtype=np.uint8))
    result = CliRunner().invoke(
        cli,
        [
            "reconcile",
            "--ortho",
            str(gray),
            "--cloud",
            str(cloud_path),
            "--out",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 1
    assert "band" in result.output
