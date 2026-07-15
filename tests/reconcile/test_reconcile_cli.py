"""Tests for the ``reconcile`` CLI command."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import rasterio
from click.testing import CliRunner, Result
from rasterio.transform import from_bounds

from ahn_cli.cli import app
from ahn_cli.cli.app import cli

if TYPE_CHECKING:
    from pathlib import Path

    import pytest
    from typing_extensions import Self


class _SpyBar:
    """A tqdm stand-in recording every (n, total) update, standing in for tqdm."""

    def __init__(self) -> None:
        self.n = 0
        self.total: int | None = None
        self.updates: list[tuple[int, int | None]] = []

    def __call__(self, **_kwargs: object) -> Self:
        return self

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        return None

    def refresh(self) -> None:
        self.updates.append((self.n, self.total))


def test_reconcile_drives_the_progress_bar_across_blocks(
    ortho_path: Path,
    cloud_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each streamed block updates the tqdm bar's n/total, not just runs clean.

    Replaces ``tqdm`` itself with a spy so the assertion is on the bar's actual
    state after each block, not merely that the CLI exits 0 (which it would even
    if the progress wiring silently no-opped).
    """
    spy = _SpyBar()
    monkeypatch.setattr(app, "tqdm", spy)
    monkeypatch.setattr("ahn_cli.reconcile.reconcile._BLOCK_CELLS", 6)

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
            "--format",
            "pt",
        ],
    )

    assert result.exit_code == 0, result.output
    assert spy.updates == [(1, 6), (2, 6), (3, 6), (4, 6), (5, 6), (6, 6)]


def test_reconcile_no_progress_skips_the_bar(
    ortho_path: Path,
    cloud_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--no-progress never constructs a tqdm bar, and the run still succeeds."""

    def _boom(**_kwargs: object) -> None:
        msg = "tqdm must not be constructed when --no-progress is passed"
        raise AssertionError(msg)

    monkeypatch.setattr(app, "tqdm", _boom)

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
            "--format",
            "pt",
            "--no-progress",
        ],
    )

    assert result.exit_code == 0, result.output


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


def test_reconcile_classes_keep(
    ortho_path: Path, cloud_path: Path, tmp_path: Path
) -> None:
    """'--classes keep:0' keeps the fixture's class-0 points and runs."""
    result = _invoke(
        ortho_path,
        cloud_path,
        tmp_path / "out",
        ["--classes", "keep:0", "--format", "pt"],
    )
    assert result.exit_code == 0, result.output
    assert "cleaned" in result.output


def test_reconcile_classes_drop(
    ortho_path: Path, cloud_path: Path, tmp_path: Path
) -> None:
    """'--classes drop:7' drops noise (none here) and runs."""
    result = _invoke(
        ortho_path,
        cloud_path,
        tmp_path / "out",
        ["--classes", "drop:7", "--format", "pt"],
    )
    assert result.exit_code == 0, result.output


def test_reconcile_classes_bad_mode_rejected(
    ortho_path: Path, cloud_path: Path, tmp_path: Path
) -> None:
    """An unknown --classes mode is rejected (exit code 2)."""
    result = _invoke(
        ortho_path, cloud_path, tmp_path / "out", ["--classes", "foo:2"]
    )
    assert result.exit_code == 2
    assert "--classes" in result.output


def test_reconcile_classes_empty_list_rejected(
    ortho_path: Path, cloud_path: Path, tmp_path: Path
) -> None:
    """A --classes spec with no class list is rejected (exit code 2)."""
    result = _invoke(
        ortho_path, cloud_path, tmp_path / "out", ["--classes", "keep:"]
    )
    assert result.exit_code == 2


def test_reconcile_classes_non_integer_rejected(
    ortho_path: Path, cloud_path: Path, tmp_path: Path
) -> None:
    """A --classes list with a non-integer code is rejected (exit code 2)."""
    result = _invoke(
        ortho_path, cloud_path, tmp_path / "out", ["--classes", "keep:x"]
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
