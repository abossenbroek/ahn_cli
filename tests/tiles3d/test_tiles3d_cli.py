"""Tests for the ``tiles3d`` CLI subcommand."""

from __future__ import annotations

from typing import TYPE_CHECKING

from click.testing import CliRunner

import ahn_cli.cli.app as app_module
from ahn_cli.cli import cli
from tests.tiles3d.conftest import (
    grid_for_ortho,
    make_ortho,
    synth_rgb,
    write_exr,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pytest
    from typing_extensions import Self


class _SpyBar:
    """A tqdm stand-in recording every (n, total) update."""

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


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    rgb = synth_rgb(6, 6, seed=21)
    ortho = make_ortho(tmp_path / "ortho.tif", rgb)
    heights = write_exr(tmp_path / "r.exr", grid_for_ortho(rgb))
    return ortho, heights


def test_tiles3d_happy_path(tmp_path: Path) -> None:
    """A matching pair converts and reports the verified summary."""
    ortho, heights = _inputs(tmp_path)
    out = tmp_path / "tiles3d"
    result = CliRunner().invoke(
        cli,
        [
            "tiles3d",
            "--ortho",
            str(ortho),
            "--heights",
            str(heights),
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "1 tile(s)" in result.output
    assert "verified" in result.output
    assert (out / "tileset.json").is_file()
    assert (out / "tiles" / "0-0-0.glb").is_file()


def test_tiles3d_mismatch_is_a_clean_error(tmp_path: Path) -> None:
    """A heights file from another ortho exits 1 with the typed message."""
    ortho, _ = _inputs(tmp_path)
    other = synth_rgb(6, 6, seed=22)
    heights = write_exr(tmp_path / "other.exr", grid_for_ortho(other))
    result = CliRunner().invoke(
        cli,
        [
            "tiles3d",
            "--ortho",
            str(ortho),
            "--heights",
            str(heights),
            "--out",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 1
    assert "plane" in result.output


def test_tiles3d_missing_ortho_is_a_usage_error(tmp_path: Path) -> None:
    """A nonexistent --ortho is Click-level validation (exit 2)."""
    _, heights = _inputs(tmp_path)
    result = CliRunner().invoke(
        cli,
        [
            "tiles3d",
            "--ortho",
            str(tmp_path / "absent.tif"),
            "--heights",
            str(heights),
            "--out",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 2


def test_tiles3d_drives_the_progress_bar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The tqdm bar is driven once per emitted tile."""
    spy = _SpyBar()
    monkeypatch.setattr(app_module, "tqdm", spy)
    ortho, heights = _inputs(tmp_path)
    result = CliRunner().invoke(
        cli,
        [
            "tiles3d",
            "--ortho",
            str(ortho),
            "--heights",
            str(heights),
            "--out",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert spy.updates == [(1, 1)]
