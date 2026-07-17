"""Tests for the ``ahn_cli pipeline run`` CLI verb."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from click.testing import CliRunner

from ahn_cli.cli.app import cli
from tests.pipeline.scenes import build_site

if TYPE_CHECKING:
    from pathlib import Path

_WIDTH = 8
_HEIGHT = 6


def _spec_dict(site: Path, tmp: Path) -> dict[str, object]:
    return {
        "aoi": {"bbox": f"0,0,{_WIDTH},{_HEIGHT}"},
        "tiling": {"tile_pixels": 256, "halo": "auto"},
        "workdir": str(tmp / "wd"),
        "output": str(tmp / "out"),
        "stages": [
            {"type": "read", "path": str(site)},
            {"type": "reconcile", "method": "idw"},
            {"type": "tiles3d", "profile": "strict"},
        ],
    }


def test_pipeline_run_yaml(tmp_path: Path) -> None:
    """A YAML spec runs end-to-end and reports the tile count."""
    site, _cloud, _ortho = build_site(tmp_path, seed=0)
    spec_dict = _spec_dict(site, tmp_path)
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(
        json.dumps(spec_dict), encoding="utf-8"
    )  # JSON is YAML

    result = CliRunner().invoke(cli, ["pipeline", "run", str(spec_path)])

    assert result.exit_code == 0, result.output
    assert "1 tile(s)" in result.output
    assert (tmp_path / "out" / "tileset.json").is_file()


def test_pipeline_run_json_with_point_spacing(tmp_path: Path) -> None:
    """A .json spec is parsed as JSON and honours --point-spacing-m."""
    site, _cloud, _ortho = build_site(tmp_path, seed=1)
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(
        json.dumps(_spec_dict(site, tmp_path)), encoding="utf-8"
    )

    result = CliRunner().invoke(
        cli,
        ["pipeline", "run", str(spec_path), "--point-spacing-m", "0.3"],
    )

    assert result.exit_code == 0, result.output
    assert (tmp_path / "out" / "tileset.json").is_file()


def test_pipeline_run_translates_pipeline_error(tmp_path: Path) -> None:
    """A fetch source (a deferred seam) surfaces as a ClickException."""
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "aoi": {"bbox": "0,0,8,6"},
                "workdir": str(tmp_path / "wd"),
                "output": str(tmp_path / "out"),
                "stages": [
                    {"type": "fetch", "source": "pdok"},
                    {"type": "reconcile", "method": "idw"},
                    {"type": "tiles3d", "profile": "strict"},
                ],
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(cli, ["pipeline", "run", str(spec_path)])

    assert result.exit_code != 0
    assert "read` source" in result.output


def test_pipeline_run_rejects_malformed_spec(tmp_path: Path) -> None:
    """A malformed spec is reported as a ClickException, not a traceback."""
    spec_path = tmp_path / "spec.json"
    spec_path.write_text("{ not valid json", encoding="utf-8")

    result = CliRunner().invoke(cli, ["pipeline", "run", str(spec_path)])

    assert result.exit_code != 0
    assert "not valid JSON" in result.output
