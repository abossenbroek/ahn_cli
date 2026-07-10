"""Tests for the ``ahn_cli`` Click group and its subcommands."""

from pathlib import Path

import click
from click.testing import CliRunner

from ahn_cli.cli import cli
from ahn_cli.fetch.acquisition import SITE_SUBDIRS


def _short_flags(command: click.Command) -> list[str]:
    """Return every single-dash short flag declared on ``command``."""
    return [
        opt
        for param in command.params
        for opt in param.opts
        if len(opt) == 2 and opt.startswith("-") and not opt.startswith("--")
    ]


def test_group_exposes_exactly_fetch_and_prep() -> None:
    """The restructured CLI is a group with the two required verbs."""
    assert set(cli.commands) == {"fetch", "prep"}


def test_no_duplicate_short_flags_anywhere() -> None:
    """Regression for the -e collision: no flag is reused, per command or overall."""
    assert set(cli.commands) >= {"fetch", "prep"}
    all_shorts: list[str] = []
    for command in cli.commands.values():
        shorts = _short_flags(command)
        assert len(shorts) == len(set(shorts)), (command.name, shorts)
        all_shorts.extend(shorts)
    assert len(all_shorts) == len(set(all_shorts)), all_shorts


def test_fetch_requires_an_area_selector() -> None:
    """With no city/bbox/geojson, fetch reports the mutual-exclusivity rule."""
    result = CliRunner().invoke(cli, ["fetch", "--out", "site"])

    assert result.exit_code == 2
    assert "exactly one of --city, --bbox, or --geojson" in result.output


def test_fetch_rejects_two_area_selectors(tmp_path: Path) -> None:
    """Supplying two selectors is a usage error, not a silent pick."""
    result = CliRunner().invoke(
        cli,
        ["fetch", "--out", str(tmp_path / "s"), "--city", "delft", "--bbox", "0,0,1,1"],
    )

    assert result.exit_code == 2
    assert "exactly one of --city, --bbox, or --geojson" in result.output


def test_fetch_with_city_builds_layout_then_reports_not_wired(
    tmp_path: Path,
) -> None:
    """A valid fetch creates the layout and stops at the un-wired seam."""
    site = tmp_path / "delft"

    result = CliRunner().invoke(cli, ["fetch", "--out", str(site), "--city", "delft"])

    assert result.exit_code == 1
    assert "wired" in result.output.lower()
    for name in SITE_SUBDIRS:
        assert (site / name).is_dir()


def test_fetch_accepts_bbox_selector(tmp_path: Path) -> None:
    """The bbox selector is a valid, mutually-exclusive area choice."""
    site = tmp_path / "bboxsite"

    result = CliRunner().invoke(
        cli,
        ["fetch", "--out", str(site), "--bbox", "0,0,1,1"],
    )

    assert result.exit_code == 1
    assert (site / SITE_SUBDIRS[0]).is_dir()


def test_fetch_accepts_geojson_selector(tmp_path: Path) -> None:
    """The geojson selector is a valid, mutually-exclusive area choice."""
    site = tmp_path / "geojsonsite"

    result = CliRunner().invoke(
        cli,
        ["fetch", "--out", str(site), "--geojson", "area.geojson"],
    )

    assert result.exit_code == 1
    assert (site / SITE_SUBDIRS[0]).is_dir()


def test_prep_parses_filters_then_reports_not_wired(tmp_path: Path) -> None:
    """A valid prep parses class filters and stops at the un-wired seam."""
    result = CliRunner().invoke(
        cli,
        ["prep", "--data", str(tmp_path), "--include-class", "2,6", "--points"],
    )

    assert result.exit_code == 1
    assert "wired" in result.output.lower()


def test_prep_with_no_filters_reports_not_wired(tmp_path: Path) -> None:
    """Prep without class filters still dispatches to the un-wired seam."""
    result = CliRunner().invoke(cli, ["prep", "--data", str(tmp_path)])

    assert result.exit_code == 1
    assert "wired" in result.output.lower()


def test_prep_rejects_non_integer_class(tmp_path: Path) -> None:
    """A non-integer class list is a bad parameter."""
    result = CliRunner().invoke(
        cli,
        ["prep", "--data", str(tmp_path), "--include-class", "ground"],
    )

    assert result.exit_code == 2
    assert "integer" in result.output.lower()


def test_prep_rejects_class_in_both_include_and_exclude(tmp_path: Path) -> None:
    """A class cannot be both kept and dropped."""
    result = CliRunner().invoke(
        cli,
        [
            "prep",
            "--data",
            str(tmp_path),
            "--include-class",
            "2,6",
            "--exclude-class",
            "6",
        ],
    )

    assert result.exit_code == 2
    assert "included and excluded" in result.output
