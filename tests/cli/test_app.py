"""Tests for the ``ahn_cli`` Click group and its subcommands."""

from pathlib import Path
from typing import cast

import click
import numpy as np
import pytest
import rasterio
from click.testing import CliRunner
from rasterio.transform import from_bounds

from ahn_cli.cli import app, cli
from ahn_cli.fetch.acquisition import SITE_SUBDIRS, AcquisitionRequest
from ahn_cli.fetch.generation import default_registry
from ahn_cli.fetch.source import SourceKind


def _write_geotiff(path: Path) -> None:
    """Write a tiny valid single-band GeoTIFF at ``path``."""
    transform = from_bounds(3.0, 50.0, 7.0, 53.0, 4, 3)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=3,
        width=4,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(np.arange(12, dtype="float32").reshape(1, 3, 4))


def _short_flags(command: click.Command) -> list[str]:
    """Return every single-dash short flag declared on ``command``."""
    return [
        opt
        for param in command.params
        for opt in param.opts
        if len(opt) == 2 and opt.startswith("-") and not opt.startswith("--")
    ]


def test_group_exposes_fetch_prep_and_import_viirs() -> None:
    """The CLI group exposes the acquisition/transform verbs plus VIIRS import."""
    assert set(cli.commands) == {"fetch", "prep", "import-viirs"}


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
        [
            "fetch",
            "--out",
            str(tmp_path / "s"),
            "--city",
            "delft",
            "--bbox",
            "0,0,1,1",
        ],
    )

    assert result.exit_code == 2
    assert "exactly one of --city, --bbox, or --geojson" in result.output


def test_fetch_with_city_builds_layout_then_reports_not_wired(
    tmp_path: Path,
) -> None:
    """A valid fetch creates the layout and stops at the un-wired seam."""
    site = tmp_path / "delft"

    result = CliRunner().invoke(
        cli, ["fetch", "--out", str(site), "--city", "delft"]
    )

    assert result.exit_code == 1
    assert "wired" in result.output.lower()
    for name in SITE_SUBDIRS:
        assert (site / name).is_dir()


class _AcquireSpy:
    """Records the request the CLI dispatched, standing in for acquire()."""

    def __init__(self) -> None:
        self.request: AcquisitionRequest | None = None

    def __call__(self, request: AcquisitionRequest) -> tuple[Path, ...]:
        self.request = request
        return ()


def test_fetch_bbox_dispatches_to_acquire(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The wired bbox selector builds the layout and dispatches to acquire."""
    site = tmp_path / "bboxsite"
    spy = _AcquireSpy()
    monkeypatch.setattr(app, "acquire", spy)

    result = CliRunner().invoke(
        cli,
        ["fetch", "--out", str(site), "--bbox", "0,0,1,1"],
    )

    assert result.exit_code == 0
    assert (site / SITE_SUBDIRS[0]).is_dir()
    assert spy.request is not None
    assert spy.request.selector.value == "bbox"
    assert spy.request.source is SourceKind.PDOK


def test_fetch_source_flag_selects_geotiles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The --source flag flows through to the acquisition request."""
    spy = _AcquireSpy()
    monkeypatch.setattr(app, "acquire", spy)

    result = CliRunner().invoke(
        cli,
        [
            "fetch",
            "--out",
            str(tmp_path / "s"),
            "--bbox",
            "0,0,1,1",
            "--source",
            "geotiles",
        ],
    )

    assert result.exit_code == 0
    assert spy.request is not None
    assert spy.request.source is SourceKind.GEOTILES


def test_fetch_rejects_unknown_source(tmp_path: Path) -> None:
    """A --source outside the registry's tokens is a usage error."""
    result = CliRunner().invoke(
        cli,
        [
            "fetch",
            "--out",
            str(tmp_path / "s"),
            "--bbox",
            "0,0,1,1",
            "--source",
            "wms",
        ],
    )

    assert result.exit_code == 2


def test_fetch_accepts_geojson_selector(tmp_path: Path) -> None:
    """The geojson selector is a valid, mutually-exclusive area choice."""
    site = tmp_path / "geojsonsite"

    result = CliRunner().invoke(
        cli,
        ["fetch", "--out", str(site), "--geojson", "area.geojson"],
    )

    assert result.exit_code == 1
    assert (site / SITE_SUBDIRS[0]).is_dir()


def test_fetch_ahn_defaults_to_auto() -> None:
    """The ``--ahn`` option defaults to auto and offers the registry tokens."""
    ahn_option = next(
        p for p in cli.commands["fetch"].params if p.name == "ahn"
    )

    assert ahn_option.default == "auto"
    choice_type = cast("click.Choice[str]", ahn_option.type)
    assert tuple(choice_type.choices) == default_registry().tokens()


def test_fetch_accepts_an_explicit_generation(tmp_path: Path) -> None:
    """An explicit ``--ahn ahn4`` is a valid choice that reaches the seam."""
    site = tmp_path / "gensite"

    result = CliRunner().invoke(
        cli,
        ["fetch", "--out", str(site), "--city", "delft", "--ahn", "ahn4"],
    )

    assert result.exit_code == 1
    assert "wired" in result.output.lower()
    for name in SITE_SUBDIRS:
        assert (site / name).is_dir()


def test_fetch_rejects_an_unknown_generation(tmp_path: Path) -> None:
    """A generation outside the registry's tokens is a usage error."""
    result = CliRunner().invoke(
        cli,
        [
            "fetch",
            "--out",
            str(tmp_path / "s"),
            "--city",
            "delft",
            "--ahn",
            "ahn9",
        ],
    )

    assert result.exit_code == 2


def test_import_viirs_copies_geotiff_with_provenance(tmp_path: Path) -> None:
    """``import-viirs`` copies the raster into <out>/viirs/ with provenance."""
    source = tmp_path / "lights.tif"
    _write_geotiff(source)
    site = tmp_path / "delft"

    result = CliRunner().invoke(
        cli, ["import-viirs", "--out", str(site), str(source)]
    )

    assert result.exit_code == 0
    assert "Imported VIIRS raster" in result.output
    dest = site / "viirs" / "lights.tif"
    assert dest.read_bytes() == source.read_bytes()
    assert (site / "viirs" / "lights.tif.provenance.json").is_file()


def test_import_viirs_rejects_a_non_raster_file(tmp_path: Path) -> None:
    """An import-viirs file that is not a raster is a Click error, not a crash."""
    source = tmp_path / "broken.tif"
    source.write_bytes(b"not a GeoTIFF")

    result = CliRunner().invoke(
        cli,
        ["import-viirs", "--out", str(tmp_path / "delft"), str(source)],
    )

    assert result.exit_code == 1
    assert "not a readable raster" in result.output


def test_import_viirs_requires_an_existing_file(tmp_path: Path) -> None:
    """A missing geotiff path is rejected by Click before importing."""
    result = CliRunner().invoke(
        cli,
        ["import-viirs", "--out", str(tmp_path / "delft"), "nope.tif"],
    )

    assert result.exit_code == 2


def test_prep_parses_filters_then_reports_not_wired(tmp_path: Path) -> None:
    """A valid prep parses class filters and stops at the un-wired seam."""
    result = CliRunner().invoke(
        cli,
        [
            "prep",
            "--data",
            str(tmp_path),
            "--include-class",
            "2,6",
            "--points",
        ],
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


def test_prep_rejects_class_in_both_include_and_exclude(
    tmp_path: Path,
) -> None:
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
