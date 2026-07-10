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
    """The CLI group exposes the acquisition/transform verbs and the exporters."""
    assert set(cli.commands) == {
        "fetch",
        "prep",
        "import-viirs",
        "export-positions",
    }


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


class _OrthoSpy:
    """Records every request the CLI dispatched to acquire_ortho()."""

    def __init__(self) -> None:
        self.requests: list[AcquisitionRequest] = []

    def __call__(self, request: AcquisitionRequest) -> None:
        self.requests.append(request)


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


class _DsmSpy:
    """Records the request the CLI dispatched, standing in for fetch_dsm."""

    def __init__(self) -> None:
        self.request: AcquisitionRequest | None = None

    def __call__(self, request: AcquisitionRequest) -> Path:
        self.request = request
        return request.site_dir / "dsm.tif"


def test_fetch_without_dsm_flag_skips_dsm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Absent --dsm, the DSM fetch is not invoked."""
    monkeypatch.setattr(app, "acquire", _AcquireSpy())
    dsm_spy = _DsmSpy()
    monkeypatch.setattr(app, "fetch_dsm", dsm_spy)

    result = CliRunner().invoke(
        cli, ["fetch", "--out", str(tmp_path / "s"), "--bbox", "0,0,1,1"]
    )

    assert result.exit_code == 0
    assert dsm_spy.request is None


def test_fetch_dsm_flag_dispatches_to_fetch_dsm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With --dsm, the same request is dispatched to the DSM fetch."""
    monkeypatch.setattr(app, "acquire", _AcquireSpy())
    dsm_spy = _DsmSpy()
    monkeypatch.setattr(app, "fetch_dsm", dsm_spy)

    result = CliRunner().invoke(
        cli,
        ["fetch", "--out", str(tmp_path / "s"), "--bbox", "0,0,1,1", "--dsm"],
    )

    assert result.exit_code == 0
    assert dsm_spy.request is not None
    assert dsm_spy.request.selector.value == "bbox"


def test_fetch_dsm_error_is_a_click_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A DSM acquisition failure becomes a tidy Click error, not a traceback."""
    monkeypatch.setattr(app, "acquire", _AcquireSpy())

    def boom(request: AcquisitionRequest) -> Path:
        del request
        msg = "no DSM sheet covers the AOI"
        raise app.AcquisitionError(msg)

    monkeypatch.setattr(app, "fetch_dsm", boom)

    result = CliRunner().invoke(
        cli,
        ["fetch", "--out", str(tmp_path / "s"), "--bbox", "0,0,1,1", "--dsm"],
    )

    assert result.exit_code == 1
    assert "no DSM sheet" in result.output


def test_fetch_ortho_flag_dispatches_to_acquire_ortho(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The --ortho flag also dispatches the request to acquire_ortho."""
    site = tmp_path / "orthosite"
    acquire_spy = _AcquireSpy()
    ortho_spy = _OrthoSpy()
    monkeypatch.setattr(app, "acquire", acquire_spy)
    monkeypatch.setattr(app, "acquire_ortho", ortho_spy)

    result = CliRunner().invoke(
        cli,
        ["fetch", "--out", str(site), "--bbox", "0,0,1,1", "--ortho"],
    )

    assert result.exit_code == 0
    assert acquire_spy.request is not None
    assert len(ortho_spy.requests) == 1
    assert ortho_spy.requests[0].selector.value == "bbox"


def test_fetch_without_ortho_flag_skips_acquire_ortho(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without --ortho, the orthophoto fetch is not invoked."""
    ortho_spy = _OrthoSpy()
    monkeypatch.setattr(app, "acquire", _AcquireSpy())
    monkeypatch.setattr(app, "acquire_ortho", ortho_spy)

    result = CliRunner().invoke(
        cli,
        ["fetch", "--out", str(tmp_path / "s"), "--bbox", "0,0,1,1"],
    )

    assert result.exit_code == 0
    assert ortho_spy.requests == []


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


def test_export_positions_writes_exr(tmp_path: Path) -> None:
    """``export-positions`` turns <data>/dsm.tif into a <data>/positions.exr."""
    site = tmp_path / "delft"
    site.mkdir()
    _write_geotiff(site / "dsm.tif")

    result = CliRunner().invoke(
        cli, ["export-positions", "--data", str(site)]
    )

    assert result.exit_code == 0
    assert "Wrote" in result.output
    exr = site / "positions.exr"
    assert exr.read_bytes().startswith(b"\x76\x2f\x31\x01")


def test_export_positions_missing_dsm_is_a_click_error(
    tmp_path: Path,
) -> None:
    """A site directory with no dsm.tif is a tidy Click error, not a crash."""
    site = tmp_path / "delft"
    site.mkdir()

    result = CliRunner().invoke(
        cli, ["export-positions", "--data", str(site)]
    )

    assert result.exit_code == 1
    assert "not readable" in result.output


def test_export_positions_requires_an_existing_directory(
    tmp_path: Path,
) -> None:
    """A missing site directory is rejected by Click before exporting."""
    result = CliRunner().invoke(
        cli, ["export-positions", "--data", str(tmp_path / "absent")]
    )

    assert result.exit_code == 2


def test_prep_parses_filters_then_reports_missing_tiles(
    tmp_path: Path,
) -> None:
    """A valid prep parses class filters, then fails tidily with no tiles."""
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
    assert "ahn" in result.output.lower()


def test_prep_with_no_filters_reports_missing_tiles(tmp_path: Path) -> None:
    """Prep without class filters dispatches and fails tidily with no tiles."""
    result = CliRunner().invoke(cli, ["prep", "--data", str(tmp_path)])

    assert result.exit_code == 1
    assert "ahn" in result.output.lower()


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


def test_prep_accepts_voxel_thinning(tmp_path: Path) -> None:
    """A valid voxel request parses and reaches the transform stage."""
    result = CliRunner().invoke(
        cli,
        [
            "prep",
            "--data",
            str(tmp_path),
            "--thin-method",
            "voxel",
            "--thin-grade",
            "3",
        ],
    )

    assert result.exit_code == 1
    assert "ahn" in result.output.lower()


def test_prep_accepts_poisson_thinning(tmp_path: Path) -> None:
    """A valid Poisson request parses and reaches the transform stage."""
    result = CliRunner().invoke(
        cli,
        [
            "prep",
            "--data",
            str(tmp_path),
            "--thin-method",
            "poisson",
            "--thin-radius",
            "1.5",
            "--thin-seed",
            "7",
        ],
    )

    assert result.exit_code == 1
    assert "ahn" in result.output.lower()


def test_prep_rejects_thin_param_without_method(tmp_path: Path) -> None:
    """A grade or radius without a method is a usage error."""
    result = CliRunner().invoke(
        cli,
        ["prep", "--data", str(tmp_path), "--thin-grade", "3"],
    )

    assert result.exit_code == 2
    assert "require --thin-method" in result.output


def test_prep_rejects_voxel_without_grade(tmp_path: Path) -> None:
    """Voxel thinning demands a grade."""
    result = CliRunner().invoke(
        cli,
        ["prep", "--data", str(tmp_path), "--thin-method", "voxel"],
    )

    assert result.exit_code == 2
    assert "requires --thin-grade" in result.output


def test_prep_rejects_voxel_with_radius(tmp_path: Path) -> None:
    """A radius is meaningless for voxel thinning."""
    result = CliRunner().invoke(
        cli,
        [
            "prep",
            "--data",
            str(tmp_path),
            "--thin-method",
            "voxel",
            "--thin-grade",
            "3",
            "--thin-radius",
            "1.0",
        ],
    )

    assert result.exit_code == 2
    assert "not valid for voxel" in result.output


def test_prep_rejects_out_of_range_voxel_grade(tmp_path: Path) -> None:
    """An out-of-range grade is a bad parameter (validated by the VO)."""
    result = CliRunner().invoke(
        cli,
        [
            "prep",
            "--data",
            str(tmp_path),
            "--thin-method",
            "voxel",
            "--thin-grade",
            "42",
        ],
    )

    assert result.exit_code == 2
    assert "voxel grade" in result.output


def test_prep_rejects_poisson_without_radius(tmp_path: Path) -> None:
    """Poisson thinning demands a radius."""
    result = CliRunner().invoke(
        cli,
        ["prep", "--data", str(tmp_path), "--thin-method", "poisson"],
    )

    assert result.exit_code == 2
    assert "requires --thin-radius" in result.output


def test_prep_rejects_poisson_with_grade(tmp_path: Path) -> None:
    """A grade is meaningless for Poisson thinning."""
    result = CliRunner().invoke(
        cli,
        [
            "prep",
            "--data",
            str(tmp_path),
            "--thin-method",
            "poisson",
            "--thin-radius",
            "1.0",
            "--thin-grade",
            "3",
        ],
    )

    assert result.exit_code == 2
    assert "not valid for poisson" in result.output


def test_prep_rejects_non_positive_poisson_radius(tmp_path: Path) -> None:
    """A non-positive radius is a bad parameter (validated by the VO)."""
    result = CliRunner().invoke(
        cli,
        [
            "prep",
            "--data",
            str(tmp_path),
            "--thin-method",
            "poisson",
            "--thin-radius",
            "-1.0",
        ],
    )

    assert result.exit_code == 2
    assert "poisson radius" in result.output
