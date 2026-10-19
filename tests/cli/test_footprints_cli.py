"""Tests for the ``ahn_cli footprints`` command (fetch faked)."""

from pathlib import Path

import pytest
from click.testing import CliRunner

from ahn_cli.cli import app, cli
from ahn_cli.domain import BBox
from ahn_cli.fetch.top10nl import KasFetch, Top10nlError, features_sha256

_BOX = "100,200,110,208"


def _fake(dates: tuple[str, ...]) -> KasFetch:
    ring = [[102, 205], [104, 205], [104, 208], [102, 208], [102, 205]]
    feats: tuple[dict[str, object], ...] = (
        {
            "type": "Feature",
            "id": "a",
            "properties": {},
            "geometry": {"type": "Polygon", "coordinates": [ring]},
        },
    )
    return KasFetch(feats, ("u",), 2, features_sha256(feats), dates)


@pytest.mark.parametrize(
    ("dates", "shown"),
    [(("2010-01-01", "2024-06-01"), "2010-01-01..2024-06-01"), ((), "?..?")],
)
def test_footprints_writes_product(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dates: tuple[str, ...],
    shown: str,
) -> None:
    """A good run writes the product and reports counts, dates and pin."""
    seen: list[BBox] = []

    def fake(bbox: BBox) -> KasFetch:
        seen.append(bbox)
        return _fake(dates)

    monkeypatch.setattr(app, "fetch_kas", fake)
    res = CliRunner().invoke(
        cli, ["footprints", "--bbox", _BOX, "--out", str(tmp_path)]
    )
    assert res.exit_code == 0, res.output
    assert seen == [(100.0, 200.0, 110.0, 208.0)]
    assert "1 of 2 buildings, 6 m2 covered at 1 m" in res.output
    assert shown in res.output
    assert (tmp_path / "footprints" / "kas_cover.tif").is_file()


def _no_fetch(_bbox: BBox) -> KasFetch:
    raise AssertionError


@pytest.mark.parametrize(
    ("args", "code", "match"),
    [
        (["--bbox", "1,2,3"], 2, "--bbox must be four"),
        (["--bbox", _BOX, "--res", "0"], 1, "positive"),
    ],
)
def test_footprints_fails_before_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    args: list[str],
    code: int,
    match: str,
) -> None:
    """Bad box or resolution fail without fetching."""
    monkeypatch.setattr(app, "fetch_kas", _no_fetch)
    res = CliRunner().invoke(
        cli, ["footprints", "--out", str(tmp_path), *args]
    )
    assert res.exit_code == code
    assert match in res.output


def test_footprints_refuses_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An existing product file stops the run before fetching."""
    (tmp_path / "footprints").mkdir()
    (tmp_path / "footprints" / "kas_cover.tif").write_bytes(b"x")
    monkeypatch.setattr(app, "fetch_kas", _no_fetch)
    res = CliRunner().invoke(
        cli, ["footprints", "--bbox", _BOX, "--out", str(tmp_path)]
    )
    assert res.exit_code == 1
    assert "refusing to overwrite" in res.output


def test_footprints_reports_fetch_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fetch failure becomes a clean Click error."""

    def boom(_bbox: BBox) -> KasFetch:
        msg = "no TOP10NL 'kas, warenhuis' polygons"
        raise Top10nlError(msg)

    monkeypatch.setattr(app, "fetch_kas", boom)
    res = CliRunner().invoke(
        cli, ["footprints", "--bbox", _BOX, "--out", str(tmp_path)]
    )
    assert res.exit_code == 1
    assert "no TOP10NL" in res.output
