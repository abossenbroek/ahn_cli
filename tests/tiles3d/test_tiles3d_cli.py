"""Tests for the ``tiles3d`` CLI subcommand."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from click.testing import CliRunner, Result

import ahn_cli.cli.app as app_module
from ahn_cli.cli import cli
from ahn_cli.tiles3d import jpeg, meshopt, quantize
from ahn_cli.tiles3d.pack import read_pack
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


def _run(out: Path, ortho: Path, heights: Path, *profile: str) -> Result:
    args = [
        "tiles3d",
        "--ortho",
        str(ortho),
        "--heights",
        str(heights),
        "--out",
        str(out),
        *profile,
    ]
    return CliRunner().invoke(cli, args)


def test_tiles3d_game_profile_writes_provenance(tmp_path: Path) -> None:
    """`--profile game` builds, echoes the profile, writes the packed set."""
    ortho, heights = _inputs(tmp_path)
    out = tmp_path / "game"
    result = _run(out, ortho, heights, "--profile", "game")
    assert result.exit_code == 0, result.output
    assert "verified. profile=game." in result.output
    assert (out / "tileset.json").is_file()
    # Packed layout: content lives in tiles.hfp, not loose tiles/.
    assert (out / "tiles.hfp").is_file()
    assert not (out / "tiles").exists()
    dataset_id = read_pack(out / "tiles.hfp").header.dataset_id.hex()
    document = json.loads((out / "provenance.json").read_text())
    assert document["profile"] == "game"
    assert document["quantization"] == {
        "position_bits": 16,
        "uv": "normalized-uint16",
        "scheme": document["quantization"]["scheme"],
    }
    assert document["jpeg"] == {
        "quality": jpeg.JPEG_QUALITY,
        "subsampling": jpeg.JPEG_SUBSAMPLING,
        "progressive": jpeg.JPEG_PROGRESSIVE,
        "optimize": jpeg.JPEG_OPTIMIZE,
        "pillow": jpeg.pillow_version(),
    }
    assert document["encoders"] == {
        "meshoptimizer": meshopt.meshoptimizer_version()
    }
    assert document["pack"]["dataset_id"] == dataset_id
    assert set(document["producer"]) == {"platform", "python"}
    assert quantize.UINT16_MAX.bit_length() == 16


def test_tiles3d_strict_profile_writes_no_provenance(tmp_path: Path) -> None:
    """The default strict profile emits no provenance.json sidecar."""
    ortho, heights = _inputs(tmp_path)
    out = tmp_path / "strict"
    result = _run(out, ortho, heights, "--profile", "strict")
    assert result.exit_code == 0, result.output
    assert "profile=game" not in result.output
    assert not (out / "provenance.json").exists()
    assert sorted(p.name for p in out.iterdir()) == ["tiles", "tileset.json"]


def test_tiles3d_default_profile_writes_no_provenance(tmp_path: Path) -> None:
    """Omitting --profile is strict: no provenance.json, plain summary."""
    ortho, heights = _inputs(tmp_path)
    out = tmp_path / "default"
    result = _run(out, ortho, heights)
    assert result.exit_code == 0, result.output
    assert not (out / "provenance.json").exists()


def test_tiles3d_unknown_profile_is_rejected(tmp_path: Path) -> None:
    """An unknown --profile is the typed error translated to exit 1."""
    ortho, heights = _inputs(tmp_path)
    result = _run(tmp_path / "out", ortho, heights, "--profile", "bogus")
    assert result.exit_code == 1
    assert "unknown tiles3d profile 'bogus'" in result.output


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


def test_tiles3d_heightfield_profile_builds_and_writes_provenance(
    tmp_path: Path,
) -> None:
    """`--profile heightfield` builds .hf + .jpg tiles and provenance.json."""
    ortho, heights = _inputs(tmp_path)
    out = tmp_path / "heightfield"
    result = _run(out, ortho, heights, "--profile", "heightfield")
    assert result.exit_code == 0, result.output
    assert "verified. profile=heightfield." in result.output
    assert (out / "tileset.json").is_file()
    assert (out / "tiles.hfp").is_file()
    assert not (out / "tiles").exists()
    pack = read_pack(out / "tiles.hfp")
    assert pack.header.content_kind == 0
    assert all(entry.texture_size > 0 for entry in pack.entries)
    document = json.loads((out / "provenance.json").read_text())
    assert document["profile"] == "heightfield"
    assert document["quantization"]["height_bits"] == 12
    assert document["pack"]["dataset_id"] == pack.header.dataset_id.hex()


def test_tiles3d_splat_profile_builds_and_writes_provenance(
    tmp_path: Path,
) -> None:
    """`--profile splat` builds .ply gaussian clouds and provenance.json."""
    ortho, heights = _inputs(tmp_path)
    out = tmp_path / "splat"
    result = _run(out, ortho, heights, "--profile", "splat")
    assert result.exit_code == 0, result.output
    assert "verified. profile=splat." in result.output
    assert (out / "tileset.json").is_file()
    assert (out / "tiles.hfp").is_file()
    assert not (out / "tiles").exists()
    pack = read_pack(out / "tiles.hfp")
    assert pack.header.content_kind == 2
    assert all(entry.texture_size == 0 for entry in pack.entries)
    document = json.loads((out / "provenance.json").read_text())
    assert document["profile"] == "splat"
    assert document["pack"]["dataset_id"] == pack.header.dataset_id.hex()
