"""Tests for the tiles3d build orchestrator."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

import pytest

import ahn_cli.tiles3d.build as build_module
import ahn_cli.tiles3d.emit as emit_module
from ahn_cli.tiles3d.build import build_tiles3d
from ahn_cli.tiles3d.errors import Tiles3dError
from tests.tiles3d.conftest import (
    grid_for_ortho,
    make_ortho,
    synth_rgb,
    write_exr,
)

if TYPE_CHECKING:
    from pathlib import Path


def _make_inputs(
    tmp_path: Path, width: int, height: int, seed: int = 11
) -> tuple[Path, Path]:
    rgb = synth_rgb(width, height, seed)
    ortho = make_ortho(tmp_path / "ortho.tif", rgb)
    heights = write_exr(tmp_path / "r.exr", grid_for_ortho(rgb))
    return ortho, heights


def _tileset(out: Path) -> dict[str, Any]:
    return cast(
        "dict[str, Any]",
        json.loads((out / "tileset.json").read_text()),
    )


def test_single_tile_build(tmp_path: Path) -> None:
    """A 6x6 grid builds one leaf tile and a version-1.1 tileset."""
    ortho, heights = _make_inputs(tmp_path, 6, 6)
    out = tmp_path / "out"
    result = build_tiles3d(ortho, heights, out)
    assert result.tileset_path == out / "tileset.json"
    assert result.tile_count == 1
    assert result.levels == 0
    assert result.vertices == 36
    assert result.triangles == 50
    document = _tileset(out)
    assert document["asset"]["version"] == "1.1"
    root = document["root"]
    assert root["refine"] == "REPLACE"
    assert root["geometricError"] == 0.0
    assert document["geometricError"] == 0.5 * 4.0
    assert (out / "tiles" / "0-0-0.glb").is_file()


def test_multi_level_build_writes_exactly_the_referenced_set(
    tmp_path: Path,
) -> None:
    """Every quadtree tile lands on disk and is referenced once."""
    ortho, heights = _make_inputs(tmp_path, 20, 14)
    out = tmp_path / "out"
    result = build_tiles3d(ortho, heights, out, tile_pixels=8)
    assert result.levels == 2
    assert result.tile_count == 21
    document = _tileset(out)
    referenced: list[str] = []

    def walk(entry: dict[str, Any]) -> None:
        referenced.append(entry["content"]["uri"])
        for child in entry.get("children", []):
            walk(child)

    walk(document["root"])
    assert len(referenced) == 21
    assert len(set(referenced)) == 21
    on_disk = {f"tiles/{p.name}" for p in (out / "tiles").iterdir()}
    assert on_disk == set(referenced)
    assert document["root"]["geometricError"] == 0.5 * 4 * 4.0
    assert document["geometricError"] == 2 * 0.5 * 4 * 4.0


def test_progress_reports_every_tile(tmp_path: Path) -> None:
    """The callback runs once per tile with a growing counter."""
    ortho, heights = _make_inputs(tmp_path, 20, 14)
    calls: list[tuple[int, int]] = []
    build_tiles3d(
        ortho,
        heights,
        tmp_path / "out",
        tile_pixels=8,
        progress=lambda done, total: calls.append((done, total)),
    )
    assert calls == [(i, 21) for i in range(1, 22)]


def test_build_is_deterministic(tmp_path: Path) -> None:
    """Two builds produce byte-identical trees."""
    ortho, heights = _make_inputs(tmp_path, 20, 14)
    first = tmp_path / "a"
    second = tmp_path / "b"
    build_tiles3d(ortho, heights, first, tile_pixels=8)
    build_tiles3d(ortho, heights, second, tile_pixels=8)
    first_files = sorted(p for p in first.rglob("*") if p.is_file())
    second_files = sorted(p for p in second.rglob("*") if p.is_file())
    assert [p.relative_to(first) for p in first_files] == [
        p.relative_to(second) for p in second_files
    ]
    for left, right in zip(first_files, second_files, strict=True):
        assert left.read_bytes() == right.read_bytes()


def test_failed_build_leaves_nothing_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure during emission writes nothing at all."""
    ortho, heights = _make_inputs(tmp_path, 20, 14)
    out = tmp_path / "out"
    calls = {"count": 0}

    def explode(*_args: object, **_kwargs: object) -> bytes:
        calls["count"] += 1
        if calls["count"] == 2:
            msg = "injected failure"
            raise Tiles3dError(msg)
        return b"unused"

    monkeypatch.setattr(emit_module, "build_glb", explode)
    with pytest.raises(Tiles3dError, match="injected failure"):
        build_tiles3d(ortho, heights, out, tile_pixels=8)
    assert not (out / "tileset.json").exists()
    assert not (out / "tiles").exists()


def test_rejected_verification_removes_every_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A build the verifier rejects removes everything it wrote.

    The tileset writer is patched to emit a compact (but semantically
    identical) rendering: the byte-identity check refuses it, and the
    already-written glbs plus the bad tileset must all be removed.
    """
    ortho, heights = _make_inputs(tmp_path, 20, 14)
    out = tmp_path / "out"

    def compact_write(document: dict[str, object], path: Path) -> None:
        path.write_text(json.dumps(document, sort_keys=True))

    monkeypatch.setattr(build_module, "write_tileset", compact_write)
    with pytest.raises(Tiles3dError, match="byte-equal"):
        build_tiles3d(ortho, heights, out, tile_pixels=8)
    assert list(out.iterdir()) == []


def test_rebuild_replaces_stale_artifacts(tmp_path: Path) -> None:
    """Rebuilding into a used directory leaves only the new build."""
    first_dir = tmp_path / "a"
    second_dir = tmp_path / "b"
    first_dir.mkdir()
    second_dir.mkdir()
    ortho_a, heights_a = _make_inputs(first_dir, 20, 14)
    out = tmp_path / "out"
    build_tiles3d(ortho_a, heights_a, out, tile_pixels=8)
    ortho_b, heights_b = _make_inputs(second_dir, 6, 6, seed=12)
    result = build_tiles3d(ortho_b, heights_b, out)
    assert result.tile_count == 1
    names = sorted(p.name for p in (out / "tiles").iterdir())
    assert names == ["0-0-0.glb"]


def test_gate_failure_preserves_the_previous_build(tmp_path: Path) -> None:
    """An input-gate failure never touches an existing deliverable."""
    ortho, heights = _make_inputs(tmp_path, 6, 6)
    out = tmp_path / "out"
    build_tiles3d(ortho, heights, out)
    before = {
        p.relative_to(out): p.read_bytes()
        for p in out.rglob("*")
        if p.is_file()
    }
    other = synth_rgb(6, 6, seed=99)
    bad_heights = write_exr(tmp_path / "bad.exr", grid_for_ortho(other))
    with pytest.raises(Tiles3dError, match="plane"):
        build_tiles3d(ortho, bad_heights, out)
    after = {
        p.relative_to(out): p.read_bytes()
        for p in out.rglob("*")
        if p.is_file()
    }
    assert after == before


def test_unwritable_output_is_a_typed_error(tmp_path: Path) -> None:
    """An output path that cannot be a directory is refused."""
    ortho, heights = _make_inputs(tmp_path, 6, 6)
    blocker = tmp_path / "blocker"
    blocker.write_text("file, not dir")
    with pytest.raises(Tiles3dError, match="not writable"):
        build_tiles3d(ortho, heights, blocker)
