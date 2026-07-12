"""Tests for the tiles3d build orchestrator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NoReturn, cast

import pytest

import ahn_cli.tiles3d.build as build_module
import ahn_cli.tiles3d.encoders as encoders_module
from ahn_cli.tiles3d.build import build_tiles3d
from ahn_cli.tiles3d.errors import Tiles3dError
from tests.tiles3d.conftest import (
    grid_for_ortho,
    make_ortho,
    synth_rgb,
    write_exr,
)


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

    monkeypatch.setattr(encoders_module, "build_glb", explode)
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
    assert sorted(p.name for p in out.iterdir()) == [
        "tiles",
        "tileset.json",
    ]


def test_failed_rebuild_restores_the_previous_deliverable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rebuild the verifier rejects puts the old build back, intact."""
    ortho, heights = _make_inputs(tmp_path, 20, 14)
    out = tmp_path / "out"
    build_tiles3d(ortho, heights, out, tile_pixels=8)
    before = {
        p.relative_to(out): p.read_bytes()
        for p in out.rglob("*")
        if p.is_file()
    }

    def compact_write(document: dict[str, object], path: Path) -> None:
        path.write_text(json.dumps(document, sort_keys=True))

    monkeypatch.setattr(build_module, "write_tileset", compact_write)
    with pytest.raises(Tiles3dError, match="byte-equal"):
        build_tiles3d(ortho, heights, out, tile_pixels=8)
    after = {
        p.relative_to(out): p.read_bytes()
        for p in out.rglob("*")
        if p.is_file()
    }
    assert after == before


def _build_then_simulate_hard_kill_after_hold(
    tmp_path: Path,
) -> tuple[Path, Path, Path, dict[Path, bytes]]:
    """Build once, then fake a SIGKILL between hold and restore.

    Returns (ortho, heights, out, good) where ``good`` snapshots the
    verified deliverable now stranded in the backup directory while
    ``out`` proper holds the killed run's unverified partial write.
    """
    ortho, heights = _make_inputs(tmp_path, 6, 6)
    out = tmp_path / "out"
    build_tiles3d(ortho, heights, out)
    good = {
        p.relative_to(out): p.read_bytes()
        for p in out.rglob("*")
        if p.is_file()
    }
    backup = out / build_module.BACKUP_SUBDIR
    backup.mkdir()
    (out / "tiles").rename(backup / "tiles")
    (out / "tileset.json").rename(backup / "tileset.json")
    (out / "tiles").mkdir()
    (out / "tiles" / "junk.glb").write_bytes(b"partial")
    (out / "tileset.json").write_text("{}")
    return ortho, heights, out, good


def test_hard_killed_rebuild_recovers_and_rebuilds(tmp_path: Path) -> None:
    """The deliverable a hard kill stranded in backup is put back first."""
    ortho, heights, out, good = _build_then_simulate_hard_kill_after_hold(
        tmp_path
    )
    result = build_tiles3d(ortho, heights, out)
    assert result.tile_count == 1
    after = {
        p.relative_to(out): p.read_bytes()
        for p in out.rglob("*")
        if p.is_file()
    }
    assert after == good


def test_hard_kill_then_failed_rebuild_still_restores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even a rebuild that fails after a hard kill restores the old build."""
    ortho, heights, out, good = _build_then_simulate_hard_kill_after_hold(
        tmp_path
    )

    def compact_write(document: dict[str, object], path: Path) -> None:
        path.write_text(json.dumps(document, sort_keys=True))

    monkeypatch.setattr(build_module, "write_tileset", compact_write)
    with pytest.raises(Tiles3dError, match="byte-equal"):
        build_tiles3d(ortho, heights, out)
    after = {
        p.relative_to(out): p.read_bytes()
        for p in out.rglob("*")
        if p.is_file()
    }
    assert after == good


def test_mid_hold_kill_state_recovers(tmp_path: Path) -> None:
    """A kill between the two hold renames leaves a recoverable union."""
    ortho, heights = _make_inputs(tmp_path, 6, 6)
    out = tmp_path / "out"
    build_tiles3d(ortho, heights, out)
    backup = out / build_module.BACKUP_SUBDIR
    backup.mkdir()
    (out / "tiles").rename(backup / "tiles")
    result = build_tiles3d(ortho, heights, out)
    assert result.tile_count == 1
    assert sorted(p.name for p in out.iterdir()) == [
        "tiles",
        "tileset.json",
    ]


def test_kill_right_after_backup_mkdir_recovers(tmp_path: Path) -> None:
    """An empty backup directory from a kill after mkdir is dissolved."""
    ortho, heights = _make_inputs(tmp_path, 6, 6)
    out = tmp_path / "out"
    build_tiles3d(ortho, heights, out)
    (out / build_module.BACKUP_SUBDIR).mkdir()
    result = build_tiles3d(ortho, heights, out)
    assert result.tile_count == 1
    assert sorted(p.name for p in out.iterdir()) == [
        "tiles",
        "tileset.json",
    ]


def test_kill_during_drop_completes_on_the_next_run(tmp_path: Path) -> None:
    """Marker present: the in-place build is verified, backup is dropped."""
    ortho, heights = _make_inputs(tmp_path, 6, 6)
    out = tmp_path / "out"
    build_tiles3d(ortho, heights, out)
    (out / build_module.ACCEPT_MARKER).touch()
    leftover = out / build_module.BACKUP_SUBDIR
    (leftover / "tiles").mkdir(parents=True)
    (leftover / "tiles" / "0-0-0.glb").write_bytes(b"stale")
    result = build_tiles3d(ortho, heights, out)
    assert result.tile_count == 1
    assert not leftover.exists()
    assert sorted(p.name for p in out.iterdir()) == [
        "tiles",
        "tileset.json",
    ]


def test_orphan_accept_marker_is_cleared(tmp_path: Path) -> None:
    """A marker with no backup (kill between rmtree and unlink) is removed."""
    ortho, heights = _make_inputs(tmp_path, 6, 6)
    out = tmp_path / "out"
    build_tiles3d(ortho, heights, out)
    (out / build_module.ACCEPT_MARKER).touch()
    build_tiles3d(ortho, heights, out)
    assert sorted(p.name for p in out.iterdir()) == [
        "tiles",
        "tileset.json",
    ]


def _snapshot(out: Path) -> dict[Path, bytes]:
    return {
        p.relative_to(out): p.read_bytes()
        for p in out.rglob("*")
        if p.is_file()
    }


def _flaky_write(monkeypatch: pytest.MonkeyPatch, failing_call: int) -> None:
    """Patch Path.write_bytes to die on call N, leaving a stray file."""
    calls = {"count": 0}
    real_write = Path.write_bytes

    def flaky(self: Path, data: bytes) -> int:
        calls["count"] += 1
        if calls["count"] == failing_call:
            self.touch()
            msg = "disk full"
            raise OSError(msg)
        return real_write(self, data)

    monkeypatch.setattr(Path, "write_bytes", flaky)


def test_partial_write_failure_restores_the_deliverable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A write dying mid-loop restores the old build, typed and whole."""
    ortho, heights = _make_inputs(tmp_path, 20, 14)
    out = tmp_path / "out"
    build_tiles3d(ortho, heights, out, tile_pixels=8)
    good = _snapshot(out)
    _flaky_write(monkeypatch, failing_call=3)
    with pytest.raises(Tiles3dError, match="not writable"):
        build_tiles3d(ortho, heights, out, tile_pixels=8)
    assert _snapshot(out) == good


def test_first_build_partial_write_leaves_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stray file from a failed write is removed with the tiles dir."""
    ortho, heights = _make_inputs(tmp_path, 20, 14)
    out = tmp_path / "out"
    _flaky_write(monkeypatch, failing_call=3)
    with pytest.raises(Tiles3dError, match="not writable"):
        build_tiles3d(ortho, heights, out, tile_pixels=8)
    assert list(out.iterdir()) == []


def _dying_tileset_write(document: dict[str, object], path: Path) -> None:
    """Stand in for write_tileset, leaving a partial file behind."""
    del document
    path.write_text("{ partial")
    msg = "disk full"
    raise OSError(msg)


def test_partial_tileset_write_leaves_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tileset write dying midway leaves no partial file behind."""
    ortho, heights = _make_inputs(tmp_path, 6, 6)
    out = tmp_path / "out"
    monkeypatch.setattr(build_module, "write_tileset", _dying_tileset_write)
    with pytest.raises(Tiles3dError, match="not writable"):
        build_tiles3d(ortho, heights, out)
    assert list(out.iterdir()) == []


def test_partial_tileset_write_restores_the_deliverable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tileset write dying midway on a rebuild restores the old build."""
    ortho, heights = _make_inputs(tmp_path, 6, 6)
    out = tmp_path / "out"
    build_tiles3d(ortho, heights, out)
    good = _snapshot(out)
    monkeypatch.setattr(build_module, "write_tileset", _dying_tileset_write)
    with pytest.raises(Tiles3dError, match="not writable"):
        build_tiles3d(ortho, heights, out)
    assert _snapshot(out) == good


def test_failed_recovery_leaves_state_for_the_next_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A recovery that cannot drop the backup never touches the build."""
    ortho, heights = _make_inputs(tmp_path, 6, 6)
    out = tmp_path / "out"
    build_tiles3d(ortho, heights, out)
    good = _snapshot(out)
    (out / build_module.ACCEPT_MARKER).touch()
    leftover = out / build_module.BACKUP_SUBDIR
    (leftover / "tiles").mkdir(parents=True)
    (leftover / "tiles" / "0-0-0.glb").write_bytes(b"stale")

    def refuse(*_args: object, **_kwargs: object) -> NoReturn:
        msg = "locked"
        raise OSError(msg)

    monkeypatch.setattr(build_module.shutil, "rmtree", refuse)
    with pytest.raises(Tiles3dError, match="not writable"):
        build_tiles3d(ortho, heights, out)
    assert (out / build_module.ACCEPT_MARKER).is_file()
    tool_paths = {build_module.BACKUP_SUBDIR, build_module.ACCEPT_MARKER}
    after = {
        rel: data
        for rel, data in _snapshot(out).items()
        if rel.parts[0] not in tool_paths
    }
    assert after == good
    monkeypatch.undo()
    result = build_tiles3d(ortho, heights, out)
    assert result.tile_count == 1
    assert sorted(p.name for p in out.iterdir()) == [
        "tiles",
        "tileset.json",
    ]


def _flaky_rename(monkeypatch: pytest.MonkeyPatch, failing_call: int) -> None:
    """Patch Path.rename to die on call N of the hold step."""
    calls = {"count": 0}
    real_rename = Path.rename

    def flaky(self: Path, target: Path) -> Path:
        calls["count"] += 1
        if calls["count"] == failing_call:
            msg = "permission denied"
            raise OSError(msg)
        return real_rename(self, target)

    monkeypatch.setattr(Path, "rename", flaky)


def test_failed_hold_preserves_the_previous_deliverable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hold step dying before any move leaves the old build in place."""
    ortho, heights = _make_inputs(tmp_path, 20, 14)
    out = tmp_path / "out"
    build_tiles3d(ortho, heights, out, tile_pixels=8)
    good = _snapshot(out)
    _flaky_rename(monkeypatch, failing_call=1)
    with pytest.raises(Tiles3dError, match="not writable"):
        build_tiles3d(ortho, heights, out, tile_pixels=8)
    assert _snapshot(out) == good
    assert not (out / build_module.BACKUP_SUBDIR).exists()


def test_hold_failing_midway_restores_the_moved_half(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hold step dying between its two moves is fully rolled back."""
    ortho, heights = _make_inputs(tmp_path, 20, 14)
    out = tmp_path / "out"
    build_tiles3d(ortho, heights, out, tile_pixels=8)
    good = _snapshot(out)
    _flaky_rename(monkeypatch, failing_call=2)
    with pytest.raises(Tiles3dError, match="not writable"):
        build_tiles3d(ortho, heights, out, tile_pixels=8)
    assert _snapshot(out) == good
    assert not (out / build_module.BACKUP_SUBDIR).exists()


def test_failed_restore_is_a_typed_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even a cleanup failure surfaces as the context's typed error."""
    ortho, heights = _make_inputs(tmp_path, 6, 6)
    out = tmp_path / "out"
    build_tiles3d(ortho, heights, out)

    def compact_write(document: dict[str, object], path: Path) -> None:
        path.write_text(json.dumps(document, sort_keys=True))

    def refuse(*_args: object, **_kwargs: object) -> NoReturn:
        msg = "locked"
        raise OSError(msg)

    monkeypatch.setattr(build_module, "write_tileset", compact_write)
    monkeypatch.setattr(build_module.shutil, "rmtree", refuse)
    with pytest.raises(Tiles3dError, match="could not restore"):
        build_tiles3d(ortho, heights, out)


def test_failed_backup_drop_is_typed_and_recoverable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A drop that fails leaves the marker; the next run completes it."""
    ortho, heights = _make_inputs(tmp_path, 6, 6)
    out = tmp_path / "out"
    build_tiles3d(ortho, heights, out)

    def refuse(*_args: object, **_kwargs: object) -> NoReturn:
        msg = "locked"
        raise OSError(msg)

    monkeypatch.setattr(build_module.shutil, "rmtree", refuse)
    with pytest.raises(Tiles3dError, match="could not drop"):
        build_tiles3d(ortho, heights, out)
    assert (out / build_module.ACCEPT_MARKER).is_file()
    assert (out / build_module.BACKUP_SUBDIR).is_dir()
    monkeypatch.undo()
    result = build_tiles3d(ortho, heights, out)
    assert result.tile_count == 1
    assert sorted(p.name for p in out.iterdir()) == [
        "tiles",
        "tileset.json",
    ]


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
