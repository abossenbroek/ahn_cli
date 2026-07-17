"""Tests for the resumable per-tile manifest store."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from ahn_cli.pipeline.errors import PipelineError
from ahn_cli.pipeline.manifest import (
    ManifestEntry,
    TileStore,
    encoded_digest,
)
from ahn_cli.pipeline.model import EncodedBlob, EncodedTile, TileKey

if TYPE_CHECKING:
    from pathlib import Path


def _encoded(
    *, level: int = 0, tx: int = 0, ty: int = 0, data: bytes = b"geo"
) -> EncodedTile:
    return EncodedTile(
        key=TileKey(level=level, tx=tx, ty=ty),
        blobs=(
            EncodedBlob(name="geometry", data=data),
            EncodedBlob(name="texture", data=b"tex"),
        ),
    )


def test_encoded_digest_is_deterministic() -> None:
    """The output hash depends only on blob names and bytes, in order."""
    assert encoded_digest(_encoded()) == encoded_digest(_encoded())


def test_encoded_digest_changes_with_content() -> None:
    """Different blob bytes hash differently."""
    assert encoded_digest(_encoded(data=b"a")) != encoded_digest(
        _encoded(data=b"b")
    )


def test_commit_then_is_done(tmp_path: Path) -> None:
    """A committed tile with a matching input hash reads back as done."""
    store = TileStore(tmp_path)
    enc = _encoded()
    store.write_blobs(enc)
    entry = store.commit(enc.key, "input-1", enc)
    assert isinstance(entry, ManifestEntry)
    assert store.is_done(enc.key, "input-1")


def test_is_done_false_without_marker(tmp_path: Path) -> None:
    """A tile whose blobs exist but marker does not is not done."""
    store = TileStore(tmp_path)
    enc = _encoded()
    store.write_blobs(enc)
    assert not store.is_done(enc.key, "input-1")


def test_is_done_false_on_input_mismatch(tmp_path: Path) -> None:
    """A committed tile with a changed input hash must be reprocessed."""
    store = TileStore(tmp_path)
    enc = _encoded()
    store.write_blobs(enc)
    store.commit(enc.key, "input-1", enc)
    assert not store.is_done(enc.key, "input-2")


def test_is_done_false_on_corrupt_marker(tmp_path: Path) -> None:
    """A truncated/garbage marker is treated as not-done (safe rebuild)."""
    store = TileStore(tmp_path)
    enc = _encoded()
    store.write_blobs(enc)
    store.commit(enc.key, "input-1", enc)
    marker = store.marker_path(enc.key)
    marker.write_text("{ not json", encoding="utf-8")
    assert not store.is_done(enc.key, "input-1")


def test_is_done_false_on_missing_blob(tmp_path: Path) -> None:
    """A committed marker whose blob file vanished is not done."""
    store = TileStore(tmp_path)
    enc = _encoded()
    store.write_blobs(enc)
    store.commit(enc.key, "input-1", enc)
    (store.tile_dir(enc.key) / "geometry").unlink()
    assert not store.is_done(enc.key, "input-1")


def test_is_done_false_on_marker_missing_fields(tmp_path: Path) -> None:
    """A marker missing required fields is rejected, not crashed on."""
    store = TileStore(tmp_path)
    enc = _encoded()
    store.write_blobs(enc)
    store.commit(enc.key, "input-1", enc)
    store.marker_path(enc.key).write_text(
        json.dumps({"input_hash": "input-1"}), encoding="utf-8"
    )
    assert not store.is_done(enc.key, "input-1")


@pytest.mark.parametrize(
    "marker_json",
    [
        "[1, 2, 3]",  # not an object
        '{"input_hash": "i", "output_hash": "o", "blobs": "nope"}',  # blobs not a list
        '{"input_hash": "i", "output_hash": "o", "blobs": [1, 2]}',  # blob not a str
    ],
)
def test_load_entry_rejects_malformed_marker(
    tmp_path: Path, marker_json: str
) -> None:
    """A structurally malformed marker yields ``None`` rather than raising."""
    store = TileStore(tmp_path)
    enc = _encoded()
    store.write_blobs(enc)
    store.marker_path(enc.key).write_text(marker_json, encoding="utf-8")
    assert store.load_entry(enc.key) is None


def test_write_blobs_cleans_prior_partial(tmp_path: Path) -> None:
    """Rewriting a tile removes stale files from a previous partial attempt."""
    store = TileStore(tmp_path)
    enc = _encoded()
    store.write_blobs(enc)
    stale = store.tile_dir(enc.key) / "stale.tmp"
    stale.write_bytes(b"leftover")
    store.write_blobs(enc)
    assert not stale.exists()


def test_write_blobs_rejects_unsafe_name(tmp_path: Path) -> None:
    """A blob name with a path separator cannot escape the tile directory."""
    store = TileStore(tmp_path)
    enc = EncodedTile(
        key=TileKey(level=0, tx=0, ty=0),
        blobs=(EncodedBlob(name="../escape", data=b"x"),),
    )
    with pytest.raises(PipelineError, match="blob name"):
        store.write_blobs(enc)


def test_commit_is_atomic_leaves_no_tmp(tmp_path: Path) -> None:
    """Committing leaves the marker in place and no temp file behind."""
    store = TileStore(tmp_path)
    enc = _encoded()
    store.write_blobs(enc)
    store.commit(enc.key, "input-1", enc)
    leftovers = [
        p for p in store.tile_dir(enc.key).iterdir() if ".tmp" in p.name
    ]
    assert leftovers == []


def test_write_manifest_is_deterministic(tmp_path: Path) -> None:
    """The aggregate manifest is sorted, stable, and round-trips the entries."""
    store = TileStore(tmp_path)
    keys = [TileKey(level=0, tx=1, ty=0), TileKey(level=0, tx=0, ty=0)]
    for key in keys:
        enc = _encoded(tx=key.tx, ty=key.ty)
        store.write_blobs(enc)
        store.commit(key, f"in-{key.tx}", enc)
    path_a = store.write_manifest(keys)
    first = path_a.read_bytes()
    second = store.write_manifest(list(reversed(keys)))
    assert first == second.read_bytes()
    doc = json.loads(first)
    assert [t["tx"] for t in doc["tiles"]] == [0, 1]


def test_load_entry_round_trips(tmp_path: Path) -> None:
    """A committed entry loads back with its hashes and blob names."""
    store = TileStore(tmp_path)
    enc = _encoded()
    store.write_blobs(enc)
    committed = store.commit(enc.key, "input-1", enc)
    loaded = store.load_entry(enc.key)
    assert loaded == committed
    assert loaded is not None
    assert loaded.blobs == ("geometry", "texture")


def test_load_entry_missing_returns_none(tmp_path: Path) -> None:
    """Loading an uncommitted tile yields ``None``."""
    store = TileStore(tmp_path)
    assert store.load_entry(TileKey(level=0, tx=9, ty=9)) is None


def test_write_manifest_requires_committed_tiles(tmp_path: Path) -> None:
    """Building the manifest over an uncommitted key is a hard error."""
    store = TileStore(tmp_path)
    with pytest.raises(PipelineError, match="not committed"):
        store.write_manifest([TileKey(level=0, tx=0, ty=0)])
