"""Tests for the tileset.json emission."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from ahn_cli.tiles3d.tileset import (
    tile_entry,
    tileset_document,
    union_region,
    write_tileset,
)

if TYPE_CHECKING:
    from pathlib import Path

_REGION = (0.09, 0.91, 0.10, 0.92, -3.0, 40.0)
_OTHER = (0.08, 0.915, 0.095, 0.93, -1.0, 55.0)


def test_union_region_is_the_envelope() -> None:
    """The union takes mins of west/south/minH and maxes of the rest."""
    assert union_region(_REGION, _OTHER) == (
        0.08,
        0.91,
        0.10,
        0.93,
        -3.0,
        55.0,
    )


def test_root_entry_declares_refine_and_children_inherit() -> None:
    """Only the root carries refine; empty children are omitted."""
    leaf = tile_entry(_REGION, 0.0, "tiles/1-0-0.glb", [], root=False)
    assert leaf == {
        "boundingVolume": {"region": list(_REGION)},
        "content": {"uri": "tiles/1-0-0.glb"},
        "geometricError": 0.0,
    }
    root = tile_entry(_OTHER, 8.0, "tiles/0-0-0.glb", [leaf], root=True)
    assert root["refine"] == "REPLACE"
    assert root["children"] == [leaf]


def test_document_is_version_1_1() -> None:
    """The document wraps the root with asset.version 1.1."""
    root = tile_entry(_REGION, 8.0, "tiles/0-0-0.glb", [], root=True)
    document = tileset_document(root, 16.0)
    assert document["asset"] == {
        "generator": "ahn_cli tiles3d",
        "version": "1.1",
    }
    assert document["geometricError"] == 16.0
    assert document["root"] is root


def test_write_tileset_is_deterministic(tmp_path: Path) -> None:
    """The rendered file is sorted, indented, newline-terminated."""
    root = tile_entry(_REGION, 8.0, "tiles/0-0-0.glb", [], root=True)
    document = tileset_document(root, 16.0)
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    write_tileset(document, first)
    write_tileset(document, second)
    data = first.read_bytes()
    assert data == second.read_bytes()
    assert data.endswith(b"}\n")
    parsed = json.loads(data)
    assert parsed["root"]["content"]["uri"] == "tiles/0-0-0.glb"
