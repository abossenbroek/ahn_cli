"""Tests for the cross-tile tiles3d assembler (regions + strict/packed write)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

import pytest

from ahn_cli.pipeline.assemble import (
    REGION_BLOB_NAME,
    assemble_tiles3d,
    region_blob_bytes,
)
from ahn_cli.pipeline.errors import PipelineError
from ahn_cli.pipeline.manifest import TileStore
from ahn_cli.pipeline.model import TileKey
from ahn_cli.tiles3d.profile import Profile
from ahn_cli.tiles3d.quadtree import plan_quadtree

if TYPE_CHECKING:
    from pathlib import Path

    from ahn_cli.tiles3d.mesh import Region
    from ahn_cli.tiles3d.quadtree import TilePlan


def _seed_node(store: TileStore, node: TilePlan, region: Region) -> None:
    """Write one node's geometry + region.json blobs into the store."""
    key = TileKey(level=node.level, tx=node.tx, ty=node.ty)
    tile_dir = store.tile_dir(key)
    tile_dir.mkdir(parents=True, exist_ok=True)
    (tile_dir / "geometry").write_bytes(
        f"glb-{node.level}-{node.tx}-{node.ty}".encode()
    )
    (tile_dir / REGION_BLOB_NAME).write_bytes(region_blob_bytes(region))


def _region(offset: float) -> Region:
    """Return a distinct 6-tuple region shifted by ``offset`` (test data)."""
    return (
        0.0 + offset,
        0.1 + offset,
        0.2 + offset,
        0.3 + offset,
        offset,
        1.0 + offset,
    )


def test_region_blob_round_trips() -> None:
    """The region blob is deterministic sorted-key JSON of the 6-tuple."""
    region = _region(0.0)
    data = json.loads(region_blob_bytes(region).decode())
    assert data == {"region": list(region)}


def test_missing_region_blob_is_an_error(tmp_path: Path) -> None:
    """A committed tile without its region blob fails assembly clearly."""
    store = TileStore(tmp_path / "store")
    tree = plan_quadtree(4, 4, tile_pixels=256)  # single root
    key = TileKey(level=0, tx=0, ty=0)
    store.tile_dir(key).mkdir(parents=True)
    (store.tile_dir(key) / "geometry").write_bytes(b"glb")
    with pytest.raises(PipelineError, match="missing its region blob"):
        assemble_tiles3d(
            store,
            tree,
            pixel_size_m=0.5,
            out_dir=tmp_path / "out",
            profile=Profile.STRICT,
        )


def test_strict_multi_tile_unions_child_regions(tmp_path: Path) -> None:
    """A parent's tileset region is the union of its own and its children's."""
    store = TileStore(tmp_path / "store")
    tree = plan_quadtree(8, 6, tile_pixels=4)  # one level of subdivision
    assert tree.levels == 1
    regions: dict[TileKey, Region] = {}

    def _walk(node: TilePlan, offset: float) -> float:
        key = TileKey(level=node.level, tx=node.tx, ty=node.ty)
        region = _region(offset)
        regions[key] = region
        _seed_node(store, node, region)
        nxt = offset + 1.0
        for child in node.children:
            nxt = _walk(child, nxt)
        return nxt

    _walk(tree.root, 0.0)

    out = tmp_path / "out"
    assemble_tiles3d(
        store, tree, pixel_size_m=0.5, out_dir=out, profile=Profile.STRICT
    )
    document = json.loads((out / "tileset.json").read_text())
    root = document["root"]
    assert "children" in root
    # Every child's own region is contained in the root's tileset region.
    root_region = cast("list[float]", root["boundingVolume"]["region"])
    for child in root["children"]:
        child_region = cast("list[float]", child["boundingVolume"]["region"])
        assert root_region[0] <= child_region[0]  # west
        assert root_region[2] >= child_region[2]  # east
        assert root_region[5] >= child_region[5]  # max height
    # Every referenced glb was written from the persisted geometry blobs.
    for node_dir in (out / "tiles").iterdir():
        assert node_dir.is_file()
        assert node_dir.read_bytes().startswith(b"glb-")
