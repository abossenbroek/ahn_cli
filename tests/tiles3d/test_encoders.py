"""Tests for the strict- and game-profile tile encoders."""

from __future__ import annotations

import json
import struct
from typing import Any, cast

import numpy as np

from ahn_cli.tiles3d.encoders import GameEncoder, StrictEncoder
from ahn_cli.tiles3d.geodesy import Geodesy
from ahn_cli.tiles3d.gltf import build_glb
from ahn_cli.tiles3d.mesh import build_tile_mesh
from ahn_cli.tiles3d.payload import EncodedTile, TileEncoder, TilePayload
from ahn_cli.tiles3d.png import encode_png
from ahn_cli.tiles3d.quadtree import geometric_error, plan_quadtree
from tests.tiles3d.conftest import make_terrain


def _glb_json(content: bytes) -> dict[str, Any]:
    """Parse the glTF JSON chunk out of an ``EncodedTile``'s glb bytes."""
    json_length = struct.unpack("<I", content[12:16])[0]
    return cast("dict[str, Any]", json.loads(content[20 : 20 + json_length]))


def _payload(width: int = 6, height: int = 5) -> TilePayload:
    terrain = make_terrain(width, height)
    tile = plan_quadtree(width, height).root
    mesh = build_tile_mesh(terrain, tile, Geodesy())
    grid = np.ix_(mesh.rows, mesh.cols)
    return TilePayload(
        level=tile.level,
        tx=tile.tx,
        ty=tile.ty,
        stride=tile.stride,
        geometric_error=geometric_error(tile.stride, 0.5),
        mesh=mesh,
        x=terrain.x[grid],
        y=terrain.y[grid],
        z=terrain.z[grid],
        rgb=terrain.rgb[grid],
    )


def test_strict_encode_reproduces_the_embedded_glb_path() -> None:
    """The encoder equals ``build_glb(mesh, encode_png(rgb))`` exactly."""
    payload = _payload()
    encoded = StrictEncoder().encode(payload)
    expected = build_glb(payload.mesh, encode_png(payload.rgb))
    assert encoded.content == expected
    assert encoded.content_name == "0-0-0.glb"
    assert encoded.texture is None
    assert encoded.texture_name is None


def test_strict_content_name_tracks_the_tile_coordinates() -> None:
    """The content name is ``<level>-<tx>-<ty>.glb`` for any tile."""
    terrain = make_terrain(20, 14)
    tree = plan_quadtree(20, 14, 8)
    leaf = tree.root.children[0].children[0]
    mesh = build_tile_mesh(terrain, leaf, Geodesy())
    grid = np.ix_(mesh.rows, mesh.cols)
    payload = TilePayload(
        level=leaf.level,
        tx=leaf.tx,
        ty=leaf.ty,
        stride=leaf.stride,
        geometric_error=geometric_error(leaf.stride, 0.5),
        mesh=mesh,
        x=terrain.x[grid],
        y=terrain.y[grid],
        z=terrain.z[grid],
        rgb=terrain.rgb[grid],
    )
    encoded = StrictEncoder().encode(payload)
    assert encoded.content_name == f"{leaf.level}-{leaf.tx}-{leaf.ty}.glb"


def test_strict_encode_is_deterministic() -> None:
    """Encoding the same payload twice yields identical bytes."""
    payload = _payload()
    assert StrictEncoder().encode(payload) == StrictEncoder().encode(payload)


def test_strict_encoder_satisfies_the_protocol() -> None:
    """StrictEncoder is usable through the ``TileEncoder`` seam."""
    encoder: TileEncoder = StrictEncoder()
    assert encoder.encode(_payload()).content_name.endswith(".glb")


def test_game_encode_returns_an_embedded_named_glb() -> None:
    """The game encoder names the glb and embeds its texture."""
    payload = _payload()
    encoded = GameEncoder().encode(payload)
    assert isinstance(encoded, EncodedTile)
    assert encoded.content_name == "0-0-0.glb"
    assert encoded.texture is None
    assert encoded.texture_name is None


def test_game_content_name_tracks_the_tile_coordinates() -> None:
    """The game content name is ``<level>-<tx>-<ty>.glb`` for any tile."""
    terrain = make_terrain(20, 14)
    tree = plan_quadtree(20, 14, 8)
    leaf = tree.root.children[0].children[0]
    mesh = build_tile_mesh(terrain, leaf, Geodesy())
    grid = np.ix_(mesh.rows, mesh.cols)
    payload = TilePayload(
        level=leaf.level,
        tx=leaf.tx,
        ty=leaf.ty,
        stride=leaf.stride,
        geometric_error=geometric_error(leaf.stride, 0.5),
        mesh=mesh,
        x=terrain.x[grid],
        y=terrain.y[grid],
        z=terrain.z[grid],
        rgb=terrain.rgb[grid],
    )
    encoded = GameEncoder().encode(payload)
    assert encoded.content_name == f"{leaf.level}-{leaf.tx}-{leaf.ty}.glb"


def test_game_encode_is_deterministic() -> None:
    """Encoding the same payload twice yields identical bytes."""
    payload = _payload()
    assert GameEncoder().encode(payload) == GameEncoder().encode(payload)


def test_game_encoder_satisfies_the_protocol() -> None:
    """GameEncoder is usable through the ``TileEncoder`` seam."""
    encoder: TileEncoder = GameEncoder()
    assert encoder.encode(_payload()).content_name.endswith(".glb")


def test_extension_declarations_are_only_under_the_game_profile() -> None:
    """The glTF extension declarations mark game glbs and not strict ones."""
    payload = _payload()
    strict = _glb_json(StrictEncoder().encode(payload).content)
    game = _glb_json(GameEncoder().encode(payload).content)
    assert "extensionsUsed" not in strict
    assert "extensionsRequired" not in strict
    expected = ["EXT_meshopt_compression", "KHR_mesh_quantization"]
    assert game["extensionsUsed"] == expected
    assert game["extensionsRequired"] == expected
