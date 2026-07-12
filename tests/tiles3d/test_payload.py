"""Tests for the tile payload / encoded-tile value objects."""

from __future__ import annotations

import numpy as np

from ahn_cli.tiles3d.geodesy import Geodesy
from ahn_cli.tiles3d.mesh import build_tile_mesh
from ahn_cli.tiles3d.payload import EncodedTile, TilePayload
from ahn_cli.tiles3d.quadtree import geometric_error, plan_quadtree
from tests.tiles3d.conftest import make_terrain


def _payload() -> TilePayload:
    terrain = make_terrain(6, 5)
    tile = plan_quadtree(6, 5).root
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


def test_payload_carries_the_sampled_planes_and_placement() -> None:
    """A payload exposes the mesh, sampled planes and tile metadata."""
    payload = _payload()
    assert payload.level == 0
    assert payload.tx == 0
    assert payload.ty == 0
    assert payload.stride == 1
    assert payload.geometric_error == 0.0
    assert payload.rgb.shape == (5, 6, 3)
    assert payload.x.shape == (5, 6)
    assert payload.y.shape == (5, 6)
    assert payload.z.shape == (5, 6)
    assert payload.mesh.positions.shape[0] == 30


def test_payload_compares_by_identity() -> None:
    """``eq=False``: two distinct payloads never compare equal."""
    assert _payload() != _payload()


def test_encoded_tile_defaults_to_an_embedded_texture() -> None:
    """Omitting the texture marks it embedded (both fields ``None``)."""
    encoded = EncodedTile(content=b"glb", content_name="0-0-0.glb")
    assert encoded.texture is None
    assert encoded.texture_name is None


def test_encoded_tile_is_a_value_object() -> None:
    """Equal fields compare equal; a separate texture is carried."""
    first = EncodedTile(b"c", "t.hf", texture=b"png", texture_name="t.png")
    second = EncodedTile(b"c", "t.hf", texture=b"png", texture_name="t.png")
    assert first == second
    assert first.texture == b"png"
    assert first.texture_name == "t.png"
