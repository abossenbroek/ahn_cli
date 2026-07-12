"""Tests for the glb container writer."""

from __future__ import annotations

import json
import struct
from typing import Any, cast

import numpy as np

from ahn_cli.tiles3d.geodesy import Geodesy
from ahn_cli.tiles3d.gltf import build_glb
from ahn_cli.tiles3d.mesh import TileMesh, build_tile_mesh
from ahn_cli.tiles3d.png import encode_png
from ahn_cli.tiles3d.quadtree import plan_quadtree
from tests.tiles3d.conftest import make_terrain


def _tile_mesh() -> TileMesh:
    terrain = make_terrain(6, 5)
    return build_tile_mesh(terrain, plan_quadtree(6, 5).root, Geodesy())


def _png() -> bytes:
    return encode_png(make_terrain(6, 5).rgb)


def _parse(glb: bytes) -> tuple[dict[str, Any], bytes]:
    magic, version, length = struct.unpack("<III", glb[:12])
    assert magic == 0x46546C67
    assert version == 2
    assert length == len(glb)
    json_len, json_kind = struct.unpack("<II", glb[12:20])
    assert json_kind == 0x4E4F534A
    assert json_len % 4 == 0
    document = cast("dict[str, Any]", json.loads(glb[20 : 20 + json_len]))
    bin_len, bin_kind = struct.unpack(
        "<II", glb[20 + json_len : 28 + json_len]
    )
    assert bin_kind == 0x004E4942
    binary = glb[28 + json_len :]
    assert len(binary) == bin_len
    assert bin_len % 4 == 0
    return document, binary


def test_container_layout_and_payloads() -> None:
    """The BIN chunk carries positions, uvs, indices and PNG verbatim."""
    mesh = _tile_mesh()
    png = _png()
    document, binary = _parse(build_glb(mesh, png))
    views = document["bufferViews"]
    slices = [
        binary[v["byteOffset"] : v["byteOffset"] + v["byteLength"]]
        for v in views
    ]
    assert slices[0] == mesh.positions.tobytes()
    assert slices[1] == mesh.uvs.tobytes()
    assert slices[2] == mesh.indices.tobytes()
    assert slices[3] == png
    assert document["buffers"][0]["byteLength"] == len(binary)
    for view in views:
        assert view["byteOffset"] % 4 == 0


def test_document_structure() -> None:
    """The JSON document carries the fixed single-primitive scene."""
    mesh = _tile_mesh()
    document, _ = _parse(build_glb(mesh, _png()))
    assert document["asset"] == {
        "generator": "ahn_cli tiles3d",
        "version": "2.0",
    }
    accessors = document["accessors"]
    assert accessors[0]["type"] == "VEC3"
    assert accessors[0]["count"] == mesh.positions.shape[0]
    assert accessors[1]["type"] == "VEC2"
    assert accessors[2]["componentType"] == 5125
    assert accessors[2]["count"] == mesh.indices.shape[0]
    primitive = document["meshes"][0]["primitives"][0]
    assert primitive["attributes"] == {"POSITION": 0, "TEXCOORD_0": 1}
    assert primitive["mode"] == 4
    assert document["images"][0]["mimeType"] == "image/png"
    material = document["materials"][0]
    assert material["doubleSided"] is True
    assert material["pbrMetallicRoughness"]["baseColorTexture"]["index"] == 0
    assert document["nodes"][0]["translation"] == list(mesh.center)
    assert document["scenes"][0]["nodes"] == [0]


def test_position_extremes_are_exact() -> None:
    """Accessor min/max equal the float32 data extremes bit for bit."""
    mesh = _tile_mesh()
    document, _ = _parse(build_glb(mesh, _png()))
    accessor = document["accessors"][0]
    for column in range(3):
        column_values = mesh.positions[:, column]
        assert accessor["min"][column] == float(column_values.min())
        assert accessor["max"][column] == float(column_values.max())


def test_glb_is_deterministic() -> None:
    """Two builds of the same tile are byte-identical."""
    assert build_glb(_tile_mesh(), _png()) == build_glb(_tile_mesh(), _png())


def test_uv_bytes_survive_the_round_trip() -> None:
    """UVs read back from the BIN chunk equal the mesh's UVs."""
    mesh = _tile_mesh()
    document, binary = _parse(build_glb(mesh, _png()))
    view = document["bufferViews"][1]
    uvs = np.frombuffer(
        binary[view["byteOffset"] : view["byteOffset"] + view["byteLength"]],
        dtype="<f4",
    ).reshape(-1, 2)
    assert np.array_equal(uvs, mesh.uvs)
