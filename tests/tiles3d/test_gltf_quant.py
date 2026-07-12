"""Tests for the game-profile glb writer (``gltf_quant.build_game_glb``).

These pin the container framing, the ``KHR_mesh_quantization`` /
``EXT_meshopt_compression`` declarations and per-stream extension objects,
the bit-exact POSITION integer extremes, the accessor/bufferView shapes,
the embedded JPEG image, and the node dequantization fold — including the
end-to-end proof that decoding the streams and applying the node transform
reproduces the source RTC vertices within the quantizer's error bound.
"""

from __future__ import annotations

import json
import struct
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from ahn_cli.tiles3d.errors import Tiles3dError
from ahn_cli.tiles3d.geodesy import Geodesy
from ahn_cli.tiles3d.gltf_quant import build_game_glb
from ahn_cli.tiles3d.jpeg import encode_jpeg
from ahn_cli.tiles3d.mesh import build_tile_mesh
from ahn_cli.tiles3d.meshopt import (
    MeshoptStream,
    decode_indices,
    decode_positions,
    decode_uvs,
    encode_indices,
    encode_positions,
    encode_uvs,
)
from ahn_cli.tiles3d.quadtree import plan_quadtree
from ahn_cli.tiles3d.quantize import (
    dequantize_positions,
    position_error_bound,
    quantize_positions,
    quantize_uvs,
)
from tests.tiles3d.conftest import make_terrain

if TYPE_CHECKING:
    from ahn_cli.tiles3d.mesh import TileMesh
    from ahn_cli.tiles3d.quantize import QuantizedPositions

_GLB_MAGIC = 0x46546C67
_CHUNK_JSON = 0x4E4F534A
_CHUNK_BIN = 0x004E4942
_UNSIGNED_SHORT = 5123
_UNSIGNED_INT = 5125


def _mesh(width: int = 6, height: int = 5) -> TileMesh:
    terrain = make_terrain(width, height)
    tile = plan_quadtree(width, height).root
    return build_tile_mesh(terrain, tile, Geodesy())


def _pieces(
    mesh: TileMesh,
) -> tuple[
    QuantizedPositions, MeshoptStream, MeshoptStream, MeshoptStream, bytes
]:
    quantized = quantize_positions(mesh.positions)
    uv_ints = quantize_uvs(mesh.uvs)
    terrain = make_terrain(6, 5)
    grid = np.ix_(mesh.rows, mesh.cols)
    jpeg = encode_jpeg(terrain.rgb[grid])
    return (
        quantized,
        encode_positions(quantized.ints),
        encode_uvs(uv_ints),
        encode_indices(mesh.indices),
        jpeg,
    )


def _build(mesh: TileMesh) -> bytes:
    quantized, positions, uvs, indices, jpeg = _pieces(mesh)
    return build_game_glb(
        quantized, positions, uvs, indices, jpeg, mesh.center
    )


def _parse(data: bytes) -> tuple[dict[str, Any], bytes]:
    magic, version, length = struct.unpack("<III", data[:12])
    assert magic == _GLB_MAGIC
    assert version == 2
    assert length == len(data)
    json_length, json_kind = struct.unpack("<II", data[12:20])
    assert json_kind == _CHUNK_JSON
    assert json_length % 4 == 0
    bin_header = 20 + json_length
    bin_length, bin_kind = struct.unpack(
        "<II", data[bin_header : bin_header + 8]
    )
    assert bin_kind == _CHUNK_BIN
    assert bin_header + 8 + bin_length == len(data)
    document = json.loads(data[20:bin_header])
    return cast("dict[str, Any]", document), data[bin_header + 8 :]


def _ext(view: dict[str, Any]) -> dict[str, Any]:
    return cast(
        "dict[str, Any]", view["extensions"]["EXT_meshopt_compression"]
    )


def _compressed_bytes(binary: bytes, ext: dict[str, Any]) -> bytes:
    offset = int(ext["byteOffset"])
    return binary[offset : offset + int(ext["byteLength"])]


def test_build_game_glb_is_deterministic() -> None:
    """Encoding the same inputs twice yields byte-identical output."""
    mesh = _mesh()
    assert _build(mesh) == _build(mesh)


def test_container_framing_is_valid() -> None:
    """The glb magic, version, length and JSON-then-BIN chunks parse."""
    document, _ = _parse(_build(_mesh()))
    assert document["asset"] == {
        "generator": "ahn_cli tiles3d",
        "version": "2.0",
    }


def test_extensions_declared_used_and_required() -> None:
    """Both extensions appear in ``extensionsUsed`` and required."""
    document, _ = _parse(_build(_mesh()))
    expected = ["EXT_meshopt_compression", "KHR_mesh_quantization"]
    assert document["extensionsUsed"] == expected
    assert document["extensionsRequired"] == expected


def test_position_accessor_int_extremes_are_bit_exact() -> None:
    """POSITION ``min``/``max`` equal the quantized data's int extremes."""
    mesh = _mesh()
    quantized, *_ = _pieces(mesh)
    document, _ = _parse(_build(mesh))
    accessor = document["accessors"][0]
    assert accessor["componentType"] == _UNSIGNED_SHORT
    assert "normalized" not in accessor
    assert accessor["type"] == "VEC3"
    assert accessor["count"] == mesh.positions.shape[0]
    assert accessor["max"] == [int(v) for v in quantized.ints.max(axis=0)]
    assert accessor["min"] == [int(v) for v in quantized.ints.min(axis=0)]


def test_texcoord_accessor_is_normalized_uint16() -> None:
    """TEXCOORD_0 is a normalized ``uint16`` VEC2 accessor."""
    document, _ = _parse(_build(_mesh()))
    accessor = document["accessors"][1]
    assert accessor["componentType"] == _UNSIGNED_SHORT
    assert accessor["normalized"] is True
    assert accessor["type"] == "VEC2"


def test_index_accessor_is_uint32_scalar() -> None:
    """The index accessor is an ``UNSIGNED_INT`` SCALAR of every index."""
    mesh = _mesh()
    document, _ = _parse(_build(mesh))
    accessor = document["accessors"][2]
    assert accessor["componentType"] == _UNSIGNED_INT
    assert accessor["type"] == "SCALAR"
    assert accessor["count"] == mesh.indices.shape[0]


def test_buffer_view_strides_and_targets() -> None:
    """POSITION view stride is 8, UV 4; the index view carries no stride."""
    document, _ = _parse(_build(_mesh()))
    views = document["bufferViews"]
    assert views[0]["byteStride"] == 8
    assert views[0]["target"] == 34962
    assert views[1]["byteStride"] == 4
    assert views[1]["target"] == 34962
    assert "byteStride" not in views[2]
    assert views[2]["target"] == 34963


def test_meshopt_extension_objects_point_into_the_bin() -> None:
    """Each compressed view's extension locates its stream in buffer 0."""
    mesh = _mesh()
    _, positions, uvs, indices, _ = _pieces(mesh)
    document, binary = _parse(_build(mesh))
    for view_index, (stream, mode) in enumerate(
        (
            (positions, "ATTRIBUTES"),
            (uvs, "ATTRIBUTES"),
            (indices, "TRIANGLES"),
        )
    ):
        ext = _ext(document["bufferViews"][view_index])
        assert ext["buffer"] == 0
        assert ext["byteStride"] == stream.byte_stride
        assert ext["count"] == stream.count
        assert ext["mode"] == mode
        assert "filter" not in ext
        assert _compressed_bytes(binary, ext) == stream.data


def test_fallback_buffers_are_declared_unallocated() -> None:
    """Buffers 1-3 are per-stream fallbacks: no uri, flagged fallback."""
    mesh = _mesh()
    _, positions, uvs, indices, _ = _pieces(mesh)
    document, _ = _parse(_build(mesh))
    buffers = document["buffers"]
    assert "uri" not in buffers[0]
    for index, stream in enumerate((positions, uvs, indices), start=1):
        buffer = buffers[index]
        assert "uri" not in buffer
        assert buffer["extensions"]["EXT_meshopt_compression"]["fallback"]
        assert buffer["byteLength"] == stream.count * stream.byte_stride
        assert document["bufferViews"][index - 1]["buffer"] == index


def test_image_is_embedded_jpeg() -> None:
    """The texture is an ``image/jpeg`` whose BIN bytes equal the JPEG."""
    mesh = _mesh()
    _, _, _, _, jpeg = _pieces(mesh)
    document, binary = _parse(_build(mesh))
    image = document["images"][0]
    assert image["mimeType"] == "image/jpeg"
    view = document["bufferViews"][image["bufferView"]]
    assert "target" not in view
    offset = int(view["byteOffset"])
    assert binary[offset : offset + int(view["byteLength"])] == jpeg


def test_material_and_texture_structure_matches_strict() -> None:
    """Sampler/material/texture wiring mirrors the strict glb."""
    document, _ = _parse(_build(_mesh()))
    assert document["textures"] == [{"sampler": 0, "source": 0}]
    material = document["materials"][0]
    assert material["pbrMetallicRoughness"]["baseColorTexture"] == {
        "index": 0
    }
    assert document["samplers"][0]["magFilter"] == 9729


def test_node_folds_centre_into_the_dequant_transform() -> None:
    """``node.translation = centre + qtrans`` and ``scale = qscale``."""
    mesh = _mesh()
    quantized, *_ = _pieces(mesh)
    document, _ = _parse(_build(mesh))
    node = document["nodes"][0]
    assert node["scale"] == list(quantized.scale)
    expected = [mesh.center[a] + quantized.translation[a] for a in range(3)]
    assert node["translation"] == expected


def test_streams_decode_to_the_encoder_inputs() -> None:
    """The BIN streams decode bit-exactly to the quantized inputs."""
    mesh = _mesh()
    quantized, *_ = _pieces(mesh)
    document, binary = _parse(_build(mesh))
    pos_ext = _ext(document["bufferViews"][0])
    uv_ext = _ext(document["bufferViews"][1])
    idx_ext = _ext(document["bufferViews"][2])
    decoded_pos = decode_positions(
        _compressed_bytes(binary, pos_ext), pos_ext["count"]
    )
    assert np.array_equal(decoded_pos, quantized.ints)
    decoded_uv = decode_uvs(
        _compressed_bytes(binary, uv_ext), uv_ext["count"]
    )
    assert np.array_equal(decoded_uv, quantize_uvs(mesh.uvs))
    decoded_idx = decode_indices(
        _compressed_bytes(binary, idx_ext), idx_ext["count"]
    )
    assert np.array_equal(decoded_idx, mesh.indices)


def test_node_fold_reproduces_source_vertices_within_error_bound() -> None:
    """Decode + node fold reproduces ``centre + RTC`` within the bound."""
    mesh = _mesh()
    quantized, *_ = _pieces(mesh)
    document, binary = _parse(_build(mesh))
    node = document["nodes"][0]
    pos_ext = _ext(document["bufferViews"][0])
    ints = decode_positions(
        _compressed_bytes(binary, pos_ext), pos_ext["count"]
    )
    scale = np.asarray(node["scale"], dtype=np.float64)
    translation = np.asarray(node["translation"], dtype=np.float64)
    world = ints.astype(np.float64) * scale + translation
    source = mesh.positions.astype(np.float64) + np.asarray(
        mesh.center, dtype=np.float64
    )
    bound = np.asarray(
        position_error_bound(quantized.scale), dtype=np.float64
    )
    assert np.all(np.abs(world - source) <= bound)
    # Cross-check: the node fold matches the direct dequantizer + centre.
    dequant = dequantize_positions(quantized) + np.asarray(
        mesh.center, dtype=np.float64
    )
    assert np.allclose(world, dequant)


def test_filtered_stream_is_refused() -> None:
    """A stream carrying a meshopt filter is rejected (game is unfiltered)."""
    mesh = _mesh()
    quantized, positions, uvs, indices, jpeg = _pieces(mesh)
    filtered = MeshoptStream(
        data=positions.data,
        count=positions.count,
        byte_stride=positions.byte_stride,
        mode=positions.mode,
        filter="OCTAHEDRAL",
    )
    with pytest.raises(Tiles3dError, match="unfiltered"):
        build_game_glb(quantized, filtered, uvs, indices, jpeg, mesh.center)
