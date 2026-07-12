"""Binary glTF (glb) assembly for one quantized, meshopt-compressed tile.

The game profile's counterpart to :mod:`ahn_cli.tiles3d.gltf`: instead of
float32 attributes and an embedded PNG, it packs a tile as
``KHR_mesh_quantization`` ``uint16`` positions, normalized ``uint16`` UVs
and ``uint32`` indices, each stream compressed with
``EXT_meshopt_compression`` (:mod:`ahn_cli.tiles3d.meshopt`), draped with a
baseline JPEG (:mod:`ahn_cli.tiles3d.jpeg`). Both extensions are declared
in ``extensionsUsed`` *and* ``extensionsRequired`` — there is no
decompressed fallback data, so a runtime must decode the streams.

Node dequantization fold (the ``KHR_mesh_quantization`` contract): the RTC
mesh's float32 offsets are quantized about the tile's own per-axis extents,
yielding a per-axis ``scale``/``translation``. The single node folds the RTC
centre into that transform — ``node.translation = centre +
quantization_translation`` (component-wise float64) and ``node.scale =
quantization_scale`` — so a runtime reconstructs each world vertex as
``node.translation + node.scale * q`` == ``centre + dequantized_offset``,
exactly reproducing the strict path's ``centre + RTC_offset``. The Rust
consumer reads node ``scale``/``translation`` per the extension, needing no
separate RTC translation.

BIN layout (buffer 0, no uri — the glb BIN chunk): the compressed POSITION,
TEXCOORD_0 and index streams then the JPEG image, each 4-byte aligned, so
the whole container is byte-deterministic. Each compressed bufferView points
its ``buffer`` at a per-stream fallback buffer (``fallback: true``, no uri,
never allocated) and carries the compressed location/params in its
``EXT_meshopt_compression`` extension. POSITION accessor ``min``/``max`` are
the exact per-axis integer extremes of the quantized data (the verifier
recomputes and bit-compares them).
"""

from __future__ import annotations

import json
import struct
from typing import TYPE_CHECKING

from ahn_cli.tiles3d.errors import Tiles3dError

if TYPE_CHECKING:
    from ahn_cli.tiles3d.meshopt import MeshoptStream
    from ahn_cli.tiles3d.quantize import QuantizedPositions

__all__ = ["build_game_glb"]

_GLB_MAGIC = 0x46546C67
_GLB_VERSION = 2
_CHUNK_JSON = 0x4E4F534A
_CHUNK_BIN = 0x004E4942
_ARRAY_BUFFER = 34962
_ELEMENT_ARRAY_BUFFER = 34963
_UNSIGNED_SHORT = 5123
_UNSIGNED_INT = 5125
_LINEAR = 9729
_CLAMP_TO_EDGE = 33071
_TRIANGLES = 4
_GENERATOR = "ahn_cli tiles3d"
_EXT_MESHOPT = "EXT_meshopt_compression"
_KHR_QUANT = "KHR_mesh_quantization"


def build_game_glb(
    quantized: QuantizedPositions,
    positions: MeshoptStream,
    uvs: MeshoptStream,
    indices: MeshoptStream,
    jpeg: bytes,
    center: tuple[float, float, float],
) -> bytes:
    """Pack quantized, meshopt-compressed streams and a JPEG into a glb.

    Contract:
        - ``quantized`` supplies the POSITION integer extremes and the
          node dequantization transform; ``positions``/``uvs``/``indices``
          are the three :class:`~ahn_cli.tiles3d.meshopt.MeshoptStream`
          bufferViews (filter ``None``); ``jpeg`` is the embedded texture;
          ``center`` is the mesh's RTC node translation (glTF y-up).
        - Returns the complete binary glTF 2.0 stream, deterministic for
          identical inputs, with both ``KHR_mesh_quantization`` and
          ``EXT_meshopt_compression`` declared used *and* required.

    Failure modes:
        - Raises :class:`~ahn_cli.tiles3d.errors.Tiles3dError` if any
          stream carries a meshopt filter (the game streams are always
          unfiltered; a filtered stream would need a ``filter`` key this
          writer deliberately omits).
    """
    streams = (positions, uvs, indices)
    for stream in streams:
        if stream.filter is not None:
            msg = (
                "game glb: meshopt streams must be unfiltered, got filter "
                f"{stream.filter!r}."
            )
            raise Tiles3dError(msg)

    sections = (positions.data, uvs.data, indices.data, jpeg)
    offsets: list[int] = []
    binary = bytearray()
    for section in sections:
        offsets.append(len(binary))
        binary.extend(section)
        binary.extend(b"\x00" * (-len(binary) % 4))
    document = _gltf_document(
        quantized,
        streams,
        center,
        compressed_offsets=offsets,
        jpeg_length=len(jpeg),
        bin_length=len(binary),
    )
    json_chunk = json.dumps(
        document, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    json_chunk += b" " * (-len(json_chunk) % 4)
    length = 12 + 8 + len(json_chunk) + 8 + len(binary)
    return b"".join(
        (
            struct.pack("<III", _GLB_MAGIC, _GLB_VERSION, length),
            struct.pack("<II", len(json_chunk), _CHUNK_JSON),
            json_chunk,
            struct.pack("<II", len(binary), _CHUNK_BIN),
            bytes(binary),
        )
    )


def _gltf_document(
    quantized: QuantizedPositions,
    streams: tuple[MeshoptStream, MeshoptStream, MeshoptStream],
    center: tuple[float, float, float],
    *,
    compressed_offsets: list[int],
    jpeg_length: int,
    bin_length: int,
) -> dict[str, object]:
    """Build the glTF JSON document for the compressed BIN layout."""
    positions, uvs, indices = streams
    vertex_count = positions.count
    # Fallback buffers 1..3 mirror each stream's decompressed length; a
    # runtime that honours the required extension never allocates them.
    fallback_lengths = (
        vertex_count * positions.byte_stride,
        uvs.count * uvs.byte_stride,
        indices.count * indices.byte_stride,
    )
    buffers: list[dict[str, object]] = [{"byteLength": bin_length}]
    buffers.extend(
        {
            "byteLength": length,
            "extensions": {_EXT_MESHOPT: {"fallback": True}},
        }
        for length in fallback_lengths
    )
    buffer_views = [
        _compressed_view(positions, compressed_offsets[0], 1, _ARRAY_BUFFER),
        _compressed_view(uvs, compressed_offsets[1], 2, _ARRAY_BUFFER),
        _index_view(indices, compressed_offsets[2], 3),
        {
            "buffer": 0,
            "byteLength": jpeg_length,
            "byteOffset": compressed_offsets[3],
        },
    ]
    accessors = [
        {
            "bufferView": 0,
            "componentType": _UNSIGNED_SHORT,
            "count": vertex_count,
            "max": [int(value) for value in quantized.ints.max(axis=0)],
            "min": [int(value) for value in quantized.ints.min(axis=0)],
            "type": "VEC3",
        },
        {
            "bufferView": 1,
            "componentType": _UNSIGNED_SHORT,
            "count": uvs.count,
            "normalized": True,
            "type": "VEC2",
        },
        {
            "bufferView": 2,
            "componentType": _UNSIGNED_INT,
            "count": indices.count,
            "type": "SCALAR",
        },
    ]
    translation = [center[a] + quantized.translation[a] for a in range(3)]
    return {
        "accessors": accessors,
        "asset": {"generator": _GENERATOR, "version": "2.0"},
        "bufferViews": buffer_views,
        "buffers": buffers,
        "extensionsRequired": sorted((_EXT_MESHOPT, _KHR_QUANT)),
        "extensionsUsed": sorted((_EXT_MESHOPT, _KHR_QUANT)),
        "images": [{"bufferView": 3, "mimeType": "image/jpeg"}],
        "materials": [
            {
                "doubleSided": True,
                "pbrMetallicRoughness": {
                    "baseColorTexture": {"index": 0},
                    "metallicFactor": 0.0,
                    "roughnessFactor": 1.0,
                },
            }
        ],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "TEXCOORD_0": 1},
                        "indices": 2,
                        "material": 0,
                        "mode": _TRIANGLES,
                    }
                ]
            }
        ],
        "nodes": [
            {
                "mesh": 0,
                "scale": list(quantized.scale),
                "translation": translation,
            }
        ],
        "samplers": [
            {
                "magFilter": _LINEAR,
                "minFilter": _LINEAR,
                "wrapS": _CLAMP_TO_EDGE,
                "wrapT": _CLAMP_TO_EDGE,
            }
        ],
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "textures": [{"sampler": 0, "source": 0}],
    }


def _compressed_view(
    stream: MeshoptStream, byte_offset: int, fallback_buffer: int, target: int
) -> dict[str, object]:
    """Build a vertex-attribute bufferView carrying its meshopt extension."""
    length = stream.count * stream.byte_stride
    return {
        "buffer": fallback_buffer,
        "byteLength": length,
        "byteOffset": 0,
        "byteStride": stream.byte_stride,
        "extensions": {_EXT_MESHOPT: _meshopt_extension(stream, byte_offset)},
        "target": target,
    }


def _index_view(
    stream: MeshoptStream, byte_offset: int, fallback_buffer: int
) -> dict[str, object]:
    """Build the index bufferView (no ``byteStride``) with its extension."""
    return {
        "buffer": fallback_buffer,
        "byteLength": stream.count * stream.byte_stride,
        "byteOffset": 0,
        "extensions": {_EXT_MESHOPT: _meshopt_extension(stream, byte_offset)},
        "target": _ELEMENT_ARRAY_BUFFER,
    }


def _meshopt_extension(
    stream: MeshoptStream, byte_offset: int
) -> dict[str, object]:
    """Build one ``EXT_meshopt_compression`` object into the BIN buffer."""
    return {
        "buffer": 0,
        "byteLength": len(stream.data),
        "byteOffset": byte_offset,
        "byteStride": stream.byte_stride,
        "count": stream.count,
        "mode": stream.mode,
    }
