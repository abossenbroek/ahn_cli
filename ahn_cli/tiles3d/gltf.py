"""Binary glTF (glb) assembly for one terrain tile.

One mesh, one primitive: POSITION + TEXCOORD_0 float32 attributes,
uint32 indices, a pbrMetallicRoughness material draped with the
embedded PNG texture, and a single node whose ``translation`` carries
the RTC centre. The BIN chunk lays out positions, UVs, indices and the
PNG in that order, each 4-byte aligned; the JSON chunk is serialised
with sorted keys and no timestamps, so the whole container is
byte-deterministic. POSITION accessor ``min``/``max`` are the exact
componentwise extremes of the written float32 data — the verifier
recomputes and compares them bit for bit.
"""

from __future__ import annotations

import json
import struct
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from ahn_cli.tiles3d.mesh import TileMesh

__all__ = ["build_glb"]

_GLB_MAGIC = 0x46546C67
_GLB_VERSION = 2
_CHUNK_JSON = 0x4E4F534A
_CHUNK_BIN = 0x004E4942
_ARRAY_BUFFER = 34962
_ELEMENT_ARRAY_BUFFER = 34963
_FLOAT = 5126
_UNSIGNED_INT = 5125
_LINEAR = 9729
_CLAMP_TO_EDGE = 33071
_TRIANGLES = 4
_GENERATOR = "ahn_cli tiles3d"


def build_glb(mesh: TileMesh, png: bytes) -> bytes:
    """Pack one tile mesh and its PNG texture into a glb container.

    Contract:
        - Returns the complete binary glTF 2.0 stream; deterministic
          for identical inputs.
    """
    sections = (
        mesh.positions.astype("<f4", copy=False).tobytes(),
        mesh.uvs.astype("<f4", copy=False).tobytes(),
        mesh.indices.astype("<u4", copy=False).tobytes(),
        png,
    )
    offsets: list[int] = []
    binary = bytearray()
    for section in sections:
        offsets.append(len(binary))
        binary.extend(section)
        binary.extend(b"\x00" * (-len(binary) % 4))
    document = _gltf_document(
        mesh, section_offsets=offsets, section_sizes=sections
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
    mesh: TileMesh,
    *,
    section_offsets: list[int],
    section_sizes: tuple[bytes, ...],
) -> dict[str, object]:
    """Build the glTF JSON document for the packed BIN layout."""
    vertex_count = int(mesh.positions.shape[0])
    total = section_offsets[-1] + len(section_sizes[-1])
    total += -total % 4
    buffer_views = [
        {
            "buffer": 0,
            "byteLength": len(section_sizes[0]),
            "byteOffset": section_offsets[0],
            "target": _ARRAY_BUFFER,
        },
        {
            "buffer": 0,
            "byteLength": len(section_sizes[1]),
            "byteOffset": section_offsets[1],
            "target": _ARRAY_BUFFER,
        },
        {
            "buffer": 0,
            "byteLength": len(section_sizes[2]),
            "byteOffset": section_offsets[2],
            "target": _ELEMENT_ARRAY_BUFFER,
        },
        {
            "buffer": 0,
            "byteLength": len(section_sizes[3]),
            "byteOffset": section_offsets[3],
        },
    ]
    accessors = [
        {
            "bufferView": 0,
            "componentType": _FLOAT,
            "count": vertex_count,
            "max": _column_extremes(mesh.positions, use_max=True),
            "min": _column_extremes(mesh.positions, use_max=False),
            "type": "VEC3",
        },
        {
            "bufferView": 1,
            "componentType": _FLOAT,
            "count": vertex_count,
            "type": "VEC2",
        },
        {
            "bufferView": 2,
            "componentType": _UNSIGNED_INT,
            "count": int(mesh.indices.shape[0]),
            "type": "SCALAR",
        },
    ]
    return {
        "accessors": accessors,
        "asset": {"generator": _GENERATOR, "version": "2.0"},
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": total}],
        "images": [{"bufferView": 3, "mimeType": "image/png"}],
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
        "nodes": [{"mesh": 0, "translation": list(mesh.center)}],
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


def _column_extremes(
    array: npt.NDArray[np.float32], *, use_max: bool
) -> list[float]:
    """Return the exact per-column extreme values as Python floats."""
    reduced = array.max(axis=0) if use_max else array.min(axis=0)
    return [float(value) for value in reduced]
