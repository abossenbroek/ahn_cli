"""Per-tile game-profile verification: the four strict-in-class families.

The strict profile stores float32 positions and an embedded PNG, so the
verifier reads them straight back and recomputes their extremes
(:mod:`ahn_cli.tiles3d.verify`). The game profile stores
``KHR_mesh_quantization`` ``uint16`` positions, ``EXT_meshopt_compression``
streams and a baseline JPEG, so the same "re-read from disk, recompute
from the sources, demand agreement" discipline needs four game-specific
check families, run per tile before the whole-file byte-identity backstop:

1.  **meshopt round-trip** — each of the three compressed streams
    (POSITION / TEXCOORD_0 / indices) decodes to exactly the bytes an
    independent re-quantization of this tile's genuine samples would
    encode.
2.  **requantization** — the decoded POSITION integers equal an
    independent :func:`~ahn_cli.tiles3d.quantize.quantize_positions` of the
    recomputed mesh bit for bit; the dequantized positions sit within the
    exported :func:`~ahn_cli.tiles3d.quantize.position_error_bound` of the
    genuine samples (zero slack); and the glb node's ``scale`` /
    ``translation`` equal the independently computed dequant fold exactly.
3.  **JPEG texture** — the embedded image is baseline sequential (SOF0
    present, SOF2 absent), byte-equals an independent
    :func:`~ahn_cli.tiles3d.jpeg.encode_jpeg` of the sampled ortho tile,
    and decodes within :func:`~ahn_cli.tiles3d.jpeg.jpeg_fidelity_ok` of it.
4.  **structure** — the glTF declares exactly the two extensions used *and*
    required; the POSITION / TEXCOORD_0 / index accessors carry the pinned
    component types, normalization and integer extremes; the meshopt
    bufferViews and their fallback buffers carry the pinned framing (no
    uri, no filter, 4-byte-aligned BIN offsets); and every element count
    matches the plan's tile span.

Plus containment: the dequantized vertices, projected back to EPSG:4979,
lie inside every enclosing region expanded only by the documented
quantization bound (a lossless-exact test is impossible for quantized
geometry, so the bound is the honest floor; the byte-identity backstop
still pins the regions themselves).

Everything is recomputed from the terrain the verifier already reloaded —
nothing is trusted from the build. The single entry point is
:func:`verify_game_tile`; all violations raise
:class:`~ahn_cli.tiles3d.errors.Tiles3dError` naming the offending tile.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from ahn_cli.tiles3d.errors import Tiles3dError
from ahn_cli.tiles3d.jpeg import (
    decode_jpeg,
    encode_jpeg,
    is_baseline_jpeg,
    jpeg_fidelity_ok,
)
from ahn_cli.tiles3d.mesh import build_tile_mesh
from ahn_cli.tiles3d.meshopt import (
    INDEX_BYTE_STRIDE,
    MODE_ATTRIBUTES,
    MODE_TRIANGLES,
    POSITION_BYTE_STRIDE,
    UV_BYTE_STRIDE,
    decode_indices,
    decode_positions,
    decode_uvs,
)
from ahn_cli.tiles3d.quantize import (
    dequantize_positions,
    position_error_bound,
    quantize_positions,
    quantize_uvs,
)

if TYPE_CHECKING:
    import numpy.typing as npt

    from ahn_cli.tiles3d.geodesy import Geodesy
    from ahn_cli.tiles3d.mesh import TileMesh
    from ahn_cli.tiles3d.quadtree import TilePlan
    from ahn_cli.tiles3d.quantize import QuantizedPositions
    from ahn_cli.tiles3d.sources import TerrainGrid

__all__ = ["verify_game_tile"]

_KHR_QUANT = "KHR_mesh_quantization"
_EXT_MESHOPT = "EXT_meshopt_compression"
_EXTENSIONS = sorted((_EXT_MESHOPT, _KHR_QUANT))
_ARRAY_BUFFER = 34962
_ELEMENT_ARRAY_BUFFER = 34963
_UNSIGNED_SHORT = 5123
_UNSIGNED_INT = 5125
_ALIGN = 4
_MIN_EARTH_RADIUS = 6_356_752.0
"""WGS84 semi-minor axis (m): a safe lower bound on any local radius, so
``bound / _MIN_EARTH_RADIUS`` upper-bounds the angular slack of a metric
displacement of ``bound``."""

_POSITION_VIEW = 0
_UV_VIEW = 1
_INDEX_VIEW = 2
_IMAGE_VIEW = 3
_BIN_BUFFER = 0

# (view index, fallback buffer, byte stride, meshopt mode, target, whether
# the bufferView itself carries a byteStride) for each compressed stream.
_STREAMS = (
    (_POSITION_VIEW, 1, POSITION_BYTE_STRIDE, MODE_ATTRIBUTES, _ARRAY_BUFFER),
    (_UV_VIEW, 2, UV_BYTE_STRIDE, MODE_ATTRIBUTES, _ARRAY_BUFFER),
    (
        _INDEX_VIEW,
        3,
        INDEX_BYTE_STRIDE,
        MODE_TRIANGLES,
        _ELEMENT_ARRAY_BUFFER,
    ),
)
_INDEX_STREAM = _INDEX_VIEW


def _require(condition: bool, message: str) -> None:  # noqa: FBT001
    """Raise the typed verification error unless ``condition`` holds."""
    if not condition:
        raise Tiles3dError(message)


def verify_game_tile(
    gltf: dict[str, Any],
    binary: bytes,
    terrain: TerrainGrid,
    tile: TilePlan,
    enclosing_regions: list[tuple[float, ...]],
    geodesy: Geodesy,
    uri: str,
) -> None:
    """Run the four game check families plus containment for one tile.

    Contract:
        - ``gltf`` / ``binary`` are the tile glb's fresh-from-disk JSON
          and BIN chunk (container framing already verified by the
          caller). ``terrain``, ``tile`` and ``geodesy`` let this
          recompute the tile's genuine mesh, quantization and JPEG
          independently; ``enclosing_regions`` are the tile's own and
          every ancestor's stored regions.
        - Returns only when the on-disk streams, texture, structure and
          quantized geometry all match the independent recomputation.

    Failure modes:
        - :class:`~ahn_cli.tiles3d.errors.Tiles3dError` naming ``uri`` and
          the first failed property.
    """
    mesh = build_tile_mesh(terrain, tile, geodesy)
    quantized = quantize_positions(mesh.positions)
    uv_ints = quantize_uvs(mesh.uvs)
    sampled_rgb = terrain.rgb[np.ix_(mesh.rows, mesh.cols)]
    _verify_structure(gltf, binary, quantized, mesh, uri)
    _verify_requantization(gltf, binary, quantized, mesh, uri)
    _verify_meshopt_streams(gltf, binary, quantized, uv_ints, mesh, uri)
    _verify_jpeg_texture(gltf, binary, sampled_rgb, uri)
    _verify_containment(quantized, mesh, enclosing_regions, geodesy, uri)


def _obj(value: object, uri: str, what: str) -> dict[str, Any]:
    """Cast a glTF value to a JSON object, raising if it is not one."""
    _require(isinstance(value, dict), f"{uri}: {what} is not a JSON object.")
    return cast("dict[str, Any]", value)


def _seq(value: object, uri: str, what: str) -> list[Any]:
    """Cast a glTF value to a JSON array, raising if it is not one."""
    _require(isinstance(value, list), f"{uri}: {what} is not a JSON array.")
    return cast("list[Any]", value)


def _verify_structure(
    gltf: dict[str, Any],
    binary: bytes,
    quantized: QuantizedPositions,
    mesh: TileMesh,
    uri: str,
) -> None:
    """Family 4: the glTF's extensions, buffers, views and accessors."""
    _require(
        sorted(_seq(gltf.get("extensionsUsed"), uri, "extensionsUsed"))
        == _EXTENSIONS
        and sorted(
            _seq(gltf.get("extensionsRequired"), uri, "extensionsRequired")
        )
        == _EXTENSIONS,
        f"{uri}: extensionsUsed/extensionsRequired are not exactly "
        f"{_EXTENSIONS}.",
    )
    counts = (
        int(mesh.positions.shape[0]),
        int(mesh.uvs.shape[0]),
        int(mesh.indices.shape[0]),
    )
    _verify_buffers(gltf, binary, counts, uri)
    _verify_stream_views(gltf, binary, counts, uri)
    _verify_image_view(gltf, binary, uri)
    _verify_accessors(gltf, quantized, counts, uri)


def _verify_buffers(
    gltf: dict[str, Any],
    binary: bytes,
    counts: tuple[int, int, int],
    uri: str,
) -> None:
    """Verify buffer 0 is the BIN chunk and 1..3 are fallback buffers."""
    buffers = _seq(gltf.get("buffers"), uri, "buffers")
    fallback_lengths = (
        counts[0] * POSITION_BYTE_STRIDE,
        counts[1] * UV_BYTE_STRIDE,
        counts[2] * INDEX_BYTE_STRIDE,
    )
    _require(
        len(buffers) == 1 + len(fallback_lengths),
        f"{uri}: expected one BIN buffer and three fallback buffers.",
    )
    bin_buffer = _obj(buffers[_BIN_BUFFER], uri, "buffer 0")
    _require(
        bin_buffer == {"byteLength": len(binary)},
        f"{uri}: buffer 0 is not exactly the BIN chunk byteLength.",
    )
    for index, length in enumerate(fallback_lengths, start=1):
        buffer = _obj(buffers[index], uri, f"buffer {index}")
        _require(
            buffer
            == {
                "byteLength": length,
                "extensions": {_EXT_MESHOPT: {"fallback": True}},
            },
            f"{uri}: fallback buffer {index} is not a uri-less "
            "fallback:true buffer of the decompressed length.",
        )


def _verify_stream_views(
    gltf: dict[str, Any],
    binary: bytes,
    counts: tuple[int, int, int],
    uri: str,
) -> None:
    """Verify the three compressed bufferViews and their meshopt framing."""
    views = _seq(gltf.get("bufferViews"), uri, "bufferViews")
    _require(
        len(views) == 4,  # noqa: PLR2004 -- 3 streams + the JPEG image
        f"{uri}: expected four bufferViews.",
    )
    for (view_index, fallback, stride, mode, target), count in zip(
        _STREAMS, counts, strict=True
    ):
        view = _obj(views[view_index], uri, f"bufferView {view_index}")
        raw_ext = view.get("extensions")
        ext_value: object = (
            cast("dict[str, Any]", raw_ext).get(_EXT_MESHOPT)
            if isinstance(raw_ext, dict)
            else None
        )
        expected: dict[str, object] = {
            "buffer": fallback,
            "byteLength": count * stride,
            "byteOffset": 0,
            "extensions": {_EXT_MESHOPT: ext_value},
            "target": target,
        }
        if view_index != _INDEX_STREAM:
            expected["byteStride"] = stride
        _require(
            view == expected,
            f"{uri}: bufferView {view_index} framing is not the pinned "
            "meshopt vertex/index view.",
        )
        _verify_meshopt_ext(view, binary, count, stride, mode, uri)


def _verify_meshopt_ext(
    view: dict[str, Any],
    binary: bytes,
    count: int,
    stride: int,
    mode: str,
    uri: str,
) -> None:
    """Verify one bufferView's ``EXT_meshopt_compression`` object."""
    ext = _obj(
        _obj(view["extensions"], uri, "extensions")[_EXT_MESHOPT],
        uri,
        _EXT_MESHOPT,
    )
    _require(
        set(ext)
        == {
            "buffer",
            "byteLength",
            "byteOffset",
            "byteStride",
            "count",
            "mode",
        },
        f"{uri}: a meshopt extension carries unexpected keys (a filter "
        "must be omitted).",
    )
    offset = int(ext["byteOffset"])
    length = int(ext["byteLength"])
    _require(
        ext["buffer"] == _BIN_BUFFER
        and ext["byteStride"] == stride
        and ext["count"] == count
        and ext["mode"] == mode
        and offset % _ALIGN == 0
        and offset >= 0
        and offset + length <= len(binary),
        f"{uri}: a meshopt extension is misframed or not 4-byte aligned.",
    )


def _verify_image_view(gltf: dict[str, Any], binary: bytes, uri: str) -> None:
    """Verify the JPEG image bufferView's framing and alignment."""
    view = _obj(
        _seq(gltf.get("bufferViews"), uri, "bufferViews")[_IMAGE_VIEW],
        uri,
        "the image bufferView",
    )
    offset = int(view.get("byteOffset", -1))
    length = int(view.get("byteLength", -1))
    _require(
        view.get("buffer") == _BIN_BUFFER
        and set(view) == {"buffer", "byteLength", "byteOffset"}
        and offset % _ALIGN == 0
        and offset >= 0
        and length >= 0
        and offset + length <= len(binary),
        f"{uri}: the image bufferView is misframed or not 4-byte aligned.",
    )


def _verify_accessors(
    gltf: dict[str, Any],
    quantized: QuantizedPositions,
    counts: tuple[int, int, int],
    uri: str,
) -> None:
    """Verify the POSITION / TEXCOORD_0 / index accessor declarations."""
    accessors = _seq(gltf.get("accessors"), uri, "accessors")
    _require(len(accessors) == 3, f"{uri}: expected three accessors.")  # noqa: PLR2004
    position = _obj(accessors[0], uri, "the POSITION accessor")
    _require(
        position
        == {
            "bufferView": _POSITION_VIEW,
            "componentType": _UNSIGNED_SHORT,
            "count": counts[0],
            "max": [int(v) for v in quantized.ints.max(axis=0)],
            "min": [int(v) for v in quantized.ints.min(axis=0)],
            "type": "VEC3",
        },
        f"{uri}: the POSITION accessor is not an unnormalized uint16 VEC3 "
        "with the recomputed integer extremes.",
    )
    uv = _obj(accessors[1], uri, "the TEXCOORD_0 accessor")
    _require(
        uv
        == {
            "bufferView": _UV_VIEW,
            "componentType": _UNSIGNED_SHORT,
            "count": counts[1],
            "normalized": True,
            "type": "VEC2",
        },
        f"{uri}: the TEXCOORD_0 accessor is not a normalized uint16 VEC2.",
    )
    index = _obj(accessors[2], uri, "the index accessor")
    _require(
        index
        == {
            "bufferView": _INDEX_VIEW,
            "componentType": _UNSIGNED_INT,
            "count": counts[2],
            "type": "SCALAR",
        },
        f"{uri}: the index accessor is not a uint32 SCALAR of the tile's "
        "triangle count.",
    )


def _stream_bytes(
    gltf: dict[str, Any], binary: bytes, view_index: int
) -> bytes:
    """Return a compressed stream's bytes via its meshopt extension."""
    view = cast("dict[str, Any]", gltf["bufferViews"][view_index])
    ext = cast("dict[str, Any]", view["extensions"][_EXT_MESHOPT])
    offset = int(ext["byteOffset"])
    return binary[offset : offset + int(ext["byteLength"])]


def _verify_requantization(
    gltf: dict[str, Any],
    binary: bytes,
    quantized: QuantizedPositions,
    mesh: TileMesh,
    uri: str,
) -> None:
    """Family 2: POSITION requantization, dequant bound and node fold."""
    decoded = decode_positions(
        _stream_bytes(gltf, binary, _POSITION_VIEW),
        int(quantized.ints.shape[0]),
    )
    _require(
        bool(np.array_equal(decoded, quantized.ints)),
        f"{uri}: the POSITION stream does not equal the independent "
        "requantization of the source samples.",
    )
    dequantized = dequantize_positions(quantized)
    bound = position_error_bound(quantized.scale)
    error = np.abs(dequantized - mesh.positions.astype(np.float64)).max(
        axis=0
    )
    _require(
        all(float(error[axis]) <= bound[axis] for axis in range(3)),
        f"{uri}: a dequantized position exceeds the documented "
        "quantization error bound.",
    )
    node = _obj(_seq(gltf.get("nodes"), uri, "nodes")[0], uri, "the node")
    fold = [
        mesh.center[axis] + quantized.translation[axis] for axis in range(3)
    ]
    _require(
        node.get("scale") == list(quantized.scale)
        and node.get("translation") == fold,
        f"{uri}: the node scale/translation is not the independently "
        "computed dequantization fold.",
    )


def _verify_meshopt_streams(
    gltf: dict[str, Any],
    binary: bytes,
    quantized: QuantizedPositions,
    uv_ints: npt.NDArray[np.uint16],
    mesh: TileMesh,
    uri: str,
) -> None:
    """Family 1: each stream decodes to the recomputed pre-encode bytes."""
    decoded_positions = decode_positions(
        _stream_bytes(gltf, binary, _POSITION_VIEW),
        int(quantized.ints.shape[0]),
    )
    decoded_uvs = decode_uvs(
        _stream_bytes(gltf, binary, _UV_VIEW), int(uv_ints.shape[0])
    )
    decoded_indices = decode_indices(
        _stream_bytes(gltf, binary, _INDEX_VIEW), int(mesh.indices.shape[0])
    )
    _require(
        bool(np.array_equal(decoded_positions, quantized.ints)),
        f"{uri}: the POSITION meshopt stream does not round-trip to the "
        "recomputed quantized integers.",
    )
    _require(
        bool(np.array_equal(decoded_uvs, uv_ints)),
        f"{uri}: the TEXCOORD_0 meshopt stream does not round-trip to the "
        "recomputed quantized UVs.",
    )
    _require(
        bool(np.array_equal(decoded_indices, mesh.indices)),
        f"{uri}: the index meshopt stream does not round-trip to the "
        "recomputed triangle list.",
    )


def _verify_jpeg_texture(
    gltf: dict[str, Any],
    binary: bytes,
    sampled_rgb: npt.NDArray[np.uint8],
    uri: str,
) -> None:
    """Family 3: the embedded baseline JPEG equals the re-encoded ortho."""
    image = _obj(_seq(gltf.get("images"), uri, "images")[0], uri, "the image")
    _require(
        image == {"bufferView": _IMAGE_VIEW, "mimeType": "image/jpeg"},
        f"{uri}: the image is not a bufferView-3 image/jpeg.",
    )
    view = cast("dict[str, Any]", gltf["bufferViews"][_IMAGE_VIEW])
    offset = int(view["byteOffset"])
    data = binary[offset : offset + int(view["byteLength"])]
    _require(
        is_baseline_jpeg(data),
        f"{uri}: the texture is not a baseline sequential JPEG "
        "(SOF0 present, SOF2 absent).",
    )
    _require(
        data == encode_jpeg(sampled_rgb),
        f"{uri}: the JPEG texture does not byte-equal the independent "
        "re-encode of the sampled ortho tile.",
    )
    _require(
        jpeg_fidelity_ok(sampled_rgb, decode_jpeg(data)),
        f"{uri}: the JPEG texture decodes below the fidelity floor of the "
        "sampled ortho tile.",
    )


def _verify_containment(
    quantized: QuantizedPositions,
    mesh: TileMesh,
    enclosing_regions: list[tuple[float, ...]],
    geodesy: Geodesy,
    uri: str,
) -> None:
    """Verify the dequantized vertices lie inside every enclosing region.

    The dequantized geometry differs from the genuine surface by at most
    the documented per-axis quantization bound, so an exact (strict-style)
    containment test is impossible for quantized vertices; each region is
    expanded by that bound's metric magnitude (angular slack via a safe
    minimum Earth radius) — the honest floor for lossy geometry. The
    region values themselves stay pinned by the byte-identity backstop.
    """
    dequantized = dequantize_positions(quantized)
    world = dequantized + np.asarray(mesh.center, dtype=np.float64)
    # Un-swizzle glTF y-up (x, z, -y) back to ECEF (x, y, z).
    ecef_x = world[:, 0]
    ecef_y = -world[:, 2]
    ecef_z = world[:, 1]
    lon, lat, height = geodesy.to_geodetic_from_ecef(ecef_x, ecef_y, ecef_z)
    bound = position_error_bound(quantized.scale)
    metric = float(np.sqrt(sum(component**2 for component in bound)))
    angular = metric / _MIN_EARTH_RADIUS
    for region in enclosing_regions:
        _require(
            _within(lon, lat, height, region, angular, metric),
            f"{uri}: a dequantized vertex lies outside an enclosing region "
            "(beyond the quantization bound).",
        )


def _within(
    lon: npt.NDArray[np.float64],
    lat: npt.NDArray[np.float64],
    height: npt.NDArray[np.float64],
    region: tuple[float, ...],
    angular: float,
    metric: float,
) -> bool:
    """Return whether every point lies in ``region`` grown by the bound."""
    return (
        bool((lon >= region[0] - angular).all())
        and bool((lat >= region[1] - angular).all())
        and bool((lon <= region[2] + angular).all())
        and bool((lat <= region[3] + angular).all())
        and bool((height >= region[4] - metric).all())
        and bool((height <= region[5] + metric).all())
    )
