"""The strictest post-write verifier for a 3D Tiles build.

:func:`verify_tiles3d` re-reads **everything from disk** and validates
it against both the OGC 3D Tiles 1.1 / glTF 2.0 rules and an
**independent recomputation from the source rasters**:

1.  tileset.json: exact key sets (we authored it — unknown keys are
    errors), ``asset.version == "1.1"``, root ``refine: "REPLACE"``,
    finite non-negative geometric errors, child error <= parent error,
    tileset error >= root error.
2.  regions: valid ranges and ordering; every child region contained
    in its parent's.
3.  content links: every uri stays inside the output directory, maps
    to a planned quadtree tile, is referenced exactly once, and exists;
    every on-disk tile file is referenced (no orphans).
4.  glb containers: magic, version, declared length == file size,
    exact two-chunk JSON+BIN framing.
5.  glTF internals: buffer/view/accessor bounds, POSITION min/max
    recomputed from the payload bit for bit, index count divisible by
    3, all indices in range, no degenerate triangles, UVs in [0, 1].
6.  textures: strict PNG decode (every chunk CRC verified), dimensions
    equal to the tile's sample counts, pixels bit-equal to the freshly
    sampled orthophoto.
7.  containment: every sampled vertex's EPSG:4979 coordinate inside
    the tile's stored region and every ancestor's stored region.
8.  coverage: the recomputed leaf spans cover every grid pixel.
9.  byte identity: every glb and tileset.json byte-equal to a full
    independent rebuild from the (re-verified) sources.

Any violation raises :class:`Tiles3dError`; the build orchestrator
then removes everything it wrote.
"""

from __future__ import annotations

import json
import math
import struct
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from ahn_cli.tiles3d.emit import (
    TILES_SUBDIR,
    TILESET_NAME,
    compute_build,
    tile_uri,
)
from ahn_cli.tiles3d.errors import Tiles3dError
from ahn_cli.tiles3d.geodesy import Geodesy
from ahn_cli.tiles3d.png import decode_png
from ahn_cli.tiles3d.quadtree import plan_quadtree, sample_indices
from ahn_cli.tiles3d.sources import load_terrain
from ahn_cli.tiles3d.tileset import render_tileset

if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt

    from ahn_cli.tiles3d.emit import ComputedBuild
    from ahn_cli.tiles3d.quadtree import TilePlan, TreePlan
    from ahn_cli.tiles3d.sources import TerrainGrid

__all__ = ["verify_tiles3d"]

_GLB_MAGIC = 0x46546C67
_CHUNK_JSON = 0x4E4F534A
_CHUNK_BIN = 0x004E4942
_REGION_LENGTH = 6
_TRIANGLE = 3

_TOP_KEYS = {"asset", "geometricError", "root"}
_ASSET = {"generator": "ahn_cli tiles3d", "version": "1.1"}
_TILE_KEYS = {"boundingVolume", "content", "geometricError"}


def _require(condition: bool, message: str) -> None:  # noqa: FBT001
    """Raise the typed verification error unless ``condition`` holds."""
    if not condition:
        raise Tiles3dError(message)


def verify_tiles3d(
    out_dir: Path,
    ortho: Path,
    heights: Path,
    *,
    tile_pixels: int = 256,
) -> None:
    """Hard-verify a written 3D Tiles build against its sources.

    Contract:
        - Returns only when every check in the module docstring holds,
          all from fresh disk reads.

    Failure modes:
        - :class:`Tiles3dError` naming the first failed check.
    """
    terrain = load_terrain(ortho, heights)
    tree = plan_quadtree(terrain.width, terrain.height, tile_pixels)
    _verify_leaf_coverage(tree, terrain)
    document = _read_tileset(out_dir / TILESET_NAME)
    flat = _verify_document(document)
    tiles_by_uri = {tile_uri(t): t for t in _walk(tree.root)}
    _verify_links(out_dir, flat, tiles_by_uri)
    geodesy = Geodesy()
    for entry, enclosing_regions in flat:
        uri = _entry_uri(entry)
        tile = tiles_by_uri[uri]
        data = (out_dir / uri).read_bytes()
        gltf, binary = _parse_glb(data, uri)
        _verify_gltf(gltf, binary, uri)
        _verify_texture(gltf, binary, terrain, tile, uri)
        _verify_containment(terrain, tile, enclosing_regions, geodesy, uri)
    computed = compute_build(terrain, tree)
    _verify_byte_identity(out_dir, computed)


def _walk(tile: TilePlan) -> list[TilePlan]:
    """Flatten the plan tree, parents before children."""
    tiles = [tile]
    for child in tile.children:
        tiles.extend(_walk(child))
    return tiles


def _verify_leaf_coverage(tree: TreePlan, terrain: TerrainGrid) -> None:
    """Verify the recomputed leaf spans cover every grid pixel."""
    coverage = np.zeros((terrain.height, terrain.width), dtype=np.int32)
    for tile in _walk(tree.root):
        if not tile.children:
            coverage[
                tile.row0 : tile.row1 + 1, tile.col0 : tile.col1 + 1
            ] += 1
    _require(
        bool(coverage.min() >= 1),
        "the quadtree's leaf spans do not cover every pixel of the "
        "grid; a pixel is missing from the output.",
    )


def _read_tileset(path: Path) -> dict[str, Any]:
    """Read and parse tileset.json from disk."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        msg = f"tileset.json at {path} is not readable: {exc}"
        raise Tiles3dError(msg) from exc
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        msg = f"tileset.json at {path} is not valid JSON: {exc}"
        raise Tiles3dError(msg) from exc
    _require(
        isinstance(document, dict),
        f"tileset.json at {path} is not a JSON object.",
    )
    return cast("dict[str, Any]", document)


_Flat = list[tuple[dict[str, Any], list[tuple[float, ...]]]]


def _verify_document(document: dict[str, Any]) -> _Flat:
    """Structurally verify the tileset document; flatten its entries.

    Returns every tile entry paired with the list of its enclosing
    stored regions (ancestors first, its own last), for the
    containment checks.
    """
    _require(
        set(document) == _TOP_KEYS,
        "tileset.json carries unexpected or missing top-level keys.",
    )
    _require(
        document["asset"] == _ASSET,
        "tileset.json's asset block is not the ahn_cli tiles3d "
        "version 1.1 asset.",
    )
    tileset_error = _finite_error(document["geometricError"], "tileset")
    root = document["root"]
    _require(
        isinstance(root, dict),
        "the root tile entry is not a JSON object.",
    )
    root_entry = cast("dict[str, Any]", root)
    _require(
        root_entry.get("refine") == "REPLACE",
        'the root tile must declare refine "REPLACE".',
    )
    flat: _Flat = []
    root_error = _verify_entry(
        root_entry,
        is_root=True,
        parent_error=math.inf,
        parent_region=None,
        ancestor_regions=[],
        flat=flat,
    )
    _require(
        tileset_error >= root_error,
        "the tileset geometricError is below the root tile's.",
    )
    return flat


def _verify_entry(
    entry: dict[str, Any],
    *,
    is_root: bool,
    parent_error: float,
    parent_region: tuple[float, ...] | None,
    ancestor_regions: list[tuple[float, ...]],
    flat: _Flat,
) -> float:
    """Verify one tile entry (recursively) and record it in ``flat``."""
    allowed = (
        _TILE_KEYS | {"children"} | ({"refine"} if is_root else set[str]())
    )
    _require(
        _TILE_KEYS <= set(entry) <= allowed,
        f"a tile entry carries unexpected or missing keys: {sorted(entry)}.",
    )
    error = _finite_error(entry["geometricError"], "a tile")
    _require(
        error <= parent_error,
        "a tile's geometricError exceeds its parent's.",
    )
    region = _verify_region(entry["boundingVolume"])
    if parent_region is not None:
        _require(
            _region_contains(parent_region, region),
            "a child tile's region is not contained in its parent's.",
        )
    content = entry["content"]
    _require(
        isinstance(content, dict)
        and set(cast("dict[str, Any]", content)) == {"uri"}
        and isinstance(cast("dict[str, Any]", content)["uri"], str),
        "a tile's content block is not exactly a string uri.",
    )
    flat.append((entry, [*ancestor_regions, region]))
    children = cast("list[Any]", entry.get("children", []))
    for child in children:
        _require(
            isinstance(child, dict),
            "a children list holds a non-object entry.",
        )
        _verify_entry(
            cast("dict[str, Any]", child),
            is_root=False,
            parent_error=error,
            parent_region=region,
            ancestor_regions=[*ancestor_regions, region],
            flat=flat,
        )
    return error


def _finite_error(value: object, owner: str) -> float:
    """Verify a geometricError is a finite non-negative number."""
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0,
        f"{owner} geometricError is not a finite non-negative number.",
    )
    return float(cast("float", value))


def _verify_region(volume: object) -> tuple[float, ...]:
    """Verify a boundingVolume is a valid 3D Tiles region."""
    _require(
        isinstance(volume, dict)
        and set(cast("dict[str, Any]", volume)) == {"region"},
        "a boundingVolume is not exactly a region volume.",
    )
    region_value = cast("dict[str, Any]", volume)["region"]
    _require(
        isinstance(region_value, list)
        and len(cast("list[Any]", region_value)) == _REGION_LENGTH
        and all(
            isinstance(v, (int, float)) and math.isfinite(float(v))
            for v in cast("list[Any]", region_value)
        ),
        "a region is not a list of six finite numbers.",
    )
    west, south, east, north, low, high = (
        float(v) for v in cast("list[Any]", region_value)
    )
    _require(
        -math.pi <= west < east <= math.pi
        and -math.pi / 2 <= south < north <= math.pi / 2
        and low <= high,
        "a region is degenerate or out of range.",
    )
    return (west, south, east, north, low, high)


def _region_contains(
    outer: tuple[float, ...], inner: tuple[float, ...]
) -> bool:
    """Return whether ``inner`` lies within ``outer``."""
    return (
        outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and inner[2] <= outer[2]
        and inner[3] <= outer[3]
        and outer[4] <= inner[4]
        and inner[5] <= outer[5]
    )


def _entry_uri(entry: dict[str, Any]) -> str:
    """Return the entry's content uri."""
    return cast("str", cast("dict[str, Any]", entry["content"])["uri"])


def _verify_links(
    out_dir: Path,
    flat: _Flat,
    tiles_by_uri: dict[str, TilePlan],
) -> None:
    """Verify uris are safe, planned, unique, present — no orphans."""
    seen: set[str] = set()
    for entry, _ in flat:
        uri = _entry_uri(entry)
        resolved = (out_dir / uri).resolve()
        _require(
            not uri.startswith("/")
            and resolved.is_relative_to(out_dir.resolve()),
            f"content uri {uri} escapes the output directory.",
        )
        _require(
            uri in tiles_by_uri,
            f"content uri {uri} does not correspond to any planned "
            "quadtree tile.",
        )
        _require(
            uri not in seen,
            f"content uri {uri} is referenced more than once.",
        )
        seen.add(uri)
        _require(
            (out_dir / uri).is_file(),
            f"missing content file: {uri}.",
        )
    on_disk = {
        f"{TILES_SUBDIR}/{path.name}"
        for path in (out_dir / TILES_SUBDIR).iterdir()
    }
    orphans = on_disk - seen
    _require(
        not orphans,
        f"tile files on disk are not referenced by tileset.json: "
        f"{sorted(orphans)}.",
    )


def _parse_glb(data: bytes, uri: str) -> tuple[dict[str, Any], bytes]:
    """Verify the glb container framing; return (glTF JSON, BIN)."""
    _require(
        len(data) >= 28,  # noqa: PLR2004 -- header + two chunk headers
        f"{uri}: glb is too short to be a container.",
    )
    magic, version, length = struct.unpack("<III", data[:12])
    _require(magic == _GLB_MAGIC, f"{uri}: bad glb magic.")
    _require(version == 2, f"{uri}: glb version is not 2.")  # noqa: PLR2004
    _require(
        length == len(data),
        f"{uri}: glb declared length does not equal the file size.",
    )
    json_length, json_kind = struct.unpack("<II", data[12:20])
    bin_header = 20 + json_length
    _require(
        json_kind == _CHUNK_JSON
        and json_length % 4 == 0
        and bin_header + 8 <= len(data),
        f"{uri}: glb chunks are not JSON-then-BIN.",
    )
    bin_length, bin_kind = struct.unpack(
        "<II", data[bin_header : bin_header + 8]
    )
    _require(
        bin_kind == _CHUNK_BIN and bin_header + 8 + bin_length == len(data),
        f"{uri}: glb BIN chunk does not exactly fill the container.",
    )
    try:
        gltf = json.loads(data[20:bin_header])
    except json.JSONDecodeError as exc:
        msg = f"{uri}: glb JSON chunk is not valid JSON: {exc}"
        raise Tiles3dError(msg) from exc
    return cast("dict[str, Any]", gltf), data[bin_header + 8 :]


def _view_bytes(
    gltf: dict[str, Any], binary: bytes, index: int, uri: str
) -> bytes:
    """Return a bufferView's bytes after bounds-checking it."""
    view = cast("dict[str, Any]", gltf["bufferViews"][index])
    offset = int(view["byteOffset"])
    length = int(view["byteLength"])
    _require(
        offset >= 0 and offset + length <= len(binary),
        f"{uri}: bufferView {index} overruns the BIN chunk.",
    )
    return binary[offset : offset + length]


def _verify_gltf(gltf: dict[str, Any], binary: bytes, uri: str) -> None:
    """Verify the glTF document's internal consistency and payloads."""
    buffers = cast("list[dict[str, Any]]", gltf["buffers"])
    _require(
        len(buffers) == 1 and int(buffers[0]["byteLength"]) == len(binary),
        f"{uri}: buffer byteLength does not equal the BIN chunk size.",
    )
    accessors = cast("list[dict[str, Any]]", gltf["accessors"])
    positions = np.frombuffer(
        _view_bytes(gltf, binary, 0, uri), dtype="<f4"
    ).reshape(-1, 3)
    uvs = np.frombuffer(
        _view_bytes(gltf, binary, 1, uri), dtype="<f4"
    ).reshape(-1, 2)
    indices = np.frombuffer(_view_bytes(gltf, binary, 2, uri), dtype="<u4")
    _require(
        int(accessors[0]["count"]) == positions.shape[0]
        and int(accessors[1]["count"]) == uvs.shape[0]
        and int(accessors[2]["count"]) == indices.shape[0],
        f"{uri}: accessor counts do not match their bufferViews.",
    )
    stored_min = [float(v) for v in accessors[0]["min"]]
    stored_max = [float(v) for v in accessors[0]["max"]]
    actual_min = [float(v) for v in positions.min(axis=0)]
    actual_max = [float(v) for v in positions.max(axis=0)]
    _require(
        stored_min == actual_min and stored_max == actual_max,
        f"{uri}: POSITION min/max do not equal the payload extremes.",
    )
    _require(
        indices.shape[0] % _TRIANGLE == 0,
        f"{uri}: index count is not a multiple of 3.",
    )
    _require(
        bool(indices.max() < positions.shape[0]),
        f"{uri}: an index is out of range of the vertex count.",
    )
    triangles = indices.reshape(-1, _TRIANGLE)
    distinct = (
        (triangles[:, 0] != triangles[:, 1])
        & (triangles[:, 1] != triangles[:, 2])
        & (triangles[:, 0] != triangles[:, 2])
    )
    _require(
        bool(distinct.all()),
        f"{uri}: a triangle references a vertex twice (degenerate).",
    )
    _require(
        bool((uvs >= 0.0).all()) and bool((uvs <= 1.0).all()),
        f"{uri}: a texture coordinate is outside [0, 1].",
    )


def _verify_texture(
    gltf: dict[str, Any],
    binary: bytes,
    terrain: TerrainGrid,
    tile: TilePlan,
    uri: str,
) -> None:
    """Verify the embedded PNG decodes and equals the sampled ortho."""
    image_view = int(
        cast("list[dict[str, Any]]", gltf["images"])[0]["bufferView"]
    )
    pixels = decode_png(_view_bytes(gltf, binary, image_view, uri))
    cols = sample_indices(tile.col0, tile.col1, tile.stride)
    rows = sample_indices(tile.row0, tile.row1, tile.stride)
    _require(
        pixels.shape == (len(rows), len(cols), 3),
        f"{uri}: texture dimensions do not equal the tile's sample counts.",
    )
    expected = terrain.rgb[np.ix_(rows, cols)]
    _require(
        bool(np.array_equal(pixels, expected)),
        f"{uri}: texture pixels do not equal the sampled orthophoto.",
    )


def _verify_containment(
    terrain: TerrainGrid,
    tile: TilePlan,
    enclosing_regions: list[tuple[float, ...]],
    geodesy: Geodesy,
    uri: str,
) -> None:
    """Verify the tile's vertices lie inside every enclosing region.

    ``enclosing_regions`` holds the stored regions of every ancestor
    plus the tile's own; the vertex geodetics are recomputed from the
    sources, so a shrunken or drifted stored region is caught here.
    """
    cols = sample_indices(tile.col0, tile.col1, tile.stride)
    rows = sample_indices(tile.row0, tile.row1, tile.stride)
    grid = np.ix_(rows, cols)
    lon, lat, height = geodesy.to_geodetic_radians(
        terrain.x[grid].astype(np.float64).ravel(),
        terrain.y[grid].astype(np.float64).ravel(),
        terrain.z[grid].astype(np.float64).ravel(),
    )
    for region in enclosing_regions:
        _require(
            _points_within(lon, lat, height, region),
            f"{uri}: a vertex lies outside an enclosing region.",
        )


def _points_within(
    lon: npt.NDArray[np.float64],
    lat: npt.NDArray[np.float64],
    height: npt.NDArray[np.float64],
    region: tuple[float, ...],
) -> bool:
    """Return whether every point lies inside the region."""
    return (
        bool((lon >= region[0]).all())
        and bool((lat >= region[1]).all())
        and bool((lon <= region[2]).all())
        and bool((lat <= region[3]).all())
        and bool((height >= region[4]).all())
        and bool((height <= region[5]).all())
    )


def _verify_byte_identity(out_dir: Path, computed: ComputedBuild) -> None:
    """Verify every artifact byte-equals its independent rebuild."""
    for uri, expected in computed.glbs.items():
        _require(
            (out_dir / uri).read_bytes() == expected,
            f"{uri} does not byte-equal its independent rebuild.",
        )
    _require(
        (out_dir / TILESET_NAME).read_bytes()
        == render_tileset(computed.document).encode("utf-8"),
        "tileset.json does not byte-equal its independent rebuild.",
    )
