"""Tests for the strictest post-write verifier.

Each negative test builds a valid tileset, corrupts exactly the bytes
one check guards, and asserts that check's message.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import struct
from typing import TYPE_CHECKING, Any, cast

import pytest

import ahn_cli.tiles3d.verify as verify_module
from ahn_cli.tiles3d.build import build_tiles3d
from ahn_cli.tiles3d.errors import Tiles3dError
from ahn_cli.tiles3d.geodesy import Geodesy
from ahn_cli.tiles3d.manifest import FileDigest, render_manifest
from ahn_cli.tiles3d.pack import read_pack
from ahn_cli.tiles3d.png import encode_png
from ahn_cli.tiles3d.profile import Profile
from ahn_cli.tiles3d.quadtree import TreePlan, plan_quadtree
from ahn_cli.tiles3d.verify import verify_tiles3d
from tests.tiles3d.conftest import (
    grid_for_ortho,
    make_ortho,
    synth_rgb,
    write_exr,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def site(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Build a valid two-level tileset; return (out, ortho, heights)."""
    rgb = synth_rgb(20, 14, seed=13)
    ortho = make_ortho(tmp_path / "ortho.tif", rgb)
    heights = write_exr(tmp_path / "r.exr", grid_for_ortho(rgb))
    out = tmp_path / "out"
    build_tiles3d(ortho, heights, out, tile_pixels=8)
    return out, ortho, heights


def _verify(site: tuple[Path, Path, Path]) -> None:
    out, ortho, heights = site
    verify_tiles3d(out, ortho, heights, tile_pixels=8)


def test_heightfield_containment_compares_nap_not_ellipsoidal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Heightfield containment compares NAP vertex heights against NAP regions.

    Regression for the r3 verifier-datum blocker: on a machine WITH the
    NLGEO2018 geoid grid, ``to_geodetic_radians`` returns ellipsoidal heights
    ~43 m above NAP. The heightfield profile stores NAP regions, so if
    ``_verify_containment`` compared the geodesy (ellipsoidal) height against
    those NAP regions, every vertex would exceed the region by the undulation
    and the profile would be **unbuildable** where the grid is installed. CI
    has no grid (undulation ~= 0), so we force a large one on the geodesy
    height to prove the check is grid-independent.

    The mock offsets only the *height* return, which the NAP-native heightfield
    path ignores everywhere (`nap_region` reads `payload.z`; `rtc_centre` uses
    `to_ecef`), so the produced bytes are unchanged and the only code path the
    offset can reach is the containment check. ``build_tiles3d`` runs the full
    verifier (incl. containment + the byte-identity backstop) as its final
    step, so a green build here is the guard: pre-fix this raised
    ``Tiles3dError`` and deleted the deliverable.
    """
    real = Geodesy.to_geodetic_radians

    def offset_height(
        self: Geodesy,
        x: object,
        y: object,
        z: object,
    ) -> tuple[object, object, object]:
        lon, lat, height = real(self, x, y, z)  # type: ignore[arg-type]
        return lon, lat, height + 43.0

    monkeypatch.setattr(Geodesy, "to_geodetic_radians", offset_height)

    rgb = synth_rgb(20, 14, seed=13)
    ortho = make_ortho(tmp_path / "ortho.tif", rgb)
    heights = write_exr(tmp_path / "r.exr", grid_for_ortho(rgb))
    out = tmp_path / "hf"
    build_tiles3d(
        ortho, heights, out, tile_pixels=8, profile=Profile.HEIGHTFIELD
    )


def _load(site: tuple[Path, Path, Path]) -> dict[str, Any]:
    return cast(
        "dict[str, Any]",
        json.loads((site[0] / "tileset.json").read_text()),
    )


def _dump(site: tuple[Path, Path, Path], document: object) -> None:
    rendered = json.dumps(document, sort_keys=True, indent=2) + "\n"
    (site[0] / "tileset.json").write_text(rendered)


def _split_glb(data: bytes) -> tuple[dict[str, Any], bytes]:
    json_length = struct.unpack("<I", data[12:16])[0]
    document = cast("dict[str, Any]", json.loads(data[20 : 20 + json_length]))
    return document, data[28 + json_length :]


def _join_glb(document: dict[str, Any], binary: bytes) -> bytes:
    payload = json.dumps(
        document, sort_keys=True, separators=(",", ":")
    ).encode()
    payload += b" " * (-len(payload) % 4)
    length = 12 + 8 + len(payload) + 8 + len(binary)
    return (
        struct.pack("<III", 0x46546C67, 2, length)
        + struct.pack("<II", len(payload), 0x4E4F534A)
        + payload
        + struct.pack("<II", len(binary), 0x004E4942)
        + binary
    )


def _leaf_glb(site: tuple[Path, Path, Path]) -> Path:
    return site[0] / "tiles" / "2-0-0.glb"


def _patch_bin(
    path: Path, view_index: int, byte_offset: int, new_bytes: bytes
) -> None:
    """Patch bytes inside one bufferView of a glb, in place."""
    document, binary = _split_glb(path.read_bytes())
    view = document["bufferViews"][view_index]
    start = view["byteOffset"] + byte_offset
    patched = binary[:start] + new_bytes + binary[start + len(new_bytes) :]
    path.write_bytes(_join_glb(document, patched))


def _replace_png(path: Path, png: bytes) -> None:
    """Replace a glb's PNG section, rebuilding the BIN layout."""
    document, binary = _split_glb(path.read_bytes())
    views = document["bufferViews"]
    sections = [
        binary[v["byteOffset"] : v["byteOffset"] + v["byteLength"]]
        for v in views
    ]
    sections[3] = png
    rebuilt = bytearray()
    for view, section in zip(views, sections, strict=True):
        view["byteOffset"] = len(rebuilt)
        view["byteLength"] = len(section)
        rebuilt.extend(section)
        rebuilt.extend(b"\x00" * (-len(rebuilt) % 4))
    document["buffers"][0]["byteLength"] = len(rebuilt)
    path.write_bytes(_join_glb(document, bytes(rebuilt)))


def test_a_fresh_build_verifies(site: tuple[Path, Path, Path]) -> None:
    """The verifier accepts what the builder just wrote."""
    _verify(site)


def test_missing_tileset(site: tuple[Path, Path, Path]) -> None:
    """A deleted tileset.json is refused."""
    (site[0] / "tileset.json").unlink()
    with pytest.raises(Tiles3dError, match="not readable"):
        _verify(site)


def test_invalid_json(site: tuple[Path, Path, Path]) -> None:
    """A non-JSON tileset is refused."""
    (site[0] / "tileset.json").write_text("nope{")
    with pytest.raises(Tiles3dError, match="not valid JSON"):
        _verify(site)


def test_non_object_tileset(site: tuple[Path, Path, Path]) -> None:
    """A JSON array is not a tileset."""
    (site[0] / "tileset.json").write_text("[]")
    with pytest.raises(Tiles3dError, match="JSON object"):
        _verify(site)


def test_unexpected_top_level_key(
    site: tuple[Path, Path, Path],
) -> None:
    """An unknown top-level key is refused (we authored this file)."""
    document = _load(site)
    document["extensionsUsed"] = []
    _dump(site, document)
    with pytest.raises(Tiles3dError, match="top-level"):
        _verify(site)


def test_wrong_asset_version(site: tuple[Path, Path, Path]) -> None:
    """asset.version must be exactly 1.1."""
    document = _load(site)
    document["asset"]["version"] = "1.0"
    _dump(site, document)
    with pytest.raises(Tiles3dError, match="asset"):
        _verify(site)


def test_missing_root_refine(site: tuple[Path, Path, Path]) -> None:
    """The root tile must declare REPLACE refinement."""
    document = _load(site)
    del document["root"]["refine"]
    _dump(site, document)
    with pytest.raises(Tiles3dError, match="REPLACE"):
        _verify(site)


def test_negative_geometric_error(
    site: tuple[Path, Path, Path],
) -> None:
    """A negative geometricError is refused."""
    document = _load(site)
    document["root"]["geometricError"] = -1.0
    _dump(site, document)
    with pytest.raises(Tiles3dError, match="finite non-negative"):
        _verify(site)


def test_tileset_error_below_root(
    site: tuple[Path, Path, Path],
) -> None:
    """The tileset geometricError must be >= the root's."""
    document = _load(site)
    document["geometricError"] = 0.001
    _dump(site, document)
    with pytest.raises(Tiles3dError, match="below the root"):
        _verify(site)


def test_child_error_above_parent(
    site: tuple[Path, Path, Path],
) -> None:
    """A child's geometricError must not exceed its parent's."""
    document = _load(site)
    document["root"]["children"][0]["geometricError"] = 999.0
    _dump(site, document)
    with pytest.raises(Tiles3dError, match="exceeds its parent"):
        _verify(site)


def test_unexpected_tile_key(site: tuple[Path, Path, Path]) -> None:
    """A tile entry with an unknown key is refused."""
    document = _load(site)
    document["root"]["children"][0]["extras"] = {}
    _dump(site, document)
    with pytest.raises(Tiles3dError, match="unexpected or missing"):
        _verify(site)


def test_degenerate_region(site: tuple[Path, Path, Path]) -> None:
    """A region with west >= east is refused."""
    document = _load(site)
    region = document["root"]["boundingVolume"]["region"]
    region[0], region[2] = region[2], region[0]
    _dump(site, document)
    with pytest.raises(Tiles3dError, match="degenerate or out of"):
        _verify(site)


def test_non_region_bounding_volume(
    site: tuple[Path, Path, Path],
) -> None:
    """A box boundingVolume is not our format."""
    document = _load(site)
    document["root"]["boundingVolume"] = {"box": [0.0] * 12}
    _dump(site, document)
    with pytest.raises(Tiles3dError, match="region volume"):
        _verify(site)


def test_child_region_not_contained(
    site: tuple[Path, Path, Path],
) -> None:
    """A parent region that does not enclose a child is refused."""
    document = _load(site)
    region = document["root"]["boundingVolume"]["region"]
    region[2] = region[0] + 1e-9  # shrink east almost to west
    _dump(site, document)
    with pytest.raises(Tiles3dError, match="not contained"):
        _verify(site)


def test_content_with_extra_key(
    site: tuple[Path, Path, Path],
) -> None:
    """A content block with more than a uri is refused."""
    document = _load(site)
    document["root"]["content"]["boundingVolume"] = {}
    _dump(site, document)
    with pytest.raises(Tiles3dError, match="content block"):
        _verify(site)


def test_uri_escaping_the_output_dir(
    site: tuple[Path, Path, Path],
) -> None:
    """A uri pointing outside the output directory is refused."""
    document = _load(site)
    document["root"]["content"]["uri"] = "../evil.glb"
    _dump(site, document)
    with pytest.raises(Tiles3dError, match="escapes"):
        _verify(site)


def test_uri_not_matching_a_planned_tile(
    site: tuple[Path, Path, Path],
) -> None:
    """A uri that maps to no quadtree tile is refused."""
    document = _load(site)
    document["root"]["content"]["uri"] = "tiles/9-9-9.glb"
    _dump(site, document)
    with pytest.raises(Tiles3dError, match="planned"):
        _verify(site)


def test_duplicate_uri(site: tuple[Path, Path, Path]) -> None:
    """Two entries sharing one uri are refused."""
    document = _load(site)
    children = document["root"]["children"]
    children[1]["content"]["uri"] = children[0]["content"]["uri"]
    _dump(site, document)
    with pytest.raises(Tiles3dError, match="more than once"):
        _verify(site)


def test_missing_content_file(site: tuple[Path, Path, Path]) -> None:
    """A referenced glb missing from disk is refused."""
    _leaf_glb(site).unlink()
    with pytest.raises(Tiles3dError, match="missing content file"):
        _verify(site)


def test_orphan_tile_file(site: tuple[Path, Path, Path]) -> None:
    """An unreferenced file under tiles/ is refused."""
    (site[0] / "tiles" / "stray.glb").write_bytes(b"stray")
    with pytest.raises(Tiles3dError, match="not referenced"):
        _verify(site)


def test_bad_glb_magic(site: tuple[Path, Path, Path]) -> None:
    """A glb without the glTF magic is refused."""
    path = _leaf_glb(site)
    data = bytearray(path.read_bytes())
    data[:4] = b"XXXX"
    path.write_bytes(bytes(data))
    with pytest.raises(Tiles3dError, match="magic"):
        _verify(site)


def test_bad_glb_version(site: tuple[Path, Path, Path]) -> None:
    """A glb version other than 2 is refused."""
    path = _leaf_glb(site)
    data = bytearray(path.read_bytes())
    data[4:8] = struct.pack("<I", 3)
    path.write_bytes(bytes(data))
    with pytest.raises(Tiles3dError, match="version is not 2"):
        _verify(site)


def test_truncated_glb(site: tuple[Path, Path, Path]) -> None:
    """A glb whose declared length disagrees with the file is refused."""
    path = _leaf_glb(site)
    path.write_bytes(path.read_bytes()[:-4])
    with pytest.raises(Tiles3dError, match="declared length"):
        _verify(site)


def test_wrong_chunk_kind(site: tuple[Path, Path, Path]) -> None:
    """A glb whose first chunk is not JSON is refused."""
    path = _leaf_glb(site)
    data = bytearray(path.read_bytes())
    data[16:20] = b"XXXX"
    path.write_bytes(bytes(data))
    with pytest.raises(Tiles3dError, match="JSON-then-BIN"):
        _verify(site)


def test_invalid_gltf_json_chunk(
    site: tuple[Path, Path, Path],
) -> None:
    """A JSON chunk that does not parse is refused."""
    path = _leaf_glb(site)
    data = bytearray(path.read_bytes())
    json_length = struct.unpack("<I", bytes(data[12:16]))[0]
    data[20 : 20 + json_length] = b"{" + b" " * (json_length - 1)
    path.write_bytes(bytes(data))
    with pytest.raises(Tiles3dError, match="not valid JSON"):
        _verify(site)


def test_position_extremes_mismatch(
    site: tuple[Path, Path, Path],
) -> None:
    """A stored POSITION min that is not the payload's is refused."""
    path = _leaf_glb(site)
    document, binary = _split_glb(path.read_bytes())
    document["accessors"][0]["min"][0] -= 1.0
    path.write_bytes(_join_glb(document, binary))
    with pytest.raises(Tiles3dError, match="POSITION min/max"):
        _verify(site)


def test_buffer_length_mismatch(
    site: tuple[Path, Path, Path],
) -> None:
    """A buffer byteLength disagreeing with the BIN chunk is refused."""
    path = _leaf_glb(site)
    document, binary = _split_glb(path.read_bytes())
    document["buffers"][0]["byteLength"] += 4
    path.write_bytes(_join_glb(document, binary))
    with pytest.raises(Tiles3dError, match="buffer byteLength"):
        _verify(site)


def test_buffer_view_overrun(site: tuple[Path, Path, Path]) -> None:
    """A bufferView reaching beyond the buffer is refused."""
    path = _leaf_glb(site)
    document, binary = _split_glb(path.read_bytes())
    document["bufferViews"][0]["byteOffset"] = len(binary)
    path.write_bytes(_join_glb(document, binary))
    with pytest.raises(Tiles3dError, match="overruns"):
        _verify(site)


def test_accessor_count_mismatch(
    site: tuple[Path, Path, Path],
) -> None:
    """An accessor count disagreeing with its view is refused."""
    path = _leaf_glb(site)
    document, binary = _split_glb(path.read_bytes())
    document["accessors"][2]["count"] -= 3
    path.write_bytes(_join_glb(document, binary))
    with pytest.raises(Tiles3dError, match="accessor counts"):
        _verify(site)


def test_out_of_range_index(site: tuple[Path, Path, Path]) -> None:
    """An index beyond the vertex count is refused."""
    _patch_bin(_leaf_glb(site), 2, 0, struct.pack("<I", 10_000_000))
    with pytest.raises(Tiles3dError, match="out of range"):
        _verify(site)


def test_degenerate_triangle(site: tuple[Path, Path, Path]) -> None:
    """A triangle repeating a vertex is refused."""
    path = _leaf_glb(site)
    document, binary = _split_glb(path.read_bytes())
    view = document["bufferViews"][2]
    start = view["byteOffset"]
    second = binary[start + 4 : start + 8]
    _patch_bin(path, 2, 0, second)
    with pytest.raises(Tiles3dError, match="degenerate"):
        _verify(site)


def test_uv_out_of_range(site: tuple[Path, Path, Path]) -> None:
    """A texture coordinate outside [0, 1] is refused."""
    _patch_bin(_leaf_glb(site), 1, 0, struct.pack("<f", 1.5))
    with pytest.raises(Tiles3dError, match=r"\[0, 1\]"):
        _verify(site)


def test_corrupt_png_crc(site: tuple[Path, Path, Path]) -> None:
    """A flipped byte in the texture PNG fails its CRC check."""
    path = _leaf_glb(site)
    document, binary = _split_glb(path.read_bytes())
    offset = int(document["bufferViews"][3]["byteOffset"]) + 20
    flip = binary[offset] ^ 0xFF
    _patch_bin(path, 3, 20, bytes([flip]))
    with pytest.raises(Tiles3dError, match="CRC"):
        _verify(site)


def test_texture_dimension_mismatch(
    site: tuple[Path, Path, Path],
) -> None:
    """A texture of the wrong size is refused."""
    _replace_png(_leaf_glb(site), encode_png(synth_rgb(2, 2)))
    with pytest.raises(Tiles3dError, match="texture dimensions"):
        _verify(site)


def test_texture_pixel_mismatch(
    site: tuple[Path, Path, Path],
) -> None:
    """A texture that is not the sampled ortho is refused."""
    tree = plan_quadtree(20, 14, tile_pixels=8)
    leaf = tree.root.children[0].children[0]
    wrong = synth_rgb(
        leaf.col1 - leaf.col0 + 1, leaf.row1 - leaf.row0 + 1, seed=99
    )
    _replace_png(_leaf_glb(site), encode_png(wrong))
    with pytest.raises(Tiles3dError, match="texture pixels"):
        _verify(site)


def test_shrunken_own_region(site: tuple[Path, Path, Path]) -> None:
    """A stored region that no longer encloses its vertices is refused."""
    document = _load(site)
    leaf = document["root"]["children"][0]["children"][0]
    region = leaf["boundingVolume"]["region"]
    region[5] = region[4] + 1e-12  # crush the height range
    _dump(site, document)
    with pytest.raises(Tiles3dError, match="outside an enclosing"):
        _verify(site)


def test_payload_bit_flip_fails_byte_identity(
    site: tuple[Path, Path, Path],
) -> None:
    """A mid-range position perturbation is caught by byte identity."""
    path = _leaf_glb(site)
    document, binary = _split_glb(path.read_bytes())
    view = document["bufferViews"][0]
    middle = view["byteOffset"] + (view["byteLength"] // 6) * 3
    offset = middle - view["byteOffset"]
    original = struct.unpack_from("<f", binary, middle)[0]
    nudged = struct.pack("<f", original + 0.0005)
    _patch_bin(path, 0, offset, nudged)
    with pytest.raises(Tiles3dError, match="byte-equal"):
        _verify(site)


def test_reserialized_tileset_fails_byte_identity(
    site: tuple[Path, Path, Path],
) -> None:
    """A semantically equal but re-rendered tileset.json is refused."""
    document = _load(site)
    (site[0] / "tileset.json").write_text(json.dumps(document))
    with pytest.raises(
        Tiles3dError, match=r"tileset\.json does not byte-equal"
    ):
        _verify(site)


def test_uncovered_pixel_is_refused(
    site: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A quadtree that misses a pixel is refused before anything else."""
    real_tree = plan_quadtree(20, 14, tile_pixels=8)
    pruned_root = dataclasses.replace(
        real_tree.root, children=real_tree.root.children[:3]
    )
    pruned = dataclasses.replace(real_tree, root=pruned_root)

    def pruned_plan(*_args: object, **_kwargs: object) -> TreePlan:
        return pruned

    monkeypatch.setattr(verify_module, "plan_quadtree", pruned_plan)
    with pytest.raises(Tiles3dError, match="do not cover"):
        _verify(site)


@pytest.fixture
def game_site(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Build a valid two-level game tileset; return (out, ortho, heights)."""
    rgb = synth_rgb(20, 14, seed=13)
    ortho = make_ortho(tmp_path / "ortho.tif", rgb)
    heights = write_exr(tmp_path / "r.exr", grid_for_ortho(rgb))
    out = tmp_path / "out"
    build_tiles3d(ortho, heights, out, tile_pixels=8, profile=Profile.GAME)
    return out, ortho, heights


def test_game_build_passes_the_game_verifier(
    game_site: tuple[Path, Path, Path],
) -> None:
    """A pristine game build re-verifies under the game profile."""
    out, ortho, heights = game_site
    verify_tiles3d(out, ortho, heights, tile_pixels=8, profile=Profile.GAME)


def test_game_provenance_corruption_is_refused(
    game_site: tuple[Path, Path, Path],
) -> None:
    """A tampered provenance.json fails the game byte-identity backstop.

    The manifest is rewritten to describe the tampered provenance (so the
    manifest-recompute check, which runs earlier, still passes) — proving the
    byte-identity backstop rejects a provenance that drifts from the source
    rebuild even when the deliverable is internally self-consistent.
    """
    out, ortho, heights = game_site
    tampered = b"{}\n"
    (out / "provenance.json").write_bytes(tampered)
    dataset_id = read_pack(out / "tiles.hfp").header.dataset_id.hex()
    files = {
        name: FileDigest(
            sha256=hashlib.sha256(data).hexdigest(), size=len(data)
        )
        for name in ("tileset.json", "provenance.json", "tiles.hfp")
        for data in ((out / name).read_bytes(),)
    }
    (out / "manifest.json").write_bytes(
        render_manifest(files, dataset_id).encode("utf-8")
    )
    with pytest.raises(
        Tiles3dError, match="provenance.json does not byte-equal"
    ):
        verify_tiles3d(
            out, ortho, heights, tile_pixels=8, profile=Profile.GAME
        )
