"""Tests for the game-profile per-tile verification families.

Each negative test builds a valid game tileset, corrupts exactly the
bytes one check guards through the real build-then-corrupt-then-verify
path, and asserts that check's message — proving each family fires
independently of the whole-file byte-identity backstop.
"""

from __future__ import annotations

import io
import json
import struct
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest
from PIL import Image

import ahn_cli.tiles3d.build as build_module
import ahn_cli.tiles3d.verify_game as verify_game_module
from ahn_cli.tiles3d.build import build_tiles3d
from ahn_cli.tiles3d.errors import Tiles3dError
from ahn_cli.tiles3d.meshopt import decode_positions, encode_positions
from ahn_cli.tiles3d.profile import Profile
from ahn_cli.tiles3d.verify import verify_tiles3d
from tests.tiles3d.conftest import (
    grid_for_ortho,
    make_ortho,
    synth_rgb,
    write_exr,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import numpy.typing as npt

_EXT_MESHOPT = "EXT_meshopt_compression"


@pytest.fixture
def game_site(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Build a valid two-level game tileset; return (out, ortho, heights)."""
    rgb = synth_rgb(20, 14, seed=13)
    ortho = make_ortho(tmp_path / "ortho.tif", rgb)
    heights = write_exr(tmp_path / "r.exr", grid_for_ortho(rgb))
    out = tmp_path / "out"
    build_tiles3d(ortho, heights, out, tile_pixels=8, profile=Profile.GAME)
    return out, ortho, heights


def _verify(site: tuple[Path, Path, Path]) -> None:
    out, ortho, heights = site
    verify_tiles3d(out, ortho, heights, tile_pixels=8, profile=Profile.GAME)


def _leaf(site: tuple[Path, Path, Path]) -> Path:
    return site[0] / "tiles" / "2-0-0.glb"


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


def _ext(document: dict[str, Any], view_index: int) -> dict[str, Any]:
    view = cast("dict[str, Any]", document["bufferViews"][view_index])
    return cast("dict[str, Any]", view["extensions"][_EXT_MESHOPT])


def _sections(document: dict[str, Any], binary: bytes) -> list[bytes]:
    """Extract the [positions, uvs, indices, jpeg] BIN sections."""
    sections: list[bytes] = []
    for view_index in range(3):
        ext = _ext(document, view_index)
        start = int(ext["byteOffset"])
        sections.append(binary[start : start + int(ext["byteLength"])])
    image = cast("dict[str, Any]", document["bufferViews"][3])
    start = int(image["byteOffset"])
    sections.append(binary[start : start + int(image["byteLength"])])
    return sections


def _rebuild_bin(document: dict[str, Any], sections: list[bytes]) -> bytes:
    """Re-lay the four BIN sections (4-byte aligned) and fix all offsets."""
    binary = bytearray()
    offsets: list[int] = []
    for section in sections:
        offsets.append(len(binary))
        binary.extend(section)
        binary.extend(b"\x00" * (-len(binary) % 4))
    for view_index in range(3):
        ext = _ext(document, view_index)
        ext["byteOffset"] = offsets[view_index]
        ext["byteLength"] = len(sections[view_index])
    image = cast("dict[str, Any]", document["bufferViews"][3])
    image["byteOffset"] = offsets[3]
    image["byteLength"] = len(sections[3])
    document["buffers"][0]["byteLength"] = len(binary)
    return bytes(binary)


def test_pristine_game_build_verifies(
    game_site: tuple[Path, Path, Path],
) -> None:
    """The game verifier accepts what the game builder just wrote."""
    _verify(game_site)


def test_meshopt_stream_bit_flip_is_refused(
    game_site: tuple[Path, Path, Path],
) -> None:
    """A flipped byte in the TEXCOORD_0 meshopt stream is caught."""
    path = _leaf(game_site)
    document, binary = _split_glb(path.read_bytes())
    ext = _ext(document, 1)  # TEXCOORD_0 stream: not the POSITION stream
    flip = int(ext["byteOffset"]) + int(ext["byteLength"]) // 2
    data = bytearray(binary)
    data[flip] ^= 0xFF
    path.write_bytes(_join_glb(document, bytes(data)))
    with pytest.raises(
        Tiles3dError, match=r"TEXCOORD_0 meshopt stream|decode"
    ):
        _verify(game_site)


def test_off_by_one_quantized_int_fires_requantization(
    game_site: tuple[Path, Path, Path],
) -> None:
    """An off-by-one quantized int fires requantization, not the backstop.

    The stream is fully re-packed (valid meshopt, offsets fixed) and the
    accessor extremes are left untouched, so only the requantization
    family — not the byte-identity backstop — attributes the failure.
    """
    path = _leaf(game_site)
    document, binary = _split_glb(path.read_bytes())
    sections = _sections(document, binary)
    count = int(_ext(document, 0)["count"])
    ints = decode_positions(sections[0], count).copy()
    middle = count // 2
    ints[middle, 0] = (int(ints[middle, 0]) + 1) % 65536  # off-by-one
    sections[0] = encode_positions(ints).data
    path.write_bytes(_join_glb(document, _rebuild_bin(document, sections)))
    with pytest.raises(
        Tiles3dError, match="does not equal the independent requantization"
    ):
        _verify(game_site)


def test_jpeg_recompressed_at_a_different_quality_is_refused(
    game_site: tuple[Path, Path, Path],
) -> None:
    """A texture re-encoded at another quality fails JPEG byte-equality."""
    path = _leaf(game_site)
    document, binary = _split_glb(path.read_bytes())
    sections = _sections(document, binary)
    with Image.open(io.BytesIO(sections[3])) as image:
        rgb = np.array(image.convert("RGB"))
    buffer = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buffer, format="JPEG", quality=60)
    sections[3] = buffer.getvalue()
    path.write_bytes(_join_glb(document, _rebuild_bin(document, sections)))
    with pytest.raises(Tiles3dError, match="does not byte-equal"):
        _verify(game_site)


def test_progressive_jpeg_is_refused(
    game_site: tuple[Path, Path, Path],
) -> None:
    """A progressive-framed texture fails the baseline JPEG check."""
    path = _leaf(game_site)
    document, binary = _split_glb(path.read_bytes())
    sections = _sections(document, binary)
    with Image.open(io.BytesIO(sections[3])) as image:
        rgb = np.array(image.convert("RGB"))
    buffer = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(
        buffer, format="JPEG", progressive=True
    )
    sections[3] = buffer.getvalue()
    path.write_bytes(_join_glb(document, _rebuild_bin(document, sections)))
    with pytest.raises(Tiles3dError, match="baseline sequential JPEG"):
        _verify(game_site)


def test_dropped_extension_declaration_is_refused(
    game_site: tuple[Path, Path, Path],
) -> None:
    """Dropping EXT_meshopt_compression from extensionsRequired is caught."""
    path = _leaf(game_site)
    document, binary = _split_glb(path.read_bytes())
    document["extensionsRequired"] = ["KHR_mesh_quantization"]
    path.write_bytes(_join_glb(document, binary))
    with pytest.raises(
        Tiles3dError, match="extensionsUsed/extensionsRequired"
    ):
        _verify(game_site)


def test_wrong_node_scale_is_refused(
    game_site: tuple[Path, Path, Path],
) -> None:
    """A node scale that is not the dequant fold is caught."""
    path = _leaf(game_site)
    document, binary = _split_glb(path.read_bytes())
    document["nodes"][0]["scale"][0] *= 2.0
    path.write_bytes(_join_glb(document, binary))
    with pytest.raises(Tiles3dError, match="dequantization fold"):
        _verify(game_site)


def test_fallback_buffer_with_a_uri_is_refused(
    game_site: tuple[Path, Path, Path],
) -> None:
    """A fallback buffer given a uri is caught."""
    path = _leaf(game_site)
    document, binary = _split_glb(path.read_bytes())
    document["buffers"][1]["uri"] = "leak.bin"
    path.write_bytes(_join_glb(document, binary))
    with pytest.raises(Tiles3dError, match="fallback buffer"):
        _verify(game_site)


def test_non_dict_stream_extensions_is_refused(
    game_site: tuple[Path, Path, Path],
) -> None:
    """A stream bufferView whose extensions is not an object is caught."""
    path = _leaf(game_site)
    document, binary = _split_glb(path.read_bytes())
    document["bufferViews"][0]["extensions"] = []
    path.write_bytes(_join_glb(document, binary))
    with pytest.raises(Tiles3dError, match="bufferView 0 framing"):
        _verify(game_site)


def test_shrunken_region_fails_dequantized_containment(
    game_site: tuple[Path, Path, Path],
) -> None:
    """A stored region crushed below the geometry fails containment."""
    out = game_site[0]
    document = cast(
        "dict[str, Any]", json.loads((out / "tileset.json").read_text())
    )
    leaf = document["root"]["children"][0]["children"][0]
    region = leaf["boundingVolume"]["region"]
    region[5] = region[4] + 1e-9  # crush the height range
    (out / "tileset.json").write_text(
        json.dumps(document, sort_keys=True, indent=2) + "\n"
    )
    with pytest.raises(
        Tiles3dError, match="lies outside an enclosing region"
    ):
        _verify(game_site)


def test_dequant_bound_is_enforced(
    game_site: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A zeroed error bound rejects the (genuine) dequantized positions.

    The dequant-bound assert guards the exported
    ``position_error_bound`` contract; shrinking the bound to zero proves
    the verifier would reject any position that ever exceeded it.
    """

    def zero_bound(
        _scale: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        return (0.0, 0.0, 0.0)

    monkeypatch.setattr(
        verify_game_module, "position_error_bound", zero_bound
    )
    with pytest.raises(
        Tiles3dError, match="exceeds the documented quantization error bound"
    ):
        _verify(game_site)


def _perturbing_encode_jpeg(
    real: Callable[[npt.NDArray[np.uint8]], bytes],
) -> Callable[[npt.NDArray[np.uint8]], bytes]:
    def encode(rgb: npt.NDArray[np.uint8]) -> bytes:
        perturbed = (rgb.astype(np.int32) ^ 3).astype(np.uint8)
        return real(perturbed)

    return encode


def test_rejected_game_rebuild_restores_the_previous_deliverable(
    game_site: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A game rebuild the verifier rejects restores the old build whole.

    Only the verifier's JPEG re-encode is perturbed, so the build writes
    a correct game tile that its own verification then refuses — the swap
    machinery must remove the rejected write and restore the held-aside
    previous deliverable, provenance.json included.
    """
    out, ortho, heights = game_site
    good = {
        p.relative_to(out): p.read_bytes()
        for p in out.rglob("*")
        if p.is_file()
    }
    monkeypatch.setattr(
        verify_game_module,
        "encode_jpeg",
        _perturbing_encode_jpeg(verify_game_module.encode_jpeg),
    )
    with pytest.raises(Tiles3dError, match="does not byte-equal"):
        build_tiles3d(
            ortho, heights, out, tile_pixels=8, profile=Profile.GAME
        )
    after = {
        p.relative_to(out): p.read_bytes()
        for p in out.rglob("*")
        if p.is_file()
    }
    assert after == good
    assert (out / "provenance.json").is_file()
    assert build_module.BACKUP_SUBDIR not in {p.name for p in out.iterdir()}
