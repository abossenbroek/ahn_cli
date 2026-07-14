"""Tests for the splat-profile per-tile verification.

Each negative test builds a valid splat tileset, corrupts exactly the
gaussian field one check guards through the real
build-then-corrupt-then-verify path, and asserts that check's message —
proving each check fires independently of the whole-file byte-identity
backstop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

import ahn_cli.tiles3d.build as build_module
import ahn_cli.tiles3d.verify_splat as verify_splat_module
from ahn_cli.tiles3d.build import build_tiles3d
from ahn_cli.tiles3d.errors import Tiles3dError
from ahn_cli.tiles3d.pack import TileKey
from ahn_cli.tiles3d.profile import Profile
from ahn_cli.tiles3d.splat import (
    _F32,  # pyright: ignore[reportPrivateUsage]
    _FIELD_COUNT,  # pyright: ignore[reportPrivateUsage]
    DecodedSplat,
    _compress,  # pyright: ignore[reportPrivateUsage]
    _header,  # pyright: ignore[reportPrivateUsage]
    decode_splat,
)
from ahn_cli.tiles3d.verify import verify_tiles3d
from tests.tiles3d.conftest import (
    grid_for_ortho,
    make_ortho,
    pack_blob,
    repack_one,
    synth_rgb,
    write_exr,
)

if TYPE_CHECKING:
    from pathlib import Path

_LEAF_KEY = TileKey(2, 0, 0)
"""The leaf tile whose blob the corruption tests target inside the pack."""


@pytest.fixture
def splat_site(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Build a valid two-level splat tileset; (out, ortho, heights)."""
    rgb = synth_rgb(20, 14, seed=13)
    ortho = make_ortho(tmp_path / "ortho.tif", rgb)
    heights = write_exr(tmp_path / "r.exr", grid_for_ortho(rgb))
    out = tmp_path / "out"
    build_tiles3d(ortho, heights, out, tile_pixels=8, profile=Profile.SPLAT)
    return out, ortho, heights


def _verify(site: tuple[Path, Path, Path]) -> None:
    out, ortho, heights = site
    verify_tiles3d(out, ortho, heights, tile_pixels=8, profile=Profile.SPLAT)


def _leaf_bytes(site: tuple[Path, Path, Path]) -> bytes:
    """Return the leaf tile's pristine ``.ply`` bytes from the pack."""
    return pack_blob(site[0] / "tiles.hfp", _LEAF_KEY)[0]


def _repack_leaf(site: tuple[Path, Path, Path], blob: bytes) -> None:
    """Repack ``tiles.hfp`` with the leaf tile's ``.ply`` replaced."""
    repack_one(
        site[0] / "tiles.hfp",
        _LEAF_KEY,
        lambda _primary, texture: (blob, texture),
    )


def _reencode(decoded: DecodedSplat) -> bytes:
    """Re-serialise a (possibly mutated) :class:`DecodedSplat`.

    Bypasses :func:`~ahn_cli.tiles3d.splat.encode_splat` (which only takes a
    genuine payload) so a test can corrupt exactly one decoded field and
    still produce a well-formed, decodable ``.ply`` blob.
    """
    arr = np.empty((decoded.count, _FIELD_COUNT), dtype=_F32)
    arr[:, 0:3] = decoded.positions
    arr[:, 3:6] = decoded.f_dc
    arr[:, 6] = decoded.opacity
    arr[:, 7:10] = decoded.scale
    arr[:, 10:14] = decoded.rot
    return _compress(_header(decoded.count) + arr.tobytes())


def test_pristine_splat_build_verifies(
    splat_site: tuple[Path, Path, Path],
) -> None:
    """The verifier accepts what the splat builder just wrote."""
    _verify(splat_site)


def test_wrong_position_is_refused(
    splat_site: tuple[Path, Path, Path],
) -> None:
    """A position that is not the tile's RTC mesh vertex is caught."""
    decoded = decode_splat(_leaf_bytes(splat_site))
    positions = decoded.positions.copy()
    positions[0, 0] += 1.0
    mutated = DecodedSplat(
        count=decoded.count,
        positions=positions,
        f_dc=decoded.f_dc,
        opacity=decoded.opacity,
        scale=decoded.scale,
        rot=decoded.rot,
    )
    _repack_leaf(splat_site, _reencode(mutated))
    with pytest.raises(Tiles3dError, match="positions do not equal"):
        _verify(splat_site)


def test_wrong_colour_is_refused(
    splat_site: tuple[Path, Path, Path],
) -> None:
    """A colour that is not the SH degree-0 recompute is caught."""
    decoded = decode_splat(_leaf_bytes(splat_site))
    f_dc = decoded.f_dc.copy()
    f_dc[0, 0] += 1.0
    mutated = DecodedSplat(
        count=decoded.count,
        positions=decoded.positions,
        f_dc=f_dc,
        opacity=decoded.opacity,
        scale=decoded.scale,
        rot=decoded.rot,
    )
    _repack_leaf(splat_site, _reencode(mutated))
    with pytest.raises(Tiles3dError, match="colour does not equal"):
        _verify(splat_site)


def test_wrong_opacity_is_refused(
    splat_site: tuple[Path, Path, Path],
) -> None:
    """An opacity that is not the fixed logit constant is caught."""
    decoded = decode_splat(_leaf_bytes(splat_site))
    opacity = decoded.opacity.copy()
    opacity[0] = np.float32(0.0)
    mutated = DecodedSplat(
        count=decoded.count,
        positions=decoded.positions,
        f_dc=decoded.f_dc,
        opacity=opacity,
        scale=decoded.scale,
        rot=decoded.rot,
    )
    _repack_leaf(splat_site, _reencode(mutated))
    with pytest.raises(Tiles3dError, match="opacity is not the fixed"):
        _verify(splat_site)


def test_wrong_scale_is_refused(
    splat_site: tuple[Path, Path, Path],
) -> None:
    """A scale that is not the measured isotropic cell spacing is caught."""
    decoded = decode_splat(_leaf_bytes(splat_site))
    scale = decoded.scale.copy()
    scale[0, 1] += 1.0
    mutated = DecodedSplat(
        count=decoded.count,
        positions=decoded.positions,
        f_dc=decoded.f_dc,
        opacity=decoded.opacity,
        scale=scale,
        rot=decoded.rot,
    )
    _repack_leaf(splat_site, _reencode(mutated))
    with pytest.raises(Tiles3dError, match="scale does not equal"):
        _verify(splat_site)


def test_wrong_rotation_is_refused(
    splat_site: tuple[Path, Path, Path],
) -> None:
    """A rotation that is not the identity quaternion is caught."""
    decoded = decode_splat(_leaf_bytes(splat_site))
    rot = decoded.rot.copy()
    rot[0, 1] = np.float32(1.0)
    mutated = DecodedSplat(
        count=decoded.count,
        positions=decoded.positions,
        f_dc=decoded.f_dc,
        opacity=decoded.opacity,
        scale=decoded.scale,
        rot=rot,
    )
    _repack_leaf(splat_site, _reencode(mutated))
    with pytest.raises(Tiles3dError, match="rotation is not the identity"):
        _verify(splat_site)


def test_wrong_gaussian_count_is_refused(
    splat_site: tuple[Path, Path, Path],
) -> None:
    """A gaussian count that doesn't match the tile's sample count is caught."""
    decoded = decode_splat(_leaf_bytes(splat_site))
    mutated = DecodedSplat(
        count=decoded.count - 1,
        positions=decoded.positions[:-1],
        f_dc=decoded.f_dc[:-1],
        opacity=decoded.opacity[:-1],
        scale=decoded.scale[:-1],
        rot=decoded.rot[:-1],
    )
    _repack_leaf(splat_site, _reencode(mutated))
    with pytest.raises(Tiles3dError, match="gaussian count"):
        _verify(splat_site)


def test_corrupt_zstd_frame_is_refused(
    splat_site: tuple[Path, Path, Path],
) -> None:
    """A flipped byte in the zstd frame is caught at decode."""
    data = bytearray(_leaf_bytes(splat_site))
    data[0] ^= 0xFF  # break the zstd frame magic
    _repack_leaf(splat_site, bytes(data))
    with pytest.raises(Tiles3dError, match="not a valid zstd frame"):
        _verify(splat_site)


def test_rejected_splat_rebuild_restores_the_previous_deliverable(
    splat_site: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rebuild the verifier rejects restores the old build whole.

    Only the verifier's independent colour recompute is perturbed (via the
    module-level ``SH_DC0`` it imported), so the build writes a correct
    splat tile that its own verification then refuses — the swap machinery
    must remove the rejected write and restore the held-aside previous
    deliverable, provenance.json and every ``.ply`` included.
    """
    out, ortho, heights = splat_site
    good = {
        p.relative_to(out): p.read_bytes()
        for p in out.rglob("*")
        if p.is_file()
    }
    monkeypatch.setattr(verify_splat_module, "SH_DC0", 1.0)
    with pytest.raises(Tiles3dError, match="colour does not equal"):
        build_tiles3d(
            ortho, heights, out, tile_pixels=8, profile=Profile.SPLAT
        )
    after = {
        p.relative_to(out): p.read_bytes()
        for p in out.rglob("*")
        if p.is_file()
    }
    assert after == good
    assert (out / "provenance.json").is_file()
    assert build_module.BACKUP_SUBDIR not in {p.name for p in out.iterdir()}
