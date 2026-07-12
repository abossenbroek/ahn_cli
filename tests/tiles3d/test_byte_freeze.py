"""Byte-freeze regression guard for the strict 3D Tiles profile.

The payload/encoder split (``TilePayload`` + ``TileEncoder`` +
``StrictEncoder``) must not change a single output byte of the strict
build. This test freezes that: it builds a tiny synthetic scene through
the **real** ``build_tiles3d`` pipeline and asserts the sha256 of every
emitted file equals a checked-in golden digest.

Why the geodesy is pinned: absolute ECEF/geodetic outputs depend on the
machine's PROJ grid availability (see ``geodesy.py``'s determinism
caveat), so hashing real-geodesy output would not be machine-stable.
This test therefore replaces :class:`Geodesy`'s two transforms with a
documented, deterministic, pure-numpy affine stand-in — every other
stage runs unmodified. The goldens were generated from the pre-refactor
code and must survive the refactor byte for byte.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import numpy as np

from ahn_cli.tiles3d.build import build_tiles3d
from ahn_cli.tiles3d.geodesy import Geodesy
from tests.tiles3d.conftest import (
    grid_for_ortho,
    make_ortho,
    synth_rgb,
    write_exr,
)

if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt
    import pytest

# Frozen scene parameters (a 12x12 grid at tile_pixels=8 -> 1 root + 4
# leaves, exercising parent/child region unions and multi-tile emission).
_WIDTH = 12
_HEIGHT = 12
_SEED = 7
_TILE_PIXELS = 8

# sha256 of every emitted file, generated from the pre-refactor strict
# build with the affine geodesy below. The refactor must reproduce these
# exactly.
_GOLDENS = {
    "tiles/0-0-0.glb": (
        "2bec627f23bfa15a58ac7e6f91dbc8d4c5ae3c7241630e87537fa839ef10ba02"
    ),
    "tiles/1-0-0.glb": (
        "b3f7732be9d6e0f89c143bc4f667a1b6b578cdeb9144dab2f05139ca3d577223"
    ),
    "tiles/1-0-1.glb": (
        "982f8c8355738eb10188cb4d40bc691953608449d4ce93201696ce12fb9a3c74"
    ),
    "tiles/1-1-0.glb": (
        "0bc59a214acbd2bd3e513a49e6c62a1f63be9502af7f8863848260f0ac88854c"
    ),
    "tiles/1-1-1.glb": (
        "dd0be2567a67d03c2375bd6b79382b6162684584679c5285ef750c6f65f1ce66"
    ),
    "tileset.json": (
        "db968e4a594737ed6558bd2668d21166aae27484197ad57ddb5a0a0c7962fe90"
    ),
}


def _fake_to_ecef(
    _self: Geodesy,
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Deterministic affine ECEF stand-in (no PROJ, machine-stable)."""
    return (
        np.asarray(1000.0 * x, dtype=np.float64),
        np.asarray(2000.0 * y, dtype=np.float64),
        np.asarray(5.0 * z + 10.0, dtype=np.float64),
    )


def _fake_to_geodetic_radians(
    _self: Geodesy,
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Deterministic affine geodetic-radians stand-in (no PROJ).

    Maps the synthetic RD grid into tiny, monotonically ordered radian
    ranges so every region is valid (west < east, south < north) and
    parent regions contain their children.
    """
    return (
        np.asarray((x - 100.0) * 1e-3, dtype=np.float64),
        np.asarray((y - 100.0) * 1e-3, dtype=np.float64),
        np.asarray(z, dtype=np.float64),
    )


def _pin_geodesy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace both geodesy transforms with the affine stand-ins."""
    monkeypatch.setattr(Geodesy, "to_ecef", _fake_to_ecef)
    monkeypatch.setattr(
        Geodesy, "to_geodetic_radians", _fake_to_geodetic_radians
    )


def test_strict_build_is_byte_frozen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every emitted file byte-equals its checked-in golden digest."""
    _pin_geodesy(monkeypatch)
    rgb = synth_rgb(_WIDTH, _HEIGHT, _SEED)
    ortho = make_ortho(tmp_path / "ortho.tif", rgb)
    heights = write_exr(tmp_path / "r.exr", grid_for_ortho(rgb))
    out = tmp_path / "out"

    result = build_tiles3d(ortho, heights, out, tile_pixels=_TILE_PIXELS)

    assert result.tile_count == 5
    assert result.levels == 1
    assert result.vertices == 218
    assert result.triangles == 314
    digests = {
        p.relative_to(out).as_posix(): hashlib.sha256(
            p.read_bytes()
        ).hexdigest()
        for p in out.rglob("*")
        if p.is_file()
    }
    assert digests == _GOLDENS
