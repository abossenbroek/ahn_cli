"""End-to-end integration tests spanning the three tiles3d profiles.

This is the epic's whole-pipeline proof. It exercises four things the
per-module suites do not cover as a unit:

* **CLI end-to-end** for the lossy profiles: a synthetic site built
  through the real ``ahn_cli tiles3d`` command (Click ``CliRunner``)
  under ``--profile game`` and ``--profile heightfield``, asserting the
  post-write verifier ran green and the expected file set landed. The
  matching negative suites already live in
  ``tests/tiles3d/test_verify_game.py`` and
  ``tests/tiles3d/test_verify_heightfield.py`` (one corrupt-then-verify
  test per check) — they are *not* duplicated here.
* **Double-build determinism**: building the same inputs twice in one
  process yields byte-identical files, for strict, game AND heightfield.
  Real geodesy is used deliberately — self-consistency within one machine
  is the documented determinism boundary (absolute ECEF/geodetic values
  depend on the local PROJ grid), so no golden pinning is needed here.
* **Size budget** (informational, non-blocking): the geometry byte/pixel
  cost of the game and heightfield leaves is printed every run against the
  plan's aspirational ~4 B/px reference. The frozen game encoder does not
  reach that figure — the regular-grid index stream alone is a structural
  ~2 B/px and the 16-bit-per-tile quantized positions add ~3.5 B/px, so
  even a perfect plane floors near ~6 B/px and this smooth scene measures
  ~9.3 B/px. The one hard assertion is therefore a *regression ceiling*
  with headroom over that baseline, which catches a genuine size blow-up
  while leaving the aspirational number as a documented, printed target.
* **Committed Rust-consumer fixtures**: the tiny ``fixtures/rust-consumer``
  tilesets (game + heightfield) the separate Rust runtime repo consumes
  are regenerated here with *pinned* geodesy and byte-compared, so fixture
  drift turns this test red. Regenerate them with::

      uv run python -m tests.tiles3d.regen_rust_fixtures
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest
from click.testing import CliRunner

from ahn_cli.cli import cli
from ahn_cli.tiles3d.build import build_tiles3d
from ahn_cli.tiles3d.geodesy import Geodesy
from ahn_cli.tiles3d.profile import Profile
from tests.tiles3d.conftest import (
    grid_for_ortho,
    make_ortho,
    synth_rgb,
    write_exr,
)

if TYPE_CHECKING:
    import numpy.typing as npt

# ---------------------------------------------------------------------------
# Pinned geodesy (machine-stable committed fixtures).
#
# The Rust-consumer fixtures are checked into git and re-derived on every
# CI run, so their absolute ECEF/geodetic values must not depend on the
# machine's PROJ grid availability (see ``geodesy.py``'s determinism
# caveat). ``test_byte_freeze.py`` pins the strict profile with a bare
# affine, but the *game* verifier couples all three transforms: it
# dequantizes ECEF vertices and re-projects them with
# ``to_geodetic_from_ecef``, then requires them inside each region grown
# only by the quantization bound converted through an Earth radius. An
# arbitrary affine violates that bound, so the pin here is a genuine
# spherical-Earth model instead: a single fixed radius, all three
# transforms exact inverses of one another (no PROJ, machine-stable), the
# grid placed on the equator so the longitude slack ``metric /
# _MIN_EARTH_RADIUS`` bounds the real angular error. Every region stays
# valid (monotonic in x/y) and parents contain their children.
# ---------------------------------------------------------------------------

_SPHERE_RADIUS = 6_378_137.0
_LON0 = 0.1
_LAT0 = 0.0
_X0 = 100.0
_Y0 = 100.0
_RAD_PER_UNIT = 1e-6


def fake_to_geodetic_radians(
    _self: Geodesy,
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Map the RD grid to equatorial lon/lat radians + NAP height."""
    return (
        np.asarray(_LON0 + (x - _X0) * _RAD_PER_UNIT, dtype=np.float64),
        np.asarray(_LAT0 + (y - _Y0) * _RAD_PER_UNIT, dtype=np.float64),
        np.asarray(z, dtype=np.float64),
    )


def fake_to_ecef(
    self: Geodesy,
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Spherical-Earth ECEF: the exact forward of the geodetic map."""
    lon, lat, height = fake_to_geodetic_radians(self, x, y, z)
    radius = _SPHERE_RADIUS + height
    return (
        np.asarray(radius * np.cos(lat) * np.cos(lon), dtype=np.float64),
        np.asarray(radius * np.cos(lat) * np.sin(lon), dtype=np.float64),
        np.asarray(radius * np.sin(lat), dtype=np.float64),
    )


def fake_to_geodetic_from_ecef(
    _self: Geodesy,
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Spherical-Earth ECEF inverse: the exact inverse of ``fake_to_ecef``."""
    radius = np.sqrt(x * x + y * y + z * z)
    return (
        np.asarray(np.arctan2(y, x), dtype=np.float64),
        np.asarray(
            np.arcsin(np.clip(z / radius, -1.0, 1.0)), dtype=np.float64
        ),
        np.asarray(radius - _SPHERE_RADIUS, dtype=np.float64),
    )


def pin_geodesy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace all three geodesy transforms with the spherical stand-ins."""
    monkeypatch.setattr(Geodesy, "to_ecef", fake_to_ecef)
    monkeypatch.setattr(
        Geodesy, "to_geodetic_radians", fake_to_geodetic_radians
    )
    monkeypatch.setattr(
        Geodesy, "to_geodetic_from_ecef", fake_to_geodetic_from_ecef
    )


# ---------------------------------------------------------------------------
# Shared helpers.
# ---------------------------------------------------------------------------


def _make_pair(
    tmp_path: Path, width: int, height: int, seed: int
) -> tuple[Path, Path]:
    """Write a matching ortho GeoTIFF + reconcile EXR pair to ``tmp_path``."""
    rgb = synth_rgb(width, height, seed)
    ortho = make_ortho(tmp_path / "ortho.tif", rgb)
    heights = write_exr(tmp_path / "r.exr", grid_for_ortho(rgb))
    return ortho, heights


def _smooth_pair(
    tmp_path: Path, width: int, height: int
) -> tuple[Path, Path]:
    """Write a smooth (real-terrain-like) ortho + EXR pair.

    The shared conftest fixtures use random per-pixel heights and colours,
    which are incompressible noise — meshopt cannot delta-code them, so
    their B/px is unrepresentative of genuine AHN relief. The size budget
    must be judged on smooth, low-frequency data (still non-flat and
    non-uniform, so the authenticity gates pass) the way real terrain
    compresses.
    """
    cols = np.arange(width, dtype=np.float64)[np.newaxis, :]
    rows = np.arange(height, dtype=np.float64)[:, np.newaxis]
    z = (
        5.0
        + 0.05 * cols
        + 0.03 * rows
        + 2.0 * np.sin(cols / 16.0)
        + 1.5 * np.cos(rows / 12.0)
    ) * np.ones((height, width))
    rgb = np.empty((height, width, 3), dtype=np.uint8)
    rgb[:, :, 0] = (128 + 100 * np.sin(cols / 20.0)).astype(np.uint8)
    rgb[:, :, 1] = (128 + 80 * np.cos(rows / 18.0)).astype(np.uint8)
    rgb[:, :, 2] = ((cols + rows) * 3.0 % 256).astype(np.uint8)
    ortho = make_ortho(tmp_path / "ortho.tif", rgb)
    heights = write_exr(tmp_path / "r.exr", grid_for_ortho(rgb, z=z))
    return ortho, heights


def _digests(root: Path) -> dict[str, str]:
    """Map each file under ``root`` to the sha256 of its bytes."""
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


def _file_set(root: Path) -> set[str]:
    """Return the posix-relative paths of every file under ``root``."""
    return {
        p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()
    }


def _glb_json(data: bytes) -> dict[str, Any]:
    """Return the JSON chunk of a binary glTF container."""
    json_length = struct.unpack("<I", data[12:16])[0]
    return cast("dict[str, Any]", json.loads(data[20 : 20 + json_length]))


_EXT_MESHOPT = "EXT_meshopt_compression"


def _game_geometry_bytes_per_pixel(glb: bytes) -> tuple[float, int]:
    """Return (geometry B/px, vertex count) for a game-profile glb.

    The plan phrases this as "glb minus JPEG bytes ≈ geometry". Realized
    precisely, geometry *post-meshopt* is the sum of the three geometry
    bufferViews' ``EXT_meshopt_compression.byteLength`` — the actual
    compressed stream sizes in the BIN chunk (POSITION / TEXCOORD_0 /
    indices). Crucially this is NOT ``bufferView.byteLength``, which for a
    meshopt bufferView is the *decompressed* size and would massively
    overstate the cost. Pixels are the tile's sampled vertices (one per
    source pixel at the tile's stride), read from the POSITION accessor
    count.
    """
    document = _glb_json(glb)
    buffer_views = cast("list[Any]", document["bufferViews"])
    geometry = sum(
        int(
            cast("dict[str, Any]", view["extensions"][_EXT_MESHOPT])[
                "byteLength"
            ]
        )
        for view in buffer_views
        if _EXT_MESHOPT in cast("dict[str, Any]", view).get("extensions", {})
    )
    primitive = cast("dict[str, Any]", document["meshes"][0]["primitives"][0])
    pos_index = int(primitive["attributes"]["POSITION"])
    accessor = cast("dict[str, Any]", document["accessors"][pos_index])
    vertices = int(accessor["count"])
    return geometry / vertices, vertices


# ---------------------------------------------------------------------------
# CLI end-to-end (game + heightfield). Negatives: test_verify_game.py /
# test_verify_heightfield.py.
# ---------------------------------------------------------------------------


def _run_cli(out: Path, ortho: Path, heights: Path, profile: str):  # noqa: ANN202
    return CliRunner().invoke(
        cli,
        [
            "tiles3d",
            "--ortho",
            str(ortho),
            "--heights",
            str(heights),
            "--out",
            str(out),
            "--profile",
            profile,
        ],
    )


def test_cli_game_end_to_end(tmp_path: Path) -> None:
    """A synthetic site builds green under ``--profile game`` via the CLI."""
    ortho, heights = _make_pair(tmp_path, 6, 6, seed=31)
    out = tmp_path / "game"
    result = _run_cli(out, ortho, heights, "game")
    assert result.exit_code == 0, result.output
    assert "verified. profile=game." in result.output
    assert _file_set(out) == {
        "tileset.json",
        "tiles/0-0-0.glb",
        "provenance.json",
    }


def test_cli_heightfield_end_to_end(tmp_path: Path) -> None:
    """A synthetic site builds green under ``--profile heightfield``."""
    ortho, heights = _make_pair(tmp_path, 6, 6, seed=32)
    out = tmp_path / "hf"
    result = _run_cli(out, ortho, heights, "heightfield")
    assert result.exit_code == 0, result.output
    assert "verified" in result.output
    assert _file_set(out) == {
        "tileset.json",
        "tiles/0-0-0.hf",
        "tiles/0-0-0.jpg",
        "provenance.json",
    }


# ---------------------------------------------------------------------------
# Double-build determinism (strict, game, heightfield).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "profile",
    [Profile.STRICT, Profile.GAME, Profile.HEIGHTFIELD],
)
def test_double_build_is_deterministic(
    tmp_path: Path, profile: Profile
) -> None:
    """Two builds of one input in one process are byte-identical.

    Real geodesy — self-consistency within a machine is the documented
    boundary, so no golden pinning is needed for equality across two runs.
    """
    ortho, heights = _make_pair(tmp_path, 12, 12, seed=33)
    first = tmp_path / "first"
    second = tmp_path / "second"
    build_tiles3d(ortho, heights, first, tile_pixels=8, profile=profile)
    build_tiles3d(ortho, heights, second, tile_pixels=8, profile=profile)
    assert _digests(first) == _digests(second)


# ---------------------------------------------------------------------------
# Size budget (informational; one hard alarm on game leaf geometry).
# ---------------------------------------------------------------------------

# The plan's aspirational leaf-geometry target. Printed as a reference on
# every run. NOTE: the frozen game encoder does not reach it — even a
# perfect plane never drops below ~6 B/px at any tile size, because the
# regular-grid index stream is a structural ~2 B/px floor and the
# 16-bit-per-tile quantized positions add ~3.5 B/px. The measured worst
# leaf on this smooth scene is ~9.3 B/px. See the module docstring and
# `.superpowers/sdd/task-9-report.md` for the coordinator resolution.
_LEAF_REFERENCE_BYTES_PER_PIXEL = 4.0

# The hard regression alarm: a ceiling with ~40% headroom over the current
# ~9.3 B/px baseline, so a genuine geometry-size regression trips it while
# normal run-to-run variation does not.
_LEAF_REGRESSION_CEILING_BYTES_PER_PIXEL = 13.0


def _emit(line: str) -> None:
    """Print one informational size-budget line (non-blocking output)."""
    print(line)  # noqa: T201


def test_size_budget_reports_and_guards_game_leaves(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Print geometry B/px per profile; alarm if game leaves exceed budget.

    A 64x64 smooth scene at ``tile_pixels=32`` gives a multi-level tree
    with real leaf tiles, meaningful for the byte budget yet fast (<10 s).
    """
    ortho, heights = _smooth_pair(tmp_path, 64, 64)

    game_out = tmp_path / "game"
    build_tiles3d(
        ortho, heights, game_out, tile_pixels=32, profile=Profile.GAME
    )
    game_tiles = sorted((game_out / "tiles").glob("*.glb"))
    leaf_level = max(int(p.name.split("-", 1)[0]) for p in game_tiles)
    worst_leaf = 0.0
    with capsys.disabled():
        _emit("\n[size-budget] game geometry (meshopt bufferViews):")
        for tile in game_tiles:
            per_px, vertices = _game_geometry_bytes_per_pixel(
                tile.read_bytes()
            )
            level = int(tile.name.split("-", 1)[0])
            marker = " (leaf)" if level == leaf_level else ""
            _emit(f"  {tile.name}: {per_px:.3f} B/px ({vertices} v){marker}")
            if level == leaf_level:
                worst_leaf = max(worst_leaf, per_px)
        _emit(
            f"[size-budget] worst game leaf: {worst_leaf:.3f} B/px "
            f"(aspirational reference {_LEAF_REFERENCE_BYTES_PER_PIXEL} B/px; "
            f"regression ceiling {_LEAF_REGRESSION_CEILING_BYTES_PER_PIXEL})"
        )

    hf_out = tmp_path / "hf"
    build_tiles3d(
        ortho, heights, hf_out, tile_pixels=32, profile=Profile.HEIGHTFIELD
    )
    with capsys.disabled():
        _emit("[size-budget] heightfield geometry (.hf chunk bytes):")
        for chunk in sorted((hf_out / "tiles").glob("*.hf")):
            data = chunk.read_bytes()
            width, height = struct.unpack_from("<II", data, 8)
            per_px = len(data) / (width * height)
            _emit(f"  {chunk.name}: {per_px:.3f} B/px ({width}x{height})")

    assert worst_leaf <= _LEAF_REGRESSION_CEILING_BYTES_PER_PIXEL, (
        f"game leaf geometry {worst_leaf:.3f} B/px exceeds the "
        f"{_LEAF_REGRESSION_CEILING_BYTES_PER_PIXEL} B/px regression ceiling"
    )


# ---------------------------------------------------------------------------
# Committed Rust-consumer fixtures (pinned geodesy, drift => red).
# ---------------------------------------------------------------------------

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "rust-consumer"
FIXTURE_WIDTH = 12
FIXTURE_HEIGHT = 12
FIXTURE_SEED = 7
FIXTURE_TILE_PIXELS = 8
FIXTURE_PROFILES = {"game": Profile.GAME, "heightfield": Profile.HEIGHTFIELD}


def build_fixture(tmp_path: Path, out: Path, profile: Profile) -> None:
    """Build one Rust-consumer fixture tileset (geodesy must be pinned).

    Shared by the drift test and the regeneration script so a regenerated
    fixture is byte-identical to what the test re-derives. The caller pins
    geodesy first (``pin_geodesy`` under pytest, direct ``setattr`` in the
    one-shot script).
    """
    rgb = synth_rgb(FIXTURE_WIDTH, FIXTURE_HEIGHT, FIXTURE_SEED)
    ortho = make_ortho(tmp_path / "ortho.tif", rgb)
    heights = write_exr(tmp_path / "r.exr", grid_for_ortho(rgb))
    build_tiles3d(
        ortho, heights, out, tile_pixels=FIXTURE_TILE_PIXELS, profile=profile
    )


@pytest.mark.parametrize("name", list(FIXTURE_PROFILES))
def test_committed_fixtures_are_not_stale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    """Regenerated fixtures byte-match the ones checked into git."""
    committed = FIXTURE_ROOT / name
    assert committed.is_dir(), (
        f"missing committed fixture {committed}; regenerate via "
        "`uv run python -m tests.tiles3d.regen_rust_fixtures`"
    )
    pin_geodesy(monkeypatch)
    out = tmp_path / "out"
    build_fixture(tmp_path, out, FIXTURE_PROFILES[name])
    assert _digests(out) == _digests(committed)
