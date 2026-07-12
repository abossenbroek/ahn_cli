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
* **Size budget** (informational print + one hard ceiling): geometry
  byte/pixel cost is measured through the **real** geodesy pipeline at the
  256px production leaf. Production leaves measure ~6 (smooth terrain) to
  ~8 (per-pixel-noise z) B/px — above the plan's optimistic 2-3 estimate /
  ~4 alarm — because ECEF-rotated quantized positions resist meshopt's
  byte-plane delta coding and the regular-grid index stream is a structural
  ~2 B/px. (An affine, axis-aligned geodesy would compress ~2x better, but
  that is not what production emits.) The one hard assertion is a ceiling on
  the deterministic worst-case leaf — the conftest-noise 256px single tile
  — with headroom over its ~8 B/px baseline. Small-tile scenes read ~2x
  higher (vertex/pixel ratio + short-stream codec overhead — expected
  geometry-codec scaling), so the budget is judged only at 256px.
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
from dataclasses import dataclass
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


@dataclass(frozen=True)
class _GameGeometry:
    """Per-stream compressed geometry cost of one game-profile glb (B/px)."""

    positions: float
    uvs: float
    indices: float
    total: float
    vertices: int


def _game_geometry_bytes_per_pixel(glb: bytes) -> _GameGeometry:
    """Return the per-stream compressed geometry cost of a game-profile glb.

    Geometry *post-meshopt* is the sum of the three geometry bufferViews'
    ``EXT_meshopt_compression.byteLength`` — the actual compressed stream
    bytes in the BIN chunk. Crucially this is NOT ``bufferView.byteLength``,
    which for a meshopt bufferView is the *decompressed* size and would
    massively overstate the cost (a BIN-chunk cross-check confirms the three
    compressed streams plus the JPEG equal the BIN chunk exactly). Each
    stream is resolved through its accessor, so the split is robust to
    bufferView ordering. Pixels are the tile's sampled vertices (one per
    source pixel at the tile's stride), read from the POSITION accessor.
    """
    document = _glb_json(glb)
    primitive = cast("dict[str, Any]", document["meshes"][0]["primitives"][0])
    accessors = cast("list[Any]", document["accessors"])
    views = cast("list[Any]", document["bufferViews"])

    def _stream(accessor_index: int) -> int:
        view_index = int(
            cast("dict[str, Any]", accessors[accessor_index])["bufferView"]
        )
        view = cast("dict[str, Any]", views[view_index])
        ext = cast("dict[str, Any]", view["extensions"][_EXT_MESHOPT])
        return int(ext["byteLength"])

    vertices = int(
        cast(
            "dict[str, Any]", accessors[primitive["attributes"]["POSITION"]]
        )["count"]
    )
    positions = _stream(int(primitive["attributes"]["POSITION"]))
    uvs = _stream(int(primitive["attributes"]["TEXCOORD_0"]))
    indices = _stream(int(primitive["indices"]))
    return _GameGeometry(
        positions=positions / vertices,
        uvs=uvs / vertices,
        indices=indices / vertices,
        total=(positions + uvs + indices) / vertices,
        vertices=vertices,
    )


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
# Size budget.
#
# Measured through the REAL geodesy pipeline (not a pinned affine), because
# geometry cost is dominated by the position stream and the position stream
# depends entirely on the geodesy: production ECEF-RTC y-up positions are a
# rotated/sheared image of the RD grid, so every quantized uint16 channel
# mixes row/col/height and the low byte-planes resist meshopt's delta coding
# (~3.6-5.6 B/px). An affine, axis-aligned geodesy (e.g. the byte-freeze
# pin) would collapse each channel to a near-constant delta and compress ~2x
# better — but that is not what production emits, so measuring through it
# would understate the real cost. Numbers below are the honest production
# figures; the plan's 2-3 B/px estimate / ~4 alarm was optimistic.
#
# The hard assertion is on the deterministic worst-case leaf: a
# conftest-noise (per-pixel white-noise z) 256x256 single tile — the CLI's
# default production leaf size — measured at ~8 B/px (BIN-cross-checked).
# The index stream is a structural ~2 B/px (regular-grid triangulation) and
# positions carry the rest. Small-tile scenes read ~2x higher purely from
# the vertex/pixel ratio and short-stream codec overhead (geometry-codec
# scaling, not a regression), so the budget is judged only at 256px.
# Possible future efficiency item (flagged, not built here): axis-aligned
# local quantization frames or per-block uint16 indices.
_LEAF_REFERENCE_BYTES_PER_PIXEL = 4.0
"""The plan's aspirational leaf-geometry target; printed as a reference."""

_LEAF_REGRESSION_CEILING_BYTES_PER_PIXEL = 10.0
"""Hard ceiling: ~25% over the ~8 B/px verified conftest-noise 256px leaf."""

_PRODUCTION_LEAF_PIXELS = 256
"""The CLI's default ``tile_pixels`` — the production leaf size."""


def _emit(line: str) -> None:
    """Print one informational size-budget line (non-blocking output)."""
    print(line)  # noqa: T201


def _build_single_leaf_glb(
    tmp_path: Path, ortho: Path, heights: Path
) -> bytes:
    """Build a 256x256 single-tile game tileset and return its one glb."""
    out = tmp_path
    build_tiles3d(
        ortho,
        heights,
        out,
        tile_pixels=_PRODUCTION_LEAF_PIXELS,
        profile=Profile.GAME,
    )
    return (out / "tiles" / "0-0-0.glb").read_bytes()


def test_size_budget_guards_the_production_leaf(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Assert the 256px production leaf's geometry stays under the ceiling.

    The hard assertion is the conftest-noise (per-pixel white noise z)
    256x256 single tile — the CLI's default leaf size and the deterministic
    worst case. Informational prints add the smooth-terrain 256px figure
    (the representative number) against the plan's ~4 reference, and a small
    64px scene to show the geometry-codec scaling. All through real geodesy.
    """
    noise_glb = _build_single_leaf_glb(
        tmp_path / "noise",
        *_make_pair(
            tmp_path,
            _PRODUCTION_LEAF_PIXELS,
            _PRODUCTION_LEAF_PIXELS,
            seed=34,
        ),
    )
    noise = _game_geometry_bytes_per_pixel(noise_glb)

    smooth_glb = _build_single_leaf_glb(
        tmp_path / "smooth",
        *_smooth_pair(
            tmp_path, _PRODUCTION_LEAF_PIXELS, _PRODUCTION_LEAF_PIXELS
        ),
    )
    smooth = _game_geometry_bytes_per_pixel(smooth_glb)

    small_out = tmp_path / "small"
    small_ortho, small_heights = _smooth_pair(tmp_path, 64, 64)
    build_tiles3d(
        small_ortho,
        small_heights,
        small_out,
        tile_pixels=32,
        profile=Profile.GAME,
    )
    small_leaf = _game_geometry_bytes_per_pixel(
        (small_out / "tiles" / "1-1-1.glb").read_bytes()
    )

    with capsys.disabled():
        _emit(
            f"\n[size-budget] game geometry, real geodesy "
            f"(reference {_LEAF_REFERENCE_BYTES_PER_PIXEL} B/px, "
            f"ceiling {_LEAF_REGRESSION_CEILING_BYTES_PER_PIXEL}):"
        )
        for label, g in (
            ("256px production leaf, conftest-noise z", noise),
            ("256px production leaf, smooth terrain", smooth),
            ("32px small-tile leaf, smooth terrain", small_leaf),
        ):
            _emit(
                f"  {label}: {g.total:.3f} B/px total "
                f"(pos {g.positions:.3f} + uv {g.uvs:.3f} + idx "
                f"{g.indices:.3f}), {g.vertices} v"
            )

    assert noise.total <= _LEAF_REGRESSION_CEILING_BYTES_PER_PIXEL, (
        f"game production-leaf geometry {noise.total:.3f} B/px exceeds the "
        f"{_LEAF_REGRESSION_CEILING_BYTES_PER_PIXEL} B/px regression ceiling"
    )


def test_size_budget_reports_heightfield_chunk_cost(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Print the heightfield .hf chunk cost at the 256px production leaf."""
    ortho, heights = _make_pair(
        tmp_path, _PRODUCTION_LEAF_PIXELS, _PRODUCTION_LEAF_PIXELS, seed=35
    )
    out = tmp_path / "hf"
    build_tiles3d(
        ortho,
        heights,
        out,
        tile_pixels=_PRODUCTION_LEAF_PIXELS,
        profile=Profile.HEIGHTFIELD,
    )
    chunk = (out / "tiles" / "0-0-0.hf").read_bytes()
    width, height = struct.unpack_from("<II", chunk, 8)
    with capsys.disabled():
        _emit(
            f"[size-budget] heightfield .hf chunk: "
            f"{len(chunk) / (width * height):.3f} B/px ({width}x{height})"
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
