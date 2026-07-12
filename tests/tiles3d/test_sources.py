"""Tests for terrain loading and the perfect-dimension-match gates."""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING

import laspy
import numpy as np
import pytest

from ahn_cli.reconcile.method import IdwInterp
from ahn_cli.reconcile.reconcile import ReconcileRequest, reconcile
from ahn_cli.reconcile.writers import OutputFormat
from ahn_cli.tiles3d.errors import Tiles3dError
from ahn_cli.tiles3d.sources import load_terrain
from tests.tiles3d.conftest import (
    MAXY,
    MINX,
    RES,
    grid_for_ortho,
    make_ortho,
    synth_rgb,
    write_exr,
)

if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt

_W, _H = 6, 4


@pytest.fixture
def rgb() -> npt.NDArray[np.uint8]:
    """Provide the deterministic test image."""
    return synth_rgb(_W, _H)


@pytest.fixture
def ortho_path(tmp_path: Path, rgb: npt.NDArray[np.uint8]) -> Path:
    """Write the matching orthophoto GeoTIFF."""
    return make_ortho(tmp_path / "ortho.tif", rgb)


@pytest.fixture
def heights_path(tmp_path: Path, rgb: npt.NDArray[np.uint8]) -> Path:
    """Write a reconciled EXR that matches the ortho perfectly."""
    return write_exr(tmp_path / "r.exr", grid_for_ortho(rgb))


def test_load_terrain_happy_path(
    ortho_path: Path, heights_path: Path, rgb: npt.NDArray[np.uint8]
) -> None:
    """A perfectly matching pair loads into a TerrainGrid."""
    terrain = load_terrain(ortho_path, heights_path)
    assert (terrain.width, terrain.height) == (_W, _H)
    assert terrain.transform == (RES, 0.0, MINX, 0.0, -RES, MAXY)
    assert np.array_equal(terrain.rgb, rgb)
    assert terrain.x.dtype == np.float32
    assert terrain.x[0, 0] == np.float32(MINX + RES / 2)
    assert terrain.y[0, 0] == np.float32(MAXY - RES / 2)
    assert np.isfinite(terrain.z).all()


def test_load_terrain_accepts_a_real_reconcile_output(
    tmp_path: Path,
) -> None:
    """The real reconcile pipeline's EXR passes every gate.

    This locks the pixel-centre and colour formulas of the two contexts
    together: if either drifts, this test fails.
    """
    rgb = synth_rgb(6, 6, seed=5)
    ortho = make_ortho(tmp_path / "ortho.tif", rgb)
    rng = np.random.default_rng(6)
    corners = np.array(
        [[MINX, 100.0], [103.0, 100.0], [MINX, MAXY], [103.0, MAXY]]
    )
    xy = np.vstack([rng.uniform(MINX, 103.0, (100, 2)), corners])
    header = laspy.LasHeader(point_format=2)
    header.offsets = [MINX, 100.0, 0.0]
    header.scales = [0.001, 0.001, 0.001]
    las = laspy.LasData(header)
    las.x = xy[:, 0]
    las.y = xy[:, 1]
    las.z = 0.1 * xy[:, 0] + 0.2 * xy[:, 1]
    cloud = tmp_path / "cloud.laz"
    las.write(str(cloud))

    reconcile(
        ReconcileRequest(
            ortho_path=ortho,
            cloud_path=cloud,
            output_dir=tmp_path / "out",
            method=IdwInterp(k=8),
            formats=(OutputFormat.EXR,),
        )
    )

    terrain = load_terrain(ortho, tmp_path / "out" / "reconciled.exr")
    assert (terrain.width, terrain.height) == (6, 6)
    assert np.array_equal(terrain.rgb, rgb)


def test_missing_ortho_is_refused(tmp_path: Path, heights_path: Path) -> None:
    """An unreadable ortho raises the typed error."""
    with pytest.raises(Tiles3dError, match="not readable"):
        load_terrain(tmp_path / "absent.tif", heights_path)


def test_two_band_ortho_is_refused(
    tmp_path: Path, heights_path: Path
) -> None:
    """Fewer than three bands cannot be an orthophoto."""
    rgb = synth_rgb(_W, _H)[:, :, :2]
    ortho = make_ortho(tmp_path / "two.tif", rgb)
    with pytest.raises(Tiles3dError, match="band"):
        load_terrain(ortho, heights_path)


def test_non_uint8_ortho_is_refused(
    tmp_path: Path, heights_path: Path
) -> None:
    """A float orthophoto is not the fetched Beeldmateriaal product."""
    ortho = make_ortho(tmp_path / "f.tif", synth_rgb(_W, _H), dtype="float32")
    with pytest.raises(Tiles3dError, match="uint8"):
        load_terrain(ortho, heights_path)


def test_wrong_crs_ortho_is_refused(
    tmp_path: Path, heights_path: Path
) -> None:
    """The ortho must be EPSG:28992."""
    ortho = make_ortho(
        tmp_path / "wgs.tif", synth_rgb(_W, _H), crs="EPSG:4326"
    )
    with pytest.raises(Tiles3dError, match="28992"):
        load_terrain(ortho, heights_path)


def test_uniform_ortho_is_refused(tmp_path: Path, heights_path: Path) -> None:
    """A single-colour placeholder grid is not real imagery."""
    flat = np.full((_H, _W, 3), 128, dtype=np.uint8)
    ortho = make_ortho(tmp_path / "flat.tif", flat)
    with pytest.raises(Tiles3dError, match="uniform"):
        load_terrain(ortho, heights_path)


def test_dimension_mismatch_is_refused(
    tmp_path: Path, ortho_path: Path
) -> None:
    """A heights grid of any other size is a hard error."""
    other = synth_rgb(_W + 1, _H)
    heights = write_exr(tmp_path / "big.exr", grid_for_ortho(other))
    with pytest.raises(Tiles3dError, match="dimensions"):
        load_terrain(ortho_path, heights)


def test_perturbed_x_plane_is_refused(
    tmp_path: Path, ortho_path: Path, rgb: npt.NDArray[np.uint8]
) -> None:
    """One nudged X coordinate breaks the bit-exact grid match."""
    grid = grid_for_ortho(rgb)
    grid[2, 3, 0] += 0.001
    heights = write_exr(tmp_path / "x.exr", grid)
    with pytest.raises(Tiles3dError, match=r"row=2, col=3"):
        load_terrain(ortho_path, heights)


def test_perturbed_y_plane_is_refused(
    tmp_path: Path, ortho_path: Path, rgb: npt.NDArray[np.uint8]
) -> None:
    """One nudged Y coordinate breaks the bit-exact grid match."""
    grid = grid_for_ortho(rgb)
    grid[1, 0, 1] -= 0.001
    heights = write_exr(tmp_path / "y.exr", grid)
    with pytest.raises(Tiles3dError, match="Y plane"):
        load_terrain(ortho_path, heights)


def test_mismatched_colour_is_refused(
    tmp_path: Path, ortho_path: Path, rgb: npt.NDArray[np.uint8]
) -> None:
    """The EXR colour must be exactly this ortho's colour."""
    grid = grid_for_ortho(rgb)
    grid[0, 0, 4] = (float(rgb[0, 0, 1]) + 1) % 256
    heights = write_exr(tmp_path / "g.exr", grid)
    with pytest.raises(Tiles3dError, match="G plane"):
        load_terrain(ortho_path, heights)


def test_non_finite_height_is_refused(
    tmp_path: Path, ortho_path: Path, rgb: npt.NDArray[np.uint8]
) -> None:
    """A NaN elevation is missing data: a hard error."""
    grid = grid_for_ortho(rgb)
    heights = write_exr(tmp_path / "nan.exr", grid)
    data = bytearray(heights.read_bytes())
    z32 = grid[0, 0, 2].astype(np.float32).tobytes()
    offset = data.index(z32)
    data[offset : offset + 4] = struct.pack("<f", float("nan"))
    heights.write_bytes(bytes(data))
    with pytest.raises(Tiles3dError, match="non-finite"):
        load_terrain(ortho_path, heights)


def test_flat_surface_is_refused(
    tmp_path: Path, ortho_path: Path, rgb: npt.NDArray[np.uint8]
) -> None:
    """A perfectly constant surface is a placeholder, not terrain."""
    grid = grid_for_ortho(rgb, z=np.full((_H, _W), 7.0))
    heights = write_exr(tmp_path / "flat.exr", grid)
    with pytest.raises(Tiles3dError, match="flat"):
        load_terrain(ortho_path, heights)
