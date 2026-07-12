"""Shared synthetic fixtures for the tiles3d tests.

The EXR fixtures are written with the *real* reconcile writer, so the
strict reader is tested against the exact bytes the pipeline produces;
``corrupt`` then performs byte surgery for the negative gates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import rasterio
from rasterio.transform import from_bounds

from ahn_cli.reconcile.writers import OutputFormat, write_reconciled

if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt


def synth_grid(
    width: int, height: int, seed: int = 0
) -> npt.NDArray[np.float64]:
    """Build a deterministic ``(h, w, 6)`` X, Y, Z, R, G, B grid."""
    rng = np.random.default_rng(seed)
    grid = np.empty((height, width, 6), dtype=np.float64)
    cols = np.arange(width, dtype=np.float64) + 0.5
    rows = np.arange(height, dtype=np.float64) + 0.5
    grid[:, :, 0] = 100.0 + 0.5 * cols[np.newaxis, :]
    grid[:, :, 1] = 103.0 - 0.5 * rows[:, np.newaxis]
    grid[:, :, 2] = rng.uniform(-5.0, 40.0, (height, width))
    grid[:, :, 3:6] = rng.integers(0, 256, (height, width, 3)).astype(
        np.float64
    )
    return grid


def write_exr(path: Path, grid: npt.NDArray[np.float64]) -> Path:
    """Write ``grid`` as reconcile's EXR (all pixels valid)."""
    mask = np.ones(grid.shape[:2], dtype=np.bool_)
    write_reconciled(OutputFormat.EXR, grid, mask, path)
    return path


def corrupt(path: Path, offset: int, new_bytes: bytes) -> None:
    """Overwrite ``len(new_bytes)`` bytes of ``path`` at ``offset``."""
    data = bytearray(path.read_bytes())
    data[offset : offset + len(new_bytes)] = new_bytes
    path.write_bytes(bytes(data))


MINX = 100.0
MAXY = 103.0
RES = 0.5


def make_ortho(
    path: Path,
    rgb: npt.NDArray[np.uint8],
    *,
    crs: str = "EPSG:28992",
    dtype: str = "uint8",
) -> Path:
    """Write ``rgb`` ``(h, w, 3)`` as a GeoTIFF anchored at (100, 103)."""
    height, width = rgb.shape[:2]
    transform = from_bounds(
        MINX, MAXY - height * RES, MINX + width * RES, MAXY, width, height
    )
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=rgb.shape[2],
        dtype=dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        for band in range(rgb.shape[2]):
            dst.write(rgb[:, :, band].astype(dtype), band + 1)
    return path


def synth_rgb(
    width: int, height: int, seed: int = 2
) -> npt.NDArray[np.uint8]:
    """Build a deterministic non-uniform ``(h, w, 3)`` uint8 image."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (height, width, 3)).astype(np.uint8)


def grid_for_ortho(
    rgb: npt.NDArray[np.uint8],
    z: npt.NDArray[np.float64] | None = None,
) -> npt.NDArray[np.float64]:
    """Build the reconciled grid matching :func:`make_ortho`'s transform."""
    height, width = rgb.shape[:2]
    if z is None:
        rng = np.random.default_rng(3)
        z = rng.uniform(-5.0, 40.0, (height, width))
    cols = (np.arange(width, dtype=np.float64) + 0.5)[np.newaxis, :]
    rows = (np.arange(height, dtype=np.float64) + 0.5)[:, np.newaxis]
    grid = np.empty((height, width, 6), dtype=np.float64)
    grid[:, :, 0] = RES * cols + 0.0 * rows + MINX
    grid[:, :, 1] = 0.0 * cols + -RES * rows + MAXY
    grid[:, :, 2] = z
    grid[:, :, 3:6] = rgb.astype(np.float64)
    return grid
