"""Shared synthetic fixtures for the tiles3d tests.

The EXR fixtures are written with the *real* reconcile writer, so the
strict reader is tested against the exact bytes the pipeline produces;
``corrupt`` then performs byte surgery for the negative gates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

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
