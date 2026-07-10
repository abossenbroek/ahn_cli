"""Reconcile orchestration: interpolate the cloud onto the ortho grid, then emit.

:func:`reconcile` is the reconcile context's single entry point. It builds the
interpolator over the AHN cloud once, then streams the orthophoto grid in
**row-blocks** -- reading each strip's RGB windowed, estimating an elevation ``Z``
at every pixel centre, colouring each pixel from the ortho, and pushing the
assembled ``(rows, w, 6)`` block to every requested writer. Because the neighbour
structure is built once and each block is independent, memory stays flat
regardless of area: a 50 km tile streams like a 50 m one.

Determinism: with identical inputs the output is byte-identical across runs, and
a blocked traversal is identical to a whole-grid one (every estimate is
per-pixel; the block schedule is a fixed function of the width). The only source
cost that scales with area is loading the cloud + building its ``cKDTree``; for a
continental run the cloud is tiled spatially, outside this function's scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ahn_cli.reconcile.interpolate import build_interpolator
from ahn_cli.reconcile.raster import (
    ReconcileError,
    block_target_coordinates,
    load_cloud,
    open_ortho,
)
from ahn_cli.reconcile.writers import OutputFormat, open_writer

if TYPE_CHECKING:
    from pathlib import Path

    from ahn_cli.domain.grid import PixelGrid
    from ahn_cli.reconcile.interpolate import Interpolator
    from ahn_cli.reconcile.method import InterpMethod
    from ahn_cli.reconcile.raster import OrthoReader

__all__ = [
    "ReconcileError",
    "ReconcileRequest",
    "ReconcileStats",
    "reconcile",
]

_RGB_CHANNELS = slice(3, 6)
"""The R, G, B channel slice of the assembled ``(rows, w, 6)`` grid block."""

_BLOCK_CELLS = 262_144
"""Target cell count per row-block; bounds peak memory (kriging builds a
``(cells, k+1, k+1)`` system per block). The row count is derived from this and
the grid width, so the block schedule is a deterministic function of the input."""


@dataclass(frozen=True)
class ReconcileRequest:
    """A validated intent to reconcile one ortho/cloud pair.

    Contract:
        - ``ortho_path`` / ``cloud_path`` are the orthophoto GeoTIFF and AHN LAZ
          to bridge; the ortho defines the output grid.
        - ``output_dir`` receives one ``reconciled.<ext>`` file per format.
        - ``method`` is the validated interpolation request.
        - ``formats`` is the non-empty set of output formats to write.

    Invariants:
        - Frozen value object, equal by field value.
    """

    ortho_path: Path
    cloud_path: Path
    output_dir: Path
    method: InterpMethod
    formats: tuple[OutputFormat, ...]


@dataclass(frozen=True)
class ReconcileStats:
    """The ledger of one reconcile run.

    Contract (fields):
        - ``width`` / ``height``: the output grid's dimensions (the ortho's).
        - ``valid_points``: the number of pixels with an interpolated elevation
          (the point count in the laz/ply/pt outputs).
        - ``outputs``: the written file paths, in requested-format order.

    Invariants:
        - Frozen value object, equal by field value.
    """

    width: int
    height: int
    valid_points: int
    outputs: tuple[Path, ...]


def reconcile(request: ReconcileRequest) -> ReconcileStats:
    """Interpolate the cloud onto the ortho grid and stream every format.

    Contract:
        - Loads ``cloud_path`` (source XYZ) and streams ``ortho_path``'s grid in
          row-blocks, estimating ``Z`` at each pixel centre via ``method`` and
          writing ``<output_dir>/reconciled.<ext>`` for each requested format.
        - Returns a :class:`ReconcileStats` with the grid dimensions, the valid
          (interpolated) pixel count, and the output paths.

    Invariants:
        - Deterministic: identical inputs yield byte-identical outputs, flat in
          memory with respect to the grid area.

    Failure modes:
        - :class:`ReconcileError` if an input is missing/unreadable or the
          orthophoto lacks three colour bands.
    """
    cloud = load_cloud(request.cloud_path)
    interpolator = build_interpolator(request.method, cloud)

    with open_ortho(request.ortho_path) as ortho:
        grid = ortho.grid
        width, height = grid.width, grid.height
        x_offset, y_offset = _grid_corner(grid)
        request.output_dir.mkdir(parents=True, exist_ok=True)
        outputs = tuple(
            request.output_dir / f"reconciled.{fmt.value}"
            for fmt in request.formats
        )
        writers = [
            open_writer(fmt, path, width, height, x_offset, y_offset)
            for fmt, path in zip(request.formats, outputs, strict=True)
        ]

        block_rows = max(1, _BLOCK_CELLS // width)
        valid_points = 0
        for row_start in range(0, height, block_rows):
            rows = min(block_rows, height - row_start)
            grid_block, mask_block = _assemble_block(
                ortho, grid, interpolator, row_start, rows
            )
            valid_points += int(mask_block.sum())
            for writer in writers:
                writer.write_block(grid_block, mask_block)

        for writer in writers:
            writer.close()

    return ReconcileStats(
        width=width,
        height=height,
        valid_points=valid_points,
        outputs=outputs,
    )


def _grid_corner(grid: PixelGrid) -> tuple[float, float]:
    """Return the floored SW corner (min easting, min northing) of the grid.

    Used as the LAS coordinate offset -- it must be known before any block and be
    ``<=`` every pixel centre, so it is computed from the four grid corners.
    """
    t = grid.transform
    cols = np.array([0.5, grid.width - 0.5])
    rows = np.array([0.5, grid.height - 0.5])
    xs = t[0] * cols[:, np.newaxis] + t[1] * rows[np.newaxis, :] + t[2]
    ys = t[3] * cols[:, np.newaxis] + t[4] * rows[np.newaxis, :] + t[5]
    return float(np.floor(xs.min())), float(np.floor(ys.min()))


def _assemble_block(
    ortho: OrthoReader,
    grid: PixelGrid,
    interpolator: Interpolator,
    row_start: int,
    rows: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Build one ``(rows, w, 6)`` grid block and its ``(rows, w)`` mask."""
    rgb = ortho.read_rows(row_start, rows)
    target_xy, eastings, northings = block_target_coordinates(
        grid, row_start, rows
    )
    z, valid = interpolator.estimate(target_xy)
    width = grid.width
    grid_block = np.empty((rows, width, 6), dtype=np.float64)
    grid_block[:, :, 0] = eastings
    grid_block[:, :, 1] = northings
    grid_block[:, :, 2] = z.reshape(rows, width)
    grid_block[:, :, _RGB_CHANNELS] = rgb.astype(np.float64)
    return grid_block, valid.reshape(rows, width)
