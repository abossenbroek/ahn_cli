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

import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ahn_cli.domain.progress import ProgressCallback
from ahn_cli.reconcile.clean import select_and_dedupe
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
    from ahn_cli.reconcile.writers import ReconciledWriter

__all__ = [
    "ProgressCallback",
    "ReconcileError",
    "ReconcileRequest",
    "ReconcileStats",
    "reconcile",
]


def _no_op_progress(_done: int, _total: int) -> None:
    """Report nothing; the default when the caller supplies no callback."""


_RGB_CHANNELS = slice(3, 6)
"""The R, G, B channel slice of the assembled ``(rows, w, 6)`` grid block."""

_BLOCK_CELLS = 262_144
"""Target cell count per row-block; bounds peak memory (kriging builds a
``(cells, k+1, k+1)`` system per block). The row count is derived from this and
the grid width, so the block schedule is a deterministic function of the input."""

_MIN_DISTINCT_POSITIONS = 2
"""A cloud needs at least two distinct XY positions to define a surface."""


@dataclass(frozen=True)
class ReconcileRequest:
    """A validated intent to reconcile one ortho/cloud pair.

    Contract:
        - ``ortho_path`` / ``cloud_path`` are the orthophoto GeoTIFF and AHN LAZ
          to bridge; the ortho defines the output grid.
        - ``output_dir`` receives one ``reconciled.<ext>`` file per format.
        - ``method`` is the validated interpolation request.
        - ``formats`` is the non-empty set of output formats to write.
        - ``include_classes`` / ``exclude_classes`` are the LAS classes to keep /
          drop before interpolation; empty tuples (the default) keep every class.
          Coincident-XY returns are always de-duplicated regardless.

    Invariants:
        - Frozen value object, equal by field value.
    """

    ortho_path: Path
    cloud_path: Path
    output_dir: Path
    method: InterpMethod
    formats: tuple[OutputFormat, ...]
    include_classes: tuple[int, ...] = ()
    exclude_classes: tuple[int, ...] = ()


@dataclass(frozen=True)
class ReconcileStats:
    """The ledger of one reconcile run.

    Contract (fields):
        - ``width`` / ``height``: the output grid's dimensions (the ortho's).
        - ``source_points``: the raw point count read from the cloud.
        - ``cleaned_points``: the point count after the class filter and XY
          de-duplication (what the interpolator actually sees).
        - ``valid_points``: the number of pixels with an interpolated elevation
          (the point count in the laz/ply/pt outputs).
        - ``outputs``: the written file paths, in requested-format order.

    Invariants:
        - Frozen value object, equal by field value.
    """

    width: int
    height: int
    source_points: int
    cleaned_points: int
    valid_points: int
    outputs: tuple[Path, ...]


def reconcile(
    request: ReconcileRequest, *, progress: ProgressCallback | None = None
) -> ReconcileStats:
    """Interpolate the cloud onto the ortho grid and stream every format.

    Contract:
        - Loads ``cloud_path`` (source XYZ) and streams ``ortho_path``'s grid in
          row-blocks, estimating ``Z`` at each pixel centre via ``method`` and
          writing ``<output_dir>/reconciled.<ext>`` for each requested format.
        - Calls ``progress(rows_done, total_rows)`` once per row-block (after it
          is written); defaults to a no-op so callers that don't care about
          progress are unaffected.
        - Returns a :class:`ReconcileStats` with the grid dimensions, the valid
          (interpolated) pixel count, and the output paths.

    Invariants:
        - Deterministic: identical inputs yield byte-identical outputs, flat in
          memory with respect to the grid area.

    Failure modes:
        - :class:`ReconcileError` if an input is missing/unreadable, the
          orthophoto lacks three colour bands, or the orthophoto is a uniform
          single colour (a placeholder grid, not real imagery). The ortho is
          validated *first*, before the (potentially huge) cloud is loaded,
          so a bad ortho fails in milliseconds.
        - :class:`ReconcileError` if the cleaned cloud is not genuine AHN
          data to interpolate from: empty, collapsed to a single XY
          position, or not covering every pixel centre of the orthophoto
          grid. Interpolating any of those would fabricate ("infill")
          elevations, which reconcile never does.
        - :class:`ReconcileError` if any pixel ends up without a genuine
          elevation estimate (e.g. outside linear's convex hull): missing
          data is a hard error, and every partial output is removed
          before raising.
    """
    report = progress if progress is not None else _no_op_progress
    with open_ortho(request.ortho_path) as ortho:
        cloud = load_cloud(request.cloud_path)
        coords = select_and_dedupe(
            cloud.coords,
            cloud.classification,
            request.include_classes,
            request.exclude_classes,
        )
        _verify_source_coords(coords, request.cloud_path)
        _verify_cloud_covers_grid(coords, ortho.grid, request)
        interpolator = build_interpolator(request.method, coords)
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
        try:
            for row_start in range(0, height, block_rows):
                rows = min(block_rows, height - row_start)
                grid_block, mask_block = _assemble_block(
                    ortho, grid, interpolator, row_start, rows
                )
                _verify_block_complete(mask_block, row_start, request)
                valid_points += int(mask_block.sum())
                for writer in writers:
                    writer.write_block(grid_block, mask_block)
                report(row_start + rows, height)

            for writer in writers:
                writer.close()
        except ReconcileError:
            _discard_outputs(writers, outputs)
            raise

    return ReconcileStats(
        width=width,
        height=height,
        source_points=cloud.coords.shape[0],
        cleaned_points=coords.shape[0],
        valid_points=valid_points,
        outputs=outputs,
    )


def _verify_source_coords(
    coords: npt.NDArray[np.float64], cloud_path: Path
) -> None:
    """Hard-verify the cleaned cloud is genuine data to interpolate from.

    An empty cloud, or one whose points all share a single XY position,
    carries no surface to estimate: every produced elevation would be
    fabricated ("infill"), so reconcile refuses to proceed.

    Failure modes:
        - :class:`ReconcileError` if the cleaned cloud is empty or has no
          XY extent.
    """
    if coords.shape[0] == 0:
        msg = (
            f"point cloud at {cloud_path} has no points left after the "
            "class filter and de-duplication; there is nothing genuine to "
            "interpolate from."
        )
        raise ReconcileError(msg)
    xy = coords[:, :2]
    if xy.shape[0] < _MIN_DISTINCT_POSITIONS or bool(np.all(xy == xy[:1])):
        msg = (
            f"point cloud at {cloud_path} collapses to a single XY "
            "position — interpolating a whole grid from it would fabricate "
            '("infill") elevations, which reconcile never does.'
        )
        raise ReconcileError(msg)


def _verify_cloud_covers_grid(
    coords: npt.NDArray[np.float64],
    grid: PixelGrid,
    request: ReconcileRequest,
) -> None:
    """Hard-verify the cloud covers every pixel centre of the ortho grid.

    The two dimensions must match perfectly: every interpolation target
    (pixel centre) must lie inside the cloud's XY bounding box. A pixel
    outside it would be estimated by extrapolating from unrelated points —
    fabricated ("infill") data — so reconcile refuses to proceed.

    Failure modes:
        - :class:`ReconcileError` if any side of the pixel-centre extent
          is not covered by the cloud's XY bbox, naming each uncovered
          side and its shortfall in metres.
    """
    t = grid.transform
    cols = np.array([0.5, grid.width - 0.5])
    rows = np.array([0.5, grid.height - 0.5])
    xs = t[0] * cols[:, np.newaxis] + t[1] * rows[np.newaxis, :] + t[2]
    ys = t[3] * cols[:, np.newaxis] + t[4] * rows[np.newaxis, :] + t[5]
    shortfalls = (
        ("west", float(xs.min()) - float(coords[:, 0].min())),
        ("east", float(coords[:, 0].max()) - float(xs.max())),
        ("south", float(ys.min()) - float(coords[:, 1].min())),
        ("north", float(coords[:, 1].max()) - float(ys.max())),
    )
    uncovered = [
        f"{side} by {-gap:.3f} m" for side, gap in shortfalls if gap < 0
    ]
    if uncovered:
        msg = (
            f"point cloud at {request.cloud_path} does not cover the "
            f"orthophoto grid of {request.ortho_path}: uncovered on the "
            f"{', '.join(uncovered)}; interpolating uncovered pixels "
            'would fabricate ("infill") data. Check that both inputs '
            "cover the same site."
        )
        raise ReconcileError(msg)


def _verify_block_complete(
    mask_block: npt.NDArray[np.bool_],
    row_start: int,
    request: ReconcileRequest,
) -> None:
    """Hard-verify every pixel of this block has a genuine estimate.

    A void cell means the method could not derive an elevation from the
    source cloud (e.g. a pixel centre outside linear's convex hull).
    Writing it would either fabricate a value or leave a silent hole, so
    missing data is a hard error instead.

    Failure modes:
        - :class:`ReconcileError` if any cell of ``mask_block`` is False,
          naming the void count and the affected row range.
    """
    if bool(mask_block.all()):
        return
    voids = int(mask_block.size - int(mask_block.sum()))
    last_row = row_start + mask_block.shape[0] - 1
    msg = (
        f"{voids} pixel(s) in rows {row_start}..{last_row} have "
        f"no genuine elevation estimate from the cloud at "
        f"{request.cloud_path}; missing data is an error — reconcile "
        'never fabricates ("infills") elevations. Use a method that '
        "covers every pixel (e.g. idw) or fix the inputs."
    )
    raise ReconcileError(msg)


def _discard_outputs(
    writers: list[ReconciledWriter], outputs: tuple[Path, ...]
) -> None:
    """Remove every partial output of a rejected run.

    A refused reconcile must leave nothing behind: writers are closed
    best-effort (a half-written stream may legitimately fail to close),
    then each output file and any PLY payload temp is unlinked.
    """
    for writer in writers:
        with contextlib.suppress(Exception):
            writer.close()
    for path in outputs:
        path.unlink(missing_ok=True)
        path.with_name(path.name + ".payload").unlink(missing_ok=True)


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
