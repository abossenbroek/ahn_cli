"""Reconcile-context raster/point-cloud IO (GDAL via rasterio, laspy).

Loads the two inputs the reconcile verb bridges:

* :func:`load_ortho` -- the Beeldmateriaal orthophoto GeoTIFF, read through
  rasterio (GDAL) into its :class:`~ahn_cli.domain.grid.PixelGrid` (the target
  grid) plus its RGB pixels; the reconciled cloud inherits this grid.
* :func:`load_cloud` -- the AHN point cloud LAZ, read through laspy into the
  ``(n, 3)`` world coordinates the interpolation samples from.

:func:`target_coordinates` expands the grid to the per-pixel-centre world XY the
interpolation estimates ``Z`` at. Expected IO failures (a missing or unreadable
file, an ortho without three colour bands) raise the typed :class:`ReconcileError`
so the CLI reports a tidy message rather than a library traceback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import laspy
import numpy as np
import numpy.typing as npt
import rasterio
from rasterio.errors import RasterioIOError

from ahn_cli.domain.grid import GeoTransform, PixelGrid

if TYPE_CHECKING:
    from pathlib import Path

_RGB_BANDS = 3
"""An orthophoto must carry at least three bands (red, green, blue)."""


class ReconcileError(Exception):
    """Raised when a reconcile run cannot proceed for an expected reason.

    Signals a missing or unreadable orthophoto/point-cloud input, an orthophoto
    without three colour bands, or an unknown output format. Defined here beside
    the IO that most often raises it and re-exported from
    :mod:`ahn_cli.reconcile.reconcile`; the CLI catches it to report a clean
    message instead of a traceback.
    """


@dataclass(frozen=True, eq=False)
class OrthoRaster:
    """A loaded orthophoto: its target grid and RGB pixels.

    Contract (fields):
        - ``grid`` is the :class:`~ahn_cli.domain.grid.PixelGrid` describing the
          ortho's dimensions and geotransform -- the grid the reconciled cloud
          is sampled on.
        - ``rgb`` is the ``(height, width, 3)`` ``uint8`` colour array, band
          order red, green, blue.

    ``eq=False``: it wraps a large array, so instances are compared by identity,
    never element-wise.
    """

    grid: PixelGrid
    rgb: npt.NDArray[np.uint8]


def load_ortho(path: Path) -> OrthoRaster:
    """Load an orthophoto GeoTIFF into its target grid and RGB pixels.

    Contract:
        - ``path`` is a readable GeoTIFF with at least three bands; the first
          three are taken as red, green, blue.
        - Returns an :class:`OrthoRaster` whose ``grid`` carries the ortho's
          dimensions and geotransform and whose ``rgb`` is ``(h, w, 3)`` uint8.

    Failure modes:
        - :class:`ReconcileError` if the file is unreadable or has fewer than
          three bands.
    """
    try:
        with rasterio.open(str(path)) as dataset:
            if dataset.count < _RGB_BANDS:
                msg = (
                    f"orthophoto {path} has {dataset.count} band(s); "
                    f"at least {_RGB_BANDS} (RGB) are required."
                )
                raise ReconcileError(msg)
            bands = dataset.read()
            affine = cast("tuple[float, ...]", dataset.transform)
            transform = cast("GeoTransform", affine[:6])
            width = int(dataset.width)
            height = int(dataset.height)
    except RasterioIOError as exc:
        msg = f"orthophoto at {path} is not readable: {exc}"
        raise ReconcileError(msg) from exc

    rgb = np.ascontiguousarray(
        np.transpose(bands[:_RGB_BANDS], (1, 2, 0)), dtype=np.uint8
    )
    grid = PixelGrid(width=width, height=height, transform=transform)
    return OrthoRaster(grid=grid, rgb=rgb)


def load_cloud(path: Path) -> npt.NDArray[np.float64]:
    """Load an AHN point-cloud LAZ into its ``(n, 3)`` world coordinates.

    Contract:
        - ``path`` is a readable LAZ/LAS file.
        - Returns the ``(n, 3)`` ``float64`` array of scaled ``(x, y, z)`` world
          coordinates.

    Failure modes:
        - :class:`ReconcileError` if the file is missing or unreadable.
    """
    try:
        with laspy.open(str(path)) as reader:
            las = reader.read()
            xyz = np.column_stack(
                [
                    np.asarray(las.x, dtype=np.float64),
                    np.asarray(las.y, dtype=np.float64),
                    np.asarray(las.z, dtype=np.float64),
                ]
            )
    except (OSError, laspy.LaspyException) as exc:
        msg = f"point cloud at {path} is not readable: {exc}"
        raise ReconcileError(msg) from exc
    return xyz


def target_coordinates(
    grid: PixelGrid,
) -> tuple[
    npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
]:
    """Return the grid's per-pixel-centre world coordinates.

    Contract:
        - Returns ``(target_xy, eastings, northings)`` where ``eastings`` and
          ``northings`` are the ``(h, w)`` pixel-centre world X / Y and
          ``target_xy`` is the ``(h*w, 2)`` row-major flattening used as the
          interpolation query points.
    """
    eastings = grid.eastings()
    northings = grid.northings()
    target_xy = np.column_stack([eastings.ravel(), northings.ravel()])
    return target_xy, eastings, northings
