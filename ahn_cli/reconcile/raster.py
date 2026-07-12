"""Reconcile-context raster/point-cloud IO (GDAL via rasterio, laspy).

Loads the two inputs the reconcile verb bridges, shaped for streaming so an
arbitrarily large area never materialises whole:

* :func:`open_ortho` returns an :class:`OrthoReader` -- the orthophoto opened
  through rasterio (GDAL), exposing its :class:`~ahn_cli.domain.grid.PixelGrid`
  (the target grid) and a windowed :meth:`OrthoReader.read_rows` so the RGB is
  read one row-block at a time.
* :func:`load_cloud` reads the AHN point cloud LAZ into a :class:`SourceCloud`
  holding the ``(n, 3)`` world coordinates the interpolation samples from and
  the matching ``(n,)`` per-point classification the cleanup step filters by
  (one tile's cloud fits in memory; a continental run tiles the cloud
  spatially, out of this module's scope).

:func:`block_target_coordinates` expands one row-block of the grid to its
per-pixel-centre world XY without touching the rest of the grid. Expected IO
failures (a missing/unreadable file, an ortho without three colour bands) raise
the typed :class:`ReconcileError`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import laspy
import numpy as np
import numpy.typing as npt
import rasterio
from rasterio.errors import RasterioIOError
from rasterio.windows import Window

from ahn_cli.domain.authenticity import uniform_image
from ahn_cli.domain.grid import GeoTransform, PixelGrid

if TYPE_CHECKING:
    from pathlib import Path
    from types import TracebackType

    from rasterio import DatasetReader
    from typing_extensions import Self

_RGB_BANDS = 3
_UNIFORMITY_SAMPLE = 512  # decimated read size for the placeholder guard
"""An orthophoto must carry at least three bands (red, green, blue)."""


class ReconcileError(Exception):
    """Raised when a reconcile run cannot proceed for an expected reason.

    Signals a missing or unreadable orthophoto/point-cloud input, an orthophoto
    without three colour bands, or an unknown output format. Defined here beside
    the IO that most often raises it and re-exported from
    :mod:`ahn_cli.reconcile.reconcile`; the CLI catches it to report a clean
    message instead of a traceback.
    """


class OrthoReader:
    """An opened orthophoto: its target grid plus windowed RGB reads.

    Holds the rasterio dataset open for the duration of a streaming run; use as
    a context manager so the dataset is always closed.
    """

    def __init__(self, dataset: DatasetReader, grid: PixelGrid) -> None:
        """Wrap an open dataset and its derived pixel grid."""
        self._dataset = dataset
        self.grid = grid

    def read_rows(
        self, row_start: int, row_count: int
    ) -> npt.NDArray[np.uint8]:
        """Read ``row_count`` rows of RGB starting at ``row_start``.

        Returns a ``(row_count, width, 3)`` ``uint8`` array (bands red, green,
        blue) for the requested horizontal strip of the orthophoto.
        """
        window = Window(0, row_start, self.grid.width, row_count)
        bands = self._dataset.read(window=window)
        return np.ascontiguousarray(
            np.transpose(bands[:_RGB_BANDS], (1, 2, 0)), dtype=np.uint8
        )

    def __enter__(self) -> Self:
        """Return self for use as a context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the underlying rasterio dataset."""
        self._dataset.close()


def open_ortho(path: Path) -> OrthoReader:
    """Open an orthophoto GeoTIFF for streaming reads.

    Contract:
        - ``path`` is a readable GeoTIFF with at least three bands (the first
          three are red, green, blue) carrying real imagery: a raster whose
          every sampled RGB pixel is one identical colour is a placeholder
          grid, not an orthophoto, and colouring a cloud from it would
          silently paint every output point that colour (the Moerkapelle
          gray-cloud incident).
        - Returns an :class:`OrthoReader` whose ``grid`` carries the ortho's
          dimensions and geotransform.

    Failure modes:
        - :class:`ReconcileError` if the file is unreadable, has fewer than
          three bands, or is a single uniform colour.
    """
    try:
        dataset = rasterio.open(str(path))
    except RasterioIOError as exc:
        msg = f"orthophoto at {path} is not readable: {exc}"
        raise ReconcileError(msg) from exc
    if dataset.count < _RGB_BANDS:
        band_count = dataset.count
        dataset.close()
        msg = (
            f"orthophoto {path} has {band_count} band(s); "
            f"at least {_RGB_BANDS} (RGB) are required."
        )
        raise ReconcileError(msg)
    sample = dataset.read(
        indexes=list(range(1, _RGB_BANDS + 1)),
        out_shape=(
            _RGB_BANDS,
            min(int(dataset.height), _UNIFORMITY_SAMPLE),
            min(int(dataset.width), _UNIFORMITY_SAMPLE),
        ),
    )
    if uniform_image(sample):
        dataset.close()
        msg = (
            f"orthophoto {path} is a single uniform colour across every "
            "sampled pixel — that is a placeholder grid, not real imagery; "
            "reconcile refuses to colour the cloud from it. Fetch the real "
            "Beeldmateriaal orthophoto (ahn_cli fetch --ortho) first."
        )
        raise ReconcileError(msg)
    affine = cast("tuple[float, ...]", dataset.transform)
    grid = PixelGrid(
        width=int(dataset.width),
        height=int(dataset.height),
        transform=cast("GeoTransform", affine[:6]),
    )
    return OrthoReader(dataset, grid)


@dataclass(frozen=True, eq=False)
class SourceCloud:
    """A loaded AHN point cloud: world coordinates and per-point class.

    Contract (fields):
        - ``coords`` is the ``(n, 3)`` ``float64`` world ``(x, y, z)``.
        - ``classification`` is the matching ``(n,)`` ``uint8`` LAS class, so the
          cleanup step can filter by class before interpolation.

    ``eq=False``: it wraps large arrays, so instances compare by identity.
    """

    coords: npt.NDArray[np.float64]
    classification: npt.NDArray[np.uint8]


def load_cloud(path: Path) -> SourceCloud:
    """Load an AHN point-cloud LAZ into its coordinates and per-point class.

    Contract:
        - ``path`` is a readable LAZ/LAS file.
        - Returns a :class:`SourceCloud` with the ``(n, 3)`` ``float64`` scaled
          world ``(x, y, z)`` and the matching ``(n,)`` ``uint8`` classification.

    Failure modes:
        - :class:`ReconcileError` if the file is missing or unreadable.
    """
    try:
        with laspy.open(str(path)) as reader:
            las = reader.read()
            coords = np.column_stack(
                [
                    np.asarray(las.x, dtype=np.float64),
                    np.asarray(las.y, dtype=np.float64),
                    np.asarray(las.z, dtype=np.float64),
                ]
            )
            classification = np.asarray(las.classification, dtype=np.uint8)
    except (OSError, laspy.LaspyException) as exc:
        msg = f"point cloud at {path} is not readable: {exc}"
        raise ReconcileError(msg) from exc
    return SourceCloud(coords=coords, classification=classification)


def block_target_coordinates(
    grid: PixelGrid, row_start: int, row_count: int
) -> tuple[
    npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
]:
    """Return one row-block's per-pixel-centre world coordinates.

    Contract:
        - Computes the coordinates of rows ``[row_start, row_start + row_count)``
          across the full width, using the grid's pixel-centre convention
          (``(col + 0.5, row + 0.5)``) so a blocked traversal is identical to the
          whole-grid mesh.
        - Returns ``(target_xy, eastings, northings)`` where ``eastings`` and
          ``northings`` are ``(row_count, width)`` and ``target_xy`` is the
          ``(row_count * width, 2)`` row-major flattening.
    """
    t = grid.transform
    cols = (np.arange(grid.width, dtype=np.float64) + 0.5)[np.newaxis, :]
    rows = (
        np.arange(row_start, row_start + row_count, dtype=np.float64) + 0.5
    )[:, np.newaxis]
    eastings = t[0] * cols + t[1] * rows + t[2]
    northings = t[3] * cols + t[4] * rows + t[5]
    target_xy = np.column_stack([eastings.ravel(), northings.ravel()])
    return target_xy, eastings, northings
