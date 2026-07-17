"""Minimal type stub for the parts of the untyped ``rasterio`` API WP7/WP9 use.

``rasterio`` ships no ``py.typed`` marker, so under pyright strict every access
to it is ``Unknown``. This stub declares only ``rasterio.open`` and the handful
of dataset attributes the VIIRS importer reads (CRS, bounds, band count, band
dtypes/descriptions), plus the windowed-read surface the DSM fetcher uses
(``transform``/``res``/``width``/``height``/``nodata``, ``read`` with a
``window``, and ``window_transform``) and the ``write`` method/keywords the
test fixtures use to synthesise GeoTIFFs. It is deliberately partial: it is
typing infrastructure, not a faithful reproduction of the library, and lives
under ``typings/`` (ruff-excluded) so its surface is never linted as
first-party source.
"""

from pathlib import Path

import numpy as np
import numpy.typing as npt

from rasterio.windows import Window

class CRS:
    """A coordinate reference system; only its ``str`` rendering is used."""

    def __str__(self) -> str: ...

class Affine:
    """An affine geo-transform, produced by ``rasterio.transform``."""

class BoundingBox:
    """A raster's spatial extent in its own CRS (left, bottom, right, top)."""

    left: float
    bottom: float
    right: float
    top: float

class DatasetReader:
    """An opened raster dataset, usable as a context manager."""

    @property
    def crs(self) -> CRS | None: ...
    @property
    def bounds(self) -> BoundingBox: ...
    @property
    def count(self) -> int: ...
    @property
    def dtypes(self) -> tuple[str, ...]: ...
    @property
    def descriptions(self) -> tuple[str | None, ...]: ...
    @property
    def transform(self) -> Affine: ...
    @property
    def res(self) -> tuple[float, float]: ...
    @property
    def width(self) -> int: ...
    @property
    def height(self) -> int: ...
    @property
    def nodata(self) -> float | None: ...
    @property
    def profile(self) -> dict[str, object]: ...
    def read(
        self,
        indexes: int | list[int] | None = ...,
        *,
        window: Window | None = ...,
        out_shape: tuple[int, ...] | None = ...,
    ) -> npt.NDArray[np.float32]: ...
    def window_transform(self, window: Window) -> Affine: ...
    def write(self, arr: object, indexes: object = ...) -> None: ...
    def close(self) -> None: ...
    def __enter__(self) -> DatasetReader: ...
    def __exit__(self, *args: object) -> None: ...

def open(
    fp: str | Path,
    mode: str = ...,
    *,
    driver: str | None = ...,
    width: int | None = ...,
    height: int | None = ...,
    count: int | None = ...,
    dtype: str | None = ...,
    crs: str | CRS | None = ...,
    transform: Affine | None = ...,
    nodata: float | None = ...,
    # GDAL creation options (photometric, compress, tiled, blockxsize, ...)
    # vary by driver; accept any so the partial stub need not enumerate them.
    **creation_options: object,
) -> DatasetReader: ...
