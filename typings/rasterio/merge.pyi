"""Minimal type stub for ``rasterio.merge.merge`` used by the ortho mosaic.

Declares two overloads of the merge signature WP8/W8 call: the in-memory form
(a sequence of source paths plus the ``bounds``/``res`` keywords that
clip-and-resample to the AOI, returning ``(mosaic_array, transform)``), and
the windowed-write form (adding ``dst_path``/``dst_kwds``, which makes
``merge`` write directly to a raster file in ``mem_limit``-bounded chunks
instead of returning an array). Partial by design: typing infrastructure, not
a faithful reproduction of the library.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, overload

import numpy.typing as npt

from rasterio import Affine, DatasetReader

@overload
def merge(
    sources: Sequence[str | Path | DatasetReader],
    *,
    bounds: tuple[float, float, float, float] | None = ...,
    res: float | tuple[float, float] | None = ...,
    nodata: float | None = ...,
    dtype: str | None = ...,
    method: str = ...,
    target_aligned_pixels: bool = ...,
) -> tuple[npt.NDArray[Any], Affine]: ...
@overload
def merge(
    sources: Sequence[str | Path | DatasetReader],
    *,
    bounds: tuple[float, float, float, float] | None = ...,
    res: float | tuple[float, float] | None = ...,
    nodata: float | None = ...,
    dtype: str | None = ...,
    method: str = ...,
    target_aligned_pixels: bool = ...,
    dst_path: str | Path,
    dst_kwds: dict[str, Any] | None = ...,
) -> None: ...
