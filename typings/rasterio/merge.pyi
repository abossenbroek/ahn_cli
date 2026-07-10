"""Minimal type stub for ``rasterio.merge.merge`` used by the ortho mosaic.

Declares only the merge signature WP8 calls -- a sequence of source paths plus
the ``bounds`` and ``res`` keywords that clip-and-resample to the AOI -- and its
``(mosaic_array, transform)`` return. Partial by design: typing infrastructure,
not a faithful reproduction of the library.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy.typing as npt

from rasterio import Affine, DatasetReader

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
