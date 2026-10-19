"""Minimal type stub for ``rasterio.features`` used by the footprint raster.

Only :func:`rasterize` with the keywords :mod:`ahn_cli.prep.footprint` passes
is declared. Deliberately partial typing infrastructure.
"""

from collections.abc import Iterable, Mapping

import numpy as np
import numpy.typing as npt

from rasterio import Affine

def rasterize(
    shapes: Iterable[tuple[Mapping[str, object], int]],
    *,
    out_shape: tuple[int, int],
    transform: Affine,
    fill: int = ...,
    dtype: str = ...,
) -> npt.NDArray[np.uint8]: ...
