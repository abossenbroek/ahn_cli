"""Pure data-authenticity predicates: decide if data is genuinely measured.

Every CLI verb's output step runs a hard verification that the data it is
about to emit (or derive an output from) is genuine measured AHN / imagery,
never a placeholder or fabricated infill. This module is the shared, pure
(arrays in, ``bool`` out, no I/O) vocabulary for those gates; each bounded
context wraps the predicates in its own typed error with an actionable
message. Born from the Moerkapelle gray-cloud incident, where a uniform
placeholder grid stood in for the Beeldmateriaal orthophoto and silently
painted every reconciled point the same gray.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

__all__ = ["degenerate_cloud", "flat_surface", "uniform_image"]


def uniform_image(sample: npt.NDArray[np.generic]) -> bool:
    """Report whether the sample carries no per-band pixel variation at all.

    Contract:
        - ``sample`` is a ``(bands, rows, cols)`` (or ``(rows, cols)``) pixel
          array, typically a decimated read of the raster.
        - Only finite values count as imagery: non-finite pixels (NaN/inf in
          float rasters) are masked voids, never variation. Integer samples
          are all-finite by construction and judged exactly as before.
        - Returns ``True`` when every band is constant among its finite
          values (each band anchored to its *own* constant, so a solid
          three-colour placeholder is uniform) or has no finite values --
          the signature of a placeholder grid, since genuine imagery always
          varies. An empty sample, or one with no finite value anywhere, is
          uniform (there is no imagery at all).
        - A single constant band beside varying ones is *not* uniform: real
          photography can saturate one channel.
    """
    if sample.size == 0:
        return True
    bands = sample.reshape(-1, sample.shape[-2] * sample.shape[-1])
    finite = np.isfinite(bands)
    for index in range(bands.shape[0]):
        valid = bands[index][finite[index]]
        if valid.size > 0 and bool(np.any(valid != valid[0])):
            return False
    return True


def flat_surface(
    values: npt.NDArray[np.float32], nodata: float | None
) -> bool:
    """Report whether an elevation raster carries no genuine relief at all.

    Contract:
        - ``values`` is the raster's elevation array; ``nodata`` is its
          declared void value (``None`` when the raster declares none).
        - Returns ``True`` when no valid (finite, non-``nodata``) sample
          exists, or when two or more valid samples are all one identical
          value -- genuine terrain always varies across a surface.
        - A single valid sample is *not* flat: one measurement carries no
          variation to judge.
    """
    finite = values[np.isfinite(values)]
    valid = finite if nodata is None else finite[finite != np.float32(nodata)]
    if valid.size == 0:
        return True
    return valid.size > 1 and bool(np.all(valid == valid[0]))


def degenerate_cloud(
    count: int, mins: npt.ArrayLike, maxs: npt.ArrayLike
) -> bool:
    """Report whether a point cloud is empty or collapsed to one position.

    Contract:
        - ``count`` is the cloud's point count; ``mins``/``maxs`` are its
          per-axis coordinate extremes (e.g. straight from a LAS header).
        - Returns ``True`` for an empty cloud, or for two or more points all
          at one identical XYZ -- stacked duplicates are fabricated data,
          never a genuine scan.
        - A single point is *not* degenerate: it is trivially at "one
          position" but carries no duplication signature.
    """
    if count == 0:
        return True
    return count > 1 and bool(
        np.all(
            np.asarray(mins, dtype=np.float64)
            == np.asarray(maxs, dtype=np.float64)
        )
    )
