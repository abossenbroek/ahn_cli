"""Minimal type stub for the ``rasterio.windows`` surface the DSM fetcher uses.

Only the :class:`Window` value (its offset/size fields and constructor) and the
:func:`from_bounds` helper that maps a bounding box to a pixel window are
declared -- exactly what the windowed COG read needs. Deliberately partial
typing infrastructure, not a faithful reproduction of the library.
"""

from rasterio import Affine

class Window:
    """A pixel window: column/row offset and width/height, in pixels."""

    col_off: float
    row_off: float
    width: float
    height: float
    def __init__(
        self,
        col_off: float,
        row_off: float,
        width: float,
        height: float,
    ) -> None: ...

def from_bounds(
    left: float,
    bottom: float,
    right: float,
    top: float,
    transform: Affine,
) -> Window: ...
