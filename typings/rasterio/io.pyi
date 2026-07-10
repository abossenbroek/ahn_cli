"""Minimal type stub for ``rasterio.io.MemoryFile`` used by the DSM fetcher.

The DSM windowed read writes its clipped raster into an in-memory GeoTIFF and
reads the resulting bytes back out, and inspection re-opens those bytes. Only
that surface -- construct from optional bytes, ``open`` with the same writer
keywords as :func:`rasterio.open`, ``read`` the bytes, context-manager use --
is declared. Deliberately partial typing infrastructure.
"""

from rasterio import CRS, Affine, DatasetReader

class MemoryFile:
    """An in-memory file backing a GeoTIFF, usable as a context manager."""

    def __init__(self, file_or_bytes: bytes | None = ...) -> None: ...
    def open(
        self,
        *,
        driver: str | None = ...,
        width: int | None = ...,
        height: int | None = ...,
        count: int | None = ...,
        dtype: str | None = ...,
        crs: str | CRS | None = ...,
        transform: Affine | None = ...,
        nodata: float | None = ...,
    ) -> DatasetReader: ...
    def read(self) -> bytes: ...
    def __enter__(self) -> MemoryFile: ...
    def __exit__(self, *args: object) -> None: ...
