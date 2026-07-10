"""Import an externally-produced VIIRS GeoTIFF into a site's ``viirs/`` dir.

The VIIRS night-lights raster is produced *outside* this repository (via Google
Earth Engine) and handed to the pipeline as a finished GeoTIFF. This module
does integration only, never a rebuild: it verify-opens the raster, records its
CRS, extent, band count and dtypes plus a content checksum, copies the file
**byte-for-byte untouched** into ``data/<site>/viirs/``, and writes a
:class:`~ahn_cli.domain.Provenance` sidecar. No reprojection, resampling,
re-colormapping or normalisation is ever performed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import datetime
    from pathlib import Path

    from ahn_cli.domain import BBox, Provenance


class ViirsImportError(ValueError):
    """Raised when the source path is not a readable raster.

    Contract:
        - Signals that ``rasterio`` could not open the file as a raster (missing,
          truncated, or not a GeoTIFF at all).
        - Subclasses :class:`ValueError`, so callers may catch either.
    """


@dataclass(frozen=True)
class ViirsRaster:
    """The metadata read from a VIIRS GeoTIFF, plus its content checksum.

    Contract (fields):
        crs: The raster's CRS rendered as a string (e.g. ``"EPSG:4326"``).
        bounds: The raster's extent ``(minx, miny, maxx, maxy)`` in *its own*
            CRS -- untouched, so not necessarily EPSG:28992.
        band_count: The number of raster bands.
        dtypes: The per-band pixel dtypes, in band order.
        checksum: The SHA-256 hex digest of the file's bytes.

    Invariants:
        - Immutable and hashable; equal iff every field is equal.
    """

    crs: str
    bounds: BBox
    band_count: int
    dtypes: tuple[str, ...]
    checksum: str


@dataclass(frozen=True)
class ViirsImport:
    """The result of importing one VIIRS GeoTIFF into a site.

    Contract (fields):
        dest_path: The byte-identical copy written under ``<site>/viirs/``.
        provenance_path: The sidecar written next to ``dest_path``.
        raster: The metadata read from the source raster.
        provenance: The provenance record written to ``provenance_path``.

    Invariants:
        - Immutable and hashable; equal iff every field is equal.
    """

    dest_path: Path
    provenance_path: Path
    raster: ViirsRaster
    provenance: Provenance


def inspect_viirs(source: Path) -> ViirsRaster:  # noqa: ARG001
    """Verify-open ``source`` and read its metadata and content checksum.

    Contract:
        - Opens ``source`` with ``rasterio`` to confirm it is a valid raster,
          then returns its CRS, extent, band count, band dtypes and the SHA-256
          of its bytes.
        - Reads only metadata; the pixel data is never decoded or altered.

    Failure modes:
        - :class:`ViirsImportError` if ``source`` cannot be opened as a raster.
    """
    raise NotImplementedError


def import_viirs(
    source: Path,
    site_dir: Path,
    *,
    clock: Callable[[], datetime] | None = None,  # noqa: ARG001
) -> ViirsImport:
    """Import ``source`` into ``site_dir/viirs/`` and write its provenance.

    Contract:
        - Inspects ``source`` (see :func:`inspect_viirs`), copies it byte-for-
          byte into ``site_dir/viirs/`` under its original name, and writes a
          ``<name>.provenance.json`` sidecar beside it.
        - The copy is byte-preserving, so the input and output checksums are
          identical.
        - ``clock`` supplies the download-window timestamps; it defaults to a
          UTC wall-clock and is injectable for deterministic tests.

    Failure modes:
        - :class:`ViirsImportError` if ``source`` is not a readable raster.
    """
    raise NotImplementedError
