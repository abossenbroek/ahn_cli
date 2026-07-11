"""Import an externally-produced VIIRS GeoTIFF into a site's ``viirs/`` dir.

The VIIRS night-lights raster is produced *outside* this repository (via Google
Earth Engine) and handed to the pipeline as a finished GeoTIFF. This module
does integration only, never a rebuild: it verify-opens the raster, records its
CRS, extent, band count and dtypes plus a content checksum, copies the file
**byte-for-byte untouched** into ``data/<site>/viirs/``, and writes a
:class:`~ahn_cli.domain.Provenance` sidecar. No reprojection, resampling,
re-colormapping or normalisation is ever performed. A decimated pixel sample
is read (never altered) to hard-verify the raster is genuine imagery, not a
single-value placeholder grid.
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from typing import TYPE_CHECKING

import rasterio
from rasterio.errors import RasterioIOError

from ahn_cli.domain import Product, Provenance
from ahn_cli.domain.authenticity import uniform_image
from ahn_cli.provenance import write_provenance

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from ahn_cli.domain import BBox

_CHUNK_SIZE = 1 << 20  # 1 MiB: bound peak memory when hashing large rasters.
_UNIFORMITY_SAMPLE = 512  # decimated read size for the placeholder guard
_VIIRS_SUBDIR = "viirs"
_SOURCE_PORTAL = "google_earth_engine"
_LICENCE = "public-domain"
_ATTRIBUTION = (
    "VIIRS Day/Night Band (NASA/NOAA), imported via Google Earth Engine"
)


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


def _utcnow() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def _sha256_hex(source: Path) -> str:
    """Return the SHA-256 hex digest of the bytes at ``source``.

    Hashes the file in fixed-size chunks so peak memory stays bounded
    regardless of raster size, matching the streaming byte-preserving copy.
    The digest is byte-identical to hashing the whole file at once.
    """
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while chunk := handle.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_viirs(source: Path) -> ViirsRaster:
    """Verify-open ``source`` and read its metadata and content checksum.

    Contract:
        - Opens ``source`` with ``rasterio`` to confirm it is a valid raster,
          then returns its CRS, extent, band count, band dtypes and the SHA-256
          of its bytes.
        - Reads metadata plus a decimated pixel sample for the authenticity
          gate; the pixel data is never altered (the copy stays byte-exact).

    Failure modes:
        - :class:`ViirsImportError` if ``source`` cannot be opened as a
          raster, or if every sampled pixel carries one identical value — a
          placeholder grid, not genuine VIIRS imagery.
    """
    try:
        with rasterio.open(source) as dataset:
            crs = str(dataset.crs)
            box = dataset.bounds
            # Native-CRS extent, recorded untouched (no reprojection); the CRS
            # itself is captured in provenance request_keys to disambiguate it.
            bounds: BBox = (box.left, box.bottom, box.right, box.top)
            band_count = dataset.count
            dtypes = tuple(dataset.dtypes)
            sample = dataset.read(
                out_shape=(
                    band_count,
                    min(int(dataset.height), _UNIFORMITY_SAMPLE),
                    min(int(dataset.width), _UNIFORMITY_SAMPLE),
                ),
            )
    except RasterioIOError as exc:
        msg = f"{source} is not a readable raster: {exc}"
        raise ViirsImportError(msg) from exc
    if uniform_image(sample):
        msg = (
            f"{source} is a single uniform value across every sampled "
            "pixel — that is a placeholder grid, not genuine VIIRS "
            "night-lights imagery; refusing to import it."
        )
        raise ViirsImportError(msg)
    return ViirsRaster(
        crs=crs,
        bounds=bounds,
        band_count=band_count,
        dtypes=dtypes,
        checksum=_sha256_hex(source),
    )


def import_viirs(
    source: Path,
    site_dir: Path,
    *,
    clock: Callable[[], datetime] | None = None,
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
    tick = _utcnow if clock is None else clock
    raster = inspect_viirs(source)

    viirs_dir = site_dir / _VIIRS_SUBDIR
    viirs_dir.mkdir(parents=True, exist_ok=True)
    dest_path = viirs_dir / source.name
    provenance_path = viirs_dir / f"{source.name}.provenance.json"

    started_at = tick()
    shutil.copyfile(source, dest_path)
    finished_at = tick()

    provenance = Provenance(
        source_portal=_SOURCE_PORTAL,
        product=Product.VIIRS,
        licence=_LICENCE,
        attribution=_ATTRIBUTION,
        bbox=raster.bounds,
        download_started_at=started_at,
        download_finished_at=finished_at,
        input_checksum=raster.checksum,
        output_checksum=raster.checksum,
        tool_version=version("ahn_cli"),
        request_keys=(
            ("source_path", str(source)),
            ("crs", raster.crs),
            ("band_count", str(raster.band_count)),
            ("band_dtypes", ",".join(raster.dtypes)),
        ),
    )
    write_provenance(provenance, provenance_path)
    return ViirsImport(
        dest_path=dest_path,
        provenance_path=provenance_path,
        raster=raster,
        provenance=provenance,
    )
