"""Footprint coverage raster: polygons -> fraction of each grid cell covered.

The grid is north-up EPSG:28992 with its upper-left corner at
``(minx, maxy)`` of the box, ``res`` metre cells, ``ceil(extent / res)``
columns and rows (row 0 = north), so a box equal to another product's extent
reproduces that product's cell edges. Each cell's value is the share of a
``supersample x supersample`` set of sub-cell centres that fall inside any
polygon (``rasterio.features.rasterize``, centre rule, no ``all_touched``):
0.0 outside, 1.0 fully inside, fractions on edges. The raster is a cue for
downstream class fusion; it never carries heights or class decisions.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds

from ahn_cli.domain import BBox, ensure_valid_bbox
from ahn_cli.fetch.top10nl import KasFetch, kas_geojson_bytes, kas_provenance
from ahn_cli.provenance import write_provenance

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from datetime import datetime
    from pathlib import Path

    import numpy.typing as npt
    from rasterio import Affine

MAX_CELLS = 400_000_000
"""Upper bound on supersampled cells; larger requests fail before allocating."""
_EDGE_EPS = 1e-9
_SUPERSAMPLE = 4


class FootprintError(ValueError):
    """Raised for an unusable grid request (bad box, resolution, size)."""


def grid_shape(bbox: BBox, res: float) -> tuple[int, int]:
    """Return ``(rows, cols)`` of the ``res`` grid covering ``bbox``.

    Failure modes:
        - :class:`FootprintError` for a degenerate box or ``res <= 0``.
    """
    try:
        ensure_valid_bbox(bbox)
    except ValueError as exc:
        raise FootprintError(str(exc)) from exc
    if not (math.isfinite(res) and res > 0):
        msg = f"resolution must be a positive number of metres; got {res}."
        raise FootprintError(msg)
    minx, miny, maxx, maxy = bbox
    cols = math.ceil((maxx - minx) / res - _EDGE_EPS)
    rows = math.ceil((maxy - miny) / res - _EDGE_EPS)
    return rows, cols


def _transform(bbox: BBox, rows: int, cols: int, cell: float) -> Affine:
    """North-up transform: upper-left at ``(minx, maxy)``, ``cell`` metres."""
    minx, maxy = bbox[0], bbox[3]
    return from_bounds(
        minx, maxy - rows * cell, minx + cols * cell, maxy, cols, rows
    )


def check_grid(bbox: BBox, res: float, supersample: int) -> tuple[int, int]:
    """Validate a cover request without allocating; return ``(rows, cols)``.

    Failure modes:
        - :class:`FootprintError` from :func:`grid_shape`, for
          ``supersample < 1``, or when the supersampled grid exceeds
          :data:`MAX_CELLS`.
    """
    rows, cols = grid_shape(bbox, res)
    if supersample < 1:
        msg = f"supersample must be >= 1; got {supersample}."
        raise FootprintError(msg)
    if rows * cols * supersample * supersample > MAX_CELLS:
        msg = (
            f"grid {rows}x{cols} at supersample {supersample} exceeds "
            f"{MAX_CELLS} cells; use a smaller box or a coarser resolution."
        )
        raise FootprintError(msg)
    return rows, cols


def cover_fraction(
    geometries: Iterable[Mapping[str, object]],
    bbox: BBox,
    res: float,
    supersample: int = _SUPERSAMPLE,
) -> npt.NDArray[np.float32]:
    """Rasterise GeoJSON ``geometries`` to a per-cell coverage fraction.

    Contract:
        - Returns a ``(rows, cols)`` float32 array in ``[0, 1]`` on the grid
          of :func:`grid_shape`; overlapping polygons count once.

    Failure modes:
        - :class:`FootprintError` from :func:`check_grid`.
    """
    rows, cols = check_grid(bbox, res, supersample)
    shapes = [(g, 1) for g in geometries]
    fine = np.zeros((rows * supersample, cols * supersample), dtype=np.uint8)
    if shapes:
        fine = rasterize(
            shapes,
            out_shape=fine.shape,
            transform=_transform(
                bbox,
                rows * supersample,
                cols * supersample,
                res / supersample,
            ),
            fill=0,
            dtype="uint8",
        )
    blocks = fine.reshape(rows, supersample, cols, supersample)
    return (blocks.mean(axis=(1, 3), dtype=np.float64)).astype(np.float32)


def write_cover(
    path: Path, cover: npt.NDArray[np.float32], bbox: BBox, res: float
) -> str:
    """Write ``cover`` as a float32 EPSG:28992 GeoTIFF; return its pixel pin.

    Contract:
        - The returned checksum is the SHA-256 of the C-order little-endian
          float32 pixel bytes (independent of GeoTIFF container bytes).

    Failure modes:
        - :class:`FootprintError` if ``cover``'s shape does not match
          :func:`grid_shape` for ``bbox`` and ``res``.
    """
    if cover.shape != grid_shape(bbox, res):
        msg = f"cover shape {cover.shape} does not match the grid of {bbox}."
        raise FootprintError(msg)
    rows, cols = cover.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=rows,
        width=cols,
        count=1,
        dtype="float32",
        crs="EPSG:28992",
        transform=_transform(bbox, rows, cols, res),
        compress="deflate",
    ) as dst:
        dst.write(cover, 1)
    pixels = np.ascontiguousarray(cover, dtype="<f4").tobytes()
    return hashlib.sha256(pixels).hexdigest()


GEOJSON_NAME = "top10nl_kas.geojson"
COVER_NAME = "kas_cover.tif"
PROVENANCE_NAME = "kas_cover.provenance.json"
_SUBDIR = "footprints"


@dataclass(frozen=True)
class FootprintProduct:
    """Paths and pins of one written footprint product."""

    geojson_path: Path
    cover_path: Path
    provenance_path: Path
    cover_sha256: str
    covered_m2: float


def product_paths(out_dir: Path) -> tuple[Path, Path, Path]:
    """Return the ``(geojson, cover, provenance)`` paths under ``out_dir``."""
    sub = out_dir / _SUBDIR
    return sub / GEOJSON_NAME, sub / COVER_NAME, sub / PROVENANCE_NAME


def preflight(out_dir: Path, bbox: BBox, res: float) -> None:
    """Fail before any network call: bad grid or an existing target file.

    Failure modes:
        - :class:`FootprintError` from :func:`check_grid` (supersample 4), or
          if any :func:`product_paths` file already exists.
    """
    check_grid(bbox, res, _SUPERSAMPLE)
    existing = [p for p in product_paths(out_dir) if p.exists()]
    if existing:
        msg = f"refusing to overwrite {existing[0]}; remove it first."
        raise FootprintError(msg)


def write_footprint_product(
    result: KasFetch,
    out_dir: Path,
    bbox: BBox,
    res: float,
    *,
    started_at: datetime,
    finished_at: datetime,
) -> FootprintProduct:
    """Write the kas GeoJSON, the coverage raster and its provenance.

    Contract:
        - Writes under ``out_dir/footprints/``; the provenance pins the
          canonical feature hash (input) and the raster pixel hash (output).

    Failure modes:
        - :class:`FootprintError` from :func:`preflight` (no silent
          overwrite).
    """
    preflight(out_dir, bbox, res)
    geo_path, cover_path, prov_path = product_paths(out_dir)
    geoms = [
        cast("Mapping[str, object]", f["geometry"]) for f in result.features
    ]
    cover = cover_fraction(geoms, bbox, res)
    geo_path.parent.mkdir(parents=True, exist_ok=True)
    geo_path.write_bytes(kas_geojson_bytes(result))
    sha = write_cover(cover_path, cover, bbox, res)
    write_provenance(
        kas_provenance(
            result,
            bbox,
            started_at=started_at,
            finished_at=finished_at,
            output_checksum=sha,
            resolution_m=res,
        ),
        prov_path,
    )
    return FootprintProduct(
        geojson_path=geo_path,
        cover_path=cover_path,
        provenance_path=prov_path,
        cover_sha256=sha,
        covered_m2=float(cover.sum(dtype=np.float64)) * res * res,
    )
