"""Terrain loading with the perfect-dimension-match gates.

:func:`load_terrain` reads the two tiles3d inputs — the fetched
orthophoto GeoTIFF (texture ground truth) and the reconciled EXR
(geometry ground truth) — and hard-verifies they describe exactly the
same grid before any tile is built:

* the ortho must be readable, uint8, EPSG:28992, >= 3 bands, and carry
  real imagery (not a uniform placeholder);
* the EXR must parse strictly (:mod:`ahn_cli.tiles3d.exr`);
* both dimensions must match **perfectly**: equal width/height, the
  EXR's X/Y planes bit-equal to the float32 pixel centres derived from
  the ortho's geotransform, and the EXR's colour planes bit-equal to
  ``float32(band / 255)`` of this ortho — proving the heights were
  reconciled from this exact imagery;
* nothing may be missing: every elevation finite, and the surface not a
  constant placeholder.

Any violation raises :class:`Tiles3dError`. Data is never infilled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import rasterio
from rasterio.errors import RasterioIOError

from ahn_cli.domain.authenticity import flat_surface, uniform_image
from ahn_cli.tiles3d.errors import Tiles3dError
from ahn_cli.tiles3d.exr import ReconciledExr, read_reconciled_exr

if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt

    from ahn_cli.domain.grid import GeoTransform

__all__ = ["TerrainGrid", "load_terrain"]

_RGB_BANDS = 3
_UNIFORMITY_SAMPLE = 512
_RD_EPSG = 28992
_RGB_SCALE = 255.0


@dataclass(frozen=True, eq=False)
class TerrainGrid:
    """The verified, perfectly matched ortho + heights grid.

    Contract (fields):
        - ``width``/``height``/``transform``: the shared pixel grid
          (the ortho's, EPSG:28992).
        - ``x``/``y``/``z``: ``(h, w)`` float32 pixel-centre EPSG:28992
          coordinates and NAP elevation, exactly as stored in the EXR.
        - ``rgb``: ``(h, w, 3)`` uint8 ortho colour.

    ``eq=False``: wraps large arrays, so instances compare by identity.
    """

    width: int
    height: int
    transform: GeoTransform
    x: npt.NDArray[np.float32]
    y: npt.NDArray[np.float32]
    z: npt.NDArray[np.float32]
    rgb: npt.NDArray[np.uint8]


def load_terrain(ortho_path: Path, heights_path: Path) -> TerrainGrid:
    """Load and cross-verify the ortho + reconciled-heights pair.

    Contract:
        - Returns a :class:`TerrainGrid` only when both inputs are
          genuine and their dimensions match perfectly (see module
          docstring); every array is exactly what is stored on disk.

    Failure modes:
        - :class:`Tiles3dError` for every violated gate: unreadable or
          non-genuine ortho, malformed EXR, any dimension or bit-level
          X/Y/RGB mismatch, non-finite elevations, or a flat surface.
    """
    rgb, transform = _load_ortho(ortho_path)
    exr = read_reconciled_exr(heights_path)
    height, width = rgb.shape[:2]
    if (exr.width, exr.height) != (width, height):
        msg = (
            f"dimensions do not match: orthophoto {ortho_path} is "
            f"{width}x{height} but reconciled EXR {heights_path} is "
            f"{exr.width}x{exr.height}."
        )
        raise Tiles3dError(msg)
    _verify_pixel_centres(exr, transform, ortho_path, heights_path)
    _verify_colours(exr, rgb, ortho_path, heights_path)
    _verify_heights(exr.z, heights_path)
    return TerrainGrid(
        width=width,
        height=height,
        transform=transform,
        x=exr.x,
        y=exr.y,
        z=exr.z,
        rgb=rgb,
    )


def _load_ortho(
    path: Path,
) -> tuple[npt.NDArray[np.uint8], GeoTransform]:
    """Read the full RGB image after the authenticity gates."""
    try:
        dataset = rasterio.open(str(path))
    except RasterioIOError as exc:
        msg = f"orthophoto at {path} is not readable: {exc}"
        raise Tiles3dError(msg) from exc
    with dataset:
        if dataset.count < _RGB_BANDS:
            msg = (
                f"orthophoto {path} has {dataset.count} band(s); "
                f"at least {_RGB_BANDS} (RGB) are required."
            )
            raise Tiles3dError(msg)
        if any(dtype != "uint8" for dtype in dataset.dtypes[:_RGB_BANDS]):
            msg = (
                f"orthophoto {path} is not 8-bit (uint8) imagery; "
                "that is not the fetched Beeldmateriaal product."
            )
            raise Tiles3dError(msg)
        crs = str(dataset.crs)
        if crs != f"EPSG:{_RD_EPSG}":
            msg = (
                f"orthophoto {path} is not in EPSG:{_RD_EPSG} "
                f"(RD New); found {crs}."
            )
            raise Tiles3dError(msg)
        sample = dataset.read(
            indexes=list(range(1, _RGB_BANDS + 1)),
            out_shape=(
                _RGB_BANDS,
                min(int(dataset.height), _UNIFORMITY_SAMPLE),
                min(int(dataset.width), _UNIFORMITY_SAMPLE),
            ),
        )
        if uniform_image(sample):
            msg = (
                f"orthophoto {path} is a single uniform colour across "
                "every sampled pixel — a placeholder grid, not real "
                "imagery; tiles3d refuses to drape it."
            )
            raise Tiles3dError(msg)
        bands = dataset.read(indexes=list(range(1, _RGB_BANDS + 1)))
        rgb = np.ascontiguousarray(
            np.transpose(bands, (1, 2, 0)), dtype=np.uint8
        )
        affine = cast("tuple[float, ...]", dataset.transform)
        return rgb, cast("GeoTransform", affine[:6])


def _verify_pixel_centres(
    exr: ReconciledExr,
    transform: GeoTransform,
    ortho_path: Path,
    heights_path: Path,
) -> None:
    """Verify the EXR X/Y planes are the ortho's pixel centres, bit-exact.

    The expected planes use the exact arithmetic of reconcile's
    ``block_target_coordinates`` (float64, then float32), so a genuine
    reconcile output matches to the last bit.
    """
    t = transform
    cols = (np.arange(exr.width, dtype=np.float64) + 0.5)[np.newaxis, :]
    rows = (np.arange(exr.height, dtype=np.float64) + 0.5)[:, np.newaxis]
    expected_x = (t[0] * cols + t[1] * rows + t[2]).astype(np.float32)
    expected_y = (t[3] * cols + t[4] * rows + t[5]).astype(np.float32)
    for plane_name, stored, expected in (
        ("X", exr.x, expected_x),
        ("Y", exr.y, expected_y),
    ):
        if not np.array_equal(stored, expected):
            row, col = np.argwhere(stored != expected)[0]
            msg = (
                f"the {plane_name} plane of {heights_path} does not "
                f"equal the pixel centres of {ortho_path}: first "
                f"mismatch at pixel (row={row}, col={col}). The heights "
                "were not reconciled onto this orthophoto's grid."
            )
            raise Tiles3dError(msg)


def _verify_colours(
    exr: ReconciledExr,
    rgb: npt.NDArray[np.uint8],
    ortho_path: Path,
    heights_path: Path,
) -> None:
    """Verify the EXR colour planes are exactly this ortho's colours."""
    for channel, plane_name, stored in (
        (0, "R", exr.r),
        (1, "G", exr.g),
        (2, "B", exr.b),
    ):
        expected = (
            rgb[:, :, channel].astype(np.float64) / _RGB_SCALE
        ).astype(np.float32)
        if not np.array_equal(stored, expected):
            row, col = np.argwhere(stored != expected)[0]
            msg = (
                f"the {plane_name} plane of {heights_path} does not "
                f"equal the colour of {ortho_path}: first mismatch at "
                f"pixel (row={row}, col={col}). The heights were not "
                "reconciled from this orthophoto."
            )
            raise Tiles3dError(msg)


def _verify_heights(z: npt.NDArray[np.float32], heights_path: Path) -> None:
    """Verify no elevation is missing and the surface is genuine."""
    finite = np.isfinite(z)
    if not bool(finite.all()):
        missing = int(z.size - int(finite.sum()))
        msg = (
            f"reconciled EXR {heights_path} carries {missing} "
            "non-finite elevation(s); missing data is a hard error — "
            "tiles3d never infills."
        )
        raise Tiles3dError(msg)
    if flat_surface(z, None):
        msg = (
            f"reconciled EXR {heights_path} is a perfectly flat "
            "surface — a placeholder, not reconciled AHN terrain."
        )
        raise Tiles3dError(msg)
