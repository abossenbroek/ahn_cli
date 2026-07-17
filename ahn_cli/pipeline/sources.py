"""Concrete, network-free pipeline inputs: the cloud source and ortho windows.

Two on-disk seams feed the tile-streaming executor without any network access:

* :class:`ReadSource` -- a :class:`~ahn_cli.pipeline.executor.TileSource` that,
  per :class:`~ahn_cli.pipeline.model.TileContext`, selects the AHN sheets whose
  extent overlaps the tile's bbox grown by ``halo_m``, reads and concatenates
  their points, crops to that grown box, and returns a
  :class:`~ahn_cli.pipeline.model.PointTile` plus a cheap content hash of the
  source files (their names, sizes and mtimes -- never a bulk re-serialization
  of the points). Sheet extents are read from each LAZ header once at
  construction, so a tile only ever loads the sheets it overlaps -- peak memory
  is bounded by the tile, not the area.

* :class:`WindowedOrtho` -- an
  :class:`~ahn_cli.pipeline.stages.reconcile.OrthoWindows` that reads the global
  orthophoto GeoTIFF windowed per tile and returns a **pixel-aligned**
  sub-window: the sub-grid's transform shifts only the translation coefficients
  (``c' = a*col0 + c``, ``f' = e*row0 + f``), keeping the pixel size, so a
  tiled estimate lands on exactly the global grid's pixel centres. This is the
  seam the reconcile stage's halo-kNN byte-identity depends on.

Both read from files a prior ``fetch`` (or hand-populated site) left on disk, so
a run is deterministic and offline.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import laspy
import numpy as np
import rasterio
from rasterio.windows import Window

from ahn_cli.domain import PixelGrid
from ahn_cli.pipeline.errors import PipelineError
from ahn_cli.pipeline.executor import SourceTile
from ahn_cli.pipeline.model import PointTile
from ahn_cli.pipeline.stages.reconcile import OrthoWindow

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import numpy.typing as npt

    from ahn_cli.domain import BBox, GeoTransform
    from ahn_cli.pipeline.model import TileContext

__all__ = ["ReadSource", "WindowedOrtho", "find_ahn_sheets"]

_LAZ_SUFFIXES = (".laz", ".las")
"""LAS/LAZ file suffixes a :class:`ReadSource` picks up (case-insensitive)."""

_RGB_BANDS = 3
"""The red/green/blue band count a :class:`WindowedOrtho` reads."""

_HASH_FIELD_LEN = 8
"""Bytes of the big-endian length prefix framing each hashed field."""


def find_ahn_sheets(cloud_dir: Path) -> tuple[Path, ...]:
    """Return the LAS/LAZ files directly under ``cloud_dir``, sorted by name.

    Failure modes:
        - :class:`~ahn_cli.pipeline.errors.PipelineError` if ``cloud_dir`` is
          not a directory or holds no LAS/LAZ file.
    """
    if not cloud_dir.is_dir():
        msg = f"cloud source directory {cloud_dir} is not a directory."
        raise PipelineError(msg)
    sheets = sorted(
        path
        for path in cloud_dir.iterdir()
        if path.is_file() and path.suffix.lower() in _LAZ_SUFFIXES
    )
    if not sheets:
        msg = f"cloud source directory {cloud_dir} holds no LAS/LAZ sheet."
        raise PipelineError(msg)
    return tuple(sheets)


@dataclass(frozen=True)
class _Sheet:
    """One AHN sheet on disk plus its horizontal extent (from the LAZ header)."""

    path: Path
    bbox: BBox


def _read_sheet_bbox(path: Path) -> BBox:
    """Read a LAZ file's horizontal ``(minx, miny, maxx, maxy)`` from its header."""
    with laspy.open(str(path)) as reader:
        header = reader.header
        mins = header.mins
        maxs = header.maxs
    return (
        float(mins[0]),
        float(mins[1]),
        float(maxs[0]),
        float(maxs[1]),
    )


def _boxes_overlap(a: BBox, b: BBox) -> bool:
    """Return whether two boxes overlap (touch-inclusive)."""
    return a[0] <= b[2] and a[2] >= b[0] and a[1] <= b[3] and a[3] >= b[1]


class ReadSource:
    """A network-free :class:`~ahn_cli.pipeline.executor.TileSource` over sheets.

    Contract:
        - Constructed from an iterable of AHN LAS/LAZ sheet paths; each sheet's
          horizontal extent is read from its header once at construction.
        - :meth:`load` selects the sheets overlapping the tile's bbox grown by
          ``ctx.halo_m``, reads and concatenates their points, crops to that
          grown box, and returns a
          :class:`~ahn_cli.pipeline.executor.SourceTile` wrapping a
          :class:`~ahn_cli.pipeline.model.PointTile` plus a content hash of the
          source files' identities.

    Invariants:
        - The content hash depends only on the source files' names, sizes and
          mtimes -- cheap and deterministic, changing iff the inputs change.

    Failure modes:
        - :class:`~ahn_cli.pipeline.errors.PipelineError` at construction if no
          sheet is given.
    """

    def __init__(self, sheets: Sequence[Path]) -> None:
        """Index each sheet's extent from its LAZ header."""
        if not sheets:
            msg = "ReadSource needs at least one AHN sheet."
            raise PipelineError(msg)
        self._sheets = tuple(
            _Sheet(path=path, bbox=_read_sheet_bbox(path)) for path in sheets
        )
        self._content_hash = _files_digest(sheets)

    @classmethod
    def from_dir(cls, cloud_dir: Path) -> ReadSource:
        """Build a :class:`ReadSource` over every sheet in ``cloud_dir``."""
        return cls(find_ahn_sheets(cloud_dir))

    def load(self, ctx: TileContext) -> SourceTile:
        """Load the tile's cloud (bbox grown by the halo) as a source tile."""
        minx, miny, maxx, maxy = ctx.bbox
        halo = ctx.halo_m
        grown: BBox = (minx - halo, miny - halo, maxx + halo, maxy + halo)
        planes = [
            _read_cropped(sheet.path, grown)
            for sheet in self._sheets
            if _boxes_overlap(sheet.bbox, grown)
        ]
        tile = _concat_point_tiles(planes)
        return SourceTile(payload=tile, content_hash=self._content_hash)


@dataclass(frozen=True)
class _CloudChunk:
    """One sheet's cropped points as structure-of-arrays planes."""

    x: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]
    z: npt.NDArray[np.float64]
    gps_time: npt.NDArray[np.float64]
    classification: npt.NDArray[np.uint8]
    rgb: npt.NDArray[np.uint16] | None


def _read_cropped(path: Path, grown: BBox) -> _CloudChunk:
    """Read ``path``'s points cropped to ``grown`` (closed box) as planes."""
    with laspy.open(str(path)) as reader:
        las = reader.read()
    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    minx, miny, maxx, maxy = grown
    keep = (x >= minx) & (x <= maxx) & (y >= miny) & (y <= maxy)
    rgb = None
    if "red" in las.point_format.dimension_names:
        rgb = np.ascontiguousarray(
            np.column_stack(
                [
                    np.asarray(las.red)[keep],
                    np.asarray(las.green)[keep],
                    np.asarray(las.blue)[keep],
                ]
            ).astype(np.uint16)
        )
    return _CloudChunk(
        x=np.ascontiguousarray(x[keep]),
        y=np.ascontiguousarray(y[keep]),
        z=np.ascontiguousarray(np.asarray(las.z, dtype=np.float64)[keep]),
        gps_time=np.ascontiguousarray(
            np.asarray(las.gps_time, dtype=np.float64)[keep]
        ),
        classification=np.ascontiguousarray(
            np.asarray(las.classification, dtype=np.uint8)[keep]
        ),
        rgb=rgb,
    )


def _concat_point_tiles(chunks: list[_CloudChunk]) -> PointTile:
    """Concatenate per-sheet chunks into one :class:`PointTile` (sheet order)."""
    if not chunks:
        empty_f = np.zeros(0, dtype=np.float64)
        return PointTile(
            x=empty_f,
            y=np.zeros(0, dtype=np.float64),
            z=np.zeros(0, dtype=np.float64),
            gps_time=np.zeros(0, dtype=np.float64),
            classification=np.zeros(0, dtype=np.uint8),
        )
    rgb_planes = [chunk.rgb for chunk in chunks if chunk.rgb is not None]
    rgb: npt.NDArray[np.uint16] | None = (
        np.ascontiguousarray(np.concatenate(rgb_planes))
        if len(rgb_planes) == len(chunks)
        else None
    )
    return PointTile(
        x=np.ascontiguousarray(np.concatenate([c.x for c in chunks])),
        y=np.ascontiguousarray(np.concatenate([c.y for c in chunks])),
        z=np.ascontiguousarray(np.concatenate([c.z for c in chunks])),
        gps_time=np.ascontiguousarray(
            np.concatenate([c.gps_time for c in chunks])
        ),
        classification=np.ascontiguousarray(
            np.concatenate([c.classification for c in chunks])
        ),
        rgb=rgb,
    )


def _files_digest(paths: Sequence[Path]) -> str:
    """Return a deterministic digest over the source files' identities."""
    digest = hashlib.sha256()
    for path in sorted(paths):
        stat = path.stat()
        for field in (
            path.name.encode("utf-8"),
            str(stat.st_size).encode("ascii"),
            str(stat.st_mtime_ns).encode("ascii"),
        ):
            digest.update(len(field).to_bytes(_HASH_FIELD_LEN, "big"))
            digest.update(field)
    return digest.hexdigest()


class WindowedOrtho:
    """An :class:`~ahn_cli.pipeline.stages.reconcile.OrthoWindows` over a GeoTIFF.

    Contract:
        - Constructed from the global orthophoto GeoTIFF path; its pixel grid is
          read once from the raster's affine transform.
        - :meth:`window` returns the tile's pixel-aligned
          :class:`~ahn_cli.pipeline.stages.reconcile.OrthoWindow`: a sub-grid
          whose pixel centres coincide with the global grid's, plus the windowed
          ``(h, w, 3)`` RGB pixels read from the raster.

    Invariants:
        - The sub-window transform shifts only the translation coefficients, so
          a tile's pixel centres are exactly the global grid's -- byte-identity
          across tilings depends on it.

    Failure modes:
        - :class:`~ahn_cli.pipeline.errors.PipelineError` if the raster has
          fewer than three bands, or a tile's window escapes the raster.
    """

    def __init__(self, ortho_path: Path) -> None:
        """Read the global grid geometry from ``ortho_path`` once."""
        self._path = ortho_path
        with rasterio.open(ortho_path) as dataset:
            if dataset.count < _RGB_BANDS:
                msg = (
                    f"ortho {ortho_path} has {dataset.count} band(s); need at "
                    f"least {_RGB_BANDS} for RGB."
                )
                raise PipelineError(msg)
            affine = cast("tuple[float, ...]", dataset.transform)
            self._grid = PixelGrid(
                width=int(dataset.width),
                height=int(dataset.height),
                transform=cast("GeoTransform", affine[:6]),
            )

    @property
    def grid(self) -> PixelGrid:
        """Return the global orthophoto pixel grid."""
        return self._grid

    def window(self, ctx: TileContext) -> OrthoWindow:
        """Return the pixel-aligned ortho window for ``ctx``'s tile."""
        col0, col1, row0, row1 = self._pixel_span(ctx.bbox)
        a, b, c, d, e, f = self._grid.transform
        sub_transform = (a, b, a * col0 + c, d, e, e * row0 + f)
        sub_grid = PixelGrid(
            width=col1 - col0, height=row1 - row0, transform=sub_transform
        )
        rgb = self._read_rgb(col0, row0, col1 - col0, row1 - row0)
        return OrthoWindow(grid=sub_grid, rgb=rgb)

    def _pixel_span(self, bbox: BBox) -> tuple[int, int, int, int]:
        """Return the ``(col0, col1, row0, row1)`` pixel span of ``bbox``."""
        a, _b, c, _d, e, f = self._grid.transform
        minx, miny, maxx, maxy = bbox
        col0 = round((minx - c) / a)
        col1 = round((maxx - c) / a)
        r_top = round((maxy - f) / e)
        r_bot = round((miny - f) / e)
        row0, row1 = min(r_top, r_bot), max(r_top, r_bot)
        if (
            col0 < 0
            or row0 < 0
            or col1 > self._grid.width
            or row1 > self._grid.height
            or col1 <= col0
            or row1 <= row0
        ):
            msg = (
                f"tile bbox {bbox} maps to pixel window "
                f"[{col0}:{col1}, {row0}:{row1}] outside the "
                f"{self._grid.width}x{self._grid.height} ortho."
            )
            raise PipelineError(msg)
        return col0, col1, row0, row1

    def _read_rgb(
        self, col0: int, row0: int, width: int, height: int
    ) -> npt.NDArray[np.uint8]:
        """Read the ``(height, width, 3)`` RGB window from the raster."""
        window = Window(col0, row0, width, height)
        with rasterio.open(self._path) as dataset:
            bands = cast(
                "npt.NDArray[np.generic]",
                dataset.read(indexes=[1, 2, 3], window=window),
            )
        return np.ascontiguousarray(
            np.transpose(bands, (1, 2, 0)).astype(np.uint8)
        )
