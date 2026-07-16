"""The tile-streaming contracts: tile identity, context, payloads, and stage.

This module is the load-bearing contract every pipeline workstream builds
against. The executor drives one spatial tile end-to-end through a chain of
:class:`Stage`s, handing an in-RAM :data:`TilePayload` from one stage to the
next; nothing between stages touches disk. The payload evolves along the chain:

    :class:`PointTile`  ->  :class:`GridTile`  ->  :class:`EncodedTile`

Every payload carries its bulk data as **structure-of-arrays contiguous numpy
planes** (separate ``x``/``y``/``z``/... arrays, never a packed structured
dtype), so downstream arithmetic stays on the numpy ufunc/BLAS fast path and the
buffers are page-alignable and zero-copy-ready for an optional GPU path. The
packed record layout is reserved for on-disk spill/output serialization only --
it is I/O, not compute.

Dtype guidance (documented, not enforced, so the §5 ``float32`` layout refactor
can retune it without breaking this contract): point coordinates are ``float64``
where key math needs the precision and ``float32`` elsewhere; ``gps_time`` is
``float64``; ``classification`` is ``uint8``; point ``rgb`` is ``uint16``
``(n, 3)``. Grid heights are ``float32`` and grid colour planes are ``uint8``.
The value objects validate only what is structural and load-bearing: array
rank, C-contiguity, and matching lengths/shapes across the planes.

The array-carrying payloads (:class:`PointTile`, :class:`GridTile`) are frozen
with ``eq=False``: equality by value would compare their numpy planes
element-wise (an ambiguous truth value) and make them unhashable, so they use
identity equality. :class:`TileKey`, :class:`TileContext`, :class:`EncodedBlob`
and :class:`EncodedTile` are fully value-typed and hashable (usable as manifest
keys).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypeAlias, runtime_checkable

from ahn_cli.domain import BBox, ensure_valid_bbox

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    import numpy.typing as npt

__all__ = [
    "EncodedBlob",
    "EncodedTile",
    "GridTile",
    "PointTile",
    "Stage",
    "TileContext",
    "TileKey",
    "TilePayload",
]

_POINT_PLANE_NAMES: tuple[str, ...] = (
    "x",
    "y",
    "z",
    "gps_time",
    "classification",
)
"""The required structure-of-arrays plane names of a :class:`PointTile`."""

_GRID_PLANE_NAMES: tuple[str, ...] = ("heights", "red", "green", "blue")
"""The structure-of-arrays plane names of a :class:`GridTile`."""

_RGB_COMPONENTS = 3
"""Colour components in a point ``rgb`` plane (``(n, 3)``: red, green, blue)."""

_IMAGE_NDIM = 2
"""Rank of a grid plane: a 2-D ``(h, w)`` image."""


@dataclass(frozen=True)
class TileKey:
    """The unified identity of one tile in the sink-driven tiling plan.

    Contract:
        - ``level`` is the quadtree LOD (0 = root/coarsest); leaves are the
          finest level.
        - ``tx``/``ty`` are the tile's column/row within its level; ``tz`` is
          the depth index, ``0`` for the 2.5D terrain tilings used today.
        - All four are non-negative integers.

    Invariants:
        - Immutable and hashable; equal iff every field is equal. This is the
          resumable manifest's key.

    Failure modes:
        - :class:`ValueError` if any component is negative.
    """

    level: int
    tx: int
    ty: int
    tz: int = 0

    def __post_init__(self) -> None:
        """Reject a negative level or tile index."""
        if self.level < 0:
            msg = f"tile level must be non-negative; got {self.level}."
            raise ValueError(msg)
        if self.tx < 0:
            msg = f"tile tx must be non-negative; got {self.tx}."
            raise ValueError(msg)
        if self.ty < 0:
            msg = f"tile ty must be non-negative; got {self.ty}."
            raise ValueError(msg)
        if self.tz < 0:
            msg = f"tile tz must be non-negative; got {self.tz}."
            raise ValueError(msg)


@dataclass(frozen=True)
class TileContext:
    """The per-tile context a stage needs to process one tile.

    Contract:
        - ``key`` is the tile's :class:`TileKey`.
        - ``bbox`` is the tile's extent as a :data:`~ahn_cli.domain.BBox`
          ``(minx, miny, maxx, maxy)`` in EPSG:28992 metres (the tile proper,
          without the halo).
        - ``halo_m`` is the source-overlap margin in metres a stage may read
          beyond ``bbox`` so a tile-edge estimate matches a global run; it is
          finite and ``>= 0`` (``0`` for stages needing no halo).
        - ``workdir`` is the scratch directory the tile may spill into.

    Invariants:
        - Immutable and hashable; equal iff every field is equal.

    Failure modes:
        - :class:`ValueError` if ``bbox`` is degenerate (see
          :func:`~ahn_cli.domain.ensure_valid_bbox`) or ``halo_m`` is
          non-finite or negative.
    """

    key: TileKey
    bbox: BBox
    halo_m: float
    workdir: Path

    def __post_init__(self) -> None:
        """Validate the extent and the halo margin."""
        ensure_valid_bbox(self.bbox)
        if not math.isfinite(self.halo_m) or self.halo_m < 0.0:
            msg = (
                f"halo_m must be finite and non-negative; got {self.halo_m}."
            )
            raise ValueError(msg)


def _ensure_soa_plane(
    name: str, plane: npt.NDArray[np.generic], count: int
) -> None:
    """Reject a point plane that is not a length-``count`` contiguous 1-D array."""
    if plane.ndim != 1:
        msg = f"point plane {name!r} must be 1-D; got ndim {plane.ndim}."
        raise ValueError(msg)
    if not plane.flags["C_CONTIGUOUS"]:
        msg = f"point plane {name!r} must be C-contiguous."
        raise ValueError(msg)
    if plane.shape[0] != count:
        msg = (
            f"point plane {name!r} length {plane.shape[0]} does not match the "
            f"point count {count}."
        )
        raise ValueError(msg)


def _ensure_rgb_plane(rgb: npt.NDArray[np.generic], count: int) -> None:
    """Reject an ``rgb`` plane that is not a contiguous ``(count, 3)`` array."""
    if rgb.shape != (count, _RGB_COMPONENTS):
        msg = (
            f"point rgb must have shape ({count}, {_RGB_COMPONENTS}); "
            f"got {rgb.shape}."
        )
        raise ValueError(msg)
    if not rgb.flags["C_CONTIGUOUS"]:
        msg = "point rgb must be C-contiguous."
        raise ValueError(msg)


@dataclass(frozen=True, eq=False)
class PointTile:
    """A tile's point cloud as structure-of-arrays contiguous planes.

    Contract:
        - ``x``/``y``/``z``/``gps_time`` are 1-D float planes and
          ``classification`` a 1-D integer plane, all of the same length ``n``
          (the point count) and all C-contiguous.
        - ``rgb`` is an optional contiguous ``(n, 3)`` colour plane
          (``None`` for raw AHN, which carries no colour until reconcile).

    Invariants:
        - Frozen; identity equality (holds numpy planes, so not value-hashable).

    Failure modes:
        - :class:`ValueError` if any plane is not the required rank, is not
          C-contiguous, or does not match the point count.
    """

    x: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]
    z: npt.NDArray[np.float64]
    gps_time: npt.NDArray[np.float64]
    classification: npt.NDArray[np.uint8]
    rgb: npt.NDArray[np.uint16] | None = None

    def __post_init__(self) -> None:
        """Validate every plane's rank, contiguity, and length."""
        count = self.x.shape[0]
        planes = (
            self.x,
            self.y,
            self.z,
            self.gps_time,
            self.classification,
        )
        for name, plane in zip(_POINT_PLANE_NAMES, planes, strict=True):
            _ensure_soa_plane(name, plane, count)
        if self.rgb is not None:
            _ensure_rgb_plane(self.rgb, count)


def _ensure_grid_plane(
    name: str, plane: npt.NDArray[np.generic], shape: tuple[int, ...]
) -> None:
    """Reject a grid plane that is not a contiguous 2-D ``shape`` array."""
    if plane.ndim != _IMAGE_NDIM:
        msg = f"grid plane {name!r} must be 2-D; got ndim {plane.ndim}."
        raise ValueError(msg)
    if plane.shape != shape:
        msg = (
            f"grid plane {name!r} shape {plane.shape} does not match the grid "
            f"shape {shape}."
        )
        raise ValueError(msg)
    if not plane.flags["C_CONTIGUOUS"]:
        msg = f"grid plane {name!r} must be C-contiguous."
        raise ValueError(msg)


@dataclass(frozen=True, eq=False)
class GridTile:
    """A tile interpolated onto the ortho pixel grid: heights plus colour.

    Contract:
        - ``heights`` is a 2-D ``(h, w)`` float plane of NAP elevations.
        - ``red``/``green``/``blue`` are 2-D ``(h, w)`` colour planes on the
          same grid.
        - Every cell is a genuine estimate: the grid is fully covered, never
          infilled (the halo floor guarantees this), so there is no validity
          mask.

    Invariants:
        - Frozen; identity equality (holds numpy planes, so not value-hashable).

    Failure modes:
        - :class:`ValueError` if ``heights`` is not 2-D, or a colour plane is
          not 2-D, differs in shape from ``heights``, or is not C-contiguous.
    """

    heights: npt.NDArray[np.float32]
    red: npt.NDArray[np.uint8]
    green: npt.NDArray[np.uint8]
    blue: npt.NDArray[np.uint8]

    def __post_init__(self) -> None:
        """Validate every plane is a contiguous 2-D grid of the same shape."""
        if self.heights.ndim != _IMAGE_NDIM:
            msg = f"grid heights must be 2-D; got ndim {self.heights.ndim}."
            raise ValueError(msg)
        shape = self.heights.shape
        planes = (self.heights, self.red, self.green, self.blue)
        for name, plane in zip(_GRID_PLANE_NAMES, planes, strict=True):
            _ensure_grid_plane(name, plane, shape)


@dataclass(frozen=True)
class EncodedBlob:
    """One named, opaque byte blob a sink encodes for a tile.

    Contract:
        - ``name`` is the blob's non-blank identifier within its tile (e.g.
          ``"geometry"`` or ``"texture"``), used as the on-disk/pack key.
        - ``data`` is the encoded bytes.

    Invariants:
        - Immutable and hashable (``bytes`` and ``str`` compare by value).

    Failure modes:
        - :class:`ValueError` if ``name`` is blank.
    """

    name: str
    data: bytes

    def __post_init__(self) -> None:
        """Reject a blank blob name."""
        if not self.name.strip():
            msg = "encoded blob name must be a non-blank identifier."
            raise ValueError(msg)


@dataclass(frozen=True)
class EncodedTile:
    """A sink's fully-encoded tile: one or more named blobs plus its key.

    Contract:
        - ``key`` is the tile's :class:`TileKey`.
        - ``blobs`` is a non-empty ordered tuple of :class:`EncodedBlob`
          (a strict profile emits one; a packed profile emits geometry +
          texture); the order is the emission order.

    Invariants:
        - Immutable and hashable; equal iff key and blobs are equal.

    Failure modes:
        - :class:`ValueError` if ``blobs`` is empty.
    """

    key: TileKey
    blobs: tuple[EncodedBlob, ...]

    def __post_init__(self) -> None:
        """Reject a tile with no blobs."""
        if not self.blobs:
            msg = "encoded tile must carry at least one blob."
            raise ValueError(msg)


TilePayload: TypeAlias = PointTile | GridTile | EncodedTile
"""The in-RAM handoff between stages: point cloud, grid, or encoded tile."""


@runtime_checkable
class Stage(Protocol):
    """One fused pipeline stage over a single tile.

    A stage is a thin adapter over an existing bounded-context core, scoped to
    one tile. It declares the source halo it needs and transforms one
    :data:`TilePayload` into the next.
    """

    def halo_m(self) -> float:
        """Return the source halo (metres) this stage needs (``0`` for most)."""
        ...

    def run(self, tile: TilePayload, ctx: TileContext) -> TilePayload:
        """Transform ``tile`` in RAM, returning the next payload."""
        ...
