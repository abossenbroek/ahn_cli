"""Prep-context point-cloud export: a LAZ/COPC cloud to a lossless binary PLY.

TouchDesigner, Blender and the flyover engine ingest PLY; this transform turns
a fetched/processed LAZ (or COPC) cloud into a ``.ply`` deliverable without
silently losing the datum, the classification, the colour or the lidar
attributes. Properties of the output:

1. **Datum kept.** Coordinates are written either as absolute ``double``
   (EPSG:28992 eastings near 194000 m need float64 for sub-centimetre
   position) or as ``float32`` centred on a recorded offset. In both cases the
   header carries ``comment ahn_cli.offset_xyz X Y Z`` with
   ``world = vertex + offset`` (zero for ``double``), plus the source LAS scale
   and offset, and the sidecar repeats them (see
   :mod:`ahn_cli.prep.ply_product`).
2. **Attributes kept, additively.** After ``x y z`` come ``red green blue``
   (``uchar``, only when the source point format has colour), then
   ``classification`` (``uchar``), ``intensity`` (``ushort``),
   ``return_number`` and ``number_of_returns`` (``uchar``) and, for point
   formats 6+, ``overlap`` (``uchar`` 0/1). A reader that only knows
   ``x y z`` (+ ``red green blue``) skips the rest by stride.
3. **Thinning by spacing, not share.** ``PoissonThinning`` keeps points at
   least ``radius`` apart (a hard minimum-distance guarantee); the old fixed
   random share survives only as :class:`ShareThinning`, and is recorded.
4. **Memory-bounded streaming** for the no-thinning path: two passes over
   bounded windows (a scan for count/colour depth/extent, then the write).
   Thinning also keeps the selected coordinates (24 bytes per point) in memory,
   so it refuses clouds above :data:`MAX_THINNING_POINTS`; clip with ``bbox``.
5. **Deterministic.** Identical input and parameters yield byte-identical
   output, whatever ``chunk_size`` is. The header has no date, host or path.

Colour: LAS colour is 16-bit. AHN data stores 8-bit values in the 16-bit
fields (maximum 255). The ``uchar`` channel is exact when every selected value
fits a byte; otherwise the high byte is written and the narrowing is recorded
in :class:`ColourRecord` (decided from all selected points, never a sample).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Final

import laspy
import numpy as np
import numpy.typing as npt
from laspy.copc import Bounds, CopcReader

from ahn_cli.domain.tile import BBox, ensure_valid_bbox
from ahn_cli.prep.decimate import (
    NumpyBackend,
    PoissonThinning,
    decimate_poisson,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from ahn_cli.domain import ProgressCallback

DEFAULT_CHUNK_SIZE = 1_000_000
"""Points read/written per streaming window when none is supplied."""

MAX_THINNING_POINTS = 60_000_000
"""Most selected points a thinning run holds in memory (24 B each)."""

_MAX_CENTRED_HALF_EXTENT_M = 16_384.0
"""Largest |vertex| for float32-centred output: half-ulp stays under 1 mm."""

_MM: Final = 3
"""Decimals of the centring offset (millimetres), so it is exact to 1 mm."""

_AXES: Final = ("x", "y", "z")
_BYTE_MAX: Final = 255
_N_CLASS_CODES: Final = 256

_PLY_TYPE_NAMES: Final = {
    ("f", 8): "double",
    ("f", 4): "float",
    ("u", 1): "uchar",
    ("u", 2): "ushort",
}


class CoordMode(Enum):
    """How vertex coordinates are stored (the ``--coords`` tokens)."""

    FLOAT64 = "float64"
    FLOAT32_CENTRED = "float32-centred"


class PlyExportError(ValueError):
    """Raised when an export cannot proceed for an expected reason.

    Signals an empty selection, a cloud too large for in-memory thinning, or an
    extent too large for float32-centred output. Subclasses :class:`ValueError`.
    """


@dataclass(frozen=True)
class ShareThinning:
    """Keep a seeded random ``share`` of the selected points (legacy rule).

    Contract:
        - ``share`` is finite and in ``(0, 1]``; ``seed`` seeds the draw.
        - Density-blind: a fixed share over-thins dense tiles and under-thins
          sparse ones. Prefer :class:`~ahn_cli.prep.decimate.PoissonThinning`.

    Failure modes:
        - :class:`ValueError` if ``share`` is outside ``(0, 1]``.
    """

    share: float
    seed: int = 0

    def __post_init__(self) -> None:
        """Reject a share outside ``(0, 1]``."""
        if not (math.isfinite(self.share) and 0.0 < self.share <= 1.0):
            msg = f"share must be in (0, 1]; got {self.share}."
            raise ValueError(msg)


PlyThinning = PoissonThinning | ShareThinning
"""Thinning rules an export accepts: spacing (Poisson-disk) or legacy share."""


@dataclass(frozen=True)
class ColourRecord:
    """How the colour channels were encoded.

    Contract:
        - ``source_max`` is the largest red/green/blue value over the selected
          points; ``narrowed`` is ``True`` when it exceeded 255 and the high
          byte was written (lossy); otherwise the byte equals the source value.
    """

    source_max: int
    narrowed: bool


@dataclass(frozen=True)
class PlyExportStats:
    """The ledger of one PLY export.

    Contract:
        - ``point_count``: vertices written.
        - ``source_points``: the source header's point count.
        - ``scanned_points``: source points inside ``bbox`` (all when no bbox).
        - ``selected_points``: scanned points passing the class filter, before
          thinning.
        - ``classes_scanned`` / ``classes_out``: sorted ``(class, count)`` pairs
          over the scanned points and over the written vertices.
        - ``properties``: ordered ``(name, ply type)`` of the vertex element.
        - ``coords`` / ``offset_xyz``: storage mode and ``world = vertex +
          offset``.
        - ``las_scale`` / ``las_offset``: the source header's scale and offset.
        - ``colour``: encoding record, ``None`` when the source has no colour.

    Invariants:
        - Frozen value object, equal by field value.
    """

    point_count: int
    source_points: int = 0
    scanned_points: int = 0
    selected_points: int = 0
    classes_scanned: tuple[tuple[int, int], ...] = ()
    classes_out: tuple[tuple[int, int], ...] = ()
    properties: tuple[tuple[str, str], ...] = ()
    coords: CoordMode = CoordMode.FLOAT64
    offset_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    las_scale: tuple[float, float, float] = (0.0, 0.0, 0.0)
    las_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    colour: ColourRecord | None = None


@dataclass(frozen=True)
class _Scan:
    """Pass-1 result: what the selection holds, before anything is written."""

    chunks: int
    scanned: int
    selected: int
    classes: npt.NDArray[np.int64]
    colour_max: int
    mins: npt.NDArray[np.float64]
    maxs: npt.NDArray[np.float64]
    coords: npt.NDArray[np.float64] | None


def _no_op_progress(_done: int, _total: int) -> None:
    """Report nothing; the default when the caller supplies no callback."""


def _is_copc(header: laspy.LasHeader) -> bool:
    """Return whether ``header`` carries a COPC info VLR."""
    return any(getattr(vlr, "user_id", "") == "copc" for vlr in header.vlrs)


def _iter_chunks(
    source: Path,
    bbox: BBox | None,
    chunk_size: int,
    *,
    copc: bool,
) -> Iterator[laspy.ScaleAwarePointRecord]:
    """Yield the cloud in windows; a COPC with a bbox reads only its nodes."""
    if copc and bbox is not None:
        bounds = Bounds(
            mins=np.array(bbox[:2], dtype=np.float64),
            maxs=np.array(bbox[2:], dtype=np.float64),
        )
        with CopcReader.open(str(source)) as copc_reader:
            records = copc_reader.query(bounds)
        for start in range(0, len(records), chunk_size):
            yield records[start : start + chunk_size]
    else:
        with laspy.open(str(source)) as reader:
            yield from reader.chunk_iterator(chunk_size)


def _bbox_mask(
    chunk: laspy.ScaleAwarePointRecord, bbox: BBox | None
) -> npt.NDArray[np.bool_]:
    """Return which chunk points fall inside ``bbox`` (all, when ``None``)."""
    x = np.asarray(chunk.x, dtype=np.float64)
    if bbox is None:
        everything: npt.NDArray[np.bool_] = np.ones(len(x), dtype=np.bool_)
        return everything
    y = np.asarray(chunk.y, dtype=np.float64)
    return (x >= bbox[0]) & (x <= bbox[2]) & (y >= bbox[1]) & (y <= bbox[3])


def _selection(
    chunk: laspy.ScaleAwarePointRecord,
    include: tuple[int, ...],
    bbox: BBox | None,
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    """Return ``(in_bbox, selected)`` masks for one window."""
    in_bbox = _bbox_mask(chunk, bbox)
    if not include:
        return in_bbox, in_bbox
    classified = np.isin(
        np.asarray(chunk.classification), np.asarray(include)
    )
    return in_bbox, in_bbox & classified


def _colour_max(
    chunk: laspy.ScaleAwarePointRecord, mask: npt.NDArray[np.bool_]
) -> int:
    """Largest red/green/blue value among the masked points (0 if none)."""
    return max(
        int(np.asarray(chunk[name])[mask].max())
        for name in ("red", "green", "blue")
    )


def _scan(
    source: Path,
    include: tuple[int, ...],
    bbox: BBox | None,
    chunk_size: int,
    *,
    copc: bool,
    has_rgb: bool,
    keep_coords: bool,
) -> _Scan:
    """Pass 1: count, histogram, colour depth, extent (and coords to thin)."""
    classes = np.zeros(_N_CLASS_CODES, dtype=np.int64)
    mins = np.full(3, np.inf)
    maxs = np.full(3, -np.inf)
    kept: list[npt.NDArray[np.float64]] = []
    chunks = scanned = selected = colour_max = 0
    for chunk in _iter_chunks(source, bbox, chunk_size, copc=copc):
        chunks += 1
        in_bbox, mask = _selection(chunk, include, bbox)
        scanned += int(in_bbox.sum())
        classes += np.bincount(
            np.asarray(chunk.classification)[in_bbox].astype(np.int64),
            minlength=_N_CLASS_CODES,
        )
        count = int(mask.sum())
        if count == 0:
            continue
        selected += count
        xyz = np.column_stack(
            [
                np.asarray(chunk[axis], dtype=np.float64)[mask]
                for axis in "xyz"
            ]
        )
        mins = np.minimum(mins, xyz.min(axis=0))
        maxs = np.maximum(maxs, xyz.max(axis=0))
        if has_rgb:
            colour_max = max(colour_max, _colour_max(chunk, mask))
        if keep_coords:
            if selected > MAX_THINNING_POINTS:
                msg = (
                    f"{selected} selected points exceed the in-memory "
                    f"thinning limit of {MAX_THINNING_POINTS}; clip with "
                    "--bbox first."
                )
                raise PlyExportError(msg)
            kept.append(xyz)
    coords = np.vstack(kept) if kept else None
    return _Scan(
        chunks, scanned, selected, classes, colour_max, mins, maxs, coords
    )


def _keep_mask(
    coords: npt.NDArray[np.float64], thinning: PlyThinning
) -> npt.NDArray[np.bool_]:
    """Return the per-selected-point keep mask of a thinning rule."""
    keep = np.zeros(coords.shape[0], dtype=np.bool_)
    if isinstance(thinning, PoissonThinning):
        kept = decimate_poisson(
            coords,
            thinning.radius,
            seed=thinning.seed,
            backend=NumpyBackend(),
        )
    else:
        total = coords.shape[0]
        want = max(1, round(total * thinning.share))
        kept = np.random.default_rng(thinning.seed).choice(
            total, size=want, replace=False
        )
    keep[kept] = True
    return keep


def _centring_offset(scan: _Scan) -> tuple[float, float, float]:
    """Return the millimetre-exact offset: x/y extent midpoint, z minimum.

    Failure modes:
        - :class:`PlyExportError` if any centred coordinate would leave
          float32's 1 mm range (half-extent above 16384 m): use ``float64``.
    """
    mid = np.round((scan.mins + scan.maxs) / 2.0, _MM)
    offset = (
        float(mid[0]),
        float(mid[1]),
        round(float(scan.mins[2]), _MM),
    )
    reach = max(
        float(np.abs(scan.mins - offset).max()),
        float(np.abs(scan.maxs - offset).max()),
    )
    if reach > _MAX_CENTRED_HALF_EXTENT_M:
        msg = (
            f"extent reaches {reach:.0f} m from the centring offset; "
            "float32 would lose more than 1 mm. Use --coords float64."
        )
        raise PlyExportError(msg)
    return offset


def _vertex_dtype(
    coords: CoordMode, *, has_rgb: bool, has_overlap: bool
) -> np.dtype[np.void]:
    """Build the packed little-endian vertex dtype (x y z [rgb] attrs...)."""
    float_type = "<f8" if coords is CoordMode.FLOAT64 else "<f4"
    fields: list[tuple[str, str]] = [(axis, float_type) for axis in "xyz"]
    if has_rgb:
        fields += [(name, "u1") for name in ("red", "green", "blue")]
    fields += [
        ("classification", "u1"),
        ("intensity", "<u2"),
        ("return_number", "u1"),
        ("number_of_returns", "u1"),
    ]
    if has_overlap:
        fields.append(("overlap", "u1"))
    return np.dtype(fields)


def _properties(dtype: np.dtype[np.void]) -> tuple[tuple[str, str], ...]:
    """Return ``(name, ply type)`` for each field of a vertex dtype."""
    names = dtype.names or ()
    return tuple(
        (name, _PLY_TYPE_NAMES[(dtype[name].kind, dtype[name].itemsize)])
        for name in names
    )


def _header(
    stats_count: int,
    properties: tuple[tuple[str, str], ...],
    coords: CoordMode,
    offset: tuple[float, float, float],
    las_scale: tuple[float, float, float],
    las_offset: tuple[float, float, float],
) -> bytes:
    """Render the deterministic PLY header (no date, host or path)."""
    lines = [
        "ply",
        "format binary_little_endian 1.0",
        "comment ahn_cli lossless ply export",
        f"comment ahn_cli.coords {coords.value}",
        "comment ahn_cli.offset_rule world = vertex + offset_xyz",
        "comment ahn_cli.offset_xyz " + " ".join(repr(v) for v in offset),
        "comment ahn_cli.las_scale " + " ".join(repr(v) for v in las_scale),
        "comment ahn_cli.las_offset " + " ".join(repr(v) for v in las_offset),
        f"element vertex {stats_count}",
        *(f"property {kind} {name}" for name, kind in properties),
        "end_header",
    ]
    return ("\n".join(lines) + "\n").encode("ascii")


def _pack(
    chunk: laspy.ScaleAwarePointRecord,
    mask: npt.NDArray[np.bool_],
    dtype: np.dtype[np.void],
    offset: tuple[float, float, float],
    *,
    shift: int,
) -> npt.NDArray[np.void]:
    """Pack the masked points of one window into vertex records."""
    block = np.empty(int(mask.sum()), dtype=dtype)
    for axis, shift_by in zip("xyz", offset, strict=True):
        block[axis] = (
            np.asarray(chunk[axis], dtype=np.float64)[mask] - shift_by
        )
    for name in block.dtype.names or ():
        if name in _AXES:
            continue
        values = np.asarray(chunk[name])[mask]
        if name in {"red", "green", "blue"}:
            values = values >> shift
        block[name] = values
    return block


def _pairs(counts: npt.NDArray[np.int64]) -> tuple[tuple[int, int], ...]:
    """Return the non-zero ``(class, count)`` pairs of a histogram."""
    return tuple((int(c), int(counts[c])) for c in np.flatnonzero(counts))


def export_ply(
    source_path: Path,
    output_path: Path,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    progress: ProgressCallback | None = None,
    include_classes: tuple[int, ...] = (),
    bbox: BBox | None = None,
    thinning: PlyThinning | None = None,
    coords: CoordMode = CoordMode.FLOAT64,
) -> PlyExportStats:
    """Stream a LAZ/COPC point cloud to a lossless binary little-endian PLY.

    Contract:
        - ``source_path`` is a readable LAS/LAZ/COPC file; ``output_path``
          receives ``binary_little_endian 1.0`` with one ``vertex`` element
          whose properties are ``x y z``, then ``red green blue`` when the
          source has colour, then ``classification intensity return_number
          number_of_returns`` and ``overlap`` for point formats 6+.
        - ``include_classes`` keeps only those LAS classes (empty keeps all);
          ``bbox`` (``minx, miny, maxx, maxy``) keeps only points inside it and,
          for a COPC source, reads only the octree nodes that intersect it.
        - ``thinning`` applies after both filters. Source order is preserved.
        - ``coords`` is ``FLOAT64`` (absolute) or ``FLOAT32_CENTRED`` (offset
          from the selected extent, rounded to 1 mm, recorded in the header).
        - ``chunk_size`` bounds the window; it must be positive. ``progress``
          is called ``(chunks_done, total_chunks)`` once per written window.

    Invariants:
        - Deterministic and independent of ``chunk_size``.
        - Without ``thinning`` the cloud is never fully loaded.

    Failure modes:
        - ``ValueError`` for a non-positive ``chunk_size`` or a degenerate
          ``bbox``.
        - :class:`PlyExportError` for an empty selection, an in-memory
          thinning above :data:`MAX_THINNING_POINTS`, or a float32-centred
          extent beyond 1 mm precision.
    """
    if chunk_size <= 0:
        msg = f"chunk_size must be a positive point count; got {chunk_size}."
        raise ValueError(msg)
    if bbox is not None:
        ensure_valid_bbox(bbox)
    report = progress if progress is not None else _no_op_progress

    with laspy.open(str(source_path)) as reader:
        header = reader.header
        source_points = int(header.point_count)
        names = set(header.point_format.dimension_names)
        copc = _is_copc(header)
        las_scale = tuple(float(v) for v in header.scales)
        las_offset = tuple(float(v) for v in header.offsets)
    has_rgb = "red" in names
    scan = _scan(
        source_path,
        include_classes,
        bbox,
        chunk_size,
        copc=copc,
        has_rgb=has_rgb,
        keep_coords=thinning is not None,
    )
    if scan.selected == 0:
        msg = "no points survived the class/bbox selection."
        raise PlyExportError(msg)
    keep = (
        None
        if thinning is None or scan.coords is None
        else _keep_mask(scan.coords, thinning)
    )
    written = scan.selected if keep is None else int(keep.sum())
    offset = (
        (0.0, 0.0, 0.0)
        if coords is CoordMode.FLOAT64
        else _centring_offset(scan)
    )
    narrowed = scan.colour_max > _BYTE_MAX
    dtype = _vertex_dtype(
        coords, has_rgb=has_rgb, has_overlap="overlap" in names
    )
    properties = _properties(dtype)
    out_classes = np.zeros(_N_CLASS_CODES, dtype=np.int64)

    with output_path.open("wb") as sink:
        sink.write(
            _header(
                written,
                properties,
                coords,
                offset,
                (las_scale[0], las_scale[1], las_scale[2]),
                (las_offset[0], las_offset[1], las_offset[2]),
            )
        )
        ordinal = 0
        for done, chunk in enumerate(
            _iter_chunks(source_path, bbox, chunk_size, copc=copc), start=1
        ):
            _, mask = _selection(chunk, include_classes, bbox)
            count = int(mask.sum())
            if keep is not None:
                final = np.zeros_like(mask)
                final[mask] = keep[ordinal : ordinal + count]
                mask = final
            ordinal += count
            block = _pack(
                chunk, mask, dtype, offset, shift=8 if narrowed else 0
            )
            out_classes += np.bincount(
                block["classification"].astype(np.int64),
                minlength=_N_CLASS_CODES,
            )
            sink.write(block.tobytes())
            report(done, scan.chunks)

    return PlyExportStats(
        point_count=written,
        source_points=source_points,
        scanned_points=scan.scanned,
        selected_points=scan.selected,
        classes_scanned=_pairs(scan.classes),
        classes_out=_pairs(out_classes),
        properties=properties,
        coords=coords,
        offset_xyz=offset,
        las_scale=(las_scale[0], las_scale[1], las_scale[2]),
        las_offset=(las_offset[0], las_offset[1], las_offset[2]),
        colour=ColourRecord(scan.colour_max, narrowed) if has_rgb else None,
    )
