"""Prep-context tile deduplication: crop-before-merge + XYZ/GPS-time sweep.

Adjacent AHN tiles from the GeoTiles.nl re-tiling overlap by a ~20-25 m band, so
a naive point-array append duplicates every seam. This transform removes those
duplicates in two mandatory stages, per the epic spec:

1. **Crop before merge.** Each tile is cropped to its canonical, non-overlapping
   extent with a half-open ``[min, max)`` rule, so a point on a shared tile edge
   is claimed by exactly one tile and the seam band never enters the merge.
2. **Post-merge exact-duplicate sweep.** Any point that still coincides exactly
   -- identical scaled ``X/Y/Z`` integers *and* GPS time -- with an
   earlier-kept point is dropped. This is cheap insurance that also catches the
   same tile ingested twice under different names.

The output is deterministic: identical input yields byte-identical output. The
sweep keeps the first occurrence in input (tile, then in-tile) order via a
stable index sort, so ordering and tie-breaking never depend on hashing,
wall-clock time, or environment.

Header harmonization across tiles with differing LAS extra dimensions reuses the
grandfathered ``ahn_cli.process`` logic via its public ``harmonize_headers``
re-export (imported with its module-load ``DeprecationWarning`` suppressed, so
the warning never leaks into this gated context). The append is re-expressed
here as an in-memory concatenation followed by a single write, because a
*global* duplicate sweep needs the whole merged record at once -- incremental
append-then-reread would be strictly more I/O for no benefit.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import laspy
import numpy as np
import numpy.typing as npt

from ahn_cli.domain import ensure_valid_bbox

# Header harmonization is reused from the grandfathered ``process`` module via
# its public ``harmonize_headers`` re-export seam (added there for this reuse, so
# no private symbol is crossed). The import emits a module-load
# ``DeprecationWarning``; wrapping it in ``catch_warnings`` keeps that warning
# from leaking into this gated context, while staying a module-top-level import.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from ahn_cli.process import harmonize_headers

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from ahn_cli.domain import BBox, ProgressCallback


@dataclass(frozen=True)
class CanonicalTile:
    """A fetched tile paired with its canonical, non-overlapping extent.

    Contract:
        - ``path`` locates the tile's LAZ/LAS file on disk.
        - ``extent`` is the tile's canonical :data:`~ahn_cli.domain.BBox` in
          EPSG:28992 -- the boundary the tile *owns*, excluding the overlap band
          it shares with its neighbours. Points outside it are cropped before
          merge.

    Invariants:
        - Frozen: an immutable, hashable value object, equal by field value.

    Failure modes:
        - ``ValueError`` if ``extent`` is degenerate (see
          :func:`~ahn_cli.domain.ensure_valid_bbox`).
    """

    path: Path
    extent: BBox

    def __post_init__(self) -> None:
        """Validate that ``extent`` is a positive-area bounding box."""
        ensure_valid_bbox(self.extent)


@dataclass(frozen=True)
class DedupStats:
    """The point-count ledger of one deduplication run.

    Contract:
        - ``input_points``: total points read across every input tile.
        - ``cropped_points``: points surviving the crop-before-merge stage,
          i.e. the size of the merged record fed to the sweep.
        - ``duplicates_removed``: exact XYZ+GPS-time duplicates the sweep
          dropped (``cropped_points - output_points``).
        - ``output_points``: points written to the output file.

    Invariants:
        - Frozen value object, equal by field value; safe to record verbatim in
          a provenance sidecar.
    """

    input_points: int
    cropped_points: int
    duplicates_removed: int
    output_points: int


def _crop_and_reproject(
    tile: CanonicalTile, header: laspy.LasHeader
) -> tuple[laspy.ScaleAwarePointRecord, int]:
    """Crop one tile to its canonical extent and cast it onto ``header``.

    Returns the cropped points expressed on the harmonized header's scale/offset
    grid, and the tile's original (pre-crop) point count. The crop is half-open
    ``[minx, maxx) x [miny, maxy)`` so a point on a shared edge is claimed by
    exactly one neighbouring tile.
    """
    with laspy.open(str(tile.path)) as reader:
        source = reader.read()
    original_count = len(source.points)

    minx, miny, maxx, maxy = tile.extent
    x = np.asarray(source.x, dtype=float)
    y = np.asarray(source.y, dtype=float)
    kept = (x >= minx) & (x < maxx) & (y >= miny) & (y < maxy)
    cropped = source.points[kept]

    out = laspy.ScaleAwarePointRecord.zeros(len(cropped), header=header)
    # The harmonized header is a superset of every tile's dimensions, so a
    # straight per-dimension copy is total (no missing-field guard needed).
    for name in cropped.point_format.dimension_names:
        out[name] = cropped[name]

    # Reproject the raw integer coordinates onto the harmonized grid, mirroring
    # the legacy offset correction so tiles written with different LAS offsets
    # share one integer lattice before keys are compared.
    offset_correction = header.offsets - source.header.offsets
    out.x = out.x - offset_correction[0]
    out.y = out.y - offset_correction[1]
    out.z = out.z - offset_correction[2]
    return out, original_count


def _sweep_indices(
    points: laspy.ScaleAwarePointRecord,
) -> npt.NDArray[np.intp]:
    """Return the sorted indices of the first occurrence of each exact point.

    Two points are exact duplicates when their scaled integer ``X/Y/Z`` and
    ``gps_time`` coincide. ``numpy.unique`` yields the first-occurrence index of
    each distinct key; sorting those indices restores input order, so the kept
    point of any duplicate group is deterministically the earliest one.
    """
    keys = np.rec.fromarrays(
        [points["X"], points["Y"], points["Z"], points["gps_time"]]
    )
    _, first_occurrence = np.unique(keys, return_index=True)
    return np.sort(first_occurrence)


def _no_op_progress(_done: int, _total: int) -> None:
    """Report nothing; the default when the caller supplies no callback."""


def deduplicate_tiles(
    tiles: Sequence[CanonicalTile],
    output_path: Path,
    *,
    progress: ProgressCallback | None = None,
) -> DedupStats:
    """Crop, merge, and exact-duplicate-sweep ``tiles`` into ``output_path``.

    Contract:
        - ``tiles`` is a non-empty sequence of :class:`CanonicalTile`; each is
          cropped to its canonical extent, the crops are merged onto one
          harmonized header, and exact XYZ+GPS-time duplicates are swept before
          a single deterministic write to ``output_path``.
        - Every input tile must carry a ``gps_time`` dimension (AHN point
          format 6+), which forms part of the duplicate key.
        - Returns a :class:`DedupStats` ledger of the point counts.
        - Calls ``progress(tiles_done, total_tiles)`` once per cropped tile;
          defaults to a no-op so callers that don't care about progress are
          unaffected.

    Invariants:
        - Deterministic: identical input yields byte-identical output, with the
          first occurrence of each duplicate kept in input order.

    Failure modes:
        - ``ValueError`` if ``tiles`` is empty (nothing to merge).
    """
    if not tiles:
        msg = "deduplicate_tiles requires at least one tile."
        raise ValueError(msg)
    report = progress if progress is not None else _no_op_progress

    files = [str(tile.path) for tile in tiles]
    header = harmonize_headers(files)

    input_points = 0
    cropped_arrays: list[npt.NDArray[np.void]] = []
    for i, tile in enumerate(tiles, start=1):
        cropped, original_count = _crop_and_reproject(tile, header)
        input_points += original_count
        cropped_arrays.append(cropped.array)
        report(i, len(tiles))

    cropped_count = sum(len(array) for array in cropped_arrays)
    merged = laspy.ScaleAwarePointRecord.zeros(cropped_count, header=header)
    merged.array[:] = np.concatenate(cropped_arrays)

    kept = _sweep_indices(merged)
    swept = laspy.ScaleAwarePointRecord.zeros(len(kept), header=header)
    swept.array[:] = merged.array[kept]

    out_las = laspy.LasData(header)
    out_las.points = swept
    out_las.write(str(output_path))

    return DedupStats(
        input_points=input_points,
        cropped_points=cropped_count,
        duplicates_removed=cropped_count - len(kept),
        output_points=len(kept),
    )
