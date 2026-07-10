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
grandfathered :func:`ahn_cli.process._harmonize_headers` (imported lazily with
its module-load ``DeprecationWarning`` suppressed, so the warning never leaks
into this gated context). The append itself is re-expressed here as an in-memory
concatenation followed by a single write, because a *global* duplicate sweep
needs the whole merged record at once -- incremental append-then-reread would be
strictly more I/O for no benefit.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import laspy
import numpy as np

from ahn_cli.domain import ensure_valid_bbox

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from ahn_cli.domain import BBox


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


def deduplicate_tiles(
    tiles: Sequence[CanonicalTile], output_path: Path
) -> DedupStats:
    """Crop, merge, and exact-duplicate-sweep ``tiles`` into ``output_path``.

    STUB (WP10 red): writes an empty output and reports a zeroed ledger so the
    behavioural tests fail at their assertions rather than at collection. The
    real implementation replaces this body.
    """
    header = laspy.LasHeader(point_format=6, version="1.4")
    empty = laspy.LasData(header)
    empty.write(str(output_path))
    _ = (tiles, warnings, np, ensure_valid_bbox)
    return DedupStats(
        input_points=0,
        cropped_points=0,
        duplicates_removed=0,
        output_points=0,
    )
