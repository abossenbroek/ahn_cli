"""Prep-context raster export: a DSM raster to a ``positions.exr`` map.

RED STUB (WP12): the public surface exists so the tests import and *run*, but
the body writes no valid EXR yet -- the assertions fail. The GREEN commit
replaces the body with the hand-written OpenEXR encoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


class PositionsExportError(Exception):
    """Raised when the DSM cannot be read for a positions export (stub)."""


@dataclass(frozen=True)
class PositionsExportStats:
    """The ledger of one positions export (stub)."""

    width: int
    height: int
    nodata_pixels: int


def export_positions(
    dsm_path: Path, output_path: Path
) -> PositionsExportStats:
    """Export a positions map (RED STUB: writes a non-EXR placeholder)."""
    output_path.write_bytes(dsm_path.suffix.encode("ascii"))
    return PositionsExportStats(width=0, height=0, nodata_pixels=0)
