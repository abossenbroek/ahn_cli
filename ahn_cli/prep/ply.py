"""Prep-context point-cloud export: a LAZ cloud to a binary ``pointcloud.ply``.

WP13 red stub: the importable public surface (:class:`PlyExportStats`,
:func:`export_ply`) with contracts declared, so the WP13 tests fail at their
assertions rather than on import. The streaming, deterministic implementation
lands in the following commit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

DEFAULT_CHUNK_SIZE = 1_000_000
"""Points read/written per streaming window when none is supplied."""


@dataclass(frozen=True)
class PlyExportStats:
    """The point-count ledger of one PLY export.

    Contract:
        - ``point_count``: vertices written, equal to the source header's count.

    Invariants:
        - Frozen value object, equal by field value.
    """

    point_count: int


def export_ply(
    source_path: Path,
    output_path: Path,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> PlyExportStats:
    """Stream a LAZ point cloud to a binary little-endian ``.ply`` file.

    Contract:
        - ``source_path`` is a readable LAZ/LAS file; ``output_path`` receives a
          ``binary_little_endian 1.0`` PLY of ``double x, y, z`` vertices.
        - ``chunk_size`` bounds points held in memory at once (must be positive).
        - Returns a :class:`PlyExportStats` equal to the source point count.

    Invariants:
        - Memory-bounded streaming; deterministic byte-identical output.

    Failure modes:
        - ``ValueError`` if ``chunk_size`` is not positive.
    """
    raise NotImplementedError
