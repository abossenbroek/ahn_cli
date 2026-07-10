"""Prep-context point-cloud export: a LAZ cloud to a binary ``pointcloud.ply``.

TouchDesigner (and most point-cloud consumers) ingest PLY; this transform turns
a fetched/processed LAZ tile into a ``pointcloud.ply`` deliverable. Two
properties make it fit the epic's guardrails:

1. **Memory-efficient.** Points are streamed through laspy's chunk iterator and
   written chunk-by-chunk, so an arbitrarily large cloud never materializes in
   memory at once. The whole record is *never* read; only bounded windows are.
2. **Deterministic.** Identical input yields byte-identical output. The header
   is static (no date, host, or path), the binary payload is little-endian
   IEEE-754 ``double`` in a fixed ``x, y, z`` order, and the vertex count is
   taken from the source header -- so a re-run reproduces every byte.

Coordinates are written as ``double`` (float64), not ``float``, on purpose: AHN
lives in EPSG:28992 where an easting near 194000 m needs double precision to
preserve sub-centimetre position. float32 would lose ~2-3 cm and break the
"coordinates preserved from source LAZ" contract. (This differs from the
DSM-grid ``positions.exr``, which stores float32 heights on a grid -- a
different artifact, not RD coordinates.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import laspy
import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from pathlib import Path

DEFAULT_CHUNK_SIZE = 1_000_000
"""Points read/written per streaming window when none is supplied.

Sized so one window of three ``double`` columns is a few tens of megabytes --
large enough to amortize I/O, small enough that a massive cloud never loads
whole.
"""

_PLY_HEADER_TEMPLATE = (
    "ply\n"
    "format binary_little_endian 1.0\n"
    "element vertex {count}\n"
    "property double x\n"
    "property double y\n"
    "property double z\n"
    "end_header\n"
)
"""The static, deterministic PLY header. Only the vertex ``count`` varies."""


@dataclass(frozen=True)
class PlyExportStats:
    """The point-count ledger of one PLY export.

    Contract:
        - ``point_count``: vertices written to the output, equal to the source
          LAZ header's point count.

    Invariants:
        - Frozen value object, equal by field value; safe to record verbatim in
          a provenance sidecar.
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
        - ``source_path`` is a readable LAZ/LAS file.
        - ``output_path`` receives a ``binary_little_endian 1.0`` PLY whose sole
          element is ``vertex`` with ``double x, y, z`` properties, in that
          order, holding each source point's scaled world coordinates.
        - ``chunk_size`` bounds how many points are held in memory at once; the
          cloud is read and written in windows of at most this many points. It
          must be a positive count.
        - Returns a :class:`PlyExportStats` whose ``point_count`` equals the
          source header's point count.

    Invariants:
        - Memory-bounded: the full point record is never read; only windows of
          up to ``chunk_size`` points are held at once.
        - Deterministic: identical input yields byte-identical output.

    Failure modes:
        - ``ValueError`` if ``chunk_size`` is not positive (laspy's chunk
          iterator requires a strictly positive window).
    """
    if chunk_size <= 0:
        msg = f"chunk_size must be a positive point count; got {chunk_size}."
        raise ValueError(msg)

    with laspy.open(str(source_path)) as reader:
        point_count = int(reader.header.point_count)
        header = _PLY_HEADER_TEMPLATE.format(count=point_count)
        with output_path.open("wb") as sink:
            sink.write(header.encode("ascii"))
            for chunk in reader.chunk_iterator(chunk_size):
                block: npt.NDArray[np.float64] = np.empty(
                    (len(chunk), 3), dtype=np.float64
                )
                block[:, 0] = np.asarray(chunk.x, dtype=np.float64)
                block[:, 1] = np.asarray(chunk.y, dtype=np.float64)
                block[:, 2] = np.asarray(chunk.z, dtype=np.float64)
                # ``<f8`` fixes little-endian bytes regardless of host byte
                # order, so the payload stays byte-identical across platforms.
                sink.write(block.astype("<f8", copy=False).tobytes())

    return PlyExportStats(point_count=point_count)
