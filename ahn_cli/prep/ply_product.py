"""A PLY product: the lossless export plus its provenance sidecar.

:func:`export_ply_product` runs :func:`~ahn_cli.prep.ply.export_ply` and writes
``<output>.provenance.json`` next to it, so no PLY leaves ahn_cli without a
record of its input (path, sha256, size), the tool and library versions, every
filter and thinning rule, the datum offset, the class histograms and point
counts in and out, and the run's wall time. See
:mod:`ahn_cli.provenance.product` for the schema.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING

import laspy

from ahn_cli.prep.decimate import PoissonThinning
from ahn_cli.prep.ply import (
    DEFAULT_CHUNK_SIZE,
    CoordMode,
    PlyExportStats,
    PlyThinning,
    export_ply,
)
from ahn_cli.provenance.product import (
    SCHEMA_VERSION,
    collect_tool_versions,
    fingerprint,
    write_product_sidecar,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    from ahn_cli.domain import BBox, ProgressCallback


class ClassOrigin(Enum):
    """Where the class filter came from, recorded in the sidecar."""

    EXPLICIT = "explicit"
    ALL = "all"


@dataclass(frozen=True)
class PlyProductRequest:
    """A validated intent to export one PLY product with its sidecar.

    Contract:
        - ``include_classes`` empty keeps every class; ``class_origin`` says
          whether the filter is user-chosen or "all" (no filter, the CLI default).
        - ``upstream`` records filters applied before this export (``prep``
          filters its LAZ first); ``None`` when there are none.
        - Other fields are those of :func:`~ahn_cli.prep.ply.export_ply`.

    Invariants:
        - Frozen value object, equal by field value.
    """

    source: Path
    output: Path
    include_classes: tuple[int, ...] = ()
    class_origin: ClassOrigin = ClassOrigin.ALL
    bbox: BBox | None = None
    thinning: PlyThinning | None = None
    coords: CoordMode = CoordMode.FLOAT64
    chunk_size: int = DEFAULT_CHUNK_SIZE
    upstream: Mapping[str, object] | None = None


def _utc_now() -> datetime:
    """Return the current time as an aware UTC datetime."""
    return datetime.now(timezone.utc)


def _thinning_record(thinning: PlyThinning | None) -> dict[str, object]:
    """Describe a thinning rule (or its absence) for the sidecar."""
    if thinning is None:
        return {"mode": "none"}
    if isinstance(thinning, PoissonThinning):
        return {
            "mode": "spacing",
            "algorithm": "poisson-disk",
            "spacing_m": thinning.radius,
            "seed": thinning.seed,
        }
    return {"mode": "share", "share": thinning.share, "seed": thinning.seed}


def _pairs(pairs: tuple[tuple[int, int], ...]) -> list[list[int]]:
    """Render ``(class, count)`` pairs as JSON lists, class ascending."""
    return [[code, count] for code, count in pairs]


def _input_block(source: Path) -> dict[str, object]:
    """Describe the input file: identity, LAS header facts, COPC flag."""
    sha, size = fingerprint(source)
    with laspy.open(str(source)) as reader:
        header = reader.header
        return {
            "path": str(source.resolve()),
            "sha256": sha,
            "bytes": size,
            "point_count": int(header.point_count),
            "point_format": int(header.point_format.id),
            "las_version": str(header.version),
            "is_copc": any(
                getattr(vlr, "user_id", "") == "copc" for vlr in header.vlrs
            ),
            "las_scale": [float(v) for v in header.scales],
            "las_offset": [float(v) for v in header.offsets],
            "mins": [float(v) for v in header.mins],
            "maxs": [float(v) for v in header.maxs],
        }


def _output_block(
    request: PlyProductRequest, stats: PlyExportStats
) -> dict[str, object]:
    """Describe the written PLY: identity, datum, properties, colour."""
    sha, size = fingerprint(request.output)
    colour = stats.colour
    return {
        "name": request.output.name,
        "format": "ply",
        "sha256": sha,
        "bytes": size,
        "coords": stats.coords.value,
        "offset_xyz": list(stats.offset_xyz),
        "offset_rule": "world = vertex + offset_xyz (millimetre-exact)",
        "properties": [
            {"name": name, "type": kind} for name, kind in stats.properties
        ],
        "colour": None
        if colour is None
        else {
            "source_max": colour.source_max,
            "narrowed_to_high_byte": colour.narrowed,
        },
    }


def export_ply_product(
    request: PlyProductRequest,
    *,
    progress: ProgressCallback | None = None,
    clock: Callable[[], datetime] = _utc_now,
    timer: Callable[[], float] = time.perf_counter,
) -> PlyExportStats:
    """Export ``request.source`` to ``request.output`` and write its sidecar.

    Contract:
        - Writes the PLY exactly as :func:`~ahn_cli.prep.ply.export_ply` does
          and ``<output>.provenance.json`` beside it; returns the export stats.
        - The PLY and every sidecar field except ``run`` are deterministic;
          ``run`` holds the UTC start time and wall seconds.
        - ``clock`` and ``timer`` are injectable for tests.

    Failure modes:
        - Whatever :func:`~ahn_cli.prep.ply.export_ply` raises; no sidecar is
          written for a failed export.
    """
    started = clock()
    t0 = timer()
    source_block = _input_block(request.source)
    stats = export_ply(
        request.source,
        request.output,
        chunk_size=request.chunk_size,
        progress=progress,
        include_classes=request.include_classes,
        bbox=request.bbox,
        thinning=request.thinning,
        coords=request.coords,
    )
    record: dict[str, object] = {
        "schema": SCHEMA_VERSION,
        "tool": collect_tool_versions(),
        "input": source_block,
        "filters": {
            "class_include": list(request.include_classes) or None,
            "class_origin": request.class_origin.value,
            "bbox": None if request.bbox is None else list(request.bbox),
            "thinning": _thinning_record(request.thinning),
            "upstream": None
            if request.upstream is None
            else dict(request.upstream),
        },
        "counts": {
            "source": stats.source_points,
            "scanned": stats.scanned_points,
            "selected": stats.selected_points,
            "out": stats.point_count,
        },
        "classes": {
            "scanned": _pairs(stats.classes_scanned),
            "out": _pairs(stats.classes_out),
        },
        "output": _output_block(request, stats),
        "run": {
            "started_utc": started.astimezone(timezone.utc).isoformat(),
            "wall_seconds": round(timer() - t0, 3),
        },
    }
    write_product_sidecar(request.output, record)
    return stats
