"""Copc-context pass 1: stream a LAZ cloud into per-bucket record files.

The COPC build never holds the whole cloud: this pass reads the input in
chunks, quantizes every coordinate onto the output grid fixed by the
:class:`~ahn_cli.copc.octree.BuildPlan`, and appends fixed-width binary
records to one temp file per level-``bucket_level`` XY column ("bucket").
Pass 2 then processes buckets one at a time, so peak memory is one chunk here
and one bucket there — never the input size.

Attribute handling normalizes the input zoo to what the PDRF 6/7 output
needs: legacy ``scan_angle_rank`` degrees become 0.006-degree ``scan_angle``
units, return numbers are lifted into the LAS-valid ``1..15`` range
(interpolated clouds legitimately carry zeros there), and the LAS bit-field
attributes (``synthetic``/``key_point``/``withheld``/``overlap``,
``scanner_channel``, ``scan_direction_flag``, ``edge_of_flight_line``) are
packed into the PDRF 6 flags byte — dims the source format lacks (legacy
formats have no ``overlap``/``scanner_channel``) stay zero.

Determinism: chunks are processed in file order and appends are sequential,
so identical input yields byte-identical bucket files. The bucket directory
is recreated empty at the start of every scatter, so stale records from an
aborted earlier run in the same workdir can never contaminate a build.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING

import laspy
import numpy as np

from ahn_cli.copc.octree import CopcError
from ahn_cli.domain.progress import (
    ProgressCallback,  # noqa: TC001 -- re-exported by build.py
)

if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt

    from ahn_cli.copc.octree import BuildPlan

RECORD_DTYPE = np.dtype(
    [
        ("x", "<i4"),
        ("y", "<i4"),
        ("z", "<i4"),
        ("intensity", "<u2"),
        ("return_number", "u1"),
        ("number_of_returns", "u1"),
        ("flags", "u1"),
        ("classification", "u1"),
        ("scan_angle", "<i2"),
        ("user_data", "u1"),
        ("point_source_id", "<u2"),
        ("red", "<u2"),
        ("green", "<u2"),
        ("blue", "<u2"),
        ("gps_time", "<f8"),
    ]
)
"""One scattered point: quantized coords + the attributes PDRF 6/7 carries."""

_MAX_RETURN = 15  # PDRF 6+ return fields are 4 bits wide
_SCAN_ANGLE_DEGREES_PER_UNIT = 0.006  # LAS 1.4 scan_angle unit

_FLAG_BITS = (
    ("synthetic", 0),
    ("key_point", 1),
    ("withheld", 2),
    ("overlap", 3),
    ("scan_direction_flag", 6),
    ("edge_of_flight_line", 7),
)
"""Single-bit dims and their positions in the PDRF 6 flags byte."""

_SCANNER_CHANNEL_SHIFT = 4  # 2-bit scanner channel, bits 4-5


@dataclass(frozen=True, eq=False)
class ScatterResult:
    """Pass-1 outcome: bucket files plus exact quantized data bounds.

    ``eq=False``: carries paths and per-run temp state; identity compares.
    """

    bucket_paths: dict[tuple[int, int], Path]
    quantized_mins: tuple[int, int, int]
    quantized_maxs: tuple[int, int, int]
    count: int
    has_rgb: bool
    has_gps: bool
    rgb_max: int


def scatter_cloud(
    cloud: Path,
    plan: BuildPlan,
    workdir: Path,
    *,
    chunk_points: int = 2_000_000,
    progress: ProgressCallback | None = None,
) -> ScatterResult:
    """Stream ``cloud`` into per-bucket record files under ``workdir``.

    Contract:
        - ``cloud`` is a readable LAS/LAZ whose points all fall inside the
          plan's cube (guaranteed when the plan came from this file's header).
        - ``workdir`` is the tool-owned bucket scratch directory: this pass
          creates it and it holds nothing but bucket record files, so if it
          already exists (an aborted earlier run in a persistent workdir) it
          is removed wholesale and recreated empty — records from another
          cloud are never appended into this build.
        - Writes ``RECORD_DTYPE`` records to ``workdir/bucket_<bx>_<by>.bin``
          in input order; returns the per-bucket paths, the exact min/max of
          the quantized coordinates, and which optional dims the input has.
        - Calls ``progress(points_done, total_points)`` after each chunk.

    Failure modes:
        - :class:`CopcError` if the file is unreadable or a point falls
          outside the planned cube (the header lied about its bounds).
    """
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    grid_dim = 2**plan.bucket_level
    bucket_paths: dict[tuple[int, int], Path] = {}
    mins = np.full(3, np.iinfo(np.int64).max, dtype=np.int64)
    maxs = np.full(3, np.iinfo(np.int64).min, dtype=np.int64)
    done = 0
    rgb_max = 0
    try:
        with laspy.open(str(cloud)) as reader:
            names = set(reader.header.point_format.dimension_names)
            total = reader.header.point_count
            for chunk in reader.chunk_iterator(chunk_points):
                records, quantized = _pack_records(chunk, plan, names, cloud)
                mins = np.minimum(mins, quantized.min(axis=0))
                maxs = np.maximum(maxs, quantized.max(axis=0))
                for channel in ("red", "green", "blue"):
                    rgb_max = max(rgb_max, int(records[channel].max()))
                _append_by_bucket(
                    records, quantized, plan, grid_dim, workdir, bucket_paths
                )
                done += len(records)
                if progress is not None:
                    progress(done, total)
    except (OSError, laspy.LaspyException) as exc:
        msg = f"point cloud at {cloud} is not readable: {exc}"
        raise CopcError(msg) from exc
    return ScatterResult(
        bucket_paths=bucket_paths,
        quantized_mins=(int(mins[0]), int(mins[1]), int(mins[2])),
        quantized_maxs=(int(maxs[0]), int(maxs[1]), int(maxs[2])),
        count=done,
        has_rgb="red" in names,
        has_gps="gps_time" in names,
        rgb_max=rgb_max,
    )


def _pack_records(
    chunk: laspy.ScaleAwarePointRecord,
    plan: BuildPlan,
    names: set[str],
    cloud: Path,
) -> tuple[npt.NDArray[np.void], npt.NDArray[np.int64]]:
    """Quantize one chunk and pack it into ``RECORD_DTYPE`` records."""
    offsets = np.asarray(plan.offsets, dtype=np.float64)
    coords = np.column_stack(
        [
            np.asarray(chunk.x, dtype=np.float64),
            np.asarray(chunk.y, dtype=np.float64),
            np.asarray(chunk.z, dtype=np.float64),
        ]
    )
    quantized = np.rint((coords - offsets) / plan.scale).astype(np.int64)
    if bool(np.any(quantized < 0) or np.any(quantized >= plan.side_units)):
        msg = (
            f"point cloud at {cloud} has points outside the planned cube; "
            "its header bounds do not cover its data"
        )
        raise CopcError(msg)

    n = quantized.shape[0]
    records = np.zeros(n, dtype=RECORD_DTYPE)
    records["x"] = quantized[:, 0]
    records["y"] = quantized[:, 1]
    records["z"] = quantized[:, 2]
    records["intensity"] = np.asarray(chunk.intensity, dtype=np.uint16)
    records["classification"] = np.asarray(
        chunk.classification, dtype=np.uint8
    )
    records["user_data"] = np.asarray(chunk.user_data, dtype=np.uint8)
    records["point_source_id"] = np.asarray(
        chunk.point_source_id, dtype=np.uint16
    )
    return_number = np.asarray(chunk.return_number, dtype=np.uint8)
    return_number = np.clip(return_number, 1, _MAX_RETURN)
    number_of_returns = np.asarray(chunk.number_of_returns, dtype=np.uint8)
    number_of_returns = np.clip(
        np.maximum(number_of_returns, return_number), 1, _MAX_RETURN
    )
    records["return_number"] = return_number
    records["number_of_returns"] = number_of_returns
    records["flags"] = _pack_flags(chunk, names, n)
    if "scan_angle" in names:
        records["scan_angle"] = np.asarray(chunk.scan_angle, dtype=np.int16)
    else:
        degrees = np.asarray(chunk.scan_angle_rank, dtype=np.float64)
        records["scan_angle"] = np.rint(
            degrees / _SCAN_ANGLE_DEGREES_PER_UNIT
        ).astype(np.int16)
    if "red" in names:
        records["red"] = np.asarray(chunk.red, dtype=np.uint16)
        records["green"] = np.asarray(chunk.green, dtype=np.uint16)
        records["blue"] = np.asarray(chunk.blue, dtype=np.uint16)
    if "gps_time" in names:
        records["gps_time"] = np.asarray(chunk.gps_time, dtype=np.float64)
    return records, quantized


def _pack_flags(
    chunk: laspy.ScaleAwarePointRecord,
    names: set[str],
    n: int,
) -> npt.NDArray[np.uint8]:
    """Pack the PDRF 6 bit-field byte from whatever dims the input has."""
    flags = np.zeros(n, dtype=np.uint8)
    for name, shift in _FLAG_BITS:
        if name in names:
            bit = np.asarray(chunk[name], dtype=np.uint8)
            flags |= ((bit & 1) << shift).astype(np.uint8)
    if "scanner_channel" in names:
        channel = np.asarray(chunk.scanner_channel, dtype=np.uint8)
        flags |= ((channel & 0b11) << _SCANNER_CHANNEL_SHIFT).astype(np.uint8)
    return flags


def _append_by_bucket(
    records: npt.NDArray[np.void],
    quantized: npt.NDArray[np.int64],
    plan: BuildPlan,
    grid_dim: int,
    workdir: Path,
    bucket_paths: dict[tuple[int, int], Path],
) -> None:
    """Append one chunk's records to their bucket files, in input order."""
    bucket_ids = (quantized[:, 0] // plan.bucket_units) * grid_dim + (
        quantized[:, 1] // plan.bucket_units
    )
    order = np.argsort(bucket_ids, kind="stable")
    sorted_ids = bucket_ids[order]
    sorted_records = records[order]
    starts = np.concatenate(
        ([0], np.flatnonzero(np.diff(sorted_ids)) + 1, [len(sorted_ids)])
    )
    for begin, end in pairwise(starts):
        bucket = int(sorted_ids[begin])
        key = (bucket // grid_dim, bucket % grid_dim)
        path = workdir / f"bucket_{key[0]:03d}_{key[1]:03d}.bin"
        with path.open("ab") as handle:
            sorted_records[begin:end].tofile(handle)
        bucket_paths[key] = path
