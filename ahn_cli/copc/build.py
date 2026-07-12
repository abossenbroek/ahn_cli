"""Copc-context build orchestrator: plan → scatter → dedup → sample → write.

One public entry point, :func:`build_copc`, turns a LAZ deliverable into a
``.copc.laz`` in two streaming passes (multi-bucket inputs first run a cheap
XY occupancy pre-scan that deepens the bucket level for irregular AOIs, so
the busiest bucket still fits the per-bucket target):

1. :func:`~ahn_cli.copc.scatter.scatter_cloud` streams the input into
   per-column bucket files (memory: one read chunk).
2. Each bucket is loaded alone, de-duplicated at the native 0.5 m voxel
   (:func:`~ahn_cli.copc.dedup.dedupe_voxels` — outlier-aware, never coarser
   than AHN's own grid), LOD-sampled onto octree nodes with copc.js-exact
   double math, and written node by node. Bucket files are deleted as soon
   as they are consumed, so scratch disk is bounded too.

Nodes above the bucket level collect points from many buckets; their (small,
grid-capped) payloads are held back and written after the last bucket.

RGB policy: inputs without RGB — or with all-black RGB — become PDRF 6;
8-bit-looking RGB (every channel ≤ 255) is widened by 257 to full 16-bit so
``copc-validator``'s ``rgbi`` heuristic doesn't flag the file; true 16-bit
RGB passes through untouched.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import laspy
import numpy as np

from ahn_cli.copc.dedup import dedupe_voxels
from ahn_cli.copc.octree import (
    BUCKET_LEVEL_CAP,
    BuildPlan,
    CopcError,
    LodSampler,
    NodeKey,
    plan_build,
    rebalance_bucket_level,
)
from ahn_cli.copc.scatter import (
    RECORD_DTYPE,
    ProgressCallback,
    ScatterResult,
    scatter_cloud,
)
from ahn_cli.copc.writer import (
    BARE_POINT_FORMAT,
    RGB_POINT_FORMAT,
    CopcNodeWriter,
    rd_new_wkt,
)
from ahn_cli.domain.authenticity import degenerate_cloud

if TYPE_CHECKING:
    import numpy.typing as npt

_EIGHT_BIT_MAX = 255
_EIGHT_TO_SIXTEEN = 257  # 0xff -> 0xffff, the standard 8->16 bit widening


def _no_op_progress(_done: int, _total: int) -> None:
    """Fallback progress sink when the caller injects none."""


@dataclass(frozen=True)
class CopcBuildResult:
    """What :func:`build_copc` produced, for provenance and reporting."""

    path: Path
    input_points: int
    written_points: int
    node_count: int
    point_format_id: int
    plan: BuildPlan


def build_copc(
    cloud: Path,
    out: Path,
    *,
    workdir: Path | None = None,
    scale: float = 0.001,
    target_bucket_points: int = 4_000_000,
    chunk_points: int = 2_000_000,
    progress: ProgressCallback | None = None,
) -> CopcBuildResult:
    """Build a fully valid COPC file from a LAZ cloud, streaming throughout.

    Contract:
        - ``cloud`` is a readable, non-empty LAS/LAZ whose header bounds
          cover its points; ``out`` is the ``.copc.laz`` to write.
        - ``progress(done, total)`` advances over roughly two passes of the
          input point count. The occupancy pre-scan that multi-bucket
          inputs run first is deliberately unreported, so the ``(done,
          total)`` arithmetic stays identical for every input.
        - Returns the build summary (written count, node count, plan).

    Failure modes:
        - :class:`CopcError` for unreadable/empty input, unwritable output,
          or a native copclib write failure; a write phase that fails after
          the output was opened removes the partial ``.copc.laz`` before
          the error is raised, so a half-written file never looks like a
          deliverable. A failed build also best-effort removes its bucket
          scratch directory, and scatter recreates that directory empty
          regardless, so a persistent ``workdir`` never feeds one run's
          leftover records into the next.

    Invariants:
        - Deterministic: identical input yields an identical octree and
          identical node payloads.
        - Peak memory is of the order of one scatter chunk or one bucket
          (occupancy-rebalanced so the busiest bucket fits the target even
          on irregular AOIs), never the whole cloud; the held-back payloads
          of nodes above the bucket level and the sampler's persistent
          shallow-occupancy sets also stay resident and grow slowly with
          area.
    """
    plan = _plan_from_header(cloud, scale, target_bucket_points)
    if plan.bucket_level > 0:
        # Uniform-fill sizing under-buckets irregular AOIs (a thin diagonal
        # strip concentrates the cloud in a few columns): measure the real
        # XY occupancy first and deepen the bucket level to fit. Single-
        # bucket clouds skip the extra pass and keep the single-read path.
        columns = _prescan_column_occupancy(cloud, plan, chunk_points)
        plan = rebalance_bucket_level(plan, columns, target_bucket_points)
    if workdir is not None:
        return _build(cloud, out, plan, workdir, chunk_points, progress)
    with tempfile.TemporaryDirectory(prefix="ahn_cli_copc_") as tmp:
        return _build(cloud, out, plan, Path(tmp), chunk_points, progress)


def _plan_from_header(
    cloud: Path, scale: float, target_bucket_points: int
) -> BuildPlan:
    """Fix the build geometry from the input header alone."""
    try:
        with laspy.open(str(cloud)) as reader:
            header = reader.header
            mins = tuple(float(v) for v in header.mins)
            maxs = tuple(float(v) for v in header.maxs)
            count = int(header.point_count)
    except (OSError, laspy.LaspyException) as exc:
        msg = f"point cloud at {cloud} is not readable: {exc}"
        raise CopcError(msg) from exc
    if count > 0 and degenerate_cloud(count, mins, maxs):
        msg = (
            f"point cloud at {cloud} stacks all {count} points at one "
            "identical position — that is fabricated data, not a genuine "
            "AHN cloud; refusing to build a COPC from it."
        )
        raise CopcError(msg)
    try:
        return plan_build(
            (mins[0], mins[1], mins[2]),
            (maxs[0], maxs[1], maxs[2]),
            count,
            scale=scale,
            target_bucket_points=target_bucket_points,
        )
    except ValueError as exc:
        msg = f"cannot build a COPC from {cloud}: {exc}"
        raise CopcError(msg) from exc


def _prescan_column_occupancy(
    cloud: Path, plan: BuildPlan, chunk_points: int
) -> npt.NDArray[np.int64]:
    """Count points per cap-level XY column in one cheap streaming pass.

    Contract:
        - Mirrors pass-1 scatter's quantization and column derivation
          exactly (the same float64 math per axis), so the histogram
          aggregates to the true per-bucket counts at every candidate
          bucket level: the plan's cap-aligned ``side_m`` makes the
          cap-level columns nest exactly into every coarser level's.
        - Returns a ``(2**BUCKET_LEVEL_CAP,) * 2`` int64 histogram
          (~0.5 MB); axis 0 is the X column, axis 1 the Y column.
        - Out-of-cube points (a header that lies about its bounds) are
          clipped onto the border columns here; pass-1 scatter still
          rejects them with the precise error.

    Failure modes:
        - :class:`CopcError` if the cloud is unreadable mid-scan.
    """
    grid = 2**BUCKET_LEVEL_CAP
    column_units = plan.side_units // grid
    counts = np.zeros((grid, grid), dtype=np.int64)
    offsets = np.asarray(plan.offsets[:2], dtype=np.float64)
    try:
        with laspy.open(str(cloud)) as reader:
            for chunk in reader.chunk_iterator(chunk_points):
                xy = np.column_stack(
                    [
                        np.asarray(chunk.x, dtype=np.float64),
                        np.asarray(chunk.y, dtype=np.float64),
                    ]
                )
                quantized = np.rint((xy - offsets) / plan.scale).astype(
                    np.int64
                )
                columns = np.clip(quantized // column_units, 0, grid - 1)
                flat = columns[:, 0] * grid + columns[:, 1]
                counts += np.bincount(flat, minlength=grid * grid).reshape(
                    grid, grid
                )
    except (OSError, laspy.LaspyException) as exc:
        msg = f"point cloud at {cloud} is not readable: {exc}"
        raise CopcError(msg) from exc
    return counts


def _build(
    cloud: Path,
    out: Path,
    plan: BuildPlan,
    workdir: Path,
    chunk_points: int,
    progress: ProgressCallback | None,
) -> CopcBuildResult:
    """Run scatter + per-bucket dedup/sample/write with a fixed plan."""
    report = progress if progress is not None else _no_op_progress
    buckets_dir = workdir / "buckets"
    try:
        scattered = scatter_cloud(
            cloud,
            plan,
            buckets_dir,
            chunk_points=chunk_points,
            progress=lambda done, total: report(done, 2 * total),
        )
        point_format_id, widen = _choose_point_format(scattered)
        writer = CopcNodeWriter(
            out, plan, point_format_id=point_format_id, wkt=rd_new_wkt()
        )
        sampler = LodSampler(plan)
        held_back: dict[NodeKey, list[npt.NDArray[np.void]]] = {}
        node_count = 0
        done = scattered.count
        total = 2 * scattered.count
        for bucket in sorted(scattered.bucket_paths):
            path = scattered.bucket_paths[bucket]
            records = np.fromfile(path, dtype=RECORD_DTYPE)
            path.unlink()  # scratch disk stays bounded to unprocessed buckets
            survivors = _dedupe_bucket(records, plan)
            if widen:
                for channel in ("red", "green", "blue"):
                    survivors[channel] = (
                        survivors[channel] * _EIGHT_TO_SIXTEEN
                    )
            sampled = sampler.sample(_decode(survivors, plan))
            for key, indices in sampled.items():
                picked = survivors[indices]
                if key.level < plan.bucket_level:
                    held_back.setdefault(key, []).append(picked)
                else:
                    writer.add_node(key, picked)
                    node_count += 1
            done += records.shape[0]
            report(done, total)
        for key in sorted(held_back):
            writer.add_node(key, np.concatenate(held_back[key]))
            node_count += 1
        written = writer.finish()
    except CopcError:
        # Never leave a half-written .copc.laz where it looks like a
        # deliverable, nor stale bucket records that a later build in the
        # same persistent workdir could mistake for its own: best-effort
        # remove the partial output and the tool-owned bucket scratch
        # directory, then re-raise.
        out.unlink(missing_ok=True)
        shutil.rmtree(buckets_dir, ignore_errors=True)
        raise
    return CopcBuildResult(
        path=out,
        input_points=scattered.count,
        written_points=written,
        node_count=node_count,
        point_format_id=point_format_id,
        plan=plan,
    )


def _choose_point_format(scattered: ScatterResult) -> tuple[int, bool]:
    """Pick the output PDRF and whether 8-bit RGB widening is needed."""
    if not scattered.has_rgb or scattered.rgb_max == 0:
        return BARE_POINT_FORMAT, False
    return RGB_POINT_FORMAT, scattered.rgb_max <= _EIGHT_BIT_MAX


def _dedupe_bucket(
    records: npt.NDArray[np.void], plan: BuildPlan
) -> npt.NDArray[np.void]:
    """Collapse one bucket to a single survivor per 0.5 m voxel."""
    quantized = np.column_stack(
        [records["x"], records["y"], records["z"]]
    ).astype(np.int64)
    return records[dedupe_voxels(quantized, plan.voxel_units)]


def _decode(
    records: npt.NDArray[np.void], plan: BuildPlan
) -> npt.NDArray[np.float64]:
    """Decode quantized records to the doubles every COPC reader sees."""
    quantized = np.column_stack(
        [records["x"], records["y"], records["z"]]
    ).astype(np.float64)
    return quantized * plan.scale + np.asarray(plan.offsets)
