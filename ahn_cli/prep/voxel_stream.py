"""Prep-context out-of-core voxel thinning (streaming + Polars group-by).

The in-memory voxel path in :mod:`ahn_cli.prep.decimate` materialises the whole
cloud at once -- ``reader.read()`` plus a full ``(n, 3)`` float64 coordinate copy
plus the ``np.unique(cells, axis=0)`` sort temporaries -- so a national-scale
cloud (hundreds of millions of points) exhausts RAM and the process is killed
(SIGKILL / exit 137). This module is the memory-bounded alternative used by the
prep pipeline for :class:`~ahn_cli.prep.decimate.VoxelThinning` requests: it never
holds more than one chunk of points at a time and offloads the group-by-voxel to
Polars' streaming engine, so peak memory is independent of the point count.

Semantics are the voxel contract of :mod:`ahn_cli.prep.decimate`: within each
occupied voxel exactly one point survives -- the one with the smallest index in
the *class-filtered* cloud -- and the survivors are emitted in ascending index
order. The voxel grid is anchored at the per-cloud coordinate minimum and its
edge length comes from :func:`~ahn_cli.prep.decimate.voxel_size_for_grade`, so a
given grade yields the identical grid the in-memory reference uses.

The flow is three streaming passes (grade 0 is the identity and needs only one):

1. **Spill.** Stream the source in chunks, apply the classification filter, and
   write each kept point's ``(x, y, z, idx)`` -- where ``idx`` is its dense index
   in the class-filtered cloud -- to a per-chunk Parquet file in a scratch dir.
2. **Group.** Have Polars scan the Parquet spill, quantise each point to its
   voxel, and reduce ``group_by(voxel).min(idx)`` to the surviving indices. Both
   the origin scan and the group-by run in Polars' streaming (out-of-core) engine.
3. **Write.** Re-stream the source and append the surviving points to the output,
   preserving every source attribute, via a temp file swapped into place.

Determinism: chunks are read in file order, the filtered index is assigned in
that order, and the group-by min-reduction is order-independent, so identical
input yields byte-identical output.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import laspy
import numpy as np
import polars as pl

from ahn_cli.prep.decimate import voxel_size_for_grade

if TYPE_CHECKING:
    import numpy.typing as npt

    from ahn_cli.domain.progress import ProgressCallback

DEFAULT_CHUNK_POINTS = 1_000_000
"""Points held in memory per streamed chunk (matches the PLY export window)."""

_SPILL_SUBDIR = "voxel_spill"
"""Scratch subdirectory (under the workdir) holding the per-chunk Parquet spill."""

_SPILL_GLOB = "chunk_*.parquet"
"""Glob matching every per-chunk Parquet file Polars scans in the group pass."""


def _no_op_progress(_done: int, _total: int) -> None:
    """Report nothing; the default when the caller supplies no callback."""


def stream_voxel_thin(
    source: Path,
    output: Path,
    grade: int,
    include_classes: tuple[int, ...],
    exclude_classes: tuple[int, ...],
    *,
    workdir: Path | None = None,
    chunk_points: int = DEFAULT_CHUNK_POINTS,
    progress: ProgressCallback | None = None,
) -> int:
    """Class-filter and voxel-thin ``source`` into ``output``, out of core.

    Contract:
        - ``source`` is a readable LAS/LAZ; ``output`` receives the class-filtered,
          voxel-thinned cloud (``source`` and ``output`` may be the same path --
          the write goes through a temp file swapped in at the end).
        - ``grade`` is a voxel grade in ``[GRADE_MIN, GRADE_MAX]``; grade 0 is the
          identity (every class-kept point survives, no thinning).
        - ``include_classes`` / ``exclude_classes`` are the classification filter;
          empty tuples mean "no filter on that side".
        - Within each occupied voxel the surviving point is the one with the
          smallest index in the class-filtered cloud; survivors are written in
          ascending index order, every source attribute preserved. Returns the
          surviving point count.
        - ``workdir`` is the scratch directory for the Parquet spill; when
          ``None`` a private temp dir is created and removed afterwards. The
          spill lives in a dedicated subdirectory that is recreated empty each
          run and removed when the thin finishes.
        - ``chunk_points`` bounds how many points are read at once (must be
          positive). ``progress`` ticks ``(chunk, total_chunks)`` as the output
          is written; defaults to a no-op.

    Invariants:
        - Memory-bounded: never more than one chunk of points plus Polars'
          streaming working set is resident, regardless of cloud size.
        - Deterministic: identical input and parameters yield byte-identical
          output.

    Failure modes:
        - :class:`ValueError` if ``chunk_points`` is not positive, or if ``grade``
          is out of range (via :func:`voxel_size_for_grade`).
    """
    if chunk_points <= 0:
        msg = f"chunk_points must be a positive point count; got {chunk_points}."
        raise ValueError(msg)
    report = progress if progress is not None else _no_op_progress
    size = voxel_size_for_grade(grade)
    if size == 0.0:
        # Grade 0 is the identity: apply only the class filter, keep everything
        # else. No voxel grouping, so no Parquet spill is needed.
        return _write_survivors(
            source,
            output,
            include_classes,
            exclude_classes,
            None,
            chunk_points,
            report,
        )
    if workdir is None:
        with tempfile.TemporaryDirectory(prefix="ahn_cli_voxel_") as tmp:
            return _thin_with_spill(
                source,
                output,
                size,
                include_classes,
                exclude_classes,
                Path(tmp),
                chunk_points,
                report,
            )
    return _thin_with_spill(
        source,
        output,
        size,
        include_classes,
        exclude_classes,
        workdir,
        chunk_points,
        report,
    )


def _thin_with_spill(
    source: Path,
    output: Path,
    size: float,
    include_classes: tuple[int, ...],
    exclude_classes: tuple[int, ...],
    workdir: Path,
    chunk_points: int,
    report: ProgressCallback,
) -> int:
    """Run the spill -> group -> write voxel thinning under a scratch ``workdir``."""
    spill = workdir / _SPILL_SUBDIR
    if spill.exists():
        shutil.rmtree(spill)
    spill.mkdir(parents=True)
    try:
        kept = _spill_pass(
            source, include_classes, exclude_classes, spill, chunk_points
        )
        keep_mask = (
            _survivor_mask(spill, size, kept)
            if kept > 0
            else np.zeros(0, dtype=np.bool_)
        )
        return _write_survivors(
            source,
            output,
            include_classes,
            exclude_classes,
            keep_mask,
            chunk_points,
            report,
        )
    finally:
        shutil.rmtree(spill, ignore_errors=True)


def _class_keep(
    classification: npt.NDArray[np.generic],
    include: tuple[int, ...],
    exclude: tuple[int, ...],
) -> npt.NDArray[np.bool_]:
    """Return the boolean keep-mask for the classification filter over a chunk.

    A point is kept when its class is in ``include`` (or ``include`` is empty) and
    not in ``exclude``. Empty on both sides keeps every point. Mirrors the
    in-memory ``transform._class_mask`` on a per-chunk classification array.
    """
    keep = np.ones(classification.shape[0], dtype=np.bool_)
    if include:
        keep &= np.isin(classification, np.asarray(include))
    if exclude:
        keep &= ~np.isin(classification, np.asarray(exclude))
    return keep


def _spill_pass(
    source: Path,
    include: tuple[int, ...],
    exclude: tuple[int, ...],
    spill: Path,
    chunk_points: int,
) -> int:
    """Stream ``source``, spilling each class-kept point's ``(x, y, z, idx)``.

    ``idx`` is the point's dense index in the class-filtered cloud, assigned in
    streamed (file) order across chunks. Returns the total class-kept count.
    Chunks with no kept point write no Parquet file.
    """
    kept = 0
    with laspy.open(str(source)) as reader:
        for chunk_no, chunk in enumerate(
            reader.chunk_iterator(chunk_points), start=1
        ):
            cls_keep = _class_keep(
                np.asarray(chunk.classification), include, exclude
            )
            count = int(cls_keep.sum())
            if count > 0:
                frame = pl.DataFrame(
                    {
                        "x": np.asarray(chunk.x, dtype=np.float64)[cls_keep],
                        "y": np.asarray(chunk.y, dtype=np.float64)[cls_keep],
                        "z": np.asarray(chunk.z, dtype=np.float64)[cls_keep],
                        "idx": np.arange(kept, kept + count, dtype=np.int64),
                    }
                )
                frame.write_parquet(spill / f"chunk_{chunk_no:06d}.parquet")
            kept += count
    return kept


def _survivor_mask(
    spill: Path, size: float, kept: int
) -> npt.NDArray[np.bool_]:
    """Reduce the Parquet spill to a survivor mask over the filtered indices.

    Anchors the voxel grid at the spilled coordinates' per-axis minimum, quantises
    each point to its voxel, and keeps the minimum ``idx`` per voxel -- all in
    Polars' streaming engine. Returns a length-``kept`` boolean mask, ``True`` at
    each surviving filtered index.
    """
    glob = str(spill / _SPILL_GLOB)
    origin = (
        pl.scan_parquet(glob)
        .select(
            pl.col("x").min().alias("x"),
            pl.col("y").min().alias("y"),
            pl.col("z").min().alias("z"),
        )
        .collect(engine="streaming")
    )
    ox = float(origin.item(0, "x"))
    oy = float(origin.item(0, "y"))
    oz = float(origin.item(0, "z"))
    survivors = (
        pl.scan_parquet(glob)
        .with_columns(
            ((pl.col("x") - ox) / size).floor().cast(pl.Int64).alias("cx"),
            ((pl.col("y") - oy) / size).floor().cast(pl.Int64).alias("cy"),
            ((pl.col("z") - oz) / size).floor().cast(pl.Int64).alias("cz"),
        )
        .group_by("cx", "cy", "cz")
        .agg(pl.col("idx").min().alias("idx"))
        .select("idx")
        .collect(engine="streaming")
        .get_column("idx")
        .to_numpy()
    )
    mask = np.zeros(kept, dtype=np.bool_)
    mask[survivors] = True
    return mask


def _write_survivors(
    source: Path,
    output: Path,
    include: tuple[int, ...],
    exclude: tuple[int, ...],
    keep_mask: npt.NDArray[np.bool_] | None,
    chunk_points: int,
    report: ProgressCallback,
) -> int:
    """Stream ``source`` and write the surviving points to ``output``.

    ``keep_mask`` is ``None`` for the grade-0 identity (every class-kept point is
    written) or a survivor mask indexed by the point's filtered index (from
    :func:`_survivor_mask`). The write goes to a sibling temp file swapped into
    ``output`` at the end, so a source-equals-output in-place thin is safe. Ticks
    ``report(chunk, total_chunks)`` per streamed chunk. Returns the written count.
    """
    tmp_out = output.with_name(f"{output.stem}.tmp{output.suffix}")
    written = 0
    filtered = 0
    with (
        laspy.open(str(source)) as reader,
        laspy.open(str(tmp_out), mode="w", header=reader.header) as writer,
    ):
        total = int(reader.header.point_count)
        total_chunks = max(1, -(-total // chunk_points))
        for chunk_no, chunk in enumerate(
            reader.chunk_iterator(chunk_points), start=1
        ):
            cls_keep = _class_keep(
                np.asarray(chunk.classification), include, exclude
            )
            if keep_mask is None:
                point_keep = cls_keep
            else:
                count = int(cls_keep.sum())
                point_keep = np.zeros(len(chunk), dtype=np.bool_)
                point_keep[cls_keep] = keep_mask[filtered : filtered + count]
                filtered += count
            selected = chunk[point_keep]
            if len(selected) > 0:
                writer.write_points(selected)
                written += len(selected)
            report(chunk_no, total_chunks)
    tmp_out.replace(output)
    return written
