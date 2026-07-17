"""The ``ahn_cli`` Click group and its ``fetch`` / ``prep`` subcommands.

This module is the interface-adapter layer: it declares the command-line
surface, parses and validates arguments, and translates the typed errors the
bounded contexts raise into user-facing Click errors. It holds no acquisition
or transform logic itself -- that lives in :mod:`ahn_cli.fetch.acquisition`
and :mod:`ahn_cli.prep.transform`.

The historical ``-e`` short-flag collision (``--exclude-class`` and ``--epsg``
both bound to ``-e``) is resolved by design: every short flag below is unique,
and a regression test asserts no flag is reused across the group.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

import click
from tqdm import tqdm

from ahn_cli.copc.build import build_copc
from ahn_cli.copc.octree import CopcError
from ahn_cli.fetch.acquisition import (
    AcquisitionError,
    AcquisitionRequest,
    AreaSelectorKind,
    acquire,
    create_site_layout,
)
from ahn_cli.fetch.dsm import fetch_dsm
from ahn_cli.fetch.generation import AUTO_CHOICE, default_registry
from ahn_cli.fetch.ortho import acquire_ortho
from ahn_cli.fetch.source import (
    SourceKind,
    resolve_source_token,
    source_kind_tokens,
)
from ahn_cli.fetch.viirs import ViirsImportError, import_viirs
from ahn_cli.prep.decimate import (
    DEFAULT_SEED,
    PoissonThinning,
    ThinMethod,
    Thinning,
    VoxelThinning,
)
from ahn_cli.prep.positions import (
    PositionsExportError,
    export_positions,
)
from ahn_cli.prep.transform import (
    PrepError,
    PrepRequest,
    prepare,
)
from ahn_cli.reconcile.method import (
    IdwInterp,
    InterpMethod,
    KrigingInterp,
    LinearInterp,
    Variogram,
    VariogramModel,
)
from ahn_cli.reconcile.reconcile import (
    ReconcileError,
    ReconcileRequest,
    reconcile,
)
from ahn_cli.reconcile.writers import OutputFormat
from ahn_cli.tiles3d.build import build_tiles3d
from ahn_cli.tiles3d.errors import Tiles3dError
from ahn_cli.tiles3d.profile import Profile

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    from ahn_cli.domain import ProgressCallback

_GENERATION_REGISTRY = default_registry()
"""The default AHN generation registry backing the ``--ahn`` choice.

Built once at import so the ``fetch`` command's ``--ahn`` token list is derived
from the registry (never a hardcoded switch); adding a generation to the
registry extends the CLI choices with no edit here.
"""


def _select_area(
    city: str | None,
    bbox: str | None,
    geojson: str | None,
) -> tuple[AreaSelectorKind, str]:
    """Return the single chosen area selector, or fail if not exactly one.

    Contract:
        - Exactly one of ``city`` / ``bbox`` / ``geojson`` must be non-``None``.
        - Returns the matching :class:`AreaSelectorKind` and its raw value.

    Failure modes:
        - :class:`click.UsageError` if zero or more than one selector is given,
          mirroring the legacy mutual-exclusivity rule without importing the
          deprecated validator.
    """
    chosen: list[tuple[AreaSelectorKind, str]] = []
    for kind, value in (
        (AreaSelectorKind.CITY, city),
        (AreaSelectorKind.BBOX, bbox),
        (AreaSelectorKind.GEOJSON, geojson),
    ):
        if value is not None:
            chosen.append((kind, value))
    if len(chosen) != 1:
        msg = "Specify exactly one of --city, --bbox, or --geojson."
        raise click.UsageError(msg)
    return chosen[0]


def _parse_classes(spec: str | None) -> tuple[int, ...]:
    """Parse a comma-separated classification-class list into integers.

    Contract:
        - ``None`` or an empty string yields the empty tuple (no filter).
        - Otherwise every comma-separated field must parse as an integer.

    Failure modes:
        - :class:`click.BadParameter` if any field is not an integer.
    """
    if not spec:
        return ()
    try:
        return tuple(int(part) for part in spec.split(","))
    except ValueError as exc:
        msg = f"class list must be comma-separated integers; got {spec!r}."
        raise click.BadParameter(msg) from exc


def _reject_class_overlap(
    include: tuple[int, ...],
    exclude: tuple[int, ...],
) -> None:
    """Reject any class that appears in both the include and exclude lists.

    Failure modes:
        - :class:`click.UsageError` listing every class requested on both sides,
          which is contradictory.
    """
    overlap = sorted(set(include) & set(exclude))
    if overlap:
        msg = f"classes cannot be both included and excluded: {overlap}."
        raise click.UsageError(msg)


def _parse_thinning(
    method: str | None,
    grade: int | None,
    radius: float | None,
    seed: int,
) -> Thinning | None:
    """Build the validated graded-thinning request from the CLI options.

    Contract:
        - ``method`` is ``None`` (no thinning), ``"voxel"`` or ``"poisson"``.
        - Voxel thinning requires ``--thin-grade`` and forbids ``--thin-radius``;
          Poisson thinning requires ``--thin-radius`` and forbids
          ``--thin-grade``. ``--thin-seed`` applies to Poisson only.
        - Returns the matching :data:`~ahn_cli.prep.decimate.Thinning`, or
          ``None`` when no method is requested.

    Failure modes:
        - :class:`click.UsageError` if a grade/radius is supplied without a
          method, or paired with the wrong method, or the required one is
          missing.
        - :class:`click.BadParameter` if the grade/radius value is out of range.
    """
    if method is None:
        if grade is not None or radius is not None:
            msg = "--thin-grade/--thin-radius require --thin-method."
            raise click.UsageError(msg)
        return None
    if method == ThinMethod.VOXEL.value:
        if grade is None:
            msg = "voxel thinning requires --thin-grade."
            raise click.UsageError(msg)
        if radius is not None:
            msg = "--thin-radius is not valid for voxel thinning."
            raise click.UsageError(msg)
        try:
            return VoxelThinning(grade=grade)
        except ValueError as exc:
            raise click.BadParameter(str(exc)) from exc
    if radius is None:
        msg = "poisson thinning requires --thin-radius."
        raise click.UsageError(msg)
    if grade is not None:
        msg = "--thin-grade is not valid for poisson thinning."
        raise click.UsageError(msg)
    try:
        return PoissonThinning(radius=radius, seed=seed)
    except ValueError as exc:
        raise click.BadParameter(str(exc)) from exc


@click.group()
def cli() -> None:
    """Acquire (``fetch``) and transform (``prep``) Dutch elevation data."""


@cli.command()
@click.option(
    "-o",
    "--out",
    "out",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Site directory to populate, e.g. data/delft.",
)
@click.option(
    "-c",
    "--city",
    "city",
    default=None,
    help="Acquire the area of a named municipality.",
)
@click.option(
    "-b",
    "--bbox",
    "bbox",
    default=None,
    help="Acquire an EPSG:28992 bounding box 'minx,miny,maxx,maxy'.",
)
@click.option(
    "-g",
    "--geojson",
    "geojson",
    default=None,
    help="Acquire the area of the polygon(s) in a GeoJSON file.",
)
@click.option(
    "--ahn",
    "ahn",
    type=click.Choice(_GENERATION_REGISTRY.tokens()),
    default=AUTO_CHOICE,
    show_default=True,
    help="AHN generation to fetch; 'auto' picks the newest available.",
)
@click.option(
    "--source",
    "source",
    type=click.Choice(source_kind_tokens()),
    default=SourceKind.PDOK.value,
    show_default=True,
    help="Distribution source; 'pdok' is primary, 'geotiles' the fallback.",
)
@click.option(
    "--dsm",
    "dsm",
    is_flag=True,
    help="Also fetch the DSM raster, windowed-clipped to <out>/dsm.tif.",
)
@click.option(
    "--ortho",
    "ortho",
    is_flag=True,
    help="Also fetch the Beeldmateriaal orthophoto (CC-BY) for the AOI.",
)
@click.option(
    "--progress/--no-progress",
    "progress",
    default=True,
    show_default=True,
    help="Show a progress bar during the run.",
)
@click.option(
    "-j",
    "--jobs",
    "jobs",
    type=int,
    default=1,
    show_default=True,
    help="Concurrent tile downloads (AHN and, with --ortho, orthophoto).",
)
def fetch(  # noqa: PLR0913 -- one CLI param per fetch option; a bag object would only hide them
    out: Path,
    city: str | None,
    bbox: str | None,
    geojson: str | None,
    ahn: str,
    source: str,
    jobs: int,
    *,
    dsm: bool,
    ortho: bool,
    progress: bool,
) -> None:
    """Acquire raw source tiles for one site (acquisition stage only).

    Validates that exactly one area selector is given, resolves the requested
    AHN generation and distribution source, creates the
    ``<out>/{ahn,ortho,viirs}/`` layout, and downloads the covering sheets
    (through the content cache) with a provenance sidecar per sheet. With
    ``--dsm`` it additionally windowed-reads the DSM COG and clips it to
    ``<out>/dsm.tif`` with its own provenance sidecar; with ``--ortho`` it also
    mosaics the Beeldmateriaal orthophoto to ``<out>/ortho/ortho.tif``.
    ``-j/--jobs`` (default 1, serial) downloads that many tiles concurrently
    for the AHN and orthophoto steps; the written output is identical to a
    serial run regardless of the job count.
    """
    selector, area = _select_area(city, bbox, geojson)
    generation = _GENERATION_REGISTRY.resolve_token(ahn)
    source_kind = resolve_source_token(source)
    create_site_layout(out)
    request = AcquisitionRequest(
        site_dir=out,
        selector=selector,
        area=area,
        source=source_kind,
        generation=generation,
    )
    try:
        with _progress_bar(
            enabled=progress, unit="tile", desc="fetch (ahn)"
        ) as bar:
            acquire(
                request,
                progress=_tqdm_progress(bar) if bar is not None else None,
                download_jobs=jobs,
            )
        if dsm:
            with _progress_bar(
                enabled=progress, unit="tile", desc="fetch (dsm)"
            ) as bar:
                fetch_dsm(
                    request,
                    progress=_tqdm_progress(bar) if bar is not None else None,
                )
        if ortho:
            with _progress_bar(
                enabled=progress, unit="tile", desc="fetch (ortho)"
            ) as bar:
                acquire_ortho(
                    request,
                    progress=_tqdm_progress(bar) if bar is not None else None,
                    download_jobs=jobs,
                )
    except AcquisitionError as exc:
        raise click.ClickException(str(exc)) from exc


@cli.command()
@click.option(
    "-d",
    "--data",
    "data",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Site directory produced by a prior fetch.",
)
@click.option(
    "-i",
    "--include-class",
    "include_class",
    default=None,
    help="Keep only these classes (comma-separated integers).",
)
@click.option(
    "-e",
    "--exclude-class",
    "exclude_class",
    default=None,
    help="Drop these classes (comma-separated integers).",
)
@click.option(
    "-p",
    "--points",
    "points",
    is_flag=True,
    help="Export the point cloud.",
)
@click.option(
    "--thin-method",
    "thin_method",
    type=click.Choice([m.value for m in ThinMethod]),
    default=None,
    help="Graded thinning method (additive to the legacy --decimate step).",
)
@click.option(
    "--thin-grade",
    "thin_grade",
    type=int,
    default=None,
    help="Voxel thinning grade 0-9 (0 keeps all; higher is coarser).",
)
@click.option(
    "--thin-radius",
    "thin_radius",
    type=float,
    default=None,
    help="Poisson-disk minimum spacing in metres.",
)
@click.option(
    "--thin-seed",
    "thin_seed",
    type=int,
    default=DEFAULT_SEED,
    show_default=True,
    help="Poisson-disk RNG seed (deterministic sampling).",
)
@click.option(
    "--workdir",
    "workdir",
    default=None,
    type=click.Path(file_okay=False, path_type=Path),
    help=(
        "Scratch directory for out-of-core voxel thinning's spill files "
        "(default: a private temp dir, cleaned up afterwards). Must not be "
        "shared by concurrent prep runs."
    ),
)
@click.option(
    "--progress/--no-progress",
    "progress",
    default=True,
    show_default=True,
    help="Show a progress bar during the run.",
)
def prep(  # noqa: PLR0913 -- one CLI param per prep option; a bag object would only hide them
    data: Path,
    include_class: str | None,
    exclude_class: str | None,
    thin_method: str | None,
    thin_grade: int | None,
    thin_radius: float | None,
    thin_seed: int,
    workdir: Path | None,
    *,
    points: bool,
    progress: bool,
) -> None:
    """Transform and export a fetched site (transform stage only).

    Parses and validates the classification filters and the graded-thinning
    request, then dispatches to the prep context, which deduplicates, filters,
    thins, writes the site provenance, and (with ``--points``) exports the
    point cloud.
    """
    include = _parse_classes(include_class)
    exclude = _parse_classes(exclude_class)
    _reject_class_overlap(include, exclude)
    thinning = _parse_thinning(
        thin_method, thin_grade, thin_radius, thin_seed
    )
    request = PrepRequest(
        data_dir=data,
        include_classes=include,
        exclude_classes=exclude,
        export_points=points,
        thinning=thinning,
        workdir=workdir,
    )
    try:
        with contextlib.ExitStack() as stack:
            dedup_bar = stack.enter_context(
                _progress_bar(
                    enabled=progress, unit="tile", desc="prep (dedup)"
                )
            )
            thin_bar = (
                stack.enter_context(
                    _progress_bar(
                        enabled=progress,
                        # Voxel thinning streams and ticks per chunk; Poisson
                        # runs in memory as one atomic phase.
                        unit="chunk"
                        if isinstance(thinning, VoxelThinning)
                        else "phase",
                        desc="prep (thin)",
                    )
                )
                if thinning is not None
                else None
            )
            export_bar = (
                stack.enter_context(
                    _progress_bar(
                        enabled=progress, unit="chunk", desc="prep (export)"
                    )
                )
                if points
                else None
            )
            prepare(
                request,
                dedup_progress=_tqdm_progress(dedup_bar)
                if dedup_bar is not None
                else None,
                thin_progress=_tqdm_progress(thin_bar)
                if thin_bar is not None
                else None,
                export_progress=_tqdm_progress(export_bar)
                if export_bar is not None
                else None,
            )
    except PrepError as exc:
        raise click.ClickException(str(exc)) from exc


@cli.command(name="import-viirs")
@click.option(
    "--out",
    "out",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Site directory to populate, e.g. data/delft.",
)
@click.argument(
    "geotiff",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--progress/--no-progress",
    "progress",
    default=True,
    show_default=True,
    help="Show a progress bar during the run.",
)
def import_viirs_command(out: Path, geotiff: Path, *, progress: bool) -> None:
    """Import an externally-produced VIIRS GeoTIFF into ``<out>/viirs/``.

    Verify-opens the raster, records its CRS/extent/bands and a content
    checksum, copies it byte-for-byte into ``<out>/viirs/``, and writes a
    provenance sidecar beside it. No reprojection or resampling is performed.
    """
    with _progress_bar(
        enabled=progress, unit="file", desc="import-viirs"
    ) as bar:
        cb = _tqdm_progress(bar) if bar is not None else None
        try:
            result = import_viirs(geotiff, out, progress=cb)
        except ViirsImportError as exc:
            raise click.ClickException(str(exc)) from exc
    click.echo(f"Imported VIIRS raster to {result.dest_path}")


@cli.command(name="export-positions")
@click.option(
    "--data",
    "data",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Site directory produced by a prior fetch (must contain dsm.tif).",
)
@click.option(
    "--progress/--no-progress",
    "progress",
    default=True,
    show_default=True,
    help="Show a progress bar during the run.",
)
def export_positions_command(data: Path, *, progress: bool) -> None:
    """Export ``<data>/dsm.tif`` to a deterministic ``<data>/positions.exr``.

    Reads the fetched DSM raster and writes a 3-channel float32 OpenEXR position
    map (R=easting, G=northing, B=elevation) for TouchDesigner. Void (nodata)
    pixels keep their easting/northing and take a Z=0.0 sentinel. The output is
    byte-identical across runs.
    """
    with _progress_bar(
        enabled=progress, unit="file", desc="export-positions"
    ) as bar:
        cb = _tqdm_progress(bar) if bar is not None else None
        try:
            stats = export_positions(
                data / "dsm.tif", data / "positions.exr", progress=cb
            )
        except PositionsExportError as exc:
            raise click.ClickException(str(exc)) from exc
    click.echo(
        f"Wrote {data / 'positions.exr'} "
        f"({stats.width}x{stats.height}, {stats.nodata_pixels} void px)"
    )


_IDW_DEFAULT = "2.0,12"
"""Default ``--idw`` spec: ``power,k``."""

_KRIGING_DEFAULT = "spherical,0.0,1.0,50.0,16"
"""Default ``--kriging`` spec: ``model,nugget,sill,range,k``."""

_IDW_FIELDS = 2
"""An ``--idw`` spec has exactly ``power,k``."""

_KRIGING_FIELDS = 5
"""A ``--kriging`` spec has exactly ``model,nugget,sill,range,k``."""


def _parse_idw(spec: str) -> IdwInterp:
    """Parse an ``--idw`` ``power,k`` spec into a validated request.

    Failure modes:
        - :class:`click.BadParameter` if the spec is not two fields or a value
          is out of range.
    """
    parts = spec.split(",")
    if len(parts) != _IDW_FIELDS:
        msg = f"--idw must be 'power,k'; got {spec!r}."
        raise click.BadParameter(msg)
    try:
        return IdwInterp(power=float(parts[0]), k=int(parts[1]))
    except ValueError as exc:
        raise click.BadParameter(str(exc)) from exc


def _parse_kriging(spec: str) -> KrigingInterp:
    """Parse a ``--kriging`` ``model,nugget,sill,range,k`` spec into a request.

    Failure modes:
        - :class:`click.BadParameter` if the spec is not five fields, names an
          unknown variogram model, or a value is out of range.
    """
    parts = spec.split(",")
    if len(parts) != _KRIGING_FIELDS:
        msg = f"--kriging must be 'model,nugget,sill,range,k'; got {spec!r}."
        raise click.BadParameter(msg)
    model, nugget, sill, vrange, neighbours = parts
    try:
        variogram = Variogram(
            VariogramModel(model), float(nugget), float(sill), float(vrange)
        )
        return KrigingInterp(variogram=variogram, k=int(neighbours))
    except ValueError as exc:
        raise click.BadParameter(str(exc)) from exc


def _parse_reconcile_method(
    method: str, idw_spec: str, kriging_spec: str
) -> InterpMethod:
    """Build the validated interpolation request from the CLI options."""
    if method == "linear":
        return LinearInterp()
    if method == "idw":
        return _parse_idw(idw_spec)
    return _parse_kriging(kriging_spec)


def _parse_formats(specs: tuple[str, ...]) -> tuple[OutputFormat, ...]:
    """Return the requested output formats, defaulting to all four when none."""
    if not specs:
        return tuple(OutputFormat)
    return tuple(OutputFormat(spec) for spec in specs)


def _parse_classes_spec(
    spec: str | None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Parse a ``--classes`` ``keep:LIST`` / ``drop:LIST`` spec.

    Returns ``(include, exclude)`` class tuples; ``None`` keeps every class.

    Failure modes:
        - :class:`click.BadParameter` on an unknown mode, an empty list, or a
          non-integer class code.
    """
    if not spec:
        return ((), ())
    mode, _, rest = spec.partition(":")
    if mode not in {"keep", "drop"} or not rest:
        msg = "--classes must be 'keep:LIST' or 'drop:LIST' (comma-int classes)."
        raise click.BadParameter(msg)
    try:
        codes = tuple(int(part) for part in rest.split(","))
    except ValueError as exc:
        msg = (
            f"--classes list must be comma-separated integers; got {rest!r}."
        )
        raise click.BadParameter(msg) from exc
    return (codes, ()) if mode == "keep" else ((), codes)


@cli.command(name="reconcile")
@click.option(
    "--ortho",
    "ortho",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Orthophoto GeoTIFF defining the target (e.g. 8 cm) grid.",
)
@click.option(
    "--cloud",
    "cloud",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="AHN point-cloud LAZ whose elevation is interpolated onto the grid.",
)
@click.option(
    "--out",
    "out",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory to write reconciled.<ext> output(s) into.",
)
@click.option(
    "--method",
    "method",
    type=click.Choice(["linear", "idw", "kriging"]),
    default="idw",
    show_default=True,
    help="Interpolation method for the elevation.",
)
@click.option(
    "--idw",
    "idw",
    default=_IDW_DEFAULT,
    show_default=True,
    help="IDW parameters as 'power,k' (used when --method idw).",
)
@click.option(
    "--kriging",
    "kriging",
    default=_KRIGING_DEFAULT,
    show_default=True,
    help=(
        "Kriging parameters as 'model,nugget,sill,range,k' "
        "(used when --method kriging)."
    ),
)
@click.option(
    "--classes",
    "classes",
    default=None,
    help=(
        "Class filter 'keep:2,6' or 'drop:7,18' (LAS codes); default keeps all. "
        "Coincident-XY returns are always de-duplicated (top Z kept)."
    ),
)
@click.option(
    "--format",
    "formats",
    type=click.Choice([fmt.value for fmt in OutputFormat]),
    multiple=True,
    help="Output format(s); repeatable. Default: all of laz, ply, pt, exr.",
)
@click.option(
    "--progress/--no-progress",
    "progress",
    default=True,
    show_default=True,
    help="Show a progress bar during the run.",
)
def reconcile_command(
    ortho: Path,
    cloud: Path,
    out: Path,
    method: str,
    idw: str,
    kriging: str,
    classes: str | None,
    formats: tuple[str, ...],
    *,
    progress: bool,
) -> None:
    """Interpolate a point cloud onto an ortho grid, emit a coloured cloud.

    Estimates an elevation at every ortho pixel centre from the AHN cloud
    (linear, IDW, or ordinary kriging), colours each pixel from the ortho, and
    writes the reconciled cloud as ``reconciled.<ext>`` for every requested
    format (laz/ply/pt/exr). The cloud is de-duplicated (and optionally class-
    filtered) before interpolation; the output is byte-deterministic.
    """
    interp_method = _parse_reconcile_method(method, idw, kriging)
    include, exclude = _parse_classes_spec(classes)
    request = ReconcileRequest(
        ortho_path=ortho,
        cloud_path=cloud,
        output_dir=out,
        method=interp_method,
        formats=_parse_formats(formats),
        include_classes=include,
        exclude_classes=exclude,
    )
    with _progress_bar(enabled=progress, unit="row", desc="reconcile") as bar:
        cb = _tqdm_progress(bar) if bar is not None else None
        try:
            stats = reconcile(request, progress=cb)
        except ReconcileError as exc:
            raise click.ClickException(str(exc)) from exc
    click.echo(
        f"Reconciled {stats.width}x{stats.height}: "
        f"{stats.source_points} pts -> {stats.cleaned_points} cleaned -> "
        f"{stats.valid_points} written; {len(stats.outputs)} file(s)."
    )


@cli.command(name="copc")
@click.option(
    "--cloud",
    "cloud",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help=(
        "Point-cloud LAZ to convert (prep's pointcloud.laz or "
        "reconcile's reconciled.laz)."
    ),
)
@click.option(
    "--out",
    "out",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Path of the .copc.laz file to write.",
)
@click.option(
    "--workdir",
    "workdir",
    default=None,
    type=click.Path(file_okay=False, path_type=Path),
    help=(
        "Scratch directory for the streaming bucket files "
        "(default: a private temp dir, cleaned up afterwards)."
    ),
)
@click.option(
    "--progress/--no-progress",
    "progress",
    default=True,
    show_default=True,
    help="Show a progress bar during the run.",
)
def copc_command(
    cloud: Path, out: Path, workdir: Path | None, *, progress: bool
) -> None:
    """Build a Cloud-Optimized Point Cloud (.copc.laz) from a LAZ cloud.

    Streams the cloud in bounded memory (arbitrarily large sites),
    de-duplicates at AHN's native 0.5 m voxel — outlier-aware, never
    coarser than the source grid — and builds the octree with cube and
    header bounds consistent by construction, so flat, below-sea-level
    Dutch terrain validates fully green under copc-validator.
    """
    with _progress_bar(enabled=progress, unit="pt", desc="copc") as bar:
        cb = _tqdm_progress(bar) if bar is not None else None
        try:
            result = build_copc(cloud, out, workdir=workdir, progress=cb)
        except CopcError as exc:
            raise click.ClickException(str(exc)) from exc
    click.echo(
        f"COPC {out}: {result.input_points} pts -> "
        f"{result.written_points} written across {result.node_count} "
        f"node(s), point format {result.point_format_id}."
    )


@cli.command(name="tiles3d")
@click.option(
    "--ortho",
    "ortho",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Orthophoto GeoTIFF (fetch --ortho output), EPSG:28992.",
)
@click.option(
    "--heights",
    "heights",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help=(
        "Reconciled EXR heights on the ortho's exact pixel grid "
        "(reconcile --format exr output)."
    ),
)
@click.option(
    "--out",
    "out",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help=(
        "Directory receiving the tileset: 'strict' writes tileset.json "
        "and the tiles/ glbs; 'game'/'heightfield'/'splat' write a packed "
        "tiles.hfp plus tileset.json, provenance.json and manifest.json "
        "sidecars."
    ),
)
@click.option(
    "--profile",
    "profile_name",
    default="strict",
    show_default=True,
    help=(
        "Export profile: 'strict' (lossless float32 glTF + PNG), 'game' "
        "(quantized, meshopt-compressed glTF + JPEG), 'heightfield' "
        "(vendor .hf height chunk + sibling JPEG), or 'splat' (3D Gaussian "
        "Splatting .ply cloud, no texture). The lossy 'game', 'heightfield' "
        "and 'splat' profiles pack the tiles into a tiles.hfp plus "
        "tileset.json, provenance.json and manifest.json sidecars."
    ),
)
@click.option(
    "--progress/--no-progress",
    "progress",
    default=True,
    show_default=True,
    help="Show a progress bar during the run.",
)
def tiles3d_command(
    ortho: Path,
    heights: Path,
    out: Path,
    profile_name: str,
    *,
    progress: bool,
) -> None:
    """Convert the ortho map to an OGC 3D Tiles 1.1 tileset.

    Drapes the orthophoto over the reconciled per-pixel elevations as a
    quadtree of binary glTF terrain tiles (OGC 22-025r4). The two
    inputs' dimensions must match perfectly — bit-exact pixel grid and
    colours — and any missing height is a hard error. Every written
    artifact is re-verified from disk against an independent rebuild
    before the tileset is accepted. ``--profile game`` emits the compact
    runtime glTF representation, ``--profile heightfield`` emits the
    vendor ``.hf`` height chunks with sibling JPEGs, and ``--profile splat``
    emits a 3D Gaussian Splatting ``.ply`` cloud (one gaussian per vertex,
    no texture); all three lossy profiles pack the tiles into a tiles.hfp
    plus tileset.json, provenance.json and manifest.json sidecars.
    ``--profile strict`` (the default) is the byte-frozen lossless profile.
    """
    try:
        profile = Profile.parse(profile_name)
    except Tiles3dError as exc:
        raise click.ClickException(str(exc)) from exc
    with _progress_bar(enabled=progress, unit="tile", desc="tiles3d") as bar:
        cb = _tqdm_progress(bar) if bar is not None else None
        try:
            result = build_tiles3d(
                ortho,
                heights,
                out,
                profile=profile,
                progress=cb,
            )
        except Tiles3dError as exc:
            raise click.ClickException(str(exc)) from exc
    # Both lossy profiles announce themselves; strict stays bare because it
    # is the byte-identical default.
    suffix = (
        f" profile={profile.value}." if profile is not Profile.STRICT else ""
    )
    click.echo(
        f"3D Tiles {result.tileset_path}: {result.tile_count} tile(s) "
        f"across {result.levels + 1} level(s), {result.vertices} "
        f"vertices, {result.triangles} triangles; verified.{suffix}"
    )


def _progress_bar(
    *, enabled: bool, unit: str, desc: str
) -> AbstractContextManager[tqdm[NoReturn] | None]:
    """Return a ``tqdm`` bar context when ``enabled``, else a no-op context.

    Contract:
        - ``enabled`` -- construct and yield a live ``tqdm`` bar.
        - not ``enabled`` -- yield ``None`` (via `contextlib.nullcontext`) so
          callers uniformly do ``with _progress_bar(...) as bar:`` and treat
          ``bar is None`` as "don't report progress".
    """
    if enabled:
        return tqdm(unit=unit, desc=desc)
    return contextlib.nullcontext()


def _tqdm_progress(bar: tqdm[NoReturn]) -> ProgressCallback:
    """Return a reconcile progress callback that drives ``bar``.

    Contract:
        - Each call sets ``bar``'s total to the reported total-row count (the
          same value on every call -- ``reconcile()`` already knows the grid's
          full height before streaming its first block) and its position to the
          reported rows-done count, then redraws.
    """

    def _report(done: int, total: int) -> None:
        bar.total = total
        bar.n = done
        bar.refresh()

    return _report
