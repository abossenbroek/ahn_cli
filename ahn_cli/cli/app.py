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

from pathlib import Path

import click

from ahn_cli.fetch.acquisition import (
    AcquisitionError,
    AcquisitionRequest,
    AreaSelectorKind,
    acquire,
    create_site_layout,
)
from ahn_cli.fetch.generation import AUTO_CHOICE, default_registry
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
from ahn_cli.prep.transform import (
    PrepRequest,
    TransformNotWiredError,
    prepare,
)

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
def fetch(
    out: Path,
    city: str | None,
    bbox: str | None,
    geojson: str | None,
    ahn: str,
    source: str,
) -> None:
    """Acquire raw source tiles for one site (acquisition stage only).

    Validates that exactly one area selector is given, resolves the requested
    AHN generation and distribution source, creates the
    ``<out>/{ahn,ortho,viirs}/`` layout, and downloads the covering sheets
    (through the content cache) with a provenance sidecar per sheet.
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
        acquire(request)
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
def prep(
    data: Path,
    include_class: str | None,
    exclude_class: str | None,
    thin_method: str | None,
    thin_grade: int | None,
    thin_radius: float | None,
    thin_seed: int,
    *,
    points: bool,
) -> None:
    """Transform and export a fetched site (transform stage only).

    Parses and validates the classification filters and the graded-thinning
    request, then dispatches to the prep context. The transforms themselves are
    not wired yet, so this reports the un-wired seam.
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
    )
    try:
        prepare(request)
    except TransformNotWiredError as exc:
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
def import_viirs_command(out: Path, geotiff: Path) -> None:
    """Import an externally-produced VIIRS GeoTIFF into ``<out>/viirs/``.

    Verify-opens the raster, records its CRS/extent/bands and a content
    checksum, copies it byte-for-byte into ``<out>/viirs/``, and writes a
    provenance sidecar beside it. No reprojection or resampling is performed.
    """
    try:
        result = import_viirs(geotiff, out)
    except ViirsImportError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Imported VIIRS raster to {result.dest_path}")
