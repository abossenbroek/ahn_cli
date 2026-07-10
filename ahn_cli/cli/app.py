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
    AcquisitionRequest,
    AreaSelectorKind,
    SourceNotWiredError,
    acquire,
    create_site_layout,
)
from ahn_cli.prep.transform import (
    PrepRequest,
    TransformNotWiredError,
    prepare,
)


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
def fetch(
    out: Path,
    city: str | None,
    bbox: str | None,
    geojson: str | None,
) -> None:
    """Acquire raw source tiles for one site (acquisition stage only).

    Validates that exactly one area selector is given, creates the
    ``<out>/{ahn,ortho,viirs}/`` layout, and dispatches to the fetch context.
    Downloading itself is not wired in WP2, so this reports the un-wired seam.
    """
    selector, area = _select_area(city, bbox, geojson)
    create_site_layout(out)
    request = AcquisitionRequest(site_dir=out, selector=selector, area=area)
    try:
        acquire(request)
    except SourceNotWiredError as exc:
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
def prep(
    data: Path,
    include_class: str | None,
    exclude_class: str | None,
    *,
    points: bool,
) -> None:
    """Transform and export a fetched site (transform stage only).

    Parses and validates the classification filters, then dispatches to the
    prep context. The transforms themselves are not wired in WP2, so this
    reports the un-wired seam.
    """
    include = _parse_classes(include_class)
    exclude = _parse_classes(exclude_class)
    _reject_class_overlap(include, exclude)
    request = PrepRequest(
        data_dir=data,
        include_classes=include,
        exclude_classes=exclude,
        export_points=points,
    )
    try:
        prepare(request)
    except TransformNotWiredError as exc:
        raise click.ClickException(str(exc)) from exc
