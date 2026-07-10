"""Fetch-context acquisition seam.

The ``fetch`` bounded context turns a validated area of interest into raw,
cached source tiles on disk. WP2 ships only the *seam*: it materialises the
canonical ``data/<site>/{ahn,ortho,viirs}/`` directory layout and records the
intent to acquire, then refuses to fabricate data by raising
:class:`SourceNotWiredError`. Real portal fetchers arrive in WP5-WP9 and
replace :func:`acquire`'s body without changing this module's public surface.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from ahn_cli.domain import Generation

SITE_SUBDIRS: tuple[str, ...] = ("ahn", "ortho", "viirs")
"""Per-product subdirectories created under every site directory, in order."""


class AreaSelectorKind(Enum):
    """Which area-of-interest selector an acquisition request was built from.

    Modelled as an enum so the selected area is a closed, immutable value the
    fetch context can record and later branch on without a stringly-typed
    switch. WP2 only records it; WP5-WP9 dispatch on it.
    """

    CITY = "city"
    BBOX = "bbox"
    GEOJSON = "geojson"


class SourceNotWiredError(NotImplementedError):
    """No real fetcher is wired for the requested source yet (WP5-WP9).

    A :class:`NotImplementedError` subclass so a caller may catch it broadly
    while still distinguishing the deliberate "not yet built" state from an
    accidental one.
    """


@dataclass(frozen=True)
class AcquisitionRequest:
    """A validated intent to acquire source data for one site.

    Contract:
        - ``site_dir`` is the site root beneath which :func:`create_site_layout`
          materialises the product subdirectories.
        - ``selector`` records which area-of-interest kind was chosen; the
          caller guarantees exactly one was given.
        - ``area`` is that selector's raw value (a city name, a bbox string, or
          a GeoJSON path), carried verbatim for the fetchers that land later.
        - ``generation`` is the requested AHN generation: an explicit
          :class:`~ahn_cli.domain.Generation`, or ``None`` (the default) to
          request automatic newest-available selection at download time. WP5
          resolves the ``--ahn`` flag to this field; WP6 consults it when the
          real fetcher actuates and records it in the provenance sidecar.

    Invariants:
        - Frozen: an immutable, hashable value object, equal by field value, so
          it is safe as a cache key and a set/dict member.
    """

    site_dir: Path
    selector: AreaSelectorKind
    area: str
    generation: Generation | None = None


def create_site_layout(site_dir: Path) -> tuple[Path, ...]:
    """Create ``data/<site>/{ahn,ortho,viirs}/`` and return the subdir paths.

    Contract:
        - Creates ``site_dir`` and each :data:`SITE_SUBDIRS` entry beneath it,
          including any missing parents.
        - Idempotent: pre-existing directories are left intact and no error is
          raised when the layout already exists.
        - Returns the subdirectory paths in :data:`SITE_SUBDIRS` order, giving
          callers a deterministic, stable result.
    """
    created: list[Path] = []
    for name in SITE_SUBDIRS:
        subdir = site_dir / name
        subdir.mkdir(parents=True, exist_ok=True)
        created.append(subdir)
    return tuple(created)


def acquire(request: AcquisitionRequest) -> None:
    """Fetch-context seam: record intent, then refuse until a fetcher is wired.

    Contract:
        - Accepts a fully validated :class:`AcquisitionRequest`.
        - Performs no network I/O in WP2 and never fabricates data.

    Failure modes:
        - :class:`SourceNotWiredError`, unconditionally: WP2 wires no portal
          client, so acquisition cannot yet proceed. WP5-WP9 replace this body.
    """
    msg = (
        f"No fetch source is wired yet for site {request.site_dir}; "
        "real acquisition lands in WP5-WP9."
    )
    raise SourceNotWiredError(msg)
