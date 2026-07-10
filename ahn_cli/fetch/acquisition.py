"""WP6 red stub: fetch-context acquisition actuation (not yet implemented)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING

from ahn_cli.fetch.geotiles_source import GeotilesSource
from ahn_cli.fetch.pdok import PdokSource
from ahn_cli.fetch.source import FetchSource, HttpGet, SourceKind

if TYPE_CHECKING:
    from pathlib import Path

    from ahn_cli.domain import Generation

SITE_SUBDIRS: tuple[str, ...] = ("ahn", "ortho", "viirs")

Clock = Callable[[], datetime]


class AreaSelectorKind(Enum):
    """AOI selector kind (stub)."""

    CITY = "city"
    BBOX = "bbox"
    GEOJSON = "geojson"


class AcquisitionError(RuntimeError):
    """Base acquisition failure (stub)."""


class MalformedBboxError(AcquisitionError):
    """Malformed bbox (stub)."""


class SelectorNotWiredError(AcquisitionError):
    """Selector AOI derivation not wired (stub)."""


@dataclass(frozen=True)
class AcquisitionRequest:
    """Acquisition intent (stub)."""

    site_dir: Path
    selector: AreaSelectorKind
    area: str
    source: SourceKind = SourceKind.PDOK
    generation: Generation | None = None


def create_site_layout(site_dir: Path) -> tuple[Path, ...]:
    """Create the site subdirectories and return their paths."""
    created: list[Path] = []
    for name in SITE_SUBDIRS:
        subdir = site_dir / name
        subdir.mkdir(parents=True, exist_ok=True)
        created.append(subdir)
    return tuple(created)


_SOURCE_REGISTRY: dict[SourceKind, FetchSource] = {
    SourceKind.PDOK: PdokSource(),
    SourceKind.GEOTILES: GeotilesSource(),
}


def source_for(kind: SourceKind) -> FetchSource:
    """Return the source registered for ``kind`` (stub)."""
    return _SOURCE_REGISTRY[kind]


def default_http_get(url: str) -> bytes:
    """Not yet implemented (stub)."""
    del url
    return b""


def _utcnow() -> datetime:
    """Return the current UTC time."""
    return datetime.now(timezone.utc)


def acquire(
    request: AcquisitionRequest,
    *,
    http_get: HttpGet = default_http_get,
    now: Clock = _utcnow,
    cache_root: Path | None = None,
    tool_version: str | None = None,
) -> tuple[Path, ...]:
    """Not yet implemented (stub): creates the layout, downloads nothing."""
    del http_get, now, cache_root, tool_version
    create_site_layout(request.site_dir)
    return ()
