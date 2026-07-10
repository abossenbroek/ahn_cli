"""Fetch-context acquisition seam (WP2 stub).

RED stub: importable, but the behaviour is deliberately not implemented so the
WP2 tests fail at assertion time. The green implementation replaces the bodies.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

SITE_SUBDIRS: tuple[str, ...] = ("ahn", "ortho", "viirs")


class AreaSelectorKind(Enum):
    """Which area-of-interest selector an acquisition request was built from."""

    CITY = "city"
    BBOX = "bbox"
    GEOJSON = "geojson"


class SourceNotWiredError(NotImplementedError):
    """No real fetcher is wired for the requested source yet (WP5-WP9)."""


@dataclass(frozen=True)
class AcquisitionRequest:
    """A validated intent to acquire source data for one site (stub)."""

    site_dir: Path
    selector: AreaSelectorKind
    area: str


def create_site_layout(_site_dir: Path) -> tuple[Path, ...]:
    """RED stub; the real impl creates the {ahn,ortho,viirs} subdirectories."""
    return ()


def acquire(_request: AcquisitionRequest) -> None:
    """RED stub; the real seam raises SourceNotWiredError."""
