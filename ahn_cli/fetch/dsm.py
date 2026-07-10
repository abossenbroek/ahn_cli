"""Fetch a DSM (Digital Surface Model) raster for an AOI and clip it.

STUB (failing-first): this module declares the public surface of the DSM
windowed-fetch-and-clip feature so the WP7 tests import and run against real,
typed signatures. Every behaviour raises :class:`NotImplementedError`, so the
specs fail at call time (an assertion-level red), not at import/collection. The
next commit fills in the real windowed COG read, clip, cache-through, and
provenance.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from ahn_cli.domain import BBox
from ahn_cli.fetch.acquisition import AcquisitionError, AcquisitionRequest
from ahn_cli.fetch.source import HttpGet, ResolvedFeed, SourceKind

if TYPE_CHECKING:
    from pathlib import Path

WindowedDsmReader = Callable[[str, BBox], bytes]
"""An injected windowed COG reader: maps ``(url, aoi)`` to clipped GeoTIFF bytes."""

Clock = Callable[[], datetime]
"""An injected UTC clock, so download timestamps are deterministic in tests."""


class DsmError(AcquisitionError):
    """Raised when a DSM fetch cannot complete for an expected reason.

    Subclasses :class:`~ahn_cli.fetch.acquisition.AcquisitionError`, so the CLI
    reports every such failure as a tidy error rather than a traceback.
    """


@dataclass(frozen=True)
class DsmStats:
    """The metadata and QA read back from a clipped DSM raster."""

    crs: str
    bounds: BBox
    resolution: float
    nodata: float | None
    nodata_fraction: float
    spike_count: int
    width: int
    height: int


@dataclass(frozen=True)
class PdokDsmSource:
    """The PDOK INSPIRE ATOM distribution source for the DSM COG."""

    feed_url: str = ""

    def resolve(self, aoi: BBox, http_get: HttpGet) -> ResolvedFeed:
        """Resolve the DSM sheets covering ``aoi`` from the ATOM feed."""
        raise NotImplementedError


def dsm_source_for(kind: SourceKind) -> PdokDsmSource:
    """Return the DSM distribution source registered for ``kind``."""
    raise NotImplementedError


def read_dsm_window(url: str, aoi: BBox) -> bytes:
    """Windowed-read the DSM COG at ``url``, clipped to ``aoi``."""
    raise NotImplementedError


def inspect_dsm(content: bytes) -> DsmStats:
    """Read a clipped DSM raster's metadata and QA from its GeoTIFF bytes."""
    raise NotImplementedError


def fetch_dsm(
    request: AcquisitionRequest,
    *,
    http_get: HttpGet | None = None,
    reader: WindowedDsmReader | None = None,
    now: Clock | None = None,
    cache_root: Path | None = None,
    tool_version: str | None = None,
) -> Path:
    """Fetch and clip the DSM for ``request``'s AOI to ``<site>/dsm.tif``."""
    raise NotImplementedError
