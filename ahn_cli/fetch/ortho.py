"""WP8 RED stub: importable placeholders for the ortho fetcher.

This is the failing-test-first commit's production surface: every public name
the ortho tests import is defined so the suite *collects*, but the behaviour is
unimplemented, so the tests fail at the call/assertion level (not at import).
The real implementation lands in the following green commit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from ahn_cli.domain import BBox, Vintage
    from ahn_cli.fetch.acquisition import AcquisitionRequest
    from ahn_cli.fetch.source import HttpGet, RemoteTile

_NOT_IMPLEMENTED = "WP8 ortho fetch is not implemented yet (red stub)."


class OrthoFeedError(ValueError):
    """Raised when a Beeldmateriaal ATOM feed cannot be parsed."""


class OrthoUnavailableError(RuntimeError):
    """Raised when no pinned ortho zone covers the AOI."""


class DuplicateOrthoTierError(ValueError):
    """Raised when registering two datasets at the same resolution tier."""


@dataclass(frozen=True)
class OrthoDataset:
    """A pinned Beeldmateriaal ortho zone (red stub: no validation yet)."""

    vintage: Vintage
    zone: str
    resolution_tier: str
    resolution_m: float
    feed_url: str
    semantics: str


@dataclass
class OrthoDatasetRegistry:
    """A preference-ordered registry of ortho zones (red stub)."""

    def register(self, dataset: OrthoDataset) -> None:
        """Unimplemented in the red stub."""
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def datasets(self) -> tuple[OrthoDataset, ...]:
        """Unimplemented in the red stub."""
        raise NotImplementedError(_NOT_IMPLEMENTED)


@dataclass(frozen=True)
class ResolvedOrthoFeed:
    """The sheets covering an AOI plus CC-BY terms (red stub)."""

    licence: str
    attribution: str
    tiles: tuple[RemoteTile, ...]


@dataclass(frozen=True)
class OrthoMosaic:
    """The mosaic-and-clip result (red stub)."""

    bbox: BBox
    crs: str
    width: int
    height: int
    resolution_m: float
    pixel_checksum: str


@dataclass(frozen=True)
class OrthoAcquisition:
    """The ortho acquisition result (red stub)."""

    mosaic_path: Path
    provenance_path: Path
    dataset: OrthoDataset
    mosaic: OrthoMosaic
    tile_paths: tuple[Path, ...]


def default_ortho_registry() -> OrthoDatasetRegistry:
    """Unimplemented in the red stub."""
    raise NotImplementedError(_NOT_IMPLEMENTED)


def resolve_ortho_tiles(
    dataset: OrthoDataset,
    aoi: BBox,
    http_get: HttpGet,
) -> ResolvedOrthoFeed:
    """Unimplemented in the red stub."""
    raise NotImplementedError(_NOT_IMPLEMENTED)


def select_ortho_dataset(
    aoi: BBox,
    registry: OrthoDatasetRegistry,
    http_get: HttpGet,
) -> OrthoDataset:
    """Unimplemented in the red stub."""
    raise NotImplementedError(_NOT_IMPLEMENTED)


def mosaic_and_clip(
    tile_paths: tuple[Path, ...],
    aoi: BBox,
    resolution_m: float,
    out_path: Path,
) -> OrthoMosaic:
    """Unimplemented in the red stub."""
    raise NotImplementedError(_NOT_IMPLEMENTED)


def acquire_ortho(
    request: AcquisitionRequest,
    **kwargs: object,
) -> OrthoAcquisition:
    """Unimplemented in the red stub."""
    raise NotImplementedError(_NOT_IMPLEMENTED)
