"""WP6 red stub: shared fetch-source vocabulary (not yet implemented)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Protocol

from ahn_cli.domain import BBox

if TYPE_CHECKING:
    from ahn_cli.fetch.generation import GenerationRegistry, GenerationSource

HttpGet = Callable[[str], bytes]


class UnknownSourceError(LookupError):
    """Unknown --source token (stub)."""


class SourceKind(Enum):
    """Distribution sources (stub)."""

    PDOK = "pdok"
    GEOTILES = "geotiles"


def source_kind_tokens() -> tuple[str, ...]:
    """Not yet implemented (stub)."""
    return ()


def resolve_source_token(token: str) -> SourceKind:
    """Not yet implemented (stub)."""
    del token
    return SourceKind.PDOK


@dataclass(frozen=True)
class RemoteTile:
    """Download address of one sheet (stub: no validation)."""

    tile_id: str
    bbox: BBox
    download_url: str


@dataclass(frozen=True)
class ResolvedFeed:
    """Resolved tiles plus licence terms (stub)."""

    licence: str
    attribution: str
    tiles: tuple[RemoteTile, ...]


class FetchSource(Protocol):
    """Source behaviour (stub)."""

    def generation_registry(self, http_get: HttpGet) -> GenerationRegistry:
        """Stub."""
        ...

    def resolve(
        self,
        generation_source: GenerationSource,
        aoi: BBox,
        http_get: HttpGet,
    ) -> ResolvedFeed:
        """Stub."""
        ...


def to_wgs84(bbox_rd: BBox) -> BBox:
    """Not yet implemented (stub: identity, no validation)."""
    return bbox_rd


def to_rd(bbox_wgs84: BBox) -> BBox:
    """Not yet implemented (stub: identity, no validation)."""
    return bbox_wgs84


def boxes_intersect(a: BBox, b: BBox) -> bool:
    """Not yet implemented (stub)."""
    del a, b
    return False
