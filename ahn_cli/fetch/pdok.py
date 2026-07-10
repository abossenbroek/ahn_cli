"""WP6 red stub: PDOK ATOM distribution source (not yet implemented)."""

from __future__ import annotations

from dataclasses import dataclass

from ahn_cli.domain import BBox, Generation
from ahn_cli.fetch.generation import GenerationRegistry, GenerationSource
from ahn_cli.fetch.source import HttpGet, ResolvedFeed


class PdokFeedError(ValueError):
    """Malformed PDOK feed (stub)."""


@dataclass(frozen=True)
class AtomTile:
    """A section tile (stub)."""

    tile_id: str
    bbox_wgs84: BBox
    download_url: str


@dataclass(frozen=True)
class AtomFeed:
    """A parsed feed (stub)."""

    licence: str
    attribution: str
    tiles: tuple[AtomTile, ...]


def parse_atom_feed(data: bytes) -> AtomFeed:
    """Not yet implemented (stub)."""
    del data
    return AtomFeed(licence="", attribution="", tiles=())


def _stub_registry() -> GenerationRegistry:
    """Return a two-generation registry with always-false probes (stub)."""
    registry = GenerationRegistry()
    for number in (5, 4):
        registry.register(
            GenerationSource(
                generation=Generation(number),
                base_url="https://stub/",
                probe=lambda aoi: bool(aoi) and False,
                semantics="stub",
            )
        )
    return registry


class PdokSource:
    """PDOK source (stub)."""

    def generation_registry(self, http_get: HttpGet) -> GenerationRegistry:
        """Not yet implemented (stub)."""
        del http_get
        return _stub_registry()

    def resolve(
        self,
        generation_source: GenerationSource,
        aoi: BBox,
        http_get: HttpGet,
    ) -> ResolvedFeed:
        """Not yet implemented (stub)."""
        del generation_source, aoi, http_get
        return ResolvedFeed(licence="", attribution="", tiles=())
