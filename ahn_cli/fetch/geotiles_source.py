"""WP6 red stub: GeoTiles.nl fallback source (not yet implemented)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from ahn_cli.domain import BBox, Generation
from ahn_cli.fetch.generation import GenerationRegistry, GenerationSource
from ahn_cli.fetch.source import HttpGet, ResolvedFeed

_BUNDLED_CATALOG: Final = (
    Path(__file__).resolve().parent.parent
    / "fetcher"
    / "data"
    / "ahn_subunit.geojson"
)


class GeotilesCatalogError(ValueError):
    """Malformed catalogue (stub)."""


@dataclass(frozen=True)
class CatalogTile:
    """A catalogue sheet (stub)."""

    tile_id: str
    bbox_wgs84: BBox


def load_catalog(path: Path = _BUNDLED_CATALOG) -> tuple[CatalogTile, ...]:
    """Not yet implemented (stub)."""
    del path
    return ()


@dataclass(frozen=True)
class GeotilesSource:
    """GeoTiles source (stub)."""

    catalog_path: Path = _BUNDLED_CATALOG

    def generation_registry(self, http_get: HttpGet) -> GenerationRegistry:
        """Not yet implemented (stub)."""
        del http_get
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

    def resolve(
        self,
        generation_source: GenerationSource,
        aoi: BBox,
        http_get: HttpGet,
    ) -> ResolvedFeed:
        """Not yet implemented (stub)."""
        del generation_source, aoi, http_get
        return ResolvedFeed(licence="", attribution="", tiles=())
