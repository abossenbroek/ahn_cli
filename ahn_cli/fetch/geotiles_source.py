"""The GeoTiles.nl distribution source (fallback).

GeoTiles.nl (TU Delft) re-tiles AHN into 1x1.25km sub-tiles addressed by a
sheet id, indexed by the bundled ``ahn_subunit.geojson`` catalogue. This module
reuses that catalogue and the GeoTiles URL convention to serve the same
:class:`~ahn_cli.fetch.source.FetchSource` contract as PDOK, but WITHOUT
importing the deprecated ``ahn_cli.fetcher`` package: the bbox tile-select is
re-implemented here (a citation to
``ahn_cli.fetcher.geotiles.ahn_subunit_indices_of_bbox``, whose bbox-index
``.cx`` selection this mirrors) over a plain-JSON read of the catalogue file, so
this gated module emits no ``DeprecationWarning``.

Selection is an axis-aligned bounding-box intersection between the AOI and each
sheet's WGS84 bounds -- a safe *superset* of the true-geometry cover; any extra
edge sheet is clipped away downstream. The catalogue geometry is WGS84 (CRS84);
an EPSG:28992 AOI is projected to match it, and each selected sheet's extent is
projected back to the Dutch grid.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

if TYPE_CHECKING:
    from collections.abc import Iterator

from ahn_cli.domain import BBox, Generation
from ahn_cli.fetch.generation import (
    AvailabilityProbe,
    GenerationRegistry,
    GenerationSource,
)
from ahn_cli.fetch.source import (
    HttpGet,
    RemoteTile,
    ResolvedFeed,
    boxes_intersect,
    to_rd,
    to_wgs84,
)

_POSITION_LEN: Final = 2

# The bundled AHN tile catalogue, read directly as data. Referencing the file
# does NOT import ``ahn_cli.fetcher`` (which would emit a DeprecationWarning);
# it is the same data file that package reads.
_BUNDLED_CATALOG: Final = (
    Path(__file__).resolve().parent.parent
    / "fetcher"
    / "data"
    / "ahn_subunit.geojson"
)
_CATALOG_ID_PROPERTY: Final = "AHN_subuni"

# GeoTiles.nl per-generation tile endpoints, newest first. Duplicated (with
# citation) from ``ahn_cli.fetch.generation`` (which itself cites the deprecated
# ``ahn_cli.config``) so this module imports no deprecated code.
_GEOTILES_FEEDS: tuple[tuple[int, str, str], ...] = (
    (
        5,
        "https://geotiles.citg.tudelft.nl/AHN5_T/",
        "AHN5 point cloud via GeoTiles.nl re-tiling.",
    ),
    (
        4,
        "https://geotiles.citg.tudelft.nl/AHN4_T/",
        "AHN4 point cloud via GeoTiles.nl re-tiling.",
    ),
)
_GEOTILES_LICENCE: Final = (
    "https://creativecommons.org/publicdomain/zero/1.0/deed.nl"
)
_GEOTILES_ATTRIBUTION: Final = (
    "AHN (Rijkswaterstaat), re-tiled by GeoTiles.nl (TU Delft)."
)


class GeotilesCatalogError(ValueError):
    """Raised when the GeoTiles tile catalogue cannot be parsed.

    Signals a malformed catalogue: not a GeoJSON ``FeatureCollection``, a
    feature missing its sheet-id property, or a geometry with no coordinates
    from which to derive a sheet extent.
    """


@dataclass(frozen=True)
class CatalogTile:
    """One catalogue sheet: its id and WGS84 extent.

    Contract:
        - ``bbox_wgs84`` is the sheet geometry's bounding box in EPSG:4326.
    """

    tile_id: str
    bbox_wgs84: BBox


def load_catalog(path: Path = _BUNDLED_CATALOG) -> tuple[CatalogTile, ...]:
    """Read the GeoTiles sheet catalogue into :class:`CatalogTile` entries.

    Contract:
        - Reads a GeoJSON ``FeatureCollection`` whose features carry an
          ``AHN_subuni`` id and a Polygon/MultiPolygon geometry; each feature
          becomes one :class:`CatalogTile` with the geometry's WGS84 bounds.
        - The order of the returned tuple mirrors the file's feature order.

    Failure modes:
        - :class:`GeotilesCatalogError` if the document is not a
          ``FeatureCollection`` or a feature is malformed.
    """
    parsed = json.loads(path.read_bytes())
    if not isinstance(parsed, dict):
        msg = "GeoTiles catalogue must be a GeoJSON object."
        raise GeotilesCatalogError(msg)
    document = cast("dict[str, object]", parsed)
    features = document.get("features")
    if not isinstance(features, list):
        msg = "GeoTiles catalogue must have a 'features' array."
        raise GeotilesCatalogError(msg)
    feature_list = cast("list[object]", features)
    return tuple(_parse_feature(feature) for feature in feature_list)


def _parse_feature(feature: object) -> CatalogTile:
    """Parse one GeoJSON feature into a :class:`CatalogTile`."""
    if not isinstance(feature, dict):
        msg = "GeoTiles catalogue feature must be an object."
        raise GeotilesCatalogError(msg)
    feature_map = cast("dict[str, object]", feature)
    tile_id = _feature_tile_id(feature_map.get("properties"))
    bbox = _geometry_bounds(feature_map.get("geometry"))
    return CatalogTile(tile_id=tile_id, bbox_wgs84=bbox)


def _feature_tile_id(properties: object) -> str:
    """Extract the non-blank sheet id from a feature's properties."""
    if not isinstance(properties, dict):
        msg = "GeoTiles catalogue feature has no properties object."
        raise GeotilesCatalogError(msg)
    property_map = cast("dict[str, object]", properties)
    tile_id = property_map.get(_CATALOG_ID_PROPERTY)
    if not isinstance(tile_id, str) or not tile_id.strip():
        msg = f"feature is missing a non-blank {_CATALOG_ID_PROPERTY}."
        raise GeotilesCatalogError(msg)
    return tile_id


def _geometry_bounds(geometry: object) -> BBox:
    """Compute the WGS84 bounding box of a Polygon/MultiPolygon geometry."""
    if not isinstance(geometry, dict):
        msg = "GeoTiles catalogue feature has no geometry object."
        raise GeotilesCatalogError(msg)
    geometry_map = cast("dict[str, object]", geometry)
    lons: list[float] = []
    lats: list[float] = []
    for lon, lat in _iter_positions(geometry_map.get("coordinates")):
        lons.append(lon)
        lats.append(lat)
    if not lons:
        msg = "GeoTiles catalogue geometry has no coordinates."
        raise GeotilesCatalogError(msg)
    return (min(lons), min(lats), max(lons), max(lats))


def _iter_positions(node: object) -> Iterator[tuple[float, float]]:
    """Yield every ``(lon, lat)`` position within nested coordinate lists.

    A ``[lon, lat]`` leaf (both entries JSON numbers, never booleans) is a
    position; any other list is a coordinate container recursed into. The
    ``isinstance`` guards both classify the node and narrow the numbers so no
    unreachable coercion branch is needed.
    """
    if not isinstance(node, list):
        return
    coordinates = cast("list[object]", node)
    if len(coordinates) >= _POSITION_LEN:
        first = coordinates[0]
        second = coordinates[1]
        if (
            not isinstance(first, bool)
            and isinstance(first, (int, float))
            and not isinstance(second, bool)
            and isinstance(second, (int, float))
        ):
            yield (float(first), float(second))
            return
    for child in coordinates:
        yield from _iter_positions(child)


def _covering_tiles(
    catalog: tuple[CatalogTile, ...],
    aoi_rd: BBox,
    base_url: str,
) -> tuple[RemoteTile, ...]:
    """Return catalogue sheets whose bbox intersects ``aoi_rd`` (a superset)."""
    aoi_wgs84 = to_wgs84(aoi_rd)
    selected = [
        RemoteTile(
            tile_id=tile.tile_id,
            bbox=to_rd(tile.bbox_wgs84),
            download_url=f"{base_url}{tile.tile_id}.LAZ",
        )
        for tile in catalog
        if boxes_intersect(aoi_wgs84, tile.bbox_wgs84)
    ]
    return tuple(sorted(selected, key=lambda tile: tile.tile_id))


@dataclass(frozen=True)
class GeotilesSource:
    """The GeoTiles.nl distribution source (fallback).

    Implements :class:`~ahn_cli.fetch.source.FetchSource` over the bundled tile
    catalogue and the GeoTiles URL convention. Coverage is a catalogue lookup,
    so the injected ``http_get`` is unused here (downloads happen in the
    acquisition stage, through the cache).

    Contract:
        - ``catalog_path`` is the GeoJSON catalogue to read; it defaults to the
          bundled AHN sheet index and is overridable for testing.
    """

    catalog_path: Path = _BUNDLED_CATALOG

    def generation_registry(self, http_get: HttpGet) -> GenerationRegistry:
        """Return a registry of GeoTiles generations with catalogue probes.

        The GeoTiles grid is generation-independent, so every generation shares
        one catalogue-coverage probe (does any sheet intersect the AOI). The
        injected ``http_get`` is not needed for a local catalogue lookup.
        """
        del http_get
        catalog = load_catalog(self.catalog_path)
        registry = GenerationRegistry()
        for number, base_url, semantics in _GEOTILES_FEEDS:
            registry.register(
                GenerationSource(
                    generation=Generation(number),
                    base_url=base_url,
                    probe=_catalog_probe(catalog),
                    semantics=semantics,
                )
            )
        return registry

    def resolve(
        self,
        generation_source: GenerationSource,
        aoi: BBox,
        http_get: HttpGet,
    ) -> ResolvedFeed:
        """Resolve the GeoTiles sheets of ``generation_source`` covering ``aoi``.

        Reads the catalogue, selects intersecting sheets, and addresses each as
        ``<base_url><tile_id>.LAZ``. ``http_get`` is unused (see the class note).
        """
        del http_get
        catalog = load_catalog(self.catalog_path)
        tiles = _covering_tiles(catalog, aoi, generation_source.base_url)
        return ResolvedFeed(
            licence=_GEOTILES_LICENCE,
            attribution=_GEOTILES_ATTRIBUTION,
            tiles=tiles,
        )


def _catalog_probe(catalog: tuple[CatalogTile, ...]) -> AvailabilityProbe:
    """Return a probe reporting whether any sheet intersects the AOI."""

    def probe(aoi: BBox) -> bool:
        aoi_wgs84 = to_wgs84(aoi)
        return any(
            boxes_intersect(aoi_wgs84, tile.bbox_wgs84) for tile in catalog
        )

    return probe
