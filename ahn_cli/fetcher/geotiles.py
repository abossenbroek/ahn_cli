import os

import geopandas as gpd
from pyproj import Transformer

from ahn_cli.fetcher.municipality import city_polygon


def geotiles() -> gpd.GeoDataFrame:
    file_path = os.path.join(
        os.path.dirname(__file__), "data/ahn_subunit.geojson"
    )

    ahn_tile_gdf = gpd.read_file(file_path)
    return ahn_tile_gdf


def ahn_subunit_indices_of_city(city_name: str) -> list[str]:
    """Return a list of AHN tile indices that intersect with the city's boundary."""  # noqa
    city_poly = city_polygon(city_name)
    geotiles_tile_gdf = geotiles()

    # Filter the DataFrame based on lowercase column values
    filtered_df = geotiles_tile_gdf.overlay(city_poly)
    tile_indices: list[str] = filtered_df["AHN_subuni"].tolist()  # noqa

    return tile_indices


def ahn_subunit_indices_of_bbox(bbox: list[float]) -> list[str]:
    """Return a list of AHN tile indices that intersect with the bbox."""  # noqa
    geotiles_tile_gdf = geotiles()

    transformer = Transformer.from_crs(
        "EPSG:28992", "EPSG:4326", always_xy=True
    )
    minx, miny = transformer.transform(bbox[0], bbox[1])
    maxx, maxy = transformer.transform(bbox[2], bbox[3])
    # Filter the DataFrame based on lowercase column values
    filtered_df = geotiles_tile_gdf.cx[minx:maxx, miny:maxy]
    tile_indices: list[str] = filtered_df["AHN_subuni"].tolist()  # noqa

    return tile_indices


def ahn_subunit_indices_of_geojson(geojson_path: str) -> list[str]:
    """Return a list of AHN tile indices that intersect with the GeoJSON polygon(s)."""
    # Read the GeoJSON file
    gdf = gpd.read_file(geojson_path)

    # Transform to Dutch national grid if needed
    if gdf.crs != "EPSG:28992":
        gdf = gdf.to_crs("EPSG:28992")

    # Get the AHN tiles (they are in EPSG:4326)
    tiles_gdf = geotiles()

    # Transform tiles to EPSG:28992 to match the input geometry
    tiles_gdf = tiles_gdf.to_crs("EPSG:28992")

    # Use spatial join for efficient intersection
    intersecting = gpd.sjoin(
        tiles_gdf, gdf, how="inner", predicate="intersects"
    )

    if intersecting.empty:
        return []

    # Get unique tile indices
    tile_indices: list[str] = intersecting["AHN_subuni"].unique().tolist()

    return tile_indices
