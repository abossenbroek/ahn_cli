# DEPRECATED; ANY LOGIC USED IN THIS CODE SHOULD BE MOVED
# Legacy pre-7rad module, pending migration into the new bounded contexts.
import warnings

warnings.warn(
    "ahn_cli.manipulator.geojson is a deprecated pre-7rad module; logic must move into the new bounded contexts",
    DeprecationWarning,
    stacklevel=2,
)

import geopandas as gdp
from shapely.geometry import Polygon


def read_geojson(path: str) -> gdp.GeoDataFrame:
    gdf = gdp.read_file(path)
    return gdf


def extract_polygon(gdf: gdp.GeoDataFrame) -> Polygon | None:
    # return first polygon
    for geom in gdf.geometry:
        if geom.geom_type == "Polygon":
            return geom.coords[0]
    return None
