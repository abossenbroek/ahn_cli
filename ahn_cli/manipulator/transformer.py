# DEPRECATED; ANY LOGIC USED IN THIS CODE SHOULD BE MOVED
# Legacy pre-7rad module, pending migration into the new bounded contexts.
import warnings

warnings.warn(
    "ahn_cli.manipulator.transformer is a deprecated pre-7rad module; logic must move into the new bounded contexts",
    DeprecationWarning,
    stacklevel=2,
)

import pyproj
import shapely
from shapely.ops import transform


# Memo: hope to support more types of geometry
def transform_polygon(
    geometry: shapely.Polygon, source_crs: str, target_crs: str
) -> shapely.Polygon | None:
    proj = pyproj.Transformer.from_crs(
        pyproj.CRS(source_crs),
        pyproj.CRS(target_crs),
        always_xy=True,
    ).transform
    if geometry.is_empty:
        return None
    elif geometry.geom_type == "Polygon":
        return transform(proj, geometry)
    return None
