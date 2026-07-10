# DEPRECATED; ANY LOGIC USED IN THIS CODE SHOULD BE MOVED
# Legacy pre-7rad module, pending migration into the new bounded contexts.
import warnings

warnings.warn(
    "ahn_cli.manipulator.rasterizer is a deprecated pre-7rad module; logic must move into the new bounded contexts",
    DeprecationWarning,
    stacklevel=2,
)

import numpy as np
from rasterio.features import rasterize
from rasterio.transform import Affine, from_origin
from shapely import Polygon


def polygon_to_raster(
    polygon: Polygon,
    resolution: float,
) -> tuple[np.ndarray, Affine]:
    """
    Converts a polygon into a rasterized numpy array.

    Args:
        polygon (Polygon): The input polygon to be rasterized.
        resolution (float): The desired resolution of the rasterized array.

    Returns:
        Tuple[np.ndarray, Affine]: A tuple containing the rasterized numpy array and the affine transformation matrix.
    """

    bbox = polygon.bounds
    height = int((bbox[3] - bbox[1]) / resolution)
    width = int((bbox[2] - bbox[0]) / resolution)

    transform = from_origin(bbox[0], bbox[3], resolution, resolution)
    shape = (height, width)
    rasterized = rasterize(
        shapes=[polygon],
        out_shape=shape,
        transform=transform,
        fill=0,
        all_touched=True,
        dtype="uint8",
    )
    return rasterized, transform
