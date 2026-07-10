# DEPRECATED; ANY LOGIC USED IN THIS CODE SHOULD BE MOVED
# Legacy pre-7rad module, pending migration into the new bounded contexts.
import warnings

warnings.warn(
    "ahn_cli.config is a deprecated pre-7rad module; logic must move into the new bounded contexts",
    DeprecationWarning,
    stacklevel=2,
)

from dataclasses import dataclass
from importlib.resources import files


@dataclass
class Config:
    geotiles_base_url = "https://geotiles.citg.tudelft.nl/AHN4_T/"
    # ahn_base_url = (
    #     "https://ns_hwh.fundaments.nl/hwh-ahn/ahn4/03a_DSM_0.5m/{tile_index}.zip"
    # )
    city_polygon_file = files("ahn_cli.fetcher.data").joinpath(
        "municipality_simple.geojson"
    )
