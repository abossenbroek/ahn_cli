# DEPRECATED; ANY LOGIC USED IN THIS CODE SHOULD BE MOVED
# Legacy pre-7rad module, pending migration into the new bounded contexts.
import warnings

warnings.warn(
    "ahn_cli.kwargs is a deprecated pre-7rad module; logic must move into the new bounded contexts",
    DeprecationWarning,
    stacklevel=2,
)

from typing import TypedDict


class CLIArgs(TypedDict):
    output: str
    city: str
    include_class: str | None
    exclude_class: str | None
    no_clip_city: bool
    clip_file: str | None
    epsg: int | None
    decimate: int | None
    bbox: list[float] | None
    geojson: str | None
    preview: bool
    no_verify: bool
    verify_pdal: bool
    bbox_tolerance: float
    strict_bbox_check: bool
