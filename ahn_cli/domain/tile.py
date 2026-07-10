"""The :class:`Tile` value object: the identity of one distributable tile.

A tile is the unit a portal hands out: a single product covering a bounded
rectangle, pinned to exactly one temporal axis -- a :class:`Generation` (for
AHN-family products) or a :class:`Vintage` (for dated imagery). This module
also exposes the shared bounding-box type and validator the domain reuses.
"""

from dataclasses import dataclass
from typing import TypeAlias

from ahn_cli.domain.generation import Generation
from ahn_cli.domain.product import Product
from ahn_cli.domain.vintage import Vintage

BBox: TypeAlias = tuple[float, float, float, float]
"""An axis-aligned bounding box ``(minx, miny, maxx, maxy)`` in EPSG:28992."""


def ensure_valid_bbox(bbox: BBox) -> None:
    """Raise if ``bbox`` is not a positive-area ``(minx, miny, maxx, maxy)`` box.

    Contract:
        - Accepts the shared :data:`BBox` 4-tuple in EPSG:28992 metres.
        - Returns ``None`` when ``minx < maxx`` and ``miny < maxy``.

    Failure modes:
        - ``ValueError`` if either extent is empty or inverted (``minx >= maxx``
          or ``miny >= maxy``), which would describe a degenerate box.
    """
    minx, miny, maxx, maxy = bbox
    if minx >= maxx or miny >= maxy:
        msg = (
            "bbox must be (minx, miny, maxx, maxy) with minx < maxx and "
            f"miny < maxy; got {bbox}."
        )
        raise ValueError(msg)


@dataclass(frozen=True)
class Tile:
    """The identity of a single distributable tile of one product.

    Contract:
        - ``tile_id`` is the portal's tile identifier (e.g. ``"37FN2"``); it
          must be non-blank.
        - ``product`` is the dataset kind the tile carries.
        - ``bbox`` is the tile's extent as :data:`BBox` in EPSG:28992.
        - Exactly one of ``generation`` / ``vintage`` pins the tile's temporal
          axis: AHN-family products carry a :class:`Generation`; dated imagery
          carries a :class:`Vintage`.

    Invariants:
        - Immutable and hashable; two tiles are equal iff every field is equal.

    Failure modes:
        - ``ValueError`` if ``tile_id`` is blank.
        - ``ValueError`` if ``bbox`` is degenerate (see :func:`ensure_valid_bbox`).
        - ``ValueError`` if neither or both of ``generation`` / ``vintage`` are
          given (a tile has exactly one temporal axis).

    Note:
        WP1 does not enforce which product maps to which axis (e.g. AHN ->
        generation); that policy belongs to the fetch context and is flagged,
        not silently added, here.

    """

    tile_id: str
    product: Product
    bbox: BBox
    generation: Generation | None = None
    vintage: Vintage | None = None

    def __post_init__(self) -> None:
        """Validate identity, extent, and the exactly-one temporal axis rule."""
        if not self.tile_id.strip():
            msg = "tile_id must be a non-blank identifier."
            raise ValueError(msg)
        ensure_valid_bbox(self.bbox)
        has_generation = self.generation is not None
        has_vintage = self.vintage is not None
        if has_generation == has_vintage:
            msg = (
                "A Tile must be pinned to exactly one temporal axis: provide "
                "either generation or vintage, not neither and not both."
            )
            raise ValueError(msg)
