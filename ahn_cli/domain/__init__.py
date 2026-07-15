"""The ``ahn_cli`` domain model: pure value objects, no I/O.

This package holds the ubiquitous-language types shared by both bounded
contexts (``fetch`` and ``prep``): :class:`Product`, :class:`Generation`,
:class:`Vintage`, :class:`Tile`, and :class:`Provenance`. It is intentionally
free of I/O, configuration, and any dependency on the legacy pre-7rad modules,
so the domain stays testable and reusable in isolation.
"""

from __future__ import annotations

from ahn_cli.domain.generation import Generation
from ahn_cli.domain.grid import GeoTransform, PixelGrid
from ahn_cli.domain.product import Product
from ahn_cli.domain.progress import ProgressCallback
from ahn_cli.domain.provenance import Provenance
from ahn_cli.domain.tile import BBox, Tile, ensure_valid_bbox
from ahn_cli.domain.vintage import Vintage

__all__ = [
    "BBox",
    "Generation",
    "GeoTransform",
    "PixelGrid",
    "Product",
    "ProgressCallback",
    "Provenance",
    "Tile",
    "Vintage",
    "ensure_valid_bbox",
]
