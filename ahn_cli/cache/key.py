"""The :class:`CacheKey` value object: the logical address of a cached tile.

A cache key is the deterministic identity under which a fetched artifact is
stored and later found. It is derived from the domain value objects that
identify a distributable tile -- :class:`~ahn_cli.domain.Product`, exactly one
of :class:`~ahn_cli.domain.Generation` / :class:`~ahn_cli.domain.Vintage`, and
the portal tile id -- exactly the ``(product, vintage-or-generation, tile-id)``
triple the acquisition spec keys the cache on. The tile's extent (bbox) is
deliberately excluded: the portal's tile id already pins the extent, and the
spec keys on the id, not the box.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ahn_cli.domain import Generation, Product, Tile, Vintage

# NOTE (RED stub): validation and the real digest are intentionally absent so
# every WP4 test fails at assertion time. Replaced by the GREEN implementation.


@dataclass(frozen=True)
class CacheKey:
    """The deterministic logical address of one cached tile artifact (RED stub)."""

    product: Product
    tile_id: str
    generation: Generation | None = None
    vintage: Vintage | None = None

    @classmethod
    def from_tile(cls, tile: Tile) -> CacheKey:
        """RED stub: drops the temporal axis so preservation/digest tests fail."""
        return cls(product=tile.product, tile_id=tile.tile_id)

    def digest(self) -> str:
        """RED stub: constant placeholder so digest-value assertions fail."""
        return ""
