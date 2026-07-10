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

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ahn_cli.domain import Generation, Product, Tile, Vintage

_KEY_FIELD_SEPARATOR = "\x00"
"""Field separator for the canonical encoding.

A NUL byte cannot occur in a portal tile id or in the controlled product /
axis codes, so joining on it yields an unambiguous, collision-free encoding.
"""


@dataclass(frozen=True)
class CacheKey:
    """The deterministic logical address of one cached tile artifact.

    Contract:
        - ``product`` is the dataset kind the artifact carries.
        - ``tile_id`` is the portal tile identifier; it must be non-blank.
        - Exactly one of ``generation`` / ``vintage`` pins the temporal axis,
          mirroring :class:`~ahn_cli.domain.Tile`.
        - Two keys are equal iff every field is equal; equal keys therefore
          produce an identical :meth:`digest`.

    Invariants:
        - Immutable and hashable, so a key is usable as a dict/set member.
        - :meth:`digest` is a pure function of the key fields: the same key
          hashes to the same digest in every process and release (no salt, no
          set/dict ordering), so it is a stable content-cache address.

    Failure modes:
        - ``ValueError`` if ``tile_id`` is blank.
        - ``ValueError`` if the temporal axis is not exactly one of
          ``generation`` / ``vintage`` (neither and both are rejected).
    """

    product: Product
    tile_id: str
    generation: Generation | None = None
    vintage: Vintage | None = None

    def __post_init__(self) -> None:
        """Reject a blank id or an ill-formed (neither/both) temporal axis."""
        if not self.tile_id.strip():
            msg = "tile_id must be a non-blank identifier."
            raise ValueError(msg)
        has_generation = self.generation is not None
        has_vintage = self.vintage is not None
        if has_generation == has_vintage:
            msg = (
                "A CacheKey must be pinned to exactly one temporal axis: "
                "provide either generation or vintage, not neither and not both."
            )
            raise ValueError(msg)

    @classmethod
    def from_tile(cls, tile: Tile) -> CacheKey:
        """Derive the cache key that addresses ``tile``.

        Contract:
            - Projects ``tile`` onto its ``(product, temporal-axis, tile_id)``
              identity, dropping the bbox (the tile id already pins the extent).
            - The result inherits :class:`~ahn_cli.domain.Tile`'s guarantee of
              exactly one temporal axis, so it always constructs successfully.
        """
        return cls(
            product=tile.product,
            tile_id=tile.tile_id,
            generation=tile.generation,
            vintage=tile.vintage,
        )

    def _canonical_bytes(self) -> bytes:
        """Return the canonical byte encoding hashed by :meth:`digest`.

        The temporal axis is emitted with an explicit ``gen:`` / ``vin:`` tag so
        a generation and a vintage that share a numeric value never collide.
        Exactly one axis is present (enforced in ``__post_init__``), so the two
        independent guards below append exactly one element.
        """
        axis_parts: list[str] = []
        if self.generation is not None:
            axis_parts.append(f"gen:{self.generation.code}")
        if self.vintage is not None:
            axis_parts.append(f"vin:{self.vintage.year}")
        fields = (self.product.value, axis_parts[0], self.tile_id)
        return _KEY_FIELD_SEPARATOR.join(fields).encode("utf-8")

    def digest(self) -> str:
        """Return the stable SHA-256 hex digest addressing this key.

        Contract:
            - Deterministic: equal keys yield the same digest across processes
              and releases; SHA-256 over the canonical encoding is unsalted and
              order-free.
            - The 64-character lowercase hex string is safe as a filesystem
              path component.
        """
        return hashlib.sha256(self._canonical_bytes()).hexdigest()
