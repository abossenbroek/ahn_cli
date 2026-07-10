"""The :class:`Generation` value object: an AHN survey generation.

An AHN generation (AHN4, AHN5, ...) identifies a nationwide acquisition
programme. This module defines the *type only*: a minimal, immutable identity.
WP5 ("AHN Generation Selection") will attach a registry keyed by this value
object -- base URL, coverage-probe function, and semantics note -- so that
adding a future generation (e.g. AHN6) is a pure registry addition that touches
zero production call sites. Deliberately, no registry, URL, or probe lives here.
"""

from __future__ import annotations

from dataclasses import dataclass

_MIN_GENERATION_NUMBER = 1


@dataclass(frozen=True)
class Generation:
    """An AHN survey generation, identified by its ordinal number.

    Contract:
        - Construct with the generation ordinal: ``Generation(4)`` is AHN4.
        - ``number`` must be a positive integer; smaller values raise.

    Invariants:
        - Immutable and hashable, so it is usable as a registry key (the key
          WP5's generation registry will map to a base URL / probe / semantics).
        - Two generations are equal iff their ``number`` is equal.

    Failure modes:
        - ``ValueError`` if ``number`` is below ``1`` (there is no AHN0).

    Note:
        No upper bound is imposed: future generations are valid identities the
        moment they exist, so adding one needs no change to this type.

    """

    number: int

    def __post_init__(self) -> None:
        """Reject non-positive generation ordinals."""
        if self.number < _MIN_GENERATION_NUMBER:
            msg = (
                "Generation number must be a positive integer "
                f"(>= {_MIN_GENERATION_NUMBER}); got {self.number}."
            )
            raise ValueError(msg)

    @property
    def code(self) -> str:
        """Return the canonical family code for this generation, e.g. ``AHN4``."""
        return f"AHN{self.number}"
