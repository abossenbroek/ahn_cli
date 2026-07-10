"""The :class:`Vintage` value object: the acquisition year of a dataset.

Orthophoto (Beeldmateriaal) products and AHN surveys are distributed as dated
*vintages*. A vintage must be pinned explicitly (never floated to the newest
"Actueel" layer), so the domain models it as an immutable, validated value
object rather than a bare integer.
"""

from __future__ import annotations

from dataclasses import dataclass

# Aerial national-survey data does not predate the 20th century; this is a
# permissive lower guard against obviously wrong years, not a data catalogue.
_MIN_ACQUISITION_YEAR = 1900


@dataclass(frozen=True)
class Vintage:
    """The acquisition year a dataset was captured.

    Contract:
        - Construct with the four-digit acquisition year: ``Vintage(2023)``.
        - ``year`` must be a plausible acquisition year (>= 1900).

    Invariants:
        - Immutable and hashable; two vintages are equal iff their ``year`` is
          equal.

    Failure modes:
        - ``ValueError`` if ``year`` is earlier than ``1900``.

    Note:
        No upper bound is imposed: a just-released vintage is valid, so future
        acquisition years are accepted without changing this type.

    """

    year: int

    def __post_init__(self) -> None:
        """Reject years that cannot be a real acquisition year."""
        if self.year < _MIN_ACQUISITION_YEAR:
            msg = (
                "Vintage year must be a plausible acquisition year "
                f"(>= {_MIN_ACQUISITION_YEAR}); got {self.year}."
            )
            raise ValueError(msg)
