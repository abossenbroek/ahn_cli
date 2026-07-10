"""Prep-context transform/export seam (WP2 stub).

RED stub: importable, but the behaviour is deliberately not implemented so the
WP2 tests fail at assertion time. The green implementation replaces the bodies.
"""

from dataclasses import dataclass
from pathlib import Path


class TransformNotWiredError(NotImplementedError):
    """No real prep transform is wired yet (WP10-WP13)."""


@dataclass(frozen=True)
class PrepRequest:
    """A validated intent to transform a fetched site (stub)."""

    data_dir: Path
    include_classes: tuple[int, ...] = ()
    exclude_classes: tuple[int, ...] = ()
    export_points: bool = False


def prepare(_request: PrepRequest) -> None:
    """RED stub; the real seam raises TransformNotWiredError."""
