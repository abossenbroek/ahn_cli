"""The tiles3d context's typed error."""

from __future__ import annotations

__all__ = ["Tiles3dError"]


class Tiles3dError(Exception):
    """A 3D Tiles export failed: bad input, mismatch, or failed verify.

    Contract:
        - Raised for every tiles3d failure mode: unreadable or
          non-genuine inputs, imperfect ortho/heights dimension match,
          missing data, unwritable outputs, and any post-write
          verification failure. The message states the offending file
          and the exact check that failed.

    Invariants:
        - The CLI adapter translates this (and only this) into a
          ``click.ClickException``; it never escapes as a bare
          ``Exception`` from the public build entry point.
    """
