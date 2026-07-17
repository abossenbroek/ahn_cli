"""The pipeline context's typed error."""

from __future__ import annotations

__all__ = ["PipelineError"]


class PipelineError(Exception):
    """A tile-streaming pipeline run failed: bad spec, tiling, or stage.

    Contract:
        - Raised for every pipeline failure mode: an unparsable or
          ill-formed spec, a source/sink that is not first/last, a tiling
          that cannot cover the area of interest, a stage that rejects its
          input, a disk-floor breach, or a resumability/manifest
          inconsistency. The message states the offending element and the
          exact check that failed.

    Invariants:
        - The CLI adapter translates this (and only this) into a
          ``click.ClickException``; it never escapes as a bare
          ``Exception`` from the public pipeline entry point.
    """
