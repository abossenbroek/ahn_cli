"""Shared progress-reporting callback type.

Used by every bounded context that streams work in reportable units (rows,
tiles, chunks, points) so a caller (typically the CLI) can drive a progress
bar without the callee owning any rendering concern.
"""

from __future__ import annotations

from collections.abc import Callable

ProgressCallback = Callable[[int, int], None]
"""An injected progress reporter: called ``(done, total)`` once per unit of
work (a row-block, tile, chunk, or phase), so a caller can drive a progress
bar without this module owning any rendering concern."""
