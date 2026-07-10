"""The content-addressed cache store (RED stub).

Real behavior arrives in the GREEN implementation; these stubs exist only so
the WP4 tests import cleanly and fail at assertion time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from ahn_cli.cache.key import CacheKey


class ChecksumMismatchError(Exception):
    """Raised when stored content does not match its recorded checksum."""


@dataclass(frozen=True)
class ContentAddressedCache:
    """A content-addressed artifact cache rooted at a directory (RED stub)."""

    root: Path

    def put(self, key: CacheKey, content: bytes) -> str:
        """RED stub: writes nothing and returns a placeholder digest."""
        del key, content
        return ""

    def get(self, key: CacheKey) -> bytes | None:
        """RED stub: returns a wrong sentinel so both miss and hit tests fail."""
        del key
        return b"__stub__"

    def get_or_fetch(
        self, key: CacheKey, fetch: Callable[[], bytes]
    ) -> bytes:
        """RED stub: always calls ``fetch`` and returns a placeholder."""
        del key
        fetch()
        return b""
