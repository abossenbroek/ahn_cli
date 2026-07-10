"""Content-addressed cache for fetched dataset artifacts (WP4).

This package makes ``fetch`` idempotent: an artifact is stored under a cache
directory addressed by its content hash, and looked up by a deterministic key
derived from the :mod:`ahn_cli.domain` value objects that identify a tile --
:class:`~ahn_cli.domain.Product`, exactly one of
:class:`~ahn_cli.domain.Generation` / :class:`~ahn_cli.domain.Vintage`, and the
tile id. Re-fetching an already-cached key performs zero network work and
writes zero new bytes; reads verify stored content against its checksum, so a
tampered artifact fails loudly instead of returning silently.
"""

from __future__ import annotations

from ahn_cli.cache.key import CacheKey
from ahn_cli.cache.store import ChecksumMismatchError, ContentAddressedCache

__all__ = [
    "CacheKey",
    "ChecksumMismatchError",
    "ContentAddressedCache",
]
