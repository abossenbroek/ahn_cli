"""The content-addressed cache store: idempotent, checksum-verified fetch.

The store maps a :class:`~ahn_cli.cache.key.CacheKey` to an artifact addressed
by the SHA-256 of its content. Writes are split into two layers under the cache
root: an ``index/`` entry per key recording the content hash, and a ``blobs/``
entry per content hash holding the bytes. On read the stored bytes are re-hashed
and checked against the recorded hash, so a tampered or corrupt blob fails
verification instead of being returned silently.

Every write (blob or index entry) lands on a temp file in the destination's own
directory, then an atomic rename swaps it into place -- so a crash mid-write,
or a concurrent :meth:`ContentAddressedCache.put` racing another writer for
the same key, can only ever leave the destination as its previous content or
the new content in full, never a torn write in between.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from ahn_cli.cache.key import CacheKey

_INDEX_DIRNAME = "index"
_BLOBS_DIRNAME = "blobs"


def _tmp_path_for(path: Path) -> Path:
    """Return a unique temp path beside ``path``, on the same filesystem.

    Sharing the destination's directory keeps the later atomic rename on one
    filesystem (a cross-filesystem replace is not atomic); the random suffix
    means concurrent writers to the same destination never collide on the
    same temp file.
    """
    return path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` crash-safely: temp file + atomic rename.

    A reader can only ever observe ``path``'s previous content or ``data`` in
    full -- never a partial write, whether from a crash mid-write or another
    writer racing to the same destination.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _tmp_path_for(path)
    try:
        tmp_path.write_bytes(data)
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` crash-safely: temp file + atomic rename.

    Mirrors :func:`_atomic_write_bytes` for the index entry's text content.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = _tmp_path_for(path)
    try:
        tmp_path.write_text(text)
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)


class ChecksumMismatchError(Exception):
    """Raised when a stored blob's bytes do not match its recorded checksum.

    Signals cache corruption or tampering: the content addressed by a key no
    longer hashes to the content hash recorded for it, so the bytes must not be
    trusted or returned.
    """


@dataclass(frozen=True)
class ContentAddressedCache:
    """A content-addressed artifact cache rooted at a directory.

    Contract:
        - ``root`` is the cache directory; its ``index/`` and ``blobs/``
          subtrees are created on first write and need not pre-exist.
        - :meth:`put` stores bytes addressed by their content hash and records
          the key -> hash mapping.
        - :meth:`get` returns the verified bytes for a key, ``None`` on a miss,
          and raises :class:`ChecksumMismatchError` if the blob is corrupt.
        - :meth:`get_or_fetch` is idempotent: on a hit it neither calls the
          fetcher nor writes any bytes.
        - :meth:`discard` evicts a key so the next :meth:`get_or_fetch`
          re-fetches it; discarding a missing key is a no-op.

    Invariants:
        - Content addressing makes storage deterministic: identical content is
          always written to the same blob path.
    """

    root: Path

    def _index_path(self, key: CacheKey) -> Path:
        """Return the index-entry path recording ``key``'s content hash."""
        return self.root / _INDEX_DIRNAME / key.digest()

    def _blob_path(self, content_hash: str) -> Path:
        """Return the blob path addressing content by its SHA-256 hash."""
        return self.root / _BLOBS_DIRNAME / content_hash

    def put(self, key: CacheKey, content: bytes) -> str:
        """Store ``content`` under ``key`` and return its content hash.

        Contract:
            - Writes the blob addressed by ``sha256(content)`` and an index
              entry mapping ``key`` to that hash, creating parent directories
              as needed.
            - Returns the 64-character lowercase hex content hash.
            - Crash-safe: both the blob and the index entry are written via a
              temp file + ``os.replace()`` (see module docstring), so a crash
              or a concurrent :meth:`put` of the same key never leaves the
              index pointing at an absent or partial blob.
        """
        content_hash = hashlib.sha256(content).hexdigest()
        blob_path = self._blob_path(content_hash)
        _atomic_write_bytes(blob_path, content)
        index_path = self._index_path(key)
        _atomic_write_text(index_path, content_hash)
        return content_hash

    def get(self, key: CacheKey) -> bytes | None:
        """Return the verified cached bytes for ``key``, or ``None`` on a miss.

        Contract:
            - Returns ``None`` when no index entry exists for ``key``.
            - On a hit, re-hashes the stored blob and returns its bytes only if
              the hash matches the recorded content hash.

        Failure modes:
            - :class:`ChecksumMismatchError` if the stored blob's bytes do not
              hash to the recorded content hash (corruption or tampering).
        """
        index_path = self._index_path(key)
        if not index_path.exists():
            return None
        content_hash = index_path.read_text()
        content = self._blob_path(content_hash).read_bytes()
        actual_hash = hashlib.sha256(content).hexdigest()
        if actual_hash != content_hash:
            msg = (
                "cached blob failed checksum verification: expected "
                f"{content_hash}, got {actual_hash}."
            )
            raise ChecksumMismatchError(msg)
        return content

    def get_or_fetch(
        self, key: CacheKey, fetch: Callable[[], bytes]
    ) -> bytes:
        """Return cached bytes for ``key``, fetching and storing them on a miss.

        Contract:
            - On a hit: returns the verified cached bytes, calls ``fetch`` zero
              times, and writes zero bytes (the idempotence guarantee).
            - On a miss: calls ``fetch`` once, stores the result under ``key``,
              and returns it.

        Failure modes:
            - Propagates :class:`ChecksumMismatchError` from :meth:`get` if the
              existing cached blob is corrupt.
        """
        cached = self.get(key)
        if cached is not None:
            return cached
        content = fetch()
        self.put(key, content)
        return content

    def discard(self, key: CacheKey) -> None:
        """Evict ``key`` so the next :meth:`get_or_fetch` re-fetches it.

        Contract:
            - Removes the index entry mapping ``key`` to its content hash;
              a key with no entry is a no-op, so eviction is idempotent.
            - The blob itself is deliberately left in place: blobs are
              addressed by content hash and may be shared by other keys, so
              removing one here could corrupt an unrelated entry. An
              orphaned blob is unreachable through :meth:`get` (which
              requires an index entry), and a later :meth:`put` of the same
              content deterministically re-uses its path.
        """
        self._index_path(key).unlink(missing_ok=True)
