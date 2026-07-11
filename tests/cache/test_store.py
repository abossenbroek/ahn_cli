"""Tests for :class:`ahn_cli.cache.ContentAddressedCache`.

The store is exercised end-to-end against ``tmp_path`` -- no real network is
used anywhere; the fetcher is an injected in-process closure.
"""

from pathlib import Path

import pytest

from ahn_cli.cache import (
    CacheKey,
    ChecksumMismatchError,
    ContentAddressedCache,
)
from ahn_cli.domain import Generation, Product

_CONTENT = b"the-artifact-bytes"


def _key() -> CacheKey:
    """Return a stable generation-pinned key for store tests."""
    return CacheKey(
        product=Product.AHN_POINT_CLOUD,
        tile_id="37FN2",
        generation=Generation(4),
    )


def _snapshot(root: Path) -> dict[str, bytes]:
    """Map every file under ``root`` to its bytes, for byte-equality checks."""
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_get_returns_none_on_miss(tmp_path: Path) -> None:
    """An empty cache reports a miss rather than raising."""
    cache = ContentAddressedCache(root=tmp_path)
    assert cache.get(_key()) is None


def test_miss_fetches_stores_then_hits(tmp_path: Path) -> None:
    """First access fetches and stores; a later get returns the same bytes."""
    cache = ContentAddressedCache(root=tmp_path)
    calls = 0

    def fetch() -> bytes:
        nonlocal calls
        calls += 1
        return _CONTENT

    fetched = cache.get_or_fetch(_key(), fetch)
    assert fetched == _CONTENT
    assert calls == 1
    assert cache.get(_key()) == _CONTENT


def test_hit_does_not_fetch_and_writes_nothing(tmp_path: Path) -> None:
    """A cached key re-fetches with zero fetch calls and zero new bytes."""
    cache = ContentAddressedCache(root=tmp_path)

    def populate() -> bytes:
        return _CONTENT

    cache.get_or_fetch(_key(), populate)
    before = _snapshot(tmp_path)

    calls = 0

    def fetch() -> bytes:
        nonlocal calls
        calls += 1
        return b"different-bytes"

    result = cache.get_or_fetch(_key(), fetch)
    assert result == _CONTENT
    assert calls == 0
    assert _snapshot(tmp_path) == before


def test_put_then_get_round_trips(tmp_path: Path) -> None:
    """An explicitly stored artifact reads back byte-identical."""
    cache = ContentAddressedCache(root=tmp_path)
    cache.put(_key(), _CONTENT)
    assert cache.get(_key()) == _CONTENT


def test_discard_evicts_the_key_so_the_next_fetch_refetches(
    tmp_path: Path,
) -> None:
    """A discarded key misses, and get_or_fetch downloads fresh bytes."""
    cache = ContentAddressedCache(root=tmp_path)
    cache.put(_key(), _CONTENT)

    cache.discard(_key())

    assert cache.get(_key()) is None
    calls = 0

    def fetch() -> bytes:
        nonlocal calls
        calls += 1
        return b"fresh-bytes"

    assert cache.get_or_fetch(_key(), fetch) == b"fresh-bytes"
    assert calls == 1


def test_discard_of_a_missing_key_is_a_no_op(tmp_path: Path) -> None:
    """Discarding a key that was never stored neither raises nor stores."""
    cache = ContentAddressedCache(root=tmp_path)

    cache.discard(_key())

    assert cache.get(_key()) is None


def test_discard_is_idempotent(tmp_path: Path) -> None:
    """Discarding the same key twice is as safe as discarding it once."""
    cache = ContentAddressedCache(root=tmp_path)
    cache.put(_key(), _CONTENT)

    cache.discard(_key())
    cache.discard(_key())

    assert cache.get(_key()) is None


def test_discard_leaves_a_shared_blob_readable_via_another_key(
    tmp_path: Path,
) -> None:
    """Discarding one key never corrupts another key sharing its content.

    Blobs are addressed by content hash and shared across keys, so discard
    removes only the key's index entry and leaves the blob in place.
    """
    cache = ContentAddressedCache(root=tmp_path)
    other = CacheKey(
        product=Product.AHN_POINT_CLOUD,
        tile_id="37FN1",
        generation=Generation(4),
    )
    cache.put(_key(), _CONTENT)
    cache.put(other, _CONTENT)

    cache.discard(_key())

    assert cache.get(_key()) is None
    assert cache.get(other) == _CONTENT


def test_tampered_content_fails_checksum_verification(tmp_path: Path) -> None:
    """A corrupted stored blob fails verification instead of returning bytes."""
    cache = ContentAddressedCache(root=tmp_path)
    cache.put(_key(), _CONTENT)
    assert cache.get(_key()) == _CONTENT

    stored = [
        path
        for path in tmp_path.rglob("*")
        if path.is_file() and path.read_bytes() == _CONTENT
    ]
    assert len(stored) == 1
    stored[0].write_bytes(b"tampered-bytes-of-different-length")

    with pytest.raises(ChecksumMismatchError):
        cache.get(_key())
