"""Tests for :class:`ahn_cli.cache.ContentAddressedCache`.

The store is exercised end-to-end against ``tmp_path`` -- no real network is
used anywhere; the fetcher is an injected in-process closure.
"""

import threading
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


def test_concurrent_put_of_the_same_key_never_corrupts_the_index(
    tmp_path: Path,
) -> None:
    """N threads racing to ``put()`` one key always leave a valid entry.

    Whichever writer's blob lands last, :meth:`get` must return one of the
    written payloads in full -- never a checksum failure or a crash from a
    torn read of two overlapping writes to the same index entry.
    """
    cache = ContentAddressedCache(root=tmp_path)
    key = _key()
    payloads = [f"payload-{i}-".encode() * 500 for i in range(16)]

    def worker(payload: bytes) -> None:
        cache.put(key, payload)

    threads = [threading.Thread(target=worker, args=(p,)) for p in payloads]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    result = cache.get(key)
    assert result in payloads


def test_crash_between_blob_and_index_write_never_leaves_a_corrupt_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash mid-index-write must never leave a dangling/corrupt entry.

    Establishes one valid entry, then simulates a hard-kill exactly while the
    *second* ``put()`` is writing the index (the blob write for the new
    content already completed): the destination file is opened, half its new
    content is written, then the write raises -- mimicking a process crash
    mid-write. Today's ``write_text`` writes directly to the real index path,
    so this leaves it truncated and pointing at no real blob (a bare
    ``FileNotFoundError`` from :meth:`get`, not a clean signal). Once ``put``
    writes through a temp file + ``os.replace``, the same fault lands on the
    temp file only: the real index entry is untouched, so ``get`` still
    resolves the *old* content -- this assertion fails on current
    ``store.py`` and must pass after the fix.
    """
    cache = ContentAddressedCache(root=tmp_path)
    key = _key()
    cache.put(key, _CONTENT)
    index_path = tmp_path / "index" / key.digest()
    before = index_path.read_text()

    def crashing_write_text(
        self: Path, data: str, *args: object, **kwargs: object
    ) -> int:
        del args, kwargs
        with self.open("w", encoding="utf-8") as handle:
            handle.write(data[: len(data) // 2])
        msg = "simulated crash mid index write"
        raise OSError(msg)

    monkeypatch.setattr(Path, "write_text", crashing_write_text)

    with pytest.raises(OSError, match="simulated crash"):
        cache.put(key, b"a completely different payload")

    assert index_path.read_text() == before
    assert cache.get(key) == _CONTENT
