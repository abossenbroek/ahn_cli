"""The resumable per-tile manifest: crash-safe, two-phase tile commits.

Each finished output tile lives in its own directory under ``<out>/tiles/`` and
is committed by an **atomic** marker file (``_tile.json``, written via a temp
file plus :meth:`Path.replace`). The marker's presence -- with an input hash
matching the tile's current inputs and every recorded blob still on disk -- is
the single commit point: a tile is "done" iff its marker commits it. This makes
the executor's per-tile work resumable and crash-safe exactly like
``tiles3d/build.py``'s accept-marker swap, but at tile granularity:

* A kill **between** a tile's blob write and its marker leaves blobs with no
  marker -> the tile reads as not-done -> it is cleanly reprocessed (its dir is
  wiped and rewritten), never double-emitted.
* A **corrupt/truncated** marker, a **stale** input hash, or a **missing** blob
  all read as not-done -> safe rebuild, never silent data loss.

The durable resumable state is the set of markers; :meth:`TileStore.write_manifest`
compiles them into an aggregate, deterministically sorted ``manifest.json``
(itself written atomically) as the final, human- and integration-facing index.
The store holds no per-tile state in memory, so resuming a run with millions of
tiles reads one marker at a time.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from ahn_cli.pipeline.errors import PipelineError

if TYPE_CHECKING:
    from pathlib import Path

    from ahn_cli.pipeline.model import EncodedTile, TileKey

__all__ = ["ManifestEntry", "TileStore", "encoded_digest"]

_TILES_DIR = "tiles"
"""Sub-directory of the output root holding the per-tile directories."""

_MARKER_NAME = "_tile.json"
"""The atomic per-tile commit marker filename."""

_MANIFEST_NAME = "manifest.json"
"""The aggregate index filename at the output root."""

_TMP_SUFFIX = ".tmp"
"""Suffix of the transient file an atomic write renames from."""

_LEN_FIELD = 8
"""Bytes of the big-endian length prefix framing each hashed field."""


def encoded_digest(encoded: EncodedTile) -> str:
    """Return a deterministic SHA-256 over a tile's blob names and bytes.

    Contract:
        - Depends only on the ordered ``(name, data)`` blob pairs, each field
          fed in length-prefixed so no two distinct tiles collide by
          concatenation.
    """
    digest = hashlib.sha256()
    for blob in encoded.blobs:
        for field in (blob.name.encode("utf-8"), blob.data):
            digest.update(len(field).to_bytes(_LEN_FIELD, "big"))
            digest.update(field)
    return digest.hexdigest()


@dataclass(frozen=True)
class ManifestEntry:
    """One committed tile's record: identity, input/output hashes, blob names.

    Contract:
        - ``key`` is the tile's :class:`~ahn_cli.pipeline.model.TileKey`.
        - ``input_hash`` is the content hash of the tile's inputs (the freshness
          key); ``output_hash`` is :func:`encoded_digest` of its blobs.
        - ``blobs`` is the ordered tuple of blob filenames written for the tile.

    Invariants:
        - Frozen value object, equal by field value.
    """

    key: TileKey
    input_hash: str
    output_hash: str
    blobs: tuple[str, ...]


def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically via a temp file and replace."""
    tmp = path.with_name(path.name + _TMP_SUFFIX)
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _validate_blob_name(name: str) -> None:
    """Reject a blob name that could escape its tile directory."""
    if "/" in name or "\\" in name or name in {".", ".."}:
        msg = f"unsafe blob name {name!r}: must not contain a path separator."
        raise PipelineError(msg)


class TileStore:
    """The on-disk home of a run's per-tile blobs, markers and index.

    Contract:
        - Rooted at ``out_dir``; tiles live under ``out_dir / "tiles"`` keyed by
          ``<level>/<tx>_<ty>_<tz>``.
        - :meth:`write_blobs` then :meth:`commit` is the two-phase write; a fault
          between them leaves an uncommitted tile that :meth:`is_done` rejects.
        - Holds no per-tile state in memory: every query reads the marker on
          disk, so a resume of a huge run stays bounded-memory.
    """

    def __init__(self, out_dir: Path) -> None:
        """Bind the store to ``out_dir`` (created lazily on first write)."""
        self._out = out_dir

    def tile_dir(self, key: TileKey) -> Path:
        """Return the directory holding ``key``'s blobs and marker."""
        return (
            self._out
            / _TILES_DIR
            / str(key.level)
            / f"{key.tx}_{key.ty}_{key.tz}"
        )

    def marker_path(self, key: TileKey) -> Path:
        """Return ``key``'s atomic commit-marker path."""
        return self.tile_dir(key) / _MARKER_NAME

    def is_done(self, key: TileKey, input_hash: str) -> bool:
        """Return whether ``key`` is committed for the given ``input_hash``.

        A tile is done iff its marker parses, records the same ``input_hash``,
        and every blob it names is still present. Any other state (no marker,
        corrupt marker, stale hash, missing blob) reads as not-done so the
        executor safely reprocesses it.
        """
        entry = self.load_entry(key)
        if entry is None or entry.input_hash != input_hash:
            return False
        tile_dir = self.tile_dir(key)
        return all((tile_dir / name).is_file() for name in entry.blobs)

    def load_entry(self, key: TileKey) -> ManifestEntry | None:
        """Return ``key``'s committed entry, or ``None`` if not committed.

        Returns ``None`` on a missing, unreadable, corrupt, or incomplete
        marker rather than raising -- a damaged marker is a reprocess signal,
        not a fatal error.
        """
        marker = self.marker_path(key)
        try:
            raw = marker.read_text(encoding="utf-8")
        except OSError:
            return None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return self._entry_from(key, data)

    @staticmethod
    def _entry_from(key: TileKey, data: object) -> ManifestEntry | None:
        """Build an entry from parsed marker ``data``, or ``None`` if malformed."""
        if not isinstance(data, dict):
            return None
        mapping = cast("dict[str, object]", data)
        input_hash = mapping.get("input_hash")
        output_hash = mapping.get("output_hash")
        blobs = mapping.get("blobs")
        if not isinstance(input_hash, str) or not isinstance(
            output_hash, str
        ):
            return None
        if not isinstance(blobs, list):
            return None
        blob_list = cast("list[object]", blobs)
        if not all(isinstance(name, str) for name in blob_list):
            return None
        return ManifestEntry(
            key=key,
            input_hash=input_hash,
            output_hash=output_hash,
            blobs=tuple(cast("list[str]", blob_list)),
        )

    def write_blobs(self, encoded: EncodedTile) -> None:
        """Write a tile's blobs into a freshly-cleaned tile directory (phase 1).

        Any prior partial attempt at this key is removed first, so a resume
        never mixes a killed run's leftovers with the fresh write. The commit
        marker is written separately by :meth:`commit`.
        """
        for blob in encoded.blobs:
            _validate_blob_name(blob.name)
        tile_dir = self.tile_dir(encoded.key)
        if tile_dir.exists():
            self._clear_dir(tile_dir)
        else:
            tile_dir.mkdir(parents=True)
        for blob in encoded.blobs:
            (tile_dir / blob.name).write_bytes(blob.data)

    @staticmethod
    def _clear_dir(tile_dir: Path) -> None:
        """Delete every file directly under ``tile_dir`` (a shallow tile dir)."""
        for child in tile_dir.iterdir():
            child.unlink()

    def commit(
        self, key: TileKey, input_hash: str, encoded: EncodedTile
    ) -> ManifestEntry:
        """Atomically commit ``key`` by writing its marker last (phase 2).

        The atomic :meth:`Path.replace` of the marker is the commit point: before it the
        tile reads as not-done, after it as done. Returns the recorded entry.
        """
        entry = ManifestEntry(
            key=key,
            input_hash=input_hash,
            output_hash=encoded_digest(encoded),
            blobs=tuple(blob.name for blob in encoded.blobs),
        )
        _atomic_write_text(
            self.marker_path(key),
            json.dumps(
                {
                    "input_hash": entry.input_hash,
                    "output_hash": entry.output_hash,
                    "blobs": list(entry.blobs),
                },
                sort_keys=True,
            ),
        )
        return entry

    def write_manifest(self, keys: list[TileKey]) -> Path:
        """Compile the committed markers for ``keys`` into ``manifest.json``.

        Contract:
            - Every key must be committed; the entries are sorted by
              ``(level, ty, tx, tz)`` and written atomically, so the index is
              deterministic regardless of the order ``keys`` arrives in.

        Failure modes:
            - :class:`PipelineError` if any key has no committed marker.
        """
        entries: list[ManifestEntry] = []
        for key in keys:
            entry = self.load_entry(key)
            if entry is None:
                msg = (
                    f"tile {key} is not committed; cannot build the manifest."
                )
                raise PipelineError(msg)
            entries.append(entry)
        entries.sort(
            key=lambda e: (e.key.level, e.key.ty, e.key.tx, e.key.tz)
        )
        document = {
            "tiles": [
                {
                    "level": e.key.level,
                    "tx": e.key.tx,
                    "ty": e.key.ty,
                    "tz": e.key.tz,
                    "input_hash": e.input_hash,
                    "output_hash": e.output_hash,
                    "blobs": list(e.blobs),
                }
                for e in entries
            ]
        }
        self._out.mkdir(parents=True, exist_ok=True)
        path = self._out / _MANIFEST_NAME
        _atomic_write_text(
            path, json.dumps(document, sort_keys=True, indent=2)
        )
        return path
