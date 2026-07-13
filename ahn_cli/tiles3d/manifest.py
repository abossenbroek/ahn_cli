r"""Deterministic ``manifest.json`` codec (integrity sidecar).

Both lossy profiles write a ``manifest.json`` alongside the pack: a
byte-oriented integrity witness over every loose file plus ``tiles.hfp``,
tying them to the pack's ``dataset_id``. The normative shape lives in
``docs/specs/2026-07-12-hfp-pack-format.md`` (*manifest.json
shape*); this module is the only place that knows it.

**Determinism.** The rendered bytes are UTF-8, ``sort_keys``, 2-space
indented, LF (``\\n``) on every platform, with a trailing newline —
:func:`json.dumps` emits LF-only separators, so an identical file/digest
set reproduces identical bytes on any OS. :func:`render_manifest` and
:func:`parse_manifest` are an exact inverse pair for a well-formed
document. Pure module, no I/O: the caller hashes the files and hands in
the precomputed digests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from ahn_cli.tiles3d.errors import Tiles3dError

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "ALGORITHM",
    "FileDigest",
    "Manifest",
    "parse_manifest",
    "render_manifest",
]

ALGORITHM = "sha256"
"""The one hash algorithm the manifest names; any other value is rejected."""

_HEX_DIGITS = 64
"""A SHA-256 hex digest is exactly 64 lowercase hex characters."""

_TOP_KEYS = frozenset({"algorithm", "dataset_id", "files"})
_FILE_KEYS = frozenset({"sha256", "size"})


@dataclass(frozen=True)
class FileDigest:
    """One manifest file record: its SHA-256 hex digest and byte size.

    Contract (fields):
        - ``sha256``: 64 lowercase hex characters.
        - ``size``: the file's length in bytes (``>= 0``).
    """

    sha256: str
    size: int


@dataclass(frozen=True)
class Manifest:
    """A parsed ``manifest.json`` document.

    Contract (fields):
        - ``dataset_id``: the pack's content root as 64 hex characters.
        - ``files``: ``{relative-path: FileDigest}`` for every hashed file.

    ``algorithm`` is fixed to :data:`ALGORITHM` and is not stored.
    """

    dataset_id: str
    files: Mapping[str, FileDigest]


def render_manifest(files: Mapping[str, FileDigest], dataset_id: str) -> str:
    """Render the manifest document to deterministic text.

    Contract:
        - Emits ``{"algorithm": "sha256", "dataset_id": ..., "files":
          {...}}`` with sorted keys, 2-space indent, LF newlines and a
          trailing newline. ``files`` maps each relative path to its
          ``{"sha256": ..., "size": ...}`` record.
        - Deterministic: identical text for identical inputs, containing
          only LF (never CR).
    """
    document = {
        "algorithm": ALGORITHM,
        "dataset_id": dataset_id,
        "files": {
            name: {"sha256": digest.sha256, "size": digest.size}
            for name, digest in files.items()
        },
    }
    return json.dumps(document, sort_keys=True, indent=2) + "\n"


def parse_manifest(text: str) -> Manifest:
    """Parse manifest text into a :class:`Manifest`, validating its shape.

    Contract:
        - Inverse of :func:`render_manifest` for a well-formed document.

    Failure modes (each a :class:`~ahn_cli.tiles3d.errors.Tiles3dError`):
        - not a JSON object, or its key set is not exactly
          ``{algorithm, dataset_id, files}``;
        - ``algorithm`` not :data:`ALGORITHM`;
        - ``dataset_id`` not a 64-char lowercase hex string;
        - ``files`` not an object, or any record whose key set is not
          exactly ``{sha256, size}``, whose ``sha256`` is not 64-char
          lowercase hex, or whose ``size`` is not a non-negative integer.
    """
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        msg = f"manifest is not valid JSON: {exc}"
        raise Tiles3dError(msg) from exc
    if (
        not isinstance(parsed, dict)
        or frozenset(cast("dict[str, Any]", parsed)) != _TOP_KEYS
    ):
        msg = "manifest must be a JSON object with keys {algorithm, dataset_id, files}."
        raise Tiles3dError(msg)
    document = cast("dict[str, Any]", parsed)
    if document["algorithm"] != ALGORITHM:
        msg = (
            f"manifest algorithm {document['algorithm']!r} is not "
            f"{ALGORITHM!r}."
        )
        raise Tiles3dError(msg)
    dataset_id = document["dataset_id"]
    _require_hex(dataset_id, "dataset_id")
    raw_files = document["files"]
    if not isinstance(raw_files, dict):
        msg = "manifest 'files' must be a JSON object."
        raise Tiles3dError(msg)
    files = {
        name: _parse_record(name, record)
        for name, record in cast("dict[str, Any]", raw_files).items()
    }
    return Manifest(dataset_id=str(dataset_id), files=files)


def _parse_record(name: str, record: object) -> FileDigest:
    """Validate and build one file record, or raise :class:`Tiles3dError`."""
    if (
        not isinstance(record, dict)
        or frozenset(cast("dict[str, Any]", record)) != _FILE_KEYS
    ):
        msg = f"manifest file {name!r} must have keys {{sha256, size}}."
        raise Tiles3dError(msg)
    fields = cast("dict[str, Any]", record)
    _require_hex(fields["sha256"], f"file {name!r} sha256")
    size = fields["size"]
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        msg = f"manifest file {name!r} size must be a non-negative integer."
        raise Tiles3dError(msg)
    return FileDigest(sha256=str(fields["sha256"]), size=size)


def _require_hex(value: object, field: str) -> None:
    """Reject anything but a 64-char lowercase-hex string for ``field``."""
    if (
        not isinstance(value, str)
        or len(value) != _HEX_DIGITS
        or any(character not in "0123456789abcdef" for character in value)
    ):
        msg = (
            f"manifest {field} must be {_HEX_DIGITS} lowercase hex "
            f"characters."
        )
        raise Tiles3dError(msg)
