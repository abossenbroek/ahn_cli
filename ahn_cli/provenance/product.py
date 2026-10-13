"""Provenance sidecar for a derived product (``<output>.provenance.json``).

The acquisition sidecar (:mod:`ahn_cli.provenance.sidecar`) records where raw
tiles came from. This module records what a *derived product* (a PLY export,
today) is made of: the input file's identity, the tool and library versions,
every filter applied, the datum offset, the class histograms and point counts
in and out, and the run's wall time. It exists so nothing that shapes a product
(a class filter, a thinning rule, a centring offset) is ever lost silently.

Determinism: everything outside the ``run`` block is a pure function of the
input bytes, the request and the tool environment, serialised with sorted keys
and a single trailing newline. Only ``run`` (UTC timestamp, wall seconds)
varies between rebuilds of a byte-identical product.
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Final, cast

import numpy as np

SCHEMA_VERSION: Final = "ahn_cli.product-provenance/1"
"""Value of the sidecar's ``schema`` field; bump on any shape change."""

SIDECAR_SUFFIX: Final = ".provenance.json"
"""Appended to the product's file name to name its sidecar."""

_HASH_BLOCK: Final = 1 << 22
_PROBE_TIMEOUT_S: Final = 20.0

_STR_OR_NONE = (str, type(None))
_LIST_OR_NONE = (list, type(None))

_SHAPE: Final[dict[str, object]] = {
    "schema": str,
    "tool": {
        "ahn_cli": str,
        "ahn_cli_commit": _STR_OR_NONE,
        "ahn_cli_dirty": (bool, type(None)),
        "python": str,
        "laspy": str,
        "lazrs": _STR_OR_NONE,
        "numpy": str,
        "pdal_python": _STR_OR_NONE,
        "pdal_cli": _STR_OR_NONE,
    },
    "input": {
        "path": str,
        "sha256": str,
        "bytes": int,
        "point_count": int,
        "point_format": int,
        "las_version": str,
        "is_copc": bool,
        "las_scale": list,
        "las_offset": list,
        "mins": list,
        "maxs": list,
    },
    "filters": {
        "class_include": _LIST_OR_NONE,
        "class_origin": str,
        "bbox": _LIST_OR_NONE,
        "thinning": dict,
        "upstream": (dict, type(None)),
    },
    "counts": {"source": int, "scanned": int, "selected": int, "out": int},
    "classes": {"scanned": list, "out": list},
    "output": {
        "name": str,
        "format": str,
        "sha256": str,
        "bytes": int,
        "coords": str,
        "offset_xyz": list,
        "offset_rule": str,
        "properties": list,
        "colour": (dict, type(None)),
    },
    "run": {"started_utc": str, "wall_seconds": float},
}
"""Required shape: nested dicts name keys, leaves are type(s) for isinstance."""


class ProductProvenanceError(ValueError):
    """Raised when a product sidecar is malformed or cannot be read.

    Contract:
        - Signals a missing/unknown key, a mistyped value, or invalid JSON.
        - Subclasses :class:`ValueError`, so callers may catch either.
    """


def sidecar_path(product: Path) -> Path:
    """Return the sidecar path for ``product`` (``<name>.provenance.json``)."""
    return product.with_name(product.name + SIDECAR_SUFFIX)


def fingerprint(path: Path) -> tuple[str, int]:
    """Return ``(sha256 hex, size in bytes)`` of ``path``, hashed in blocks."""
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_HASH_BLOCK), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def run_text(argv: list[str], cwd: Path | None = None) -> str | None:
    """Run a fixed argv and return stripped stdout, or ``None`` on any failure.

    Public so tests (and callers probing other tools) can patch or reuse it.
    """
    try:
        done = subprocess.run(  # noqa: S603 -- fixed argv, shell never used
            argv,
            check=True,
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_S,
            cwd=cwd,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip()


def _package_version(name: str) -> str | None:
    """Return an installed distribution's version, or ``None`` if absent."""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def collect_tool_versions() -> dict[str, object]:
    """Return the ``tool`` block: ahn_cli (+git commit), laspy, numpy, PDAL.

    Contract:
        - ``ahn_cli_commit``/``ahn_cli_dirty`` are ``None`` outside a git
          checkout; ``pdal_python``/``pdal_cli`` are ``None`` when PDAL is not
          installed. Nothing here is required to exist for a run to succeed.
    """
    here = Path(__file__).resolve().parent
    commit = run_text(["git", "rev-parse", "HEAD"], here)
    status = run_text(["git", "status", "--porcelain"], here)
    pdal_cli: str | None = None
    pdal_path = shutil.which("pdal")
    if pdal_path is not None:
        banner = run_text([pdal_path, "--version"]) or ""
        pdal_cli = next(
            (line.strip() for line in banner.splitlines() if "pdal" in line),
            None,
        )
    return {
        "ahn_cli": _package_version("ahn_cli") or "unknown",
        "ahn_cli_commit": commit or None,
        "ahn_cli_dirty": None if status is None else bool(status),
        "python": platform.python_version(),
        "laspy": _package_version("laspy") or "unknown",
        "lazrs": _package_version("lazrs"),
        "numpy": np.__version__,
        "pdal_python": _package_version("pdal"),
        "pdal_cli": pdal_cli,
    }


def validate_product_record(record: object) -> None:
    """Check ``record`` has exactly the sidecar shape and leaf types.

    Failure modes:
        - :class:`ProductProvenanceError` on a non-object, a missing or
          unknown key at any level, a mistyped leaf, or a wrong ``schema``.
    """
    _check(record, _SHAPE, "$")
    if cast("dict[str, object]", record)["schema"] != SCHEMA_VERSION:
        msg = f"unsupported sidecar schema; expected {SCHEMA_VERSION!r}."
        raise ProductProvenanceError(msg)


def _check(value: object, shape: object, where: str) -> None:
    """Recursively match ``value`` against ``shape`` (dicts name keys)."""
    if not isinstance(shape, dict):
        if not isinstance(value, cast("type | tuple[type, ...]", shape)):
            msg = f"{where} has the wrong type: {type(value).__name__}."
            raise ProductProvenanceError(msg)
        return
    if not isinstance(value, dict):
        msg = f"{where} must be a JSON object."
        raise ProductProvenanceError(msg)
    mapping = cast("dict[str, object]", value)
    expected = cast("dict[str, object]", shape)
    if set(mapping) != set(expected):
        missing = sorted(set(expected) - set(mapping))
        unknown = sorted(set(mapping) - set(expected))
        msg = f"{where} keys mismatch; missing={missing}, unknown={unknown}."
        raise ProductProvenanceError(msg)
    for key, sub in expected.items():
        _check(mapping[key], sub, f"{where}.{key}")


def product_record_to_bytes(record: dict[str, object]) -> bytes:
    """Validate and serialise ``record`` to deterministic sidecar bytes."""
    validate_product_record(record)
    text = json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True)
    return (text + "\n").encode("utf-8")


def write_product_sidecar(product: Path, record: dict[str, object]) -> Path:
    """Write ``record`` next to ``product`` and return the sidecar path."""
    target = sidecar_path(product)
    target.write_bytes(product_record_to_bytes(record))
    return target


def read_product_sidecar(path: Path) -> dict[str, object]:
    """Read and validate a product sidecar.

    Failure modes:
        - :class:`ProductProvenanceError` on invalid JSON or a shape error.
    """
    try:
        parsed: object = json.loads(path.read_bytes())
    except json.JSONDecodeError as exc:
        msg = "product sidecar is not valid JSON."
        raise ProductProvenanceError(msg) from exc
    validate_product_record(parsed)
    return cast("dict[str, object]", parsed)
