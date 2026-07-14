"""Deterministic tileset.json emission (OGC 3D Tiles 1.1).

Builds the tileset document from the quadtree's per-tile regions,
geometric errors and content uris: ``asset.version "1.1"``, ``refine:
"REPLACE"`` declared on the root and inherited below, ``region``
bounding volumes in EPSG:4979 radians. Serialisation is sorted-key,
2-space-indented UTF-8 with a trailing newline — byte-deterministic.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from ahn_cli.tiles3d.mesh import Region

__all__ = [
    "render_tileset",
    "tile_entry",
    "tileset_document",
    "union_region",
    "write_tileset",
]

_TILES3D_VERSION = "1.1"
_GENERATOR = "ahn_cli tiles3d"
_REFINE = "REPLACE"


def union_region(a: Region, b: Region) -> Region:
    """Return the smallest region containing both ``a`` and ``b``."""
    return (
        min(a[0], b[0]),
        min(a[1], b[1]),
        max(a[2], b[2]),
        max(a[3], b[3]),
        min(a[4], b[4]),
        max(a[5], b[5]),
    )


def tile_entry(
    region: Region,
    geometric_error: float,
    uri: str,
    children: list[dict[str, object]],
    *,
    root: bool,
) -> dict[str, object]:
    """Build one tile's tileset.json entry.

    ``refine`` is declared only on the root (children inherit it, per
    the spec); ``children`` is omitted when empty.
    """
    entry: dict[str, object] = {
        "boundingVolume": {"region": list(region)},
        "content": {"uri": uri},
        "geometricError": geometric_error,
    }
    if root:
        entry["refine"] = _REFINE
    if children:
        entry["children"] = children
    return entry


def tileset_document(
    root_entry: dict[str, object], tileset_geometric_error: float
) -> dict[str, object]:
    """Wrap the root entry in the 3D Tiles 1.1 tileset document."""
    return {
        "asset": {
            "generator": _GENERATOR,
            "version": _TILES3D_VERSION,
        },
        "geometricError": tileset_geometric_error,
        "root": root_entry,
    }


def render_tileset(document: dict[str, object]) -> str:
    """Serialise the tileset document deterministically."""
    return json.dumps(document, sort_keys=True, indent=2) + "\n"


def write_tileset(document: dict[str, object], path: Path) -> None:
    r"""Write the tileset document deterministically, LF on every platform.

    Written via :meth:`~pathlib.Path.write_bytes` (not ``write_text``) so the
    ``\\n`` separators :func:`render_tileset` emits are never translated to
    ``\\r\\n`` by the platform's text mode — the sidecar the manifest hashes
    must be byte-identical across OSes.
    """
    path.write_bytes(render_tileset(document).encode("utf-8"))
