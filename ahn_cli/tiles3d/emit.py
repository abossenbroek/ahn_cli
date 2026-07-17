"""Pure tileset emission: verified terrain + plan -> bytes in memory.

:func:`compute_build` deterministically derives every artifact of a
3D Tiles build — one glb per quadtree tile plus the ``tileset.json``
bytes — without touching disk. ``build`` writes exactly these bytes;
``verify`` recomputes them from fresh source reads and demands the
on-disk files match byte for byte. Parents are emitted after their
children so every parent's region is the union of its own vertices and
all descendant regions (content containment by construction).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ahn_cli.domain.progress import (
    ProgressCallback,  # noqa: TC001 -- re-exported by build.py
)
from ahn_cli.tiles3d.geodesy import Geodesy
from ahn_cli.tiles3d.mesh import build_tile_mesh
from ahn_cli.tiles3d.pack import (
    CONTENT_KIND_GAME,
    CONTENT_KIND_HEIGHTFIELD,
    CONTENT_KIND_SPLAT,
    PackEntry,
    TileKey,
)
from ahn_cli.tiles3d.payload import TilePayload
from ahn_cli.tiles3d.profile import Profile
from ahn_cli.tiles3d.quadtree import geometric_error
from ahn_cli.tiles3d.tileset import (
    tile_entry,
    tileset_document,
    union_region,
)

if TYPE_CHECKING:
    from ahn_cli.tiles3d.mesh import Region
    from ahn_cli.tiles3d.payload import EncodedTile, TileEncoder
    from ahn_cli.tiles3d.quadtree import TilePlan, TreePlan
    from ahn_cli.tiles3d.sources import TerrainGrid

__all__ = [
    "MANIFEST_NAME",
    "TILES_HFP_NAME",
    "ComputedBuild",
    "PackedBuild",
    "compute_build",
    "compute_packed_build",
    "pixel_size",
    "texture_uri",
    "tile_uri",
]

TILES_SUBDIR = "tiles"
TILESET_NAME = "tileset.json"
TILES_HFP_NAME = "tiles.hfp"
MANIFEST_NAME = "manifest.json"
_LEAF_ERROR_FACTOR = 4.0

_CONTENT_KIND = {
    Profile.GAME: CONTENT_KIND_GAME,
    Profile.HEIGHTFIELD: CONTENT_KIND_HEIGHTFIELD,
    Profile.SPLAT: CONTENT_KIND_SPLAT,
}
"""The pack ``content_kind`` each lossy profile stamps into ``tiles.hfp``."""

BlobSource = Callable[[TileKey], "tuple[bytes, bytes | None]"]
"""``key -> (primary, texture)`` lazy per-tile encoder shared by every build."""


@dataclass(frozen=True, eq=False)
class ComputedBuild:
    """The plan for one strict build: document + a lazy per-tile blob source.

    Contract (fields):
        - ``document``: the tileset.json document (pre-serialisation).
        - ``order``: every tile's :class:`~ahn_cli.tiles3d.pack.TileKey` in
          children-first emit order.
        - ``uri_of``: each tile key's output-relative ``.glb`` uri.
        - ``blob_source``: ``key -> (content, texture)`` re-encoding one tile
          on demand (the strict profile embeds its texture, so ``texture`` is
          always ``None``); holds **no** encoded blobs, so a build streams the
          glbs to disk with at most a bounded window resident.
        - ``vertices`` / ``triangles``: totals across every tile.

    ``eq=False``: holds a live closure, so instances compare by identity.
    """

    document: dict[str, object]
    order: list[TileKey]
    uri_of: dict[TileKey, str]
    blob_source: BlobSource
    vertices: int
    triangles: int


def tile_uri(tile: TilePlan, profile: Profile) -> str:
    """Return the tile's output-relative content uri for ``profile``."""
    base = f"{tile.level}-{tile.tx}-{tile.ty}"
    return f"{TILES_SUBDIR}/{base}{profile.content_suffix()}"


def texture_uri(tile: TilePlan, profile: Profile) -> str | None:
    """Return the tile's sibling texture uri, or ``None`` when embedded."""
    suffix = profile.texture_suffix()
    if suffix is None:
        return None
    base = f"{tile.level}-{tile.tx}-{tile.ty}"
    return f"{TILES_SUBDIR}/{base}{suffix}"


def pixel_size(terrain: TerrainGrid) -> float:
    """Return the grid's ground pixel size in metres (the larger axis)."""
    return max(abs(terrain.transform[0]), abs(terrain.transform[4]))


def compute_build(
    terrain: TerrainGrid,
    tree: TreePlan,
    *,
    encoder: TileEncoder,
    progress: ProgressCallback | None = None,
) -> ComputedBuild:
    """Derive the tileset document and lazy per-tile blob plan, children first.

    ``encoder`` is the strict profile's :class:`TileEncoder`. Walks the
    quadtree to build the tileset document (the same regions/errors every path
    computes) and records each tile's key/uri, but holds **no** encoded glbs:
    the returned ``blob_source`` re-encodes a single tile on demand, so a build
    streams the glbs to disk with only a bounded window resident (identical
    bytes to encoding them inline).
    """
    emitter = _Emitter(terrain, tree, encoder, progress)
    root_entry, _, root_error = emitter.emit(tree.root)
    tileset_error = (
        2.0 * root_error
        if root_error > 0.0
        else pixel_size(terrain) * _LEAF_ERROR_FACTOR
    )
    geodesy = emitter.geodesy
    tiles = emitter.tiles

    def blob_source(key: TileKey) -> tuple[bytes, bytes | None]:
        encoded = _encode_tile(terrain, tiles[key], geodesy, encoder)
        return encoded.content, encoded.texture

    return ComputedBuild(
        document=tileset_document(root_entry, tileset_error),
        order=emitter.order,
        uri_of=emitter.uri_of,
        blob_source=blob_source,
        vertices=emitter.vertices,
        triangles=emitter.triangles,
    )


class _Emitter:
    """Children-first strict walk collecting the plan (no glbs held)."""

    def __init__(
        self,
        terrain: TerrainGrid,
        tree: TreePlan,
        encoder: TileEncoder,
        progress: ProgressCallback | None,
    ) -> None:
        self._terrain = terrain
        self._tree = tree
        self._progress = progress
        self.geodesy = Geodesy()
        self._encoder = encoder
        self._done = 0
        self.order: list[TileKey] = []
        self.uri_of: dict[TileKey, str] = {}
        self.tiles: dict[TileKey, TilePlan] = {}
        self.vertices = 0
        self.triangles = 0

    def emit(self, tile: TilePlan) -> tuple[dict[str, object], Region, float]:
        """Emit ``tile``'s subtree; return its entry, region and error."""
        child_entries: list[dict[str, object]] = []
        region: Region | None = None
        for child in tile.children:
            entry, child_region, _ = self.emit(child)
            child_entries.append(entry)
            region = (
                child_region
                if region is None
                else union_region(region, child_region)
            )
        mesh = build_tile_mesh(self._terrain, tile, self.geodesy)
        grid = np.ix_(mesh.rows, mesh.cols)
        error = geometric_error(tile.stride, pixel_size(self._terrain))
        # A transient payload (cheap array views, no blobs) so the encoder can
        # report the tile's own region; the glb bytes are produced lazily by
        # `blob_source`, so this holds nothing heavy.
        payload = TilePayload(
            level=tile.level,
            tx=tile.tx,
            ty=tile.ty,
            stride=tile.stride,
            geometric_error=error,
            mesh=mesh,
            z=self._terrain.z[grid],
            rgb=self._terrain.rgb[grid],
        )
        self.vertices += int(mesh.positions.shape[0])
        self.triangles += int(mesh.indices.shape[0]) // 3
        self._done += 1
        if self._progress is not None:
            self._progress(self._done, self._tree.tile_count)
        own_region = self._encoder.region_of(payload)
        region = (
            own_region if region is None else union_region(region, own_region)
        )
        key = TileKey(tile.level, tile.tx, tile.ty)
        # The strict encoder embeds its texture in the glb (no sibling file);
        # its content_name is `{level}-{tx}-{ty}.glb`, matching tile_uri.
        uri = tile_uri(tile, Profile.STRICT)
        self.order.append(key)
        self.uri_of[key] = uri
        self.tiles[key] = tile
        entry = tile_entry(
            region,
            error,
            uri,
            child_entries,
            root=tile.level == 0,
        )
        return entry, region, error


@dataclass(frozen=True, eq=False)
class PackedBuild:
    """Everything the lossy profiles need to write/verify one ``tiles.hfp``.

    Contract (fields):
        - ``document``: the ``tileset.json`` document (pre-serialisation);
          its top-level ``geometricError`` is ``root_geometric_error``.
        - ``entries``: one :class:`~ahn_cli.tiles3d.pack.PackEntry` per tile
          (key + enclosing EPSG:4979 region + geometric error), bit-equal to
          the tileset bounding volumes; the pack writer sorts them itself.
        - ``root_geometric_error``: the tileset document's top-level error.
        - ``content_kind``: the pack ``content_kind`` (heightfield ``0`` /
          game ``1`` / splat ``2``).
        - ``blob_source``: ``key -> (primary, texture)`` re-encoding one tile
          on demand, so the pack writer streams tiles without the whole set
          of encoded blobs ever being resident.
        - ``vertices`` / ``triangles``: totals across every tile.

    ``eq=False``: holds a live closure, so instances compare by identity.
    """

    document: dict[str, object]
    entries: list[PackEntry]
    root_geometric_error: float
    content_kind: int
    blob_source: BlobSource
    vertices: int
    triangles: int


def compute_packed_build(
    terrain: TerrainGrid,
    tree: TreePlan,
    *,
    profile: Profile,
    progress: ProgressCallback | None = None,
) -> PackedBuild:
    """Derive the tileset document and pack plan for a lossy ``profile``.

    Walks the quadtree children-first to build the tileset document and one
    :class:`~ahn_cli.tiles3d.pack.PackEntry` per tile (enclosing region +
    geometric error) — the same regions/errors the strict path computes — but
    holds **no** encoded blobs. The returned ``blob_source`` re-encodes a
    single tile on demand, so the pack writer streams the whole tree with at
    most one tile's blobs resident.
    """
    emitter = _PackedEmitter(terrain, tree, profile, progress)
    root_entry, _, root_error = emitter.emit(tree.root)
    tileset_error = (
        2.0 * root_error
        if root_error > 0.0
        else pixel_size(terrain) * _LEAF_ERROR_FACTOR
    )
    encoder = profile.encoder()
    geodesy = emitter.geodesy
    tiles = emitter.tiles

    def blob_source(key: TileKey) -> tuple[bytes, bytes | None]:
        encoded = _encode_tile(terrain, tiles[key], geodesy, encoder)
        return encoded.content, encoded.texture

    return PackedBuild(
        document=tileset_document(root_entry, tileset_error),
        entries=emitter.entries,
        root_geometric_error=tileset_error,
        content_kind=_CONTENT_KIND[profile],
        blob_source=blob_source,
        vertices=emitter.vertices,
        triangles=emitter.triangles,
    )


def _encode_tile(
    terrain: TerrainGrid,
    tile: TilePlan,
    geodesy: Geodesy,
    encoder: TileEncoder,
) -> EncodedTile:
    """Sample and encode one tile exactly as the strict walk would.

    Shared by the packed emitter's lazy ``blob_source`` so a tile's bytes are
    a pure, deterministic function of the tile and the terrain — identical to
    what the strict path produces for the same encoder.
    """
    mesh = build_tile_mesh(terrain, tile, geodesy)
    grid = np.ix_(mesh.rows, mesh.cols)
    error = geometric_error(tile.stride, pixel_size(terrain))
    payload = TilePayload(
        level=tile.level,
        tx=tile.tx,
        ty=tile.ty,
        stride=tile.stride,
        geometric_error=error,
        mesh=mesh,
        z=terrain.z[grid],
        rgb=terrain.rgb[grid],
    )
    return encoder.encode(payload)


class _PackedEmitter:
    """Children-first walk collecting pack entries (no blobs held)."""

    def __init__(
        self,
        terrain: TerrainGrid,
        tree: TreePlan,
        profile: Profile,
        progress: ProgressCallback | None,
    ) -> None:
        self._terrain = terrain
        self._tree = tree
        self._profile = profile
        self._encoder = profile.encoder()
        self._progress = progress
        self.geodesy = Geodesy()
        self._done = 0
        self.entries: list[PackEntry] = []
        self.tiles: dict[TileKey, TilePlan] = {}
        self.vertices = 0
        self.triangles = 0

    def emit(self, tile: TilePlan) -> tuple[dict[str, object], Region, float]:
        """Emit ``tile``'s subtree; return its entry, region and error."""
        child_entries: list[dict[str, object]] = []
        region: Region | None = None
        for child in tile.children:
            entry, child_region, _ = self.emit(child)
            child_entries.append(entry)
            region = (
                child_region
                if region is None
                else union_region(region, child_region)
            )
        mesh = build_tile_mesh(self._terrain, tile, self.geodesy)
        grid = np.ix_(mesh.rows, mesh.cols)
        error = geometric_error(tile.stride, pixel_size(self._terrain))
        # A transient payload (cheap array views, no blobs) so the encoder can
        # report the tile's own region in the profile's height datum — NAP for
        # heightfield, ellipsoidal for the rest. The encoded bytes are still
        # produced lazily by `blob_source`; this holds nothing heavy.
        payload = TilePayload(
            level=tile.level,
            tx=tile.tx,
            ty=tile.ty,
            stride=tile.stride,
            geometric_error=error,
            mesh=mesh,
            z=self._terrain.z[grid],
            rgb=self._terrain.rgb[grid],
        )
        self.vertices += int(mesh.positions.shape[0])
        self.triangles += int(mesh.indices.shape[0]) // 3
        self._done += 1
        if self._progress is not None:
            self._progress(self._done, self._tree.tile_count)
        own_region = self._encoder.region_of(payload)
        region = (
            own_region if region is None else union_region(region, own_region)
        )
        key = TileKey(tile.level, tile.tx, tile.ty)
        self.entries.append(
            PackEntry(key=key, region=region, geometric_error=error)
        )
        self.tiles[key] = tile
        uri = tile_uri(tile, self._profile)
        entry = tile_entry(
            region, error, uri, child_entries, root=tile.level == 0
        )
        return entry, region, error
