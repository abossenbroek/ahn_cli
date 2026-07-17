"""Byte-identity harness shared by every pipeline workstream's tests.

The load-bearing correctness gate for the pipeline epic is that a tiled,
fused run is **byte-identical** to running the standalone verbs in sequence.
This helper is the shared machinery every later stage test reuses to assert
that: deterministic serialization/hashing of an in-RAM :data:`TilePayload`
(:func:`hash_payload`), streamed hashing of on-disk outputs (:func:`sha256_file`
/ :func:`hash_tree`), a stage-chain runner (:func:`run_stages`), tiny synthetic
AOI builders (in-RAM payloads and on-disk PDRF-6 LAZ / GeoTIFF), and an
:class:`IdentityStage` reference. It is entirely network-free.

A later workstream diffs its pipeline output against a standalone reference by
comparing :func:`hash_payload` (for in-RAM handoffs) or :func:`hash_tree` (for
written deliverables); the harness self-test proves the mechanism on an identity
transform, where the two sides are equal by construction.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import laspy
import numpy as np
import rasterio
from rasterio.transform import from_bounds

from ahn_cli.pipeline.model import (
    EncodedBlob,
    EncodedTile,
    GridTile,
    PointTile,
    TileContext,
    TileKey,
    TilePayload,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import numpy.typing as npt

    from ahn_cli.domain import BBox
    from ahn_cli.pipeline.model import Stage

_HASH_CHUNK = 1 << 20
"""Streamed-hash read block (1 MiB), so hashing stays bounded-memory."""

_RECORD_SEP = b"\x1e"
"""ASCII record separator delimiting a payload's serialized planes/blobs."""


def sha256_bytes(data: bytes) -> str:
    """Return the SHA-256 hex digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Return a file's SHA-256 hex, hashed in bounded-memory chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_HASH_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_tree(root: Path) -> dict[str, str]:
    """Map every file under ``root`` to its SHA-256, keyed by relative POSIX path.

    Deterministic (sorted keys) so two deliverable trees compare by value --
    the shape a standalone-vs-pipeline diff over a verb's output directory needs.
    """
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _plane_bytes(plane: npt.NDArray[np.generic]) -> bytes:
    """Serialize one numpy plane with its dtype and shape tag, deterministically."""
    tag = f"{plane.dtype.str}:{plane.shape}".encode()
    return tag + b"|" + np.ascontiguousarray(plane).tobytes()


def _serialize_point(tile: PointTile) -> bytes:
    """Serialize a :class:`PointTile` to deterministic bytes."""
    parts = [
        b"PointTile",
        _plane_bytes(tile.x),
        _plane_bytes(tile.y),
        _plane_bytes(tile.z),
        _plane_bytes(tile.gps_time),
        _plane_bytes(tile.classification),
    ]
    if tile.rgb is not None:
        parts.append(_plane_bytes(tile.rgb))
    return _RECORD_SEP.join(parts)


def _serialize_grid(tile: GridTile) -> bytes:
    """Serialize a :class:`GridTile` to deterministic bytes."""
    return _RECORD_SEP.join(
        (
            b"GridTile",
            _plane_bytes(tile.heights),
            _plane_bytes(tile.red),
            _plane_bytes(tile.green),
            _plane_bytes(tile.blue),
        )
    )


def _serialize_encoded(tile: EncodedTile) -> bytes:
    """Serialize an :class:`EncodedTile` to deterministic bytes."""
    key = tile.key
    parts = [
        b"EncodedTile",
        f"{key.level},{key.tx},{key.ty},{key.tz}".encode(),
    ]
    parts.extend(blob.name.encode() + b"=" + blob.data for blob in tile.blobs)
    return _RECORD_SEP.join(parts)


def serialize_payload(payload: TilePayload) -> bytes:
    """Return the deterministic byte serialization of any ``payload``."""
    if isinstance(payload, PointTile):
        return _serialize_point(payload)
    if isinstance(payload, GridTile):
        return _serialize_grid(payload)
    return _serialize_encoded(payload)


def hash_payload(payload: TilePayload) -> str:
    """Return the SHA-256 hex of a payload's deterministic serialization."""
    return sha256_bytes(serialize_payload(payload))


def run_stages(
    payload: TilePayload, ctx: TileContext, stages: Sequence[Stage]
) -> TilePayload:
    """Fold ``stages`` over ``payload`` in order (the fused pipeline side)."""
    current = payload
    for stage in stages:
        current = stage.run(current, ctx)
    return current


@dataclass(frozen=True)
class IdentityStage:
    """A :class:`~ahn_cli.pipeline.model.Stage` that returns its tile unchanged."""

    halo: float = 0.0

    def halo_m(self) -> float:
        """Return this stage's configured source halo (default ``0``)."""
        return self.halo

    def run(
        self,
        tile: TilePayload,
        ctx: TileContext,  # noqa: ARG002 -- identity ignores context
    ) -> TilePayload:
        """Return ``tile`` unchanged (the byte-identity reference transform)."""
        return tile


def make_tile_key(
    *, level: int = 0, tx: int = 0, ty: int = 0, tz: int = 0
) -> TileKey:
    """Build a :class:`TileKey` (defaults to the root tile)."""
    return TileKey(level=level, tx=tx, ty=ty, tz=tz)


def make_tile_context(
    workdir: Path,
    *,
    key: TileKey | None = None,
    bbox: BBox = (0.0, 0.0, 10.0, 10.0),
    halo_m: float = 0.0,
) -> TileContext:
    """Build a :class:`TileContext` over ``workdir`` with a valid extent."""
    return TileContext(
        key=key if key is not None else make_tile_key(),
        bbox=bbox,
        halo_m=halo_m,
        workdir=workdir,
    )


def make_point_tile(
    *, count: int = 8, seed: int = 0, with_rgb: bool = False
) -> PointTile:
    """Build a deterministic synthetic :class:`PointTile` of ``count`` points."""
    rng = np.random.default_rng(seed)
    x = np.ascontiguousarray(rng.uniform(0.0, 10.0, count))
    y = np.ascontiguousarray(rng.uniform(0.0, 10.0, count))
    z = np.ascontiguousarray(rng.uniform(-5.0, 5.0, count))
    gps_time = np.ascontiguousarray(rng.uniform(0.0, 1.0, count))
    classification = np.ascontiguousarray(
        rng.integers(0, 7, count).astype(np.uint8)
    )
    rgb = None
    if with_rgb:
        rgb = np.ascontiguousarray(
            rng.integers(0, 65536, (count, 3)).astype(np.uint16)
        )
    return PointTile(
        x=x,
        y=y,
        z=z,
        gps_time=gps_time,
        classification=classification,
        rgb=rgb,
    )


def make_grid_tile(
    *, height: int = 4, width: int = 4, seed: int = 0
) -> GridTile:
    """Build a deterministic synthetic ``(height, width)`` :class:`GridTile`."""
    rng = np.random.default_rng(seed)
    heights = np.ascontiguousarray(
        rng.uniform(-5.0, 5.0, (height, width)).astype(np.float32)
    )
    red = np.ascontiguousarray(
        rng.integers(0, 256, (height, width)).astype(np.uint8)
    )
    green = np.ascontiguousarray(
        rng.integers(0, 256, (height, width)).astype(np.uint8)
    )
    blue = np.ascontiguousarray(
        rng.integers(0, 256, (height, width)).astype(np.uint8)
    )
    return GridTile(heights=heights, red=red, green=green, blue=blue)


def make_encoded_tile(
    *,
    key: TileKey | None = None,
    blobs: tuple[EncodedBlob, ...] | None = None,
) -> EncodedTile:
    """Build an :class:`EncodedTile` with two named blobs by default."""
    return EncodedTile(
        key=key if key is not None else make_tile_key(),
        blobs=blobs
        if blobs is not None
        else (
            EncodedBlob(name="geometry", data=b"geometry-blob"),
            EncodedBlob(name="texture", data=b"texture-blob"),
        ),
    )


def write_synthetic_laz(
    path: Path, points: npt.NDArray[np.float64], *, scale: float = 0.01
) -> None:
    """Write a synthetic PDRF-6 LAZ (``points`` rows ``x, y, z, gps, class``)."""
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.offsets = np.array([0.0, 0.0, 0.0], dtype=float)
    header.scales = np.array([scale, scale, scale], dtype=float)
    las = laspy.LasData(header)
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]
    las.gps_time = points[:, 3]
    las.classification = points[:, 4].astype(np.uint8)
    las.write(str(path))


def write_synthetic_ortho(
    path: Path, rgb: npt.NDArray[np.uint8], bounds: BBox
) -> None:
    """Write a deterministic 3-band uint8 EPSG:28992 ortho (``rgb`` is CHW)."""
    _, height, width = rgb.shape
    minx, miny, maxx, maxy = bounds
    transform = from_bounds(minx, miny, maxx, maxy, width, height)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype="uint8",
        crs="EPSG:28992",
        transform=transform,
    ) as dst:
        dst.write(rgb)
