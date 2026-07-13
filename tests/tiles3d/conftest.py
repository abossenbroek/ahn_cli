"""Shared synthetic fixtures for the tiles3d tests.

The EXR fixtures are written with the *real* reconcile writer, so the
strict reader is tested against the exact bytes the pipeline produces;
``corrupt`` then performs byte surgery for the negative gates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import rasterio
from rasterio.transform import from_bounds

from ahn_cli.reconcile.writers import OutputFormat, write_reconciled
from ahn_cli.tiles3d.pack import PackEntry, TileKey, read_pack, write_pack
from ahn_cli.tiles3d.sources import TerrainGrid

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import numpy.typing as npt


def synth_grid(
    width: int, height: int, seed: int = 0
) -> npt.NDArray[np.float64]:
    """Build a deterministic ``(h, w, 6)`` X, Y, Z, R, G, B grid."""
    rng = np.random.default_rng(seed)
    grid = np.empty((height, width, 6), dtype=np.float64)
    cols = np.arange(width, dtype=np.float64) + 0.5
    rows = np.arange(height, dtype=np.float64) + 0.5
    grid[:, :, 0] = 100.0 + 0.5 * cols[np.newaxis, :]
    grid[:, :, 1] = 103.0 - 0.5 * rows[:, np.newaxis]
    grid[:, :, 2] = rng.uniform(-5.0, 40.0, (height, width))
    grid[:, :, 3:6] = rng.integers(0, 256, (height, width, 3)).astype(
        np.float64
    )
    return grid


def write_exr(path: Path, grid: npt.NDArray[np.float64]) -> Path:
    """Write ``grid`` as reconcile's EXR (all pixels valid)."""
    mask = np.ones(grid.shape[:2], dtype=np.bool_)
    write_reconciled(OutputFormat.EXR, grid, mask, path)
    return path


def corrupt(path: Path, offset: int, new_bytes: bytes) -> None:
    """Overwrite ``len(new_bytes)`` bytes of ``path`` at ``offset``."""
    data = bytearray(path.read_bytes())
    data[offset : offset + len(new_bytes)] = new_bytes
    path.write_bytes(bytes(data))


def pack_blob(hfp_path: Path, key: TileKey) -> tuple[bytes, bytes | None]:
    """Return one tile's ``(primary, texture)`` blobs from a packed build."""
    pack = read_pack(hfp_path)
    for index, entry in enumerate(pack.entries):
        if TileKey(entry.level, entry.tx, entry.ty, entry.tz) == key:
            return pack.primary_blob(index), pack.texture_blob(index)
    msg = f"tile {key} is not in the pack"
    raise AssertionError(msg)


def repack_one(
    hfp_path: Path,
    key: TileKey,
    corrupt_blobs: Callable[
        [bytes, bytes | None], tuple[bytes, bytes | None]
    ],
) -> None:
    """Rewrite ``tiles.hfp`` with one tile's blobs replaced.

    The replacement is packed through the real :func:`write_pack`, so the
    container stays integrity-valid (offsets, CRCs, hash section, dataset_id
    all recomputed): a per-tile *semantic* verifier — not the pack reader's
    checksum — is what a corruption test then exercises. The corruption
    callback receives the tile's pristine ``(primary, texture)`` and returns
    the replacement pair.
    """
    pack = read_pack(hfp_path)
    entries = [
        PackEntry(
            key=TileKey(entry.level, entry.tx, entry.ty, entry.tz),
            region=entry.region,
            geometric_error=entry.geometric_error,
        )
        for entry in pack.entries
    ]
    blobs: dict[TileKey, tuple[bytes, bytes | None]] = {}
    for index, entry in enumerate(pack.entries):
        entry_key = TileKey(entry.level, entry.tx, entry.ty, entry.tz)
        primary = pack.primary_blob(index)
        texture = pack.texture_blob(index)
        if entry_key == key:
            primary, texture = corrupt_blobs(primary, texture)
        blobs[entry_key] = (primary, texture)
    write_pack(
        hfp_path,
        entries,
        lambda k: blobs[k],
        root_geometric_error=pack.header.root_geometric_error,
        content_kind=pack.header.content_kind,
    )


def rewrite_pack(
    hfp_path: Path,
    mutate: Callable[[list[PackEntry]], list[PackEntry]],
    *,
    root_geometric_error: float | None = None,
) -> None:
    """Rewrite ``tiles.hfp`` with mutated index entries, blobs preserved.

    Reads the pack, hands the reconstructed :class:`PackEntry` list to
    ``mutate`` (which returns the replacement list — e.g. one entry's region
    or geometric error altered), and repacks through the real
    :func:`write_pack` so the container stays integrity-valid (CRCs, hash
    section, ``dataset_id`` recomputed). Lets a corruption test drive the
    verifier's *index*-level checks — the two-encodings witness and the
    chunk↔entry cross-check — rather than a container reject. ``dataset_id``
    naturally changes with the index; pass ``root_geometric_error`` to also
    override the header field.
    """
    pack = read_pack(hfp_path)
    entries = [
        PackEntry(
            key=TileKey(entry.level, entry.tx, entry.ty, entry.tz),
            region=entry.region,
            geometric_error=entry.geometric_error,
        )
        for entry in pack.entries
    ]
    blobs = {
        TileKey(entry.level, entry.tx, entry.ty, entry.tz): (
            pack.primary_blob(index),
            pack.texture_blob(index),
        )
        for index, entry in enumerate(pack.entries)
    }
    header_error = (
        pack.header.root_geometric_error
        if root_geometric_error is None
        else root_geometric_error
    )
    write_pack(
        hfp_path,
        mutate(entries),
        lambda key: blobs[key],
        root_geometric_error=header_error,
        content_kind=pack.header.content_kind,
    )


MINX = 100.0
MAXY = 103.0
RES = 0.5


def make_ortho(
    path: Path,
    rgb: npt.NDArray[np.uint8],
    *,
    crs: str = "EPSG:28992",
    dtype: str = "uint8",
) -> Path:
    """Write ``rgb`` ``(h, w, 3)`` as a GeoTIFF anchored at (100, 103)."""
    height, width = rgb.shape[:2]
    transform = from_bounds(
        MINX, MAXY - height * RES, MINX + width * RES, MAXY, width, height
    )
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=rgb.shape[2],
        dtype=dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        for band in range(rgb.shape[2]):
            dst.write(rgb[:, :, band].astype(dtype), band + 1)
    return path


def synth_rgb(
    width: int, height: int, seed: int = 2
) -> npt.NDArray[np.uint8]:
    """Build a deterministic non-uniform ``(h, w, 3)`` uint8 image."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (height, width, 3)).astype(np.uint8)


def make_terrain(width: int, height: int, seed: int = 4) -> TerrainGrid:
    """Build an in-memory TerrainGrid matching :func:`make_ortho`'s grid."""
    rgb = synth_rgb(width, height, seed)
    grid = grid_for_ortho(rgb)
    return TerrainGrid(
        width=width,
        height=height,
        transform=(RES, 0.0, MINX, 0.0, -RES, MAXY),
        x=grid[:, :, 0].astype(np.float32),
        y=grid[:, :, 1].astype(np.float32),
        z=grid[:, :, 2].astype(np.float32),
        rgb=rgb,
    )


def grid_for_ortho(
    rgb: npt.NDArray[np.uint8],
    z: npt.NDArray[np.float64] | None = None,
) -> npt.NDArray[np.float64]:
    """Build the reconciled grid matching :func:`make_ortho`'s transform."""
    height, width = rgb.shape[:2]
    if z is None:
        rng = np.random.default_rng(3)
        z = rng.uniform(-5.0, 40.0, (height, width))
    cols = (np.arange(width, dtype=np.float64) + 0.5)[np.newaxis, :]
    rows = (np.arange(height, dtype=np.float64) + 0.5)[:, np.newaxis]
    grid = np.empty((height, width, 6), dtype=np.float64)
    grid[:, :, 0] = RES * cols + 0.0 * rows + MINX
    grid[:, :, 1] = 0.0 * cols + -RES * rows + MAXY
    grid[:, :, 2] = z
    grid[:, :, 3:6] = rgb.astype(np.float64)
    return grid
