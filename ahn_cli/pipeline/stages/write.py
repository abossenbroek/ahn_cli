"""The cloud/``write`` sink: encode a tile's reconciled grid deterministically.

:class:`GridWriteSink` is the sink for a cloud/``write`` pipeline (as opposed to
the ``tiles3d`` sink): it turns a tile's
:class:`~ahn_cli.pipeline.model.GridTile` (the reconciled per-pixel heights plus
ortho colour) into an :class:`~ahn_cli.pipeline.model.EncodedTile` carrying one
blob named ``"grid"`` -- a small, self-describing binary record of the tile's
shape and its four planes (heights ``float32`` then the ``red``/``green``/
``blue`` ``uint8`` planes, all C-order little-endian). The executor persists
that blob per tile; a later assembly places each tile's grid at its pixel
offset to reconstruct the whole-area deliverable.

The encoding is deterministic and lossless, so a tiled ``reconcile`` run's
stitched grid is byte-identical to a whole-area standalone ``reconcile`` (the
halo floor guarantees identical edge kNN). :meth:`halo_m` is ``0``: the tile's
grid is already sampled, no source overlap needed.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ahn_cli.pipeline.errors import PipelineError
from ahn_cli.pipeline.model import EncodedBlob, EncodedTile, GridTile

if TYPE_CHECKING:
    import numpy.typing as npt

    from ahn_cli.pipeline.model import TileContext, TilePayload

__all__ = ["GRID_BLOB_NAME", "GridWriteSink", "decode_grid_blob"]

GRID_BLOB_NAME = "grid"
"""The single blob name a :class:`GridWriteSink` emits per tile."""

_MAGIC = b"AHNG"
"""Four-byte magic prefixing a grid blob."""

_HEADER = struct.Struct("<4sII")
"""Grid blob header: magic, height, width (little-endian)."""


def _encode_grid_blob(grid: GridTile) -> bytes:
    """Serialize a :class:`GridTile` to the self-describing grid-blob bytes."""
    height, width = grid.heights.shape
    parts = [
        _HEADER.pack(_MAGIC, height, width),
        np.ascontiguousarray(grid.heights, dtype=np.float32).tobytes(),
        np.ascontiguousarray(grid.red, dtype=np.uint8).tobytes(),
        np.ascontiguousarray(grid.green, dtype=np.uint8).tobytes(),
        np.ascontiguousarray(grid.blue, dtype=np.uint8).tobytes(),
    ]
    return b"".join(parts)


def decode_grid_blob(data: bytes) -> GridTile:
    """Reconstruct a :class:`GridTile` from grid-blob bytes.

    The inverse of :class:`GridWriteSink`'s encoding, used by an assembly step
    to place a tile's grid into the whole-area mosaic.

    Failure modes:
        - :class:`~ahn_cli.pipeline.errors.PipelineError` if the magic is wrong
          or the byte length does not match the declared shape.
    """
    magic, height, width = _HEADER.unpack_from(data)
    if magic != _MAGIC:
        msg = f"grid blob has a bad magic {magic!r}; expected {_MAGIC!r}."
        raise PipelineError(msg)
    cells = height * width
    heights_end = _HEADER.size + cells * 4
    expected = heights_end + cells * 3
    if len(data) != expected:
        msg = (
            f"grid blob length {len(data)} does not match the declared "
            f"{height}x{width} shape (expected {expected})."
        )
        raise PipelineError(msg)
    heights = np.frombuffer(
        data, dtype="<f4", count=cells, offset=_HEADER.size
    ).reshape(height, width)
    planes: list[npt.NDArray[np.uint8]] = []
    offset = heights_end
    for _ in range(3):
        plane = np.frombuffer(
            data, dtype=np.uint8, count=cells, offset=offset
        ).reshape(height, width)
        planes.append(plane)
        offset += cells
    red, green, blue = planes
    return GridTile(
        heights=np.ascontiguousarray(heights),
        red=np.ascontiguousarray(red),
        green=np.ascontiguousarray(green),
        blue=np.ascontiguousarray(blue),
    )


@dataclass(frozen=True)
class GridWriteSink:
    """Encode a tile's reconciled :class:`GridTile` into one ``"grid"`` blob.

    Contract:
        - :meth:`run` accepts only a
          :class:`~ahn_cli.pipeline.model.GridTile` and returns an
          :class:`~ahn_cli.pipeline.model.EncodedTile` with a single blob named
          :data:`GRID_BLOB_NAME`, a deterministic lossless serialization of the
          tile's four planes.
        - :meth:`halo_m` is always ``0``.

    Invariants:
        - Frozen value object; the encoding is a pure function of the grid, so
          two runs over the same tile produce byte-identical blobs.

    Failure modes:
        - :class:`~ahn_cli.pipeline.errors.PipelineError` if the tile is not a
          :class:`~ahn_cli.pipeline.model.GridTile`.
    """

    def halo_m(self) -> float:
        """Return ``0``: the tile's grid is already sampled."""
        return 0.0

    def run(self, tile: TilePayload, ctx: TileContext) -> TilePayload:
        """Encode ``tile`` into a single ``"grid"`` blob.

        Failure modes:
            - :class:`~ahn_cli.pipeline.errors.PipelineError` if ``tile`` is not
              a :class:`~ahn_cli.pipeline.model.GridTile`.
        """
        if not isinstance(tile, GridTile):
            msg = (
                f"tile {ctx.key} is not a GridTile; GridWriteSink got "
                f"{type(tile).__name__}. It must run after the reconcile stage."
            )
            raise PipelineError(msg)
        blob = EncodedBlob(name=GRID_BLOB_NAME, data=_encode_grid_blob(tile))
        return EncodedTile(key=ctx.key, blobs=(blob,))
