"""The encoder seam: per-tile payload, encoded result, encoder protocol.

:class:`TilePayload` is the sampling stage's output and every encoder's
input: it carries one tile's fully sampled data — the RTC mesh exactly
as :func:`ahn_cli.tiles3d.mesh.build_tile_mesh` computes it, plus the
raw sampled source planes and ortho pixels — together with its
placement metadata (stride, geometric error, quadtree coordinates), so
an encoder produces bytes without re-sampling or re-projecting.

:class:`TileEncoder` is the tiles3d context's single extension point: a
profile swaps the encoder to change the on-disk representation, and
nothing else in the pipeline (sampling, emission, the crash-safe swap)
knows how a tile is packed. :class:`EncodedTile` is what an encoder
returns — the content bytes and name, plus an optional separate texture
for encoders that do not embed it. The strict profile's encoder lives
in :mod:`ahn_cli.tiles3d.encoders`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from ahn_cli.tiles3d.mesh import TileMesh

__all__ = ["EncodedTile", "TileEncoder", "TilePayload"]


@dataclass(frozen=True, eq=False)
class TilePayload:
    """One tile's sampled data and placement, ready for any encoder.

    Contract (fields):
        - ``level``/``tx``/``ty``: the tile's quadtree coordinates.
        - ``stride``: the LOD sampling stride (``1`` for leaves).
        - ``geometric_error``: the tile's 3D Tiles geometric error, in
          metres (``0`` for leaves).
        - ``mesh``: the RTC float32 y-up mesh exactly as
          :func:`ahn_cli.tiles3d.mesh.build_tile_mesh` computes it —
          positions, texel-centre uvs, indices, float64 centre, the
          EPSG:4979 region, and the sampled col/row indices.
        - ``x``/``y``/``z``: the ``(rows, cols)`` float32 source planes
          sampled at ``stride`` (EPSG:28992 pixel centres and NAP
          height), carried so an encoder needs no geodesy of its own.
        - ``rgb``: the ``(rows, cols, 3)`` uint8 ortho pixels sampled at
          ``stride``.

    Invariants:
        - Every array is a genuine strided source sample (no averaging).
        - ``eq=False``: wraps large arrays, so instances compare by
          identity.
    """

    level: int
    tx: int
    ty: int
    stride: int
    geometric_error: float
    mesh: TileMesh
    x: npt.NDArray[np.float32]
    y: npt.NDArray[np.float32]
    z: npt.NDArray[np.float32]
    rgb: npt.NDArray[np.uint8]


@dataclass(frozen=True)
class EncodedTile:
    """One tile's encoded bytes, ready to write.

    Contract (fields):
        - ``content``: the tile's content-file bytes (e.g. the glb).
        - ``content_name``: the content file's output name, e.g.
          ``"0-0-0.glb"`` (written under the tileset's ``tiles/``).
        - ``texture``: separate texture-file bytes, or ``None`` when the
          encoder embeds the texture inside ``content`` (the strict
          path embeds its PNG, so this is ``None``).
        - ``texture_name``: the texture file's output name when
          ``texture`` is present, else ``None``.

    Invariants:
        - ``texture`` and ``texture_name`` are set together or both
          ``None``.
        - Frozen value object, equal by field value.
    """

    content: bytes
    content_name: str
    texture: bytes | None = None
    texture_name: str | None = None


class TileEncoder(Protocol):
    """Turns a sampled :class:`TilePayload` into an :class:`EncodedTile`.

    Contract:
        - :meth:`encode` is a pure, deterministic function of its
          payload: identical bytes for identical input, and no I/O.

    Invariants:
        - The tiles3d context's single extension point — a profile
          selects the encoder; the rest of the pipeline stays agnostic
          to the on-disk representation.
    """

    def encode(self, payload: TilePayload) -> EncodedTile:
        """Encode one payload into its content (and optional texture)."""
        ...
