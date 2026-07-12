"""The strict-profile tile encoder: a float32 glb with an embedded PNG.

:class:`StrictEncoder` is the original, byte-frozen tiles3d
representation extracted behind the
:class:`~ahn_cli.tiles3d.payload.TileEncoder` seam. It packs a tile's
RTC float32 mesh and its sampled ortho pixels into one self-contained
glb — POSITION / TEXCOORD_0 / indices plus an embedded PNG texture — so
no separate texture file is written. All glTF/PNG knowledge lives here
(and in :mod:`ahn_cli.tiles3d.gltf` / :mod:`ahn_cli.tiles3d.png`): the
emitter drives this module's ``build_glb`` / ``encode_png`` globals, and
future profiles swap in their own encoder without the rest of the
pipeline changing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ahn_cli.tiles3d.gltf import build_glb
from ahn_cli.tiles3d.payload import EncodedTile
from ahn_cli.tiles3d.png import encode_png

if TYPE_CHECKING:
    from ahn_cli.tiles3d.payload import TilePayload

__all__ = ["StrictEncoder"]


class StrictEncoder:
    """Encode a tile as a float32 glb with an embedded PNG texture.

    Contract:
        - :meth:`encode` reproduces the pre-split strict output exactly:
          ``build_glb(payload.mesh, encode_png(payload.rgb))``, named
          ``"<level>-<tx>-<ty>.glb"``, with the texture embedded
          (``EncodedTile.texture is None``).

    Invariants:
        - Deterministic: a pure function of the payload.
        - Satisfies the :class:`~ahn_cli.tiles3d.payload.TileEncoder`
          protocol.
    """

    def encode(self, payload: TilePayload) -> EncodedTile:
        """Pack the payload's mesh and pixels into a self-contained glb."""
        content = build_glb(payload.mesh, encode_png(payload.rgb))
        return EncodedTile(
            content=content,
            content_name=f"{payload.level}-{payload.tx}-{payload.ty}.glb",
        )
