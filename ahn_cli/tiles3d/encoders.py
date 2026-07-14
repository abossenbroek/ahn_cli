"""The tile encoders behind the profile seam.

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

:class:`GameEncoder` is the compact runtime profile: it quantizes the
same RTC mesh (``KHR_mesh_quantization``), meshopt-compresses the three
streams (``EXT_meshopt_compression``) and drapes the tile with a baseline
JPEG — all still a pure, deterministic function of the payload, assembled
by :mod:`ahn_cli.tiles3d.gltf_quant`. Both glTF encoders embed their
texture, so their :class:`~ahn_cli.tiles3d.payload.EncodedTile` carries no
separate file.

:class:`HeightfieldEncoder` is the vendor Approach-C profile: it packs the
tile's quantized NAP height plane into a self-describing ``.hf`` chunk
(:mod:`ahn_cli.tiles3d.heightfield`) and drapes it with the *same* baseline
JPEG as the game profile, written **alongside** as a separate texture file
— the first encoder whose ``EncodedTile.texture`` is not ``None``.

:class:`SplatEncoder` encodes a tile as a 3D Gaussian Splatting cloud
(:mod:`ahn_cli.tiles3d.splat`): one isotropic gaussian per mesh vertex,
coloured by the sampled ortho pixel as an SH degree-0 coefficient — no
separate texture (colour lives in the gaussians themselves), so
``EncodedTile.texture`` is ``None`` like the two glTF encoders.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ahn_cli.tiles3d.gltf import build_glb
from ahn_cli.tiles3d.gltf_quant import build_game_glb
from ahn_cli.tiles3d.heightfield import encode_heightfield, nap_region
from ahn_cli.tiles3d.jpeg import encode_jpeg
from ahn_cli.tiles3d.meshopt import (
    encode_indices,
    encode_positions,
    encode_uvs,
)
from ahn_cli.tiles3d.payload import EncodedTile
from ahn_cli.tiles3d.png import encode_png
from ahn_cli.tiles3d.quantize import quantize_positions, quantize_uvs
from ahn_cli.tiles3d.splat import encode_splat

if TYPE_CHECKING:
    from ahn_cli.tiles3d.mesh import Region
    from ahn_cli.tiles3d.payload import TilePayload

__all__ = [
    "GameEncoder",
    "HeightfieldEncoder",
    "SplatEncoder",
    "StrictEncoder",
]


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

    def region_of(self, payload: TilePayload) -> Region:
        """Return the mesh's ellipsoidal region (strict is globe-correct)."""
        return payload.mesh.region


class HeightfieldEncoder:
    """Encode a tile as a ``.hf`` height chunk plus a sibling JPEG.

    Contract:
        - :meth:`encode` packs ``payload``'s quantized NAP height plane
          into a ``.hf`` chunk (:func:`ahn_cli.tiles3d.heightfield.encode_heightfield`),
          names it ``"<level>-<tx>-<ty>.hf"``, and returns the same sampled
          ortho as a baseline JPEG in ``texture`` named
          ``"<level>-<tx>-<ty>.jpg"`` — a separate file, not embedded.

    Invariants:
        - Deterministic: a pure function of the payload.
        - Satisfies the :class:`~ahn_cli.tiles3d.payload.TileEncoder`
          protocol; the only encoder with ``EncodedTile.texture`` set.
    """

    def encode(self, payload: TilePayload) -> EncodedTile:
        """Pack the height plane into a ``.hf`` and drape it with a JPEG."""
        base = f"{payload.level}-{payload.tx}-{payload.ty}"
        return EncodedTile(
            content=encode_heightfield(payload),
            content_name=f"{base}.hf",
            texture=encode_jpeg(payload.rgb),
            texture_name=f"{base}.jpg",
        )

    def region_of(self, payload: TilePayload) -> Region:
        """Return the tile's **NAP** region.

        Heights are self-consistent with the ``.hf`` plane, so the emitted
        tileset/pack regions are NAP too (v3, NAP-native; see
        :func:`ahn_cli.tiles3d.heightfield.nap_region`).
        """
        return nap_region(payload)


class GameEncoder:
    """Encode a tile as a quantized, meshopt-compressed glb with a JPEG.

    Contract:
        - :meth:`encode` quantizes ``payload.mesh`` positions and UVs
          (``KHR_mesh_quantization``), meshopt-compresses all three
          streams (``EXT_meshopt_compression``), JPEG-encodes
          ``payload.rgb`` and assembles them via
          :func:`ahn_cli.tiles3d.gltf_quant.build_game_glb`, named
          ``"<level>-<tx>-<ty>.glb"`` with the texture embedded
          (``EncodedTile.texture is None``).

    Invariants:
        - Deterministic: a pure function of the payload.
        - Satisfies the :class:`~ahn_cli.tiles3d.payload.TileEncoder`
          protocol.
    """

    def encode(self, payload: TilePayload) -> EncodedTile:
        """Quantize, compress and JPEG-drape the payload into a glb."""
        quantized = quantize_positions(payload.mesh.positions)
        uv_ints = quantize_uvs(payload.mesh.uvs)
        content = build_game_glb(
            quantized,
            encode_positions(quantized.ints),
            encode_uvs(uv_ints),
            encode_indices(payload.mesh.indices),
            encode_jpeg(payload.rgb),
            payload.mesh.center,
        )
        return EncodedTile(
            content=content,
            content_name=f"{payload.level}-{payload.tx}-{payload.ty}.glb",
        )

    def region_of(self, payload: TilePayload) -> Region:
        """Return the mesh's ellipsoidal region (game is globe-correct)."""
        return payload.mesh.region


class SplatEncoder:
    """Encode a tile as a zstd-wrapped binary 3DGS ``.ply`` gaussian cloud.

    Contract:
        - :meth:`encode` packs ``payload`` into a splat ``.ply`` blob
          (:func:`ahn_cli.tiles3d.splat.encode_splat`), named
          ``"<level>-<tx>-<ty>.ply"``. Colour lives in the gaussians'
          SH coefficients, so there is no separate texture
          (``EncodedTile.texture is None``).

    Invariants:
        - Deterministic: a pure function of the payload.
        - Satisfies the :class:`~ahn_cli.tiles3d.payload.TileEncoder`
          protocol.
    """

    def encode(self, payload: TilePayload) -> EncodedTile:
        """Pack the payload's gaussians into a zstd-wrapped ``.ply``."""
        content = encode_splat(payload)
        return EncodedTile(
            content=content,
            content_name=f"{payload.level}-{payload.tx}-{payload.ty}.ply",
        )

    def region_of(self, payload: TilePayload) -> Region:
        """Return the mesh's ellipsoidal region (splat is globe-correct)."""
        return payload.mesh.region
