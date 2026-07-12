"""The lossy-profile provenance sidecar: a deterministic settings record.

The strict profile is lossless and byte-frozen, so it writes no sidecar.
The two lossy profiles — game and heightfield — are version-pinned, so
each records exactly *how* a tile was packed next to the tileset: the
quantization scheme, the JPEG settings, and the encoder library versions
that fix the bytes. A consumer (or a future audit) reads
``provenance.json`` to know which lossy profile produced the deliverable
and at what pinned settings.

Every field is sourced from the encoder-layer modules that own it — the
quantization bit depth from :mod:`ahn_cli.tiles3d.quantize`, the JPEG
constants and Pillow version from :mod:`ahn_cli.tiles3d.jpeg`, the
meshopt version from :mod:`ahn_cli.tiles3d.meshopt`, the ``.hf`` magic /
version / zstd settings from :mod:`ahn_cli.tiles3d.heightfield` — never
re-derived here, so the record cannot drift from the code that produced
the bytes.

The rendering is a pure function of the profile and the pinned library
versions: sorted keys, two-space indent, a trailing newline, no
timestamps. The build writes these exact bytes and the verifier recomputes
them in-process and demands byte identity.
"""

from __future__ import annotations

import json

from ahn_cli.tiles3d.heightfield import (
    MAGIC,
    VERSION,
    ZSTD_LEVEL,
    zstandard_version,
)
from ahn_cli.tiles3d.jpeg import (
    JPEG_OPTIMIZE,
    JPEG_PROGRESSIVE,
    JPEG_QUALITY,
    JPEG_SUBSAMPLING,
    pillow_version,
)
from ahn_cli.tiles3d.meshopt import meshoptimizer_version
from ahn_cli.tiles3d.profile import Profile
from ahn_cli.tiles3d.quantize import UINT16_MAX

__all__ = [
    "PROVENANCE_NAME",
    "game_provenance_document",
    "heightfield_provenance_document",
    "render_game_provenance",
    "render_heightfield_provenance",
    "render_provenance",
]

PROVENANCE_NAME = "provenance.json"
"""Output-relative name of the lossy profiles' provenance sidecar."""

_QUANTIZATION_SCHEME = (
    "per-tile per-axis affine over the tile's actual data extents; "
    "round-half-even; KHR_mesh_quantization node scale/translation "
    "dequantization; a zero-extent axis uses the epsilon scale."
)
"""One stable prose note of the position quantization contract."""

_HEIGHTFIELD_QUANTIZATION_SCHEME = (
    "per-tile single-axis affine over the NAP height range; "
    "round-half-even; dequantization z = level * z_scale + z_offset; "
    "a zero-extent (flat) axis uses the epsilon scale."
)
"""One stable prose note of the heightfield height-axis quantization."""


def _jpeg_block() -> dict[str, object]:
    """Return the pinned JPEG settings block shared by lossy profiles."""
    return {
        "quality": JPEG_QUALITY,
        "subsampling": JPEG_SUBSAMPLING,
        "progressive": JPEG_PROGRESSIVE,
        "optimize": JPEG_OPTIMIZE,
        "pillow": pillow_version(),
    }


def game_provenance_document() -> dict[str, object]:
    """Return the game profile's provenance document (pre-serialisation).

    Contract:
        - ``profile`` is ``"game"``; ``quantization`` records the position
          bit depth (from :data:`~ahn_cli.tiles3d.quantize.UINT16_MAX`), the
          normalized-uint16 UV scheme and the one-line quantization note;
          ``jpeg`` records the pinned JPEG constants plus the Pillow
          version; ``encoders`` records the meshopt version.
        - Pure and deterministic given the pinned library versions — no
          timestamps or environment reads beyond the version helpers.
    """
    return {
        "profile": "game",
        "quantization": {
            "position_bits": UINT16_MAX.bit_length(),
            "uv": "normalized-uint16",
            "scheme": _QUANTIZATION_SCHEME,
        },
        "jpeg": {
            "quality": JPEG_QUALITY,
            "subsampling": JPEG_SUBSAMPLING,
            "progressive": JPEG_PROGRESSIVE,
            "optimize": JPEG_OPTIMIZE,
            "pillow": pillow_version(),
        },
        "encoders": {"meshoptimizer": meshoptimizer_version()},
    }


def render_game_provenance() -> str:
    """Serialise the game provenance deterministically (sorted keys).

    Contract:
        - Returns sorted-key, two-space-indented JSON with a trailing
          newline; byte-identical for identical pinned versions.
    """
    return (
        json.dumps(game_provenance_document(), sort_keys=True, indent=2)
        + "\n"
    )


def heightfield_provenance_document() -> dict[str, object]:
    """Return the heightfield profile's provenance document.

    Contract:
        - ``profile`` is ``"heightfield"``; ``quantization`` records the
          height-axis bit depth (from
          :data:`~ahn_cli.tiles3d.quantize.UINT16_MAX`) and the one-line
          height quantization note; ``jpeg`` records the pinned JPEG
          constants plus the Pillow version; ``chunk`` records the ``.hf``
          magic, version, the pinned zstd level and the ``zstandard``
          version that fix the payload bytes.
        - Pure and deterministic given the pinned library versions.
    """
    return {
        "profile": "heightfield",
        "quantization": {
            "height_bits": UINT16_MAX.bit_length(),
            "scheme": _HEIGHTFIELD_QUANTIZATION_SCHEME,
        },
        "jpeg": _jpeg_block(),
        "chunk": {
            "magic": MAGIC.decode("ascii"),
            "version": VERSION,
            "zstd_level": ZSTD_LEVEL,
            "zstandard": zstandard_version(),
        },
    }


def render_heightfield_provenance() -> str:
    """Serialise the heightfield provenance deterministically (sorted keys).

    Contract:
        - Returns sorted-key, two-space-indented JSON with a trailing
          newline; byte-identical for identical pinned versions.
    """
    return (
        json.dumps(
            heightfield_provenance_document(), sort_keys=True, indent=2
        )
        + "\n"
    )


def render_provenance(profile: Profile) -> str | None:
    """Return ``profile``'s provenance JSON, or ``None`` if it writes none.

    Contract:
        - The lossy ``game`` and ``heightfield`` profiles return their
          deterministic sidecar text; the byte-frozen ``strict`` profile
          returns ``None`` (it writes no sidecar). The build writes exactly
          this and the verifier recomputes it for the byte-identity check.
    """
    renderers = {
        Profile.GAME: render_game_provenance,
        Profile.HEIGHTFIELD: render_heightfield_provenance,
    }
    renderer = renderers.get(profile)
    return renderer() if renderer is not None else None
