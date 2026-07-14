"""The lossy-profile provenance sidecar: a deterministic settings record.

The strict profile is lossless and byte-frozen, so it writes no sidecar.
The three lossy profiles — game, heightfield and splat — are
version-pinned, so each records exactly *how* a tile was packed next to
the tileset: the quantization scheme, the JPEG settings (game and
heightfield), and the encoder library versions that fix the bytes. A
consumer (or a future audit) reads ``provenance.json`` to know which
lossy profile produced the deliverable and at what pinned settings.

Every field is sourced from the encoder-layer modules that own it — the
quantization bit depth from :mod:`ahn_cli.tiles3d.quantize`, the JPEG
constants and Pillow version from :mod:`ahn_cli.tiles3d.jpeg`, the
meshopt version from :mod:`ahn_cli.tiles3d.meshopt`, the ``.hf`` magic /
version / zstd settings from :mod:`ahn_cli.tiles3d.heightfield`, the
splat ``.ply``/zstd settings from :mod:`ahn_cli.tiles3d.splat` — never
re-derived here, so the record cannot drift from the code that produced
the bytes.

The rendering is a pure function of the profile and the pinned library
versions: sorted keys, two-space indent, a trailing newline, no
timestamps. The build writes these exact bytes and the verifier recomputes
them in-process and demands byte identity.
"""

from __future__ import annotations

import json
import platform

from ahn_cli.tiles3d.errors import Tiles3dError
from ahn_cli.tiles3d.heightfield import (
    MAGIC,
    MAX_AXIS_ERROR_M,
    MAX_LEVEL,
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
from ahn_cli.tiles3d.manifest import ALGORITHM
from ahn_cli.tiles3d.meshopt import meshoptimizer_version
from ahn_cli.tiles3d.pack import BLOB_ALIGNMENT, FORMAT_VERSION
from ahn_cli.tiles3d.pack import MAGIC as PACK_MAGIC
from ahn_cli.tiles3d.profile import Profile
from ahn_cli.tiles3d.quantize import UINT16_MAX
from ahn_cli.tiles3d.splat import OPACITY, SH_DC0
from ahn_cli.tiles3d.splat import ZSTD_LEVEL as SPLAT_ZSTD_LEVEL
from ahn_cli.tiles3d.splat import zstandard_version as splat_zstandard_version

__all__ = [
    "PROVENANCE_NAME",
    "game_provenance_document",
    "heightfield_provenance_document",
    "producer_platform",
    "producer_python",
    "render_game_provenance",
    "render_heightfield_provenance",
    "render_provenance",
    "render_splat_provenance",
    "splat_provenance_document",
]

PROVENANCE_NAME = "provenance.json"
"""Output-relative name of the lossy profiles' provenance sidecar."""


def producer_platform() -> str:
    """Return the producing machine's platform string (deterministic per host).

    Recorded in the ``producer`` block because the pinned Pillow /
    libjpeg-turbo build — and therefore the JPEG bytes the pack carries — is a
    property of the producing platform. The fixture-generation path pins this
    (like geodesy) so the committed cross-machine fixtures stay byte-stable.
    """
    return platform.platform()


def producer_python() -> str:
    """Return the producing interpreter's version (deterministic per host)."""
    return platform.python_version()


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

_SPLAT_GAUSSIAN_SCHEME = (
    "one isotropic gaussian per tile vertex; position copied as-is (no "
    "quantization); colour is the sampled ortho pixel as an SH degree-0 "
    "coefficient (no sRGB decode); scale is the tile's measured cell "
    "spacing, log-stored (isotropic, one value for every axis); opacity "
    "is a fixed logit(OPACITY) and rotation is the identity quaternion "
    "for every gaussian."
)
"""One stable prose note of the splat per-gaussian construction."""


def _jpeg_block() -> dict[str, object]:
    """Return the pinned JPEG settings block shared by lossy profiles."""
    return {
        "quality": JPEG_QUALITY,
        "subsampling": JPEG_SUBSAMPLING,
        "progressive": JPEG_PROGRESSIVE,
        "optimize": JPEG_OPTIMIZE,
        "pillow": pillow_version(),
    }


def _pack_block(dataset_id: str) -> dict[str, object]:
    """Return the ``AHNP`` pack block: container pins + content version.

    Every field is sourced from :mod:`ahn_cli.tiles3d.pack` (and the shared
    hash algorithm from :mod:`ahn_cli.tiles3d.manifest`) so the record cannot
    drift from the container the build actually wrote; ``dataset_id`` is the
    pack's content-derived Merkle root (64 lowercase hex characters).
    """
    return {
        "magic": PACK_MAGIC.decode("ascii"),
        "format_version": FORMAT_VERSION,
        "alignment": BLOB_ALIGNMENT,
        "hash_algorithm": ALGORITHM,
        "dataset_id": dataset_id,
    }


def _producer_block() -> dict[str, object]:
    """Return the producing host's platform / interpreter (no timestamps)."""
    return {
        "platform": producer_platform(),
        "python": producer_python(),
    }


def game_provenance_document(dataset_id: str) -> dict[str, object]:
    """Return the game profile's provenance document (pre-serialisation).

    Contract:
        - ``profile`` is ``"game"``; ``quantization`` records the position
          bit depth (from :data:`~ahn_cli.tiles3d.quantize.UINT16_MAX`), the
          normalized-uint16 UV scheme and the one-line quantization note;
          ``jpeg`` records the pinned JPEG constants plus the Pillow
          version; ``encoders`` records the meshopt version; ``pack`` records
          the ``AHNP`` container pins and ``dataset_id``; ``producer`` records
          the producing host.
        - Pure and deterministic given the pinned library versions and the
          host — no timestamps or environment reads beyond the version /
          producer helpers.
    """
    return {
        "profile": "game",
        "quantization": {
            "position_bits": UINT16_MAX.bit_length(),
            "uv": "normalized-uint16",
            "scheme": _QUANTIZATION_SCHEME,
        },
        "jpeg": _jpeg_block(),
        "encoders": {"meshoptimizer": meshoptimizer_version()},
        "pack": _pack_block(dataset_id),
        "producer": _producer_block(),
    }


def render_game_provenance(dataset_id: str) -> str:
    """Serialise the game provenance deterministically (sorted keys).

    Contract:
        - Returns sorted-key, two-space-indented JSON with a trailing
          newline; byte-identical for identical pinned versions, host and
          ``dataset_id``.
    """
    return (
        json.dumps(
            game_provenance_document(dataset_id), sort_keys=True, indent=2
        )
        + "\n"
    )


def heightfield_provenance_document(dataset_id: str) -> dict[str, object]:
    """Return the heightfield profile's provenance document.

    Contract:
        - ``profile`` is ``"heightfield"``; ``quantization`` records the
          12-bit height-axis depth, maximum level and absolute error cap
          (from :data:`~ahn_cli.tiles3d.heightfield.MAX_LEVEL` and
          :data:`~ahn_cli.tiles3d.heightfield.MAX_AXIS_ERROR_M`) plus the
          one-line height quantization note; ``jpeg`` records the pinned JPEG
          constants plus the Pillow version; ``chunk`` records the ``.hf``
          magic, version, the pinned zstd level and the ``zstandard``
          version that fix the payload bytes; ``pack`` records the ``AHNP``
          container pins and ``dataset_id``; ``producer`` records the host.
        - Pure and deterministic given the pinned library versions and host.
    """
    return {
        "profile": "heightfield",
        "quantization": {
            "height_bits": MAX_LEVEL.bit_length(),
            "max_level": MAX_LEVEL,
            "max_axis_error_m": MAX_AXIS_ERROR_M,
            "scheme": _HEIGHTFIELD_QUANTIZATION_SCHEME,
        },
        "jpeg": _jpeg_block(),
        "chunk": {
            "magic": MAGIC.decode("ascii"),
            "version": VERSION,
            "zstd_level": ZSTD_LEVEL,
            "zstandard": zstandard_version(),
        },
        "pack": _pack_block(dataset_id),
        "producer": _producer_block(),
    }


def render_heightfield_provenance(dataset_id: str) -> str:
    """Serialise the heightfield provenance deterministically (sorted keys).

    Contract:
        - Returns sorted-key, two-space-indented JSON with a trailing
          newline; byte-identical for identical pinned versions, host and
          ``dataset_id``.
    """
    return (
        json.dumps(
            heightfield_provenance_document(dataset_id),
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )


def splat_provenance_document(dataset_id: str) -> dict[str, object]:
    """Return the splat profile's provenance document (pre-serialisation).

    Contract:
        - ``profile`` is ``"splat"``; ``gaussian`` records the fixed
          opacity (:data:`~ahn_cli.tiles3d.splat.OPACITY`), the SH
          degree-0 normalization constant
          (:data:`~ahn_cli.tiles3d.splat.SH_DC0`) and the one-line
          per-gaussian construction note; ``ply`` records the pinned zstd
          level and the ``zstandard`` version that fix the blob bytes;
          ``pack`` records the ``AHNP`` container pins and ``dataset_id``;
          ``producer`` records the producing host.
        - Pure and deterministic given the pinned library version and the
          host — no timestamps or environment reads beyond the version /
          producer helpers.
    """
    return {
        "profile": "splat",
        "gaussian": {
            "opacity": OPACITY,
            "sh_dc0": SH_DC0,
            "scheme": _SPLAT_GAUSSIAN_SCHEME,
        },
        "ply": {
            "zstd_level": SPLAT_ZSTD_LEVEL,
            "zstandard": splat_zstandard_version(),
        },
        "pack": _pack_block(dataset_id),
        "producer": _producer_block(),
    }


def render_splat_provenance(dataset_id: str) -> str:
    """Serialise the splat provenance deterministically (sorted keys).

    Contract:
        - Returns sorted-key, two-space-indented JSON with a trailing
          newline; byte-identical for identical pinned versions, host and
          ``dataset_id``.
    """
    return (
        json.dumps(
            splat_provenance_document(dataset_id), sort_keys=True, indent=2
        )
        + "\n"
    )


def render_provenance(
    profile: Profile, *, dataset_id: str | None = None
) -> str | None:
    """Return ``profile``'s provenance JSON, or ``None`` if it writes none.

    Contract:
        - The lossy ``game``, ``heightfield`` and ``splat`` profiles return
          their deterministic sidecar text (embedding ``dataset_id``, which
          is required for them); the byte-frozen ``strict`` profile returns
          ``None`` (it writes no sidecar and ignores ``dataset_id``). The
          build writes exactly this and the verifier recomputes it for the
          byte-identity check.
    """
    if profile is Profile.GAME:
        return render_game_provenance(
            _require_dataset_id(dataset_id, profile)
        )
    if profile is Profile.HEIGHTFIELD:
        return render_heightfield_provenance(
            _require_dataset_id(dataset_id, profile)
        )
    if profile is Profile.SPLAT:
        return render_splat_provenance(
            _require_dataset_id(dataset_id, profile)
        )
    return None


def _require_dataset_id(dataset_id: str | None, profile: Profile) -> str:
    """Return ``dataset_id`` or raise: the lossy profiles need the pack root."""
    if dataset_id is None:
        msg = (
            f"the {profile.value} profile's provenance needs the pack "
            "dataset_id."
        )
        raise Tiles3dError(msg)
    return dataset_id
