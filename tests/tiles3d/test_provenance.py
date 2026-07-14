"""Tests for the game-profile provenance sidecar."""

from __future__ import annotations

import json

import pytest

from ahn_cli.tiles3d import heightfield, jpeg, meshopt, pack, quantize, splat
from ahn_cli.tiles3d.errors import Tiles3dError
from ahn_cli.tiles3d.manifest import ALGORITHM
from ahn_cli.tiles3d.profile import Profile
from ahn_cli.tiles3d.provenance import (
    PROVENANCE_NAME,
    game_provenance_document,
    heightfield_provenance_document,
    producer_platform,
    producer_python,
    render_game_provenance,
    render_heightfield_provenance,
    render_provenance,
    render_splat_provenance,
    splat_provenance_document,
)

_DATASET_ID = "ab" * 32
"""A stand-in 64-char pack dataset_id for the pure-function tests."""


def test_document_records_profile_and_quantization() -> None:
    """The document names the game profile and the quantization scheme."""
    document = game_provenance_document(_DATASET_ID)
    assert document["profile"] == "game"
    quant = document["quantization"]
    assert isinstance(quant, dict)
    assert quant["position_bits"] == quantize.UINT16_MAX.bit_length()
    assert quant["position_bits"] == 16
    assert quant["uv"] == "normalized-uint16"
    assert isinstance(quant["scheme"], str)
    assert "KHR_mesh_quantization" in quant["scheme"]


def test_document_sources_jpeg_and_encoder_versions() -> None:
    """JPEG settings and encoder versions come from the owning modules."""
    document = game_provenance_document(_DATASET_ID)
    jpeg_block = document["jpeg"]
    assert isinstance(jpeg_block, dict)
    assert jpeg_block == {
        "quality": jpeg.JPEG_QUALITY,
        "subsampling": jpeg.JPEG_SUBSAMPLING,
        "progressive": jpeg.JPEG_PROGRESSIVE,
        "optimize": jpeg.JPEG_OPTIMIZE,
        "pillow": jpeg.pillow_version(),
    }
    assert document["encoders"] == {
        "meshoptimizer": meshopt.meshoptimizer_version()
    }


def test_document_records_pack_and_producer_blocks() -> None:
    """Both lossy documents carry the AHNP pack pins and the producer."""
    for document in (
        game_provenance_document(_DATASET_ID),
        heightfield_provenance_document(_DATASET_ID),
        splat_provenance_document(_DATASET_ID),
    ):
        assert document["pack"] == {
            "magic": pack.MAGIC.decode("ascii"),
            "format_version": pack.FORMAT_VERSION,
            "alignment": pack.BLOB_ALIGNMENT,
            "hash_algorithm": ALGORITHM,
            "dataset_id": _DATASET_ID,
        }
        assert document["producer"] == {
            "platform": producer_platform(),
            "python": producer_python(),
        }


def test_render_is_sorted_deterministic_json_with_newline() -> None:
    """The rendering is sorted-key JSON, newline-terminated, repeatable."""
    rendered = render_game_provenance(_DATASET_ID)
    assert rendered == render_game_provenance(_DATASET_ID)
    assert rendered.endswith("\n")
    assert "\r" not in rendered
    assert json.loads(rendered) == game_provenance_document(_DATASET_ID)
    reserialised = (
        json.dumps(json.loads(rendered), sort_keys=True, indent=2) + "\n"
    )
    assert rendered == reserialised


def test_provenance_name_is_the_sidecar_filename() -> None:
    """The exported name is the on-disk sidecar filename."""
    assert PROVENANCE_NAME == "provenance.json"


def test_heightfield_document_records_profile_and_quantization() -> None:
    """The heightfield document names its profile and height quantization."""
    document = heightfield_provenance_document(_DATASET_ID)
    assert document["profile"] == "heightfield"
    quant = document["quantization"]
    assert isinstance(quant, dict)
    assert quant["height_bits"] == heightfield.MAX_LEVEL.bit_length()
    assert quant["height_bits"] == 12
    assert quant["max_level"] == heightfield.MAX_LEVEL
    assert quant["max_axis_error_m"] == heightfield.MAX_AXIS_ERROR_M
    assert isinstance(quant["scheme"], str)
    assert "NAP height" in quant["scheme"]


def test_heightfield_document_sources_jpeg_and_chunk_versions() -> None:
    """JPEG settings and the .hf chunk versions come from the owning modules."""
    document = heightfield_provenance_document(_DATASET_ID)
    assert document["jpeg"] == {
        "quality": jpeg.JPEG_QUALITY,
        "subsampling": jpeg.JPEG_SUBSAMPLING,
        "progressive": jpeg.JPEG_PROGRESSIVE,
        "optimize": jpeg.JPEG_OPTIMIZE,
        "pillow": jpeg.pillow_version(),
    }
    assert document["chunk"] == {
        "magic": heightfield.MAGIC.decode("ascii"),
        "version": heightfield.VERSION,
        "vertical_datum": heightfield.VERTICAL_DATUM,
        "zstd_level": heightfield.ZSTD_LEVEL,
        "zstandard": heightfield.zstandard_version(),
    }


def test_render_heightfield_is_sorted_deterministic_json() -> None:
    """The heightfield rendering is sorted-key JSON, newline-terminated."""
    rendered = render_heightfield_provenance(_DATASET_ID)
    assert rendered == render_heightfield_provenance(_DATASET_ID)
    assert rendered.endswith("\n")
    assert "\r" not in rendered
    assert json.loads(rendered) == heightfield_provenance_document(
        _DATASET_ID
    )


def test_splat_document_records_profile_and_gaussian_scheme() -> None:
    """The splat document names its profile and per-gaussian construction."""
    document = splat_provenance_document(_DATASET_ID)
    assert document["profile"] == "splat"
    gaussian = document["gaussian"]
    assert isinstance(gaussian, dict)
    assert gaussian["opacity"] == splat.OPACITY
    assert gaussian["sh_dc0"] == splat.SH_DC0
    assert isinstance(gaussian["scheme"], str)
    assert "gaussian" in gaussian["scheme"]


def test_splat_document_sources_ply_versions() -> None:
    """The ply block's zstd settings come from the owning splat module."""
    document = splat_provenance_document(_DATASET_ID)
    assert document["ply"] == {
        "zstd_level": splat.ZSTD_LEVEL,
        "zstandard": splat.zstandard_version(),
    }


def test_render_splat_is_sorted_deterministic_json() -> None:
    """The splat rendering is sorted-key JSON, newline-terminated."""
    rendered = render_splat_provenance(_DATASET_ID)
    assert rendered == render_splat_provenance(_DATASET_ID)
    assert rendered.endswith("\n")
    assert "\r" not in rendered
    assert json.loads(rendered) == splat_provenance_document(_DATASET_ID)


def test_render_provenance_dispatches_by_profile() -> None:
    """The dispatch returns each profile's sidecar text, None for strict."""
    assert render_provenance(Profile.STRICT) is None
    assert render_provenance(
        Profile.GAME, dataset_id=_DATASET_ID
    ) == render_game_provenance(_DATASET_ID)
    assert render_provenance(
        Profile.HEIGHTFIELD, dataset_id=_DATASET_ID
    ) == render_heightfield_provenance(_DATASET_ID)
    assert render_provenance(
        Profile.SPLAT, dataset_id=_DATASET_ID
    ) == render_splat_provenance(_DATASET_ID)


def test_render_provenance_requires_dataset_id_for_lossy() -> None:
    """A lossy profile without a dataset_id is the context's typed error."""
    for profile in (Profile.GAME, Profile.HEIGHTFIELD, Profile.SPLAT):
        with pytest.raises(Tiles3dError, match="needs the pack dataset_id"):
            render_provenance(profile)
