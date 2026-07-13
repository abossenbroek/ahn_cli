"""Tests for the game-profile provenance sidecar."""

from __future__ import annotations

import json

from ahn_cli.tiles3d import heightfield, jpeg, meshopt, quantize
from ahn_cli.tiles3d.profile import Profile
from ahn_cli.tiles3d.provenance import (
    PROVENANCE_NAME,
    game_provenance_document,
    heightfield_provenance_document,
    render_game_provenance,
    render_heightfield_provenance,
    render_provenance,
)


def test_document_records_profile_and_quantization() -> None:
    """The document names the game profile and the quantization scheme."""
    document = game_provenance_document()
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
    document = game_provenance_document()
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


def test_render_is_sorted_deterministic_json_with_newline() -> None:
    """The rendering is sorted-key JSON, newline-terminated, repeatable."""
    rendered = render_game_provenance()
    assert rendered == render_game_provenance()
    assert rendered.endswith("\n")
    assert json.loads(rendered) == game_provenance_document()
    reserialised = (
        json.dumps(json.loads(rendered), sort_keys=True, indent=2) + "\n"
    )
    assert rendered == reserialised


def test_provenance_name_is_the_sidecar_filename() -> None:
    """The exported name is the on-disk sidecar filename."""
    assert PROVENANCE_NAME == "provenance.json"


def test_heightfield_document_records_profile_and_quantization() -> None:
    """The heightfield document names its profile and height quantization."""
    document = heightfield_provenance_document()
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
    document = heightfield_provenance_document()
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
        "zstd_level": heightfield.ZSTD_LEVEL,
        "zstandard": heightfield.zstandard_version(),
    }


def test_render_heightfield_is_sorted_deterministic_json() -> None:
    """The heightfield rendering is sorted-key JSON, newline-terminated."""
    rendered = render_heightfield_provenance()
    assert rendered == render_heightfield_provenance()
    assert rendered.endswith("\n")
    assert json.loads(rendered) == heightfield_provenance_document()


def test_render_provenance_dispatches_by_profile() -> None:
    """The dispatch returns each profile's sidecar text, None for strict."""
    assert render_provenance(Profile.STRICT) is None
    assert render_provenance(Profile.GAME) == render_game_provenance()
    assert render_provenance(Profile.HEIGHTFIELD) == (
        render_heightfield_provenance()
    )
