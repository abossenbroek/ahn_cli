"""Tests for the game-profile provenance sidecar."""

from __future__ import annotations

import json

from ahn_cli.tiles3d import jpeg, meshopt, quantize
from ahn_cli.tiles3d.provenance import (
    PROVENANCE_NAME,
    game_provenance_document,
    render_game_provenance,
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
