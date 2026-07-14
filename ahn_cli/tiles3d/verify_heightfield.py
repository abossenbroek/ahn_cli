"""Per-tile heightfield-profile verification: header, requant, texture.

The heightfield profile stores each tile as a ``.hf`` chunk (a fixed
header plus a zstd-framed ``uint16`` height plane) and a sibling baseline
JPEG, instead of an embedded-texture glTF. So the same "re-read from disk,
recompute from the sources, demand agreement" discipline the strict/game
verifiers use needs a heightfield-specific check set, run per tile before
the whole-file byte-identity backstop:

1.  **chunk decode** — the ``.hf`` bytes decode cleanly
    (:func:`~ahn_cli.tiles3d.heightfield.decode_heightfield` enforces
    magic, version, the declared payload length, a valid zstd frame and a
    ``width*height*2`` decompressed length).
2.  **header** — the decoded dims, ``z_offset``/``z_scale``, ``rtc_centre``
    and ``region`` equal an independent recomputation of this tile from the
    reloaded terrain (the same mesh + :func:`~ahn_cli.tiles3d.quantize.quantize_axis`
    the builder used).
3.  **requantization** — the decoded height levels equal an independent
    12-bit (:data:`~ahn_cli.tiles3d.heightfield.MAX_LEVEL`)
    :func:`~ahn_cli.tiles3d.quantize.quantize_axis` of the genuine source
    heights bit for bit, the dequantized heights sit within the exported
    :func:`~ahn_cli.tiles3d.quantize.axis_error_bound` (zero slack), and that
    bound is within the absolute
    :data:`~ahn_cli.tiles3d.heightfield.MAX_AXIS_ERROR_M` cap.
4.  **JPEG texture** — the sibling ``.jpg`` is baseline sequential,
    byte-equals an independent :func:`~ahn_cli.tiles3d.jpeg.encode_jpeg` of
    the sampled ortho tile, and decodes within
    :func:`~ahn_cli.tiles3d.jpeg.jpeg_fidelity_ok` of it (the same three-part
    check the game profile applies to its embedded JPEG).

Region containment (the tile's genuine source vertices inside every
enclosing stored region) is exact for heightfield — its region is computed
from the same sources, not quantized geometry — so the caller runs the
strict source-based containment check; it is not duplicated here.
Everything is recomputed from the terrain the verifier already reloaded.
The single entry point is :func:`verify_heightfield_tile`; all violations
raise :class:`~ahn_cli.tiles3d.errors.Tiles3dError` naming the tile.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ahn_cli.tiles3d.errors import Tiles3dError
from ahn_cli.tiles3d.heightfield import (
    MAX_AXIS_ERROR_M,
    MAX_LEVEL,
    decode_heightfield,
)
from ahn_cli.tiles3d.jpeg import (
    decode_jpeg,
    encode_jpeg,
    is_baseline_jpeg,
    jpeg_fidelity_ok,
)
from ahn_cli.tiles3d.mesh import build_tile_mesh
from ahn_cli.tiles3d.quantize import (
    axis_error_bound,
    dequantize_axis,
    quantize_axis,
)

if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt

    from ahn_cli.tiles3d.geodesy import Geodesy
    from ahn_cli.tiles3d.heightfield import DecodedHeightfield
    from ahn_cli.tiles3d.mesh import Region, TileMesh
    from ahn_cli.tiles3d.quadtree import TilePlan
    from ahn_cli.tiles3d.quantize import QuantizedAxis
    from ahn_cli.tiles3d.sources import TerrainGrid

__all__ = ["verify_heightfield_tile"]


def _require(condition: bool, message: str) -> None:  # noqa: FBT001
    """Raise the typed verification error unless ``condition`` holds."""
    if not condition:
        raise Tiles3dError(message)


def verify_heightfield_tile(
    out_dir: Path,
    content_uri: str,
    texture_uri: str,
    terrain: TerrainGrid,
    tile: TilePlan,
    geodesy: Geodesy,
) -> None:
    """Verify one heightfield tile's chunk and texture against the sources.

    Contract:
        - Reads ``content_uri`` (the ``.hf``) and ``texture_uri`` (the
          sibling ``.jpg``) fresh from ``out_dir`` and checks them against
          an independent recomputation of the tile from ``terrain`` /
          ``tile`` / ``geodesy``: chunk decode, header fields,
          requantization + dequant bound, and the baseline-JPEG texture.

    Failure modes:
        - :class:`~ahn_cli.tiles3d.errors.Tiles3dError` naming
          ``content_uri`` and the first failed property.
    """
    mesh = build_tile_mesh(terrain, tile, geodesy)
    heights = terrain.z[np.ix_(mesh.rows, mesh.cols)]
    grid_height, grid_width = heights.shape
    quantized = quantize_axis(heights.reshape(-1), MAX_LEVEL)
    decoded = decode_heightfield((out_dir / content_uri).read_bytes())
    # Independent recomputation of the tile's own NAP region (v3, NAP-native):
    # the mesh's horizontal doubles + the tile's own NAP height min/max —
    # mirrors `heightfield.nap_region` without calling it.
    west, south, east, north, *_ = mesh.region
    expected_region = (
        west,
        south,
        east,
        north,
        float(heights.min()),
        float(heights.max()),
    )
    _verify_header(
        decoded,
        quantized,
        mesh,
        expected_region,
        (grid_width, grid_height),
        content_uri,
    )
    _verify_requantization(
        decoded, quantized, heights, (grid_width, grid_height), content_uri
    )
    _verify_texture(out_dir, texture_uri, terrain, mesh)


def _verify_header(
    decoded: DecodedHeightfield,
    quantized: QuantizedAxis,
    mesh: TileMesh,
    expected_region: Region,
    dims: tuple[int, int],
    uri: str,
) -> None:
    """Check the decoded header equals the independent recomputation."""
    width, height = dims
    _require(
        decoded.width == width and decoded.height == height,
        f"{uri}: heightfield dims {decoded.width}x{decoded.height} do not "
        f"equal the tile's sampled {width}x{height}.",
    )
    _require(
        decoded.z_offset == quantized.offset
        and decoded.z_scale == quantized.scale,
        f"{uri}: heightfield z_offset/z_scale do not equal the independent "
        "quantization of the source heights.",
    )
    _require(
        decoded.rtc_centre == mesh.center,
        f"{uri}: heightfield rtc_centre does not equal the tile's RTC "
        "centre.",
    )
    _require(
        decoded.region == expected_region,
        f"{uri}: heightfield region does not equal the tile's own NAP "
        "region (v3: radian lon/lat + NAP height bounds).",
    )


def _verify_requantization(
    decoded: DecodedHeightfield,
    quantized: QuantizedAxis,
    heights: npt.NDArray[np.float32],
    dims: tuple[int, int],
    uri: str,
) -> None:
    """Check the decoded levels requantize and dequantize within bound."""
    width, height = dims
    expected = quantized.ints.reshape(height, width)
    _require(
        bool(np.array_equal(decoded.z_ints, expected)),
        f"{uri}: the heightfield levels do not equal the independent "
        "requantization of the source heights.",
    )
    dequantized = dequantize_axis(quantized)
    bound = axis_error_bound(quantized.scale)
    error = float(
        np.abs(dequantized - heights.reshape(-1).astype(np.float64)).max()
    )
    _require(
        error <= bound,
        f"{uri}: a dequantized height exceeds the documented quantization "
        "error bound.",
    )
    _require(
        bound <= MAX_AXIS_ERROR_M,
        f"{uri}: the heightfield error bound {bound} m exceeds the "
        f"{MAX_AXIS_ERROR_M} m absolute cap.",
    )


def _verify_texture(
    out_dir: Path,
    texture_uri: str,
    terrain: TerrainGrid,
    mesh: TileMesh,
) -> None:
    """Check the sibling JPEG equals the re-encoded sampled ortho tile."""
    sampled_rgb = terrain.rgb[np.ix_(mesh.rows, mesh.cols)]
    data = (out_dir / texture_uri).read_bytes()
    _require(
        is_baseline_jpeg(data),
        f"{texture_uri}: the texture is not a baseline sequential JPEG "
        "(SOF0 present, SOF2 absent).",
    )
    _require(
        data == encode_jpeg(sampled_rgb),
        f"{texture_uri}: the JPEG texture does not byte-equal the "
        "independent re-encode of the sampled ortho tile.",
    )
    _require(
        jpeg_fidelity_ok(sampled_rgb, decode_jpeg(data)),
        f"{texture_uri}: the JPEG texture decodes below the fidelity floor "
        "of the sampled ortho tile.",
    )
