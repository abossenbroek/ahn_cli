"""Per-tile splat-profile verification: decode, per-gaussian recompute.

The splat profile stores each tile as a zstd-wrapped binary 3DGS ``.ply``
gaussian cloud, instead of a mesh or a height chunk. So the same "re-read
from disk, recompute from the sources, demand agreement" discipline the
other profile verifiers use needs a splat-specific check set, run per tile
before the whole-file byte-identity backstop:

1.  **decode** — the ``.ply`` bytes decode cleanly
    (:func:`~ahn_cli.tiles3d.splat.decode_splat` enforces the zstd frame's
    content checksum, the fixed ASCII header and the exact body length).
2.  **positions** — the decoded gaussians' positions equal an independent
    recomputation of this tile's mesh, bit for bit (no quantization).
3.  **colour** — the decoded ``f_dc`` equals an independent SH degree-0
    recompute of the sampled ortho tile, bit for bit.
4.  **opacity / scale / rotation** — every gaussian carries the fixed
    logit opacity, the tile's independently measured isotropic log scale,
    and the identity quaternion.

Region containment (the tile's genuine source vertices inside every
enclosing stored region) is exact for splat — its region is computed from
the same sources, not quantized geometry — so the caller runs the strict
source-based containment check; it is not duplicated here. Everything is
recomputed from the terrain the verifier already reloaded. The single
entry point is :func:`verify_splat_tile`; all violations raise
:class:`~ahn_cli.tiles3d.errors.Tiles3dError` naming the tile.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from ahn_cli.tiles3d.errors import Tiles3dError
from ahn_cli.tiles3d.mesh import build_tile_mesh
from ahn_cli.tiles3d.splat import OPACITY, SH_DC0, decode_splat

if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt

    from ahn_cli.tiles3d.geodesy import Geodesy
    from ahn_cli.tiles3d.mesh import TileMesh
    from ahn_cli.tiles3d.quadtree import TilePlan
    from ahn_cli.tiles3d.sources import TerrainGrid
    from ahn_cli.tiles3d.splat import DecodedSplat

__all__ = ["verify_splat_tile"]

_OPACITY_LOGIT = math.log(OPACITY / (1.0 - OPACITY))


def _require(condition: bool, message: str) -> None:  # noqa: FBT001
    """Raise the typed verification error unless ``condition`` holds."""
    if not condition:
        raise Tiles3dError(message)


def verify_splat_tile(
    out_dir: Path,
    content_uri: str,
    terrain: TerrainGrid,
    tile: TilePlan,
    geodesy: Geodesy,
) -> None:
    """Verify one splat tile's gaussian cloud against the sources.

    Contract:
        - Reads ``content_uri`` (the ``.ply``) fresh from ``out_dir`` and
          checks it against an independent recomputation of the tile from
          ``terrain`` / ``tile`` / ``geodesy``: decode, position equality,
          colour equality, and the fixed opacity/scale/rotation.

    Failure modes:
        - :class:`~ahn_cli.tiles3d.errors.Tiles3dError` naming
          ``content_uri`` and the first failed property.
    """
    mesh = build_tile_mesh(terrain, tile, geodesy)
    sampled_rgb = terrain.rgb[np.ix_(mesh.rows, mesh.cols)]
    decoded = decode_splat((out_dir / content_uri).read_bytes())
    _require(
        decoded.count == mesh.positions.shape[0],
        f"{content_uri}: splat gaussian count {decoded.count} does not "
        f"equal the tile's sampled vertex count {mesh.positions.shape[0]}.",
    )
    _require(
        bool(np.array_equal(decoded.positions, mesh.positions)),
        f"{content_uri}: splat positions do not equal the tile's RTC mesh "
        "vertices.",
    )
    expected_f_dc = _expected_f_dc(sampled_rgb)
    _require(
        bool(np.array_equal(decoded.f_dc, expected_f_dc)),
        f"{content_uri}: splat colour does not equal the independent SH "
        "degree-0 recompute of the sampled ortho tile.",
    )
    _require(
        bool((decoded.opacity == np.float32(_OPACITY_LOGIT)).all()),
        f"{content_uri}: a splat gaussian's opacity is not the fixed "
        "logit(OPACITY) constant.",
    )
    _verify_scale(decoded, mesh, content_uri)
    expected_rot = np.zeros((decoded.count, 4), dtype=np.float32)
    expected_rot[:, 0] = 1.0
    _require(
        bool(np.array_equal(decoded.rot, expected_rot)),
        f"{content_uri}: a splat gaussian's rotation is not the identity "
        "quaternion.",
    )


def _expected_f_dc(
    sampled_rgb: npt.NDArray[np.uint8],
) -> npt.NDArray[np.float32]:
    """Return the independent SH degree-0 recompute of the sampled ortho."""
    rgb = sampled_rgb.reshape(-1, 3).astype(np.float64) / 255.0
    return ((rgb - 0.5) / SH_DC0).astype(np.float32)


def _verify_scale(
    decoded: DecodedSplat, mesh: TileMesh, content_uri: str
) -> None:
    """Verify the decoded scale is isotropic and matches the measured spacing."""
    positions = mesh.positions.astype(np.float64)
    delta = positions[1] - positions[0]
    spacing = float(np.sqrt(np.sum(delta * delta)))
    expected = np.float32(math.log(spacing))
    _require(
        bool((decoded.scale == expected).all()),
        f"{content_uri}: a splat gaussian's scale does not equal the "
        "independently measured isotropic cell spacing.",
    )
