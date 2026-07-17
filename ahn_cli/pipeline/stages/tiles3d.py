"""The tiles3d sink: encodes one tile via the existing profile encoders.

:class:`Tiles3dSink` is a :class:`~ahn_cli.pipeline.model.Stage` that turns a
tile's already-sampled :class:`~ahn_cli.pipeline.model.GridTile` (heights plus
ortho colour, at whatever resolution the tile's own LOD calls for) into a
:class:`~ahn_cli.pipeline.model.EncodedTile`, reusing the standalone
``tiles3d`` verb's mesh/geodesy/encoder machinery unchanged
(:mod:`ahn_cli.tiles3d.mesh`, :mod:`ahn_cli.tiles3d.geodesy`,
:mod:`ahn_cli.tiles3d.profile`). Only the *encoding* of one tile is this
module's job: the executor drives which tiles exist and at what stride
(root through leaf), and a later stage assembles the per-tile blobs into a
tileset/pack -- this sink never holds more than one tile's data.

Coordinates. A :class:`~ahn_cli.pipeline.model.GridTile` carries no X/Y planes
(see the model's module docstring), so this module reconstructs them from the
tile's :class:`~ahn_cli.pipeline.model.TileContext` bbox and the grid's own
shape via the exact pixel-centre convention
(:class:`~ahn_cli.domain.PixelGrid`) the standalone verb's EXR/ortho pair
uses, then rounds to ``float32`` before handing them to
:func:`~ahn_cli.tiles3d.mesh.build_tile_mesh` -- reproducing the
float64 -> float32 -> float64 round trip the standalone terrain grid goes
through, so a single-tile run is byte-identical to the standalone build.

LOD. A tile's stride is ``2 ** (levels - level)`` -- the same convention
:mod:`ahn_cli.tiles3d.quadtree` uses for its quadtree -- from the sink's
configured tree depth (``levels``) and the tile's own
:class:`~ahn_cli.pipeline.model.TileKey` level; :func:`region_of` and
:func:`geometric_error_of` expose the per-tile metadata a later assembly
stage needs (the tileset/pack region and geometric error), since neither
rides along in :class:`~ahn_cli.pipeline.model.EncodedTile` itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ahn_cli.domain import PixelGrid
from ahn_cli.pipeline.errors import PipelineError
from ahn_cli.pipeline.model import EncodedBlob, EncodedTile, GridTile
from ahn_cli.tiles3d import payload as tiles3d_payload
from ahn_cli.tiles3d.geodesy import Geodesy
from ahn_cli.tiles3d.mesh import build_tile_mesh
from ahn_cli.tiles3d.quadtree import TilePlan, geometric_error
from ahn_cli.tiles3d.sources import TerrainGrid

if TYPE_CHECKING:
    from ahn_cli.domain import GeoTransform
    from ahn_cli.pipeline.model import TileContext, TilePayload
    from ahn_cli.tiles3d.mesh import Region
    from ahn_cli.tiles3d.profile import Profile

__all__ = ["Tiles3dSink"]


def _tile_transform(
    bbox: tuple[float, float, float, float], width: int, height: int
) -> GeoTransform:
    """Return the north-up geotransform of a tile's own bbox and grid shape.

    Matches ``rasterio.transform.from_bounds`` exactly (the convention every
    tiles3d/reconcile fixture in this codebase writes its rasters with).
    """
    minx, miny, maxx, maxy = bbox
    pixel_width = (maxx - minx) / width
    pixel_height = (maxy - miny) / height
    return (pixel_width, 0.0, minx, 0.0, -pixel_height, maxy)


@dataclass(frozen=True)
class Tiles3dSink:
    """Encode one tile through a tiles3d :class:`Profile`'s encoder.

    Contract:
        - ``profile`` selects the on-disk representation (strict/game/
          heightfield/splat), exactly as the standalone ``tiles3d`` verb's
          ``--profile`` does.
        - ``native_pixel_size_m`` is the source dataset's finest ground
          sampling distance (the leaf resolution) -- a single scalar, the
          same role :func:`ahn_cli.tiles3d.emit.pixel_size` plays for a
          whole-terrain build.
        - ``levels`` is the quadtree's depth (0 for a single-level/root-only
          run); a tile's LOD stride is ``2 ** (levels - key.level)``.
        - :meth:`halo_m` is always ``0`` -- a tile's mesh, colour and height
          plane are entirely determined by its own already-sampled
          :class:`~ahn_cli.pipeline.model.GridTile`, no source overlap
          needed.
        - :meth:`run` accepts only a
          :class:`~ahn_cli.pipeline.model.GridTile` and returns an
          :class:`~ahn_cli.pipeline.model.EncodedTile` carrying one blob
          named ``"geometry"`` (the encoder's primary content) plus, for
          encoders with a separate texture (only ``heightfield`` today), a
          second blob named ``"texture"`` -- matching
          :class:`~ahn_cli.pipeline.model.EncodedTile`'s documented
          ``geometry``/``texture`` naming.
        - :meth:`region_of` / :meth:`geometric_error_of` return the same
          per-tile region and geometric error the standalone build's
          ``PackEntry``/tileset entries carry, for an assembly stage that
          needs them (:class:`~ahn_cli.pipeline.model.EncodedTile` itself
          carries neither).

    Invariants:
        - Deterministic per machine (inherits the tiles3d geodesy caveat:
          absolute ECEF/geodetic output depends on the installed PROJ grids,
          never on run-to-run variation).

    Failure modes:
        - :class:`ValueError` at construction if ``native_pixel_size_m`` is
          not finite and positive, or ``levels`` is negative.
        - :class:`~ahn_cli.pipeline.errors.PipelineError` at call time if
          the tile is not a :class:`~ahn_cli.pipeline.model.GridTile`, its
          grid is empty, or its key's level exceeds ``levels``.
    """

    profile: Profile
    native_pixel_size_m: float
    levels: int = 0

    def __post_init__(self) -> None:
        """Reject a non-finite/non-positive pixel size or a negative depth."""
        if (
            not math.isfinite(self.native_pixel_size_m)
            or self.native_pixel_size_m <= 0.0
        ):
            msg = (
                "native_pixel_size_m must be finite and positive; got "
                f"{self.native_pixel_size_m}."
            )
            raise ValueError(msg)
        if self.levels < 0:
            msg = f"levels must be non-negative; got {self.levels}."
            raise ValueError(msg)

    def halo_m(self) -> float:
        """Return ``0`` -- a tile's own grid is all this sink needs."""
        return 0.0

    def run(self, tile: TilePayload, ctx: TileContext) -> TilePayload:
        """Encode ``tile`` through :attr:`profile`'s encoder.

        Failure modes:
            - :class:`~ahn_cli.pipeline.errors.PipelineError` if ``tile`` is
              not a :class:`~ahn_cli.pipeline.model.GridTile`, or via
              :meth:`_payload`.
        """
        grid = _require_grid_tile(tile, ctx)
        payload = self._payload(grid, ctx)
        encoded = self.profile.encoder().encode(payload)
        blobs = [EncodedBlob(name="geometry", data=encoded.content)]
        if encoded.texture is not None:
            blobs.append(EncodedBlob(name="texture", data=encoded.texture))
        return EncodedTile(key=ctx.key, blobs=tuple(blobs))

    def region_of(self, tile: GridTile, ctx: TileContext) -> Region:
        """Return the tile's own bounding region, in the profile's datum.

        Failure modes:
            - :class:`~ahn_cli.pipeline.errors.PipelineError` via
              :meth:`_payload`.
        """
        payload = self._payload(tile, ctx)
        return self.profile.encoder().region_of(payload)

    def geometric_error_of(self, ctx: TileContext) -> float:
        """Return the tile's 3D Tiles geometric error, in metres.

        Failure modes:
            - :class:`~ahn_cli.pipeline.errors.PipelineError` if ``ctx``'s
              key level exceeds :attr:`levels`.
        """
        stride = self._stride_of(ctx)
        return geometric_error(stride, self.native_pixel_size_m)

    def _stride_of(self, ctx: TileContext) -> int:
        """Return the LOD sampling stride for ``ctx``'s tile key."""
        level = ctx.key.level
        if level > self.levels:
            msg = (
                f"tile {ctx.key} has level {level}, deeper than this sink's "
                f"configured depth (levels={self.levels})."
            )
            raise PipelineError(msg)
        return 2 ** (self.levels - level)

    def _payload(
        self, grid: GridTile, ctx: TileContext
    ) -> tiles3d_payload.TilePayload:
        """Build the tiles3d ``TilePayload`` for ``grid`` at ``ctx``.

        Reconstructs a single-tile :class:`~ahn_cli.tiles3d.sources.TerrainGrid`
        from ``grid`` and ``ctx.bbox`` and runs it through the unchanged
        :func:`~ahn_cli.tiles3d.mesh.build_tile_mesh`, so the resulting mesh
        is byte-identical to what the standalone verb would build for the
        same source pixels.
        """
        height, width = grid.heights.shape
        if width == 0 or height == 0:
            msg = (
                f"tile {ctx.key} has an empty grid ({height}x{width}); "
                "tiles3d needs at least one pixel per axis."
            )
            raise PipelineError(msg)
        stride = self._stride_of(ctx)
        transform = _tile_transform(ctx.bbox, width, height)
        pixel_grid = PixelGrid(
            width=width, height=height, transform=transform
        )
        x = pixel_grid.eastings().astype(np.float32)
        y = pixel_grid.northings().astype(np.float32)
        rgb = np.ascontiguousarray(
            np.stack([grid.red, grid.green, grid.blue], axis=-1)
        )
        terrain = TerrainGrid(
            width=width,
            height=height,
            transform=transform,
            x=x,
            y=y,
            z=grid.heights,
            rgb=rgb,
        )
        plan = TilePlan(
            level=ctx.key.level,
            tx=ctx.key.tx,
            ty=ctx.key.ty,
            col0=0,
            row0=0,
            col1=width - 1,
            row1=height - 1,
            stride=1,
            children=(),
        )
        mesh = build_tile_mesh(terrain, plan, Geodesy())
        grid_index = np.ix_(mesh.rows, mesh.cols)
        error = geometric_error(stride, self.native_pixel_size_m)
        return tiles3d_payload.TilePayload(
            level=ctx.key.level,
            tx=ctx.key.tx,
            ty=ctx.key.ty,
            stride=stride,
            geometric_error=error,
            mesh=mesh,
            z=terrain.z[grid_index],
            rgb=terrain.rgb[grid_index],
        )


def _require_grid_tile(tile: TilePayload, ctx: TileContext) -> GridTile:
    """Return ``tile`` as a :class:`GridTile`, or raise for a wrong-stage input."""
    if not isinstance(tile, GridTile):
        msg = (
            f"tile {ctx.key} is not a GridTile; Tiles3dSink got "
            f"{type(tile).__name__}. Tiles3dSink must run after a stage "
            "that samples the ortho/height grid."
        )
        raise PipelineError(msg)
    return tile
