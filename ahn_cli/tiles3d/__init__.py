"""3D Tiles export bounded context: ortho map -> OGC 3D Tiles 1.1.

This context owns the final export stage of the pipeline: it converts the
orthophoto map and the reconciled per-pixel heights (``reconcile``'s EXR
output) into an OGC 3D Tiles 1.1 tileset (OGC 22-025r4) — a quadtree of
binary glTF terrain tiles draped with the orthophoto. It never fetches
and never interpolates: both inputs must already exist on disk, their
grids must match perfectly, and any missing data is a hard
:class:`Tiles3dError`. Every written artifact is re-verified from disk
against an independent recomputation before the build is accepted.
"""

from ahn_cli.tiles3d.errors import Tiles3dError

__all__ = ["Tiles3dError"]
