"""Reconcile orchestration: interpolate the cloud onto the ortho grid, then emit.

:func:`reconcile` is the reconcile context's single entry point. It loads the
orthophoto (the target grid + RGB) and the AHN point cloud, estimates an
elevation ``Z`` at every ortho pixel centre with the requested interpolation
method and backend, assembles the coloured ``(h, w, 6)`` grid -- ``X, Y`` from
the geotransform, ``Z`` interpolated, ``R, G, B`` picked directly from the ortho
pixel -- and writes it in every requested format.

Determinism: with the default numpy backend the whole pipeline is a pure
function of the inputs, so every output is byte-identical across runs. (The
opt-in Metal backend is ``allclose``-equivalent, not byte-identical.)

Scale ceiling: the grid is materialised whole, so a full 12500x12500 ortho tile
(~156 M pixels, ~3.75 GB for the grid) is the memory ceiling; windowed inputs
(a clipped AOI) are the intended use, and block-streaming is a future perf step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ahn_cli.reconcile.interpolate import interpolate
from ahn_cli.reconcile.raster import (
    ReconcileError,
    load_cloud,
    load_ortho,
    target_coordinates,
)
from ahn_cli.reconcile.writers import OutputFormat, write_reconciled

if TYPE_CHECKING:
    from pathlib import Path

    from ahn_cli.reconcile.backend import InterpBackend
    from ahn_cli.reconcile.method import InterpMethod

__all__ = [
    "ReconcileError",
    "ReconcileRequest",
    "ReconcileStats",
    "reconcile",
]

_RGB_CHANNELS = slice(3, 6)
"""The R, G, B channel slice of the assembled ``(h, w, 6)`` grid."""


@dataclass(frozen=True)
class ReconcileRequest:
    """A validated intent to reconcile one ortho/cloud pair.

    Contract:
        - ``ortho_path`` / ``cloud_path`` are the orthophoto GeoTIFF and AHN LAZ
          to bridge; the ortho defines the output grid.
        - ``output_dir`` receives one ``reconciled.<ext>`` file per format.
        - ``method`` is the validated interpolation request.
        - ``formats`` is the non-empty set of output formats to write.
        - ``backend`` is the interpolation backend (numpy reference or Metal).

    Invariants:
        - Frozen value object, equal by field value.
    """

    ortho_path: Path
    cloud_path: Path
    output_dir: Path
    method: InterpMethod
    formats: tuple[OutputFormat, ...]
    backend: InterpBackend


@dataclass(frozen=True)
class ReconcileStats:
    """The ledger of one reconcile run.

    Contract (fields):
        - ``width`` / ``height``: the output grid's dimensions (the ortho's).
        - ``valid_points``: the number of pixels with an interpolated elevation
          (the point count in the laz/ply/pt outputs).
        - ``outputs``: the written file paths, in requested-format order.

    Invariants:
        - Frozen value object, equal by field value.
    """

    width: int
    height: int
    valid_points: int
    outputs: tuple[Path, ...]


def reconcile(request: ReconcileRequest) -> ReconcileStats:
    """Interpolate the cloud onto the ortho grid and write every format.

    Contract:
        - Loads ``ortho_path`` (target grid + RGB) and ``cloud_path`` (source
          XYZ), estimates ``Z`` at each pixel centre via ``method``/``backend``,
          and writes ``<output_dir>/reconciled.<ext>`` for each requested format.
        - Returns a :class:`ReconcileStats` with the grid dimensions, the valid
          (interpolated) pixel count, and the output paths.

    Invariants:
        - Deterministic on the numpy backend: identical inputs yield
          byte-identical outputs.

    Failure modes:
        - :class:`ReconcileError` if an input is missing/unreadable or the
          orthophoto lacks three colour bands.
    """
    ortho = load_ortho(request.ortho_path)
    cloud = load_cloud(request.cloud_path)
    target_xy, eastings, northings = target_coordinates(ortho.grid)

    z, valid = interpolate(
        request.method, cloud, target_xy, backend=request.backend
    )
    height, width = ortho.rgb.shape[:2]
    grid = np.empty((height, width, 6), dtype=np.float64)
    grid[:, :, 0] = eastings
    grid[:, :, 1] = northings
    grid[:, :, 2] = z.reshape(height, width)
    grid[:, :, _RGB_CHANNELS] = ortho.rgb.astype(np.float64)
    mask = valid.reshape(height, width)

    request.output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for output_format in request.formats:
        out_path = request.output_dir / f"reconciled.{output_format.value}"
        write_reconciled(output_format, grid, mask, out_path)
        outputs.append(out_path)

    return ReconcileStats(
        width=width,
        height=height,
        valid_points=int(mask.sum()),
        outputs=tuple(outputs),
    )
