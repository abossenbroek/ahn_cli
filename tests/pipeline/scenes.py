"""Shared synthetic-site builders for the pipeline integration tests.

A ``read`` source run needs an on-disk site: an ``ahn/`` directory of LAZ
sheets plus an ``ortho.tif`` on the same EPSG:28992 grid. :func:`build_site`
writes one from a deterministic jittered point grid (uniform spacing so the
``sqrt(k) * spacing`` halo floor is reliably sufficient, continuous jitter so
every squared distance is distinct -- no kNN tie ambiguity), matching the
reconcile stage tests' scene shape. :func:`linux_probe` builds the deterministic
free-RAM probe the executor's ``halo: auto`` sizing reads.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ahn_cli.pipeline.machine import SystemProbe
from tests.pipeline.harness import write_synthetic_laz, write_synthetic_ortho

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import numpy.typing as npt


_SPACING_M = 0.3
_JITTER = _SPACING_M / 3.0


def build_cloud_points(
    width: int, height: int, seed: int
) -> npt.NDArray[np.float64]:
    """Return a deterministic ``(n, 5)`` ``x, y, z, gps, class`` cloud.

    A jittered regular grid at :data:`_SPACING_M` covering the area padded by
    1.5 m, so a tile-edge pixel's kNN reach is tightly bounded everywhere.
    """
    rng = np.random.default_rng(seed)
    pad = 1.5
    axis_x = np.arange(-pad, width + pad, _SPACING_M)
    axis_y = np.arange(-pad, height + pad, _SPACING_M)
    grid_x, grid_y = np.meshgrid(axis_x, axis_y)
    total = grid_x.size
    x = grid_x.ravel() + rng.uniform(-_JITTER, _JITTER, total)
    y = grid_y.ravel() + rng.uniform(-_JITTER, _JITTER, total)
    z = rng.uniform(-5.0, 5.0, total)
    gps = rng.uniform(0.0, 1.0, total)
    classification = rng.integers(1, 7, total)
    return np.column_stack([x, y, z, gps, classification]).astype(np.float64)


def build_site(
    tmp_path: Path,
    *,
    width: int = 8,
    height: int = 6,
    seed: int = 0,
) -> tuple[Path, Path, Path]:
    """Write a read-source site; return ``(site_dir, cloud_path, ortho_path)``.

    The site holds ``ahn/cloud.laz`` (a jittered ``width x height`` metre cloud)
    and ``ortho.tif`` (a random 1 m RGB raster over ``(0, 0, width, height)``).
    """
    site = tmp_path / "site"
    (site / "ahn").mkdir(parents=True)
    cloud = site / "ahn" / "cloud.laz"
    write_synthetic_laz(cloud, build_cloud_points(width, height, seed))
    rng = np.random.default_rng(seed + 100)
    rgb_chw = rng.integers(0, 256, (3, height, width)).astype(np.uint8)
    ortho = site / "ortho.tif"
    write_synthetic_ortho(
        ortho, rgb_chw, (0.0, 0.0, float(width), float(height))
    )
    return site, cloud, ortho


def spec_text(
    site: Path,
    workdir: Path,
    output: Path,
    *,
    width: int,
    height: int,
    tile_pixels: int,
    sink: str,
) -> str:
    """Return a pipeline spec (YAML) for a read -> reconcile -> ``sink`` run."""
    return f"""
aoi: {{ bbox: "0,0,{width},{height}" }}
tiling: {{ tile_pixels: {tile_pixels}, halo: auto }}
workdir: {workdir}
output: {output}
stages:
  - {{ type: read, path: {site} }}
  - {{ type: reconcile, method: idw, idw: {{ power: 2, neighbors: 12 }} }}
  - {sink}
"""


def linux_probe(free_ram_bytes: int) -> SystemProbe:
    """Build a deterministic Linux probe reporting ``free_ram_bytes`` free RAM."""
    meminfo = f"MemTotal: 1 kB\nMemAvailable: {free_ram_bytes // 1024} kB\n"
    sizes = {"SC_LEVEL1_DCACHE_LINESIZE": 128, "SC_PAGE_SIZE": 4096}

    def _run(_args: Sequence[str]) -> str:
        raise AssertionError

    return SystemProbe(
        platform="linux",
        sysconf=lambda name: sizes[name],
        run=_run,
        read_text=lambda _path: meminfo,
    )
