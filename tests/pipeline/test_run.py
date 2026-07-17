"""End-to-end verification gates for :func:`ahn_cli.pipeline.run.run_spec`.

These are the load-bearing invariants of the pipeline epic, exercised through
the full wiring (planner, read source, ortho windows, stage chain, sink,
assembly):

* **Fusion identity** -- a single-tile tiles3d run is byte-identical to the
  standalone ``reconcile`` -> ``tiles3d`` verbs.
* **Multi-tile-with-halo identity** -- a 2x2 cloud run stitches back to the
  whole-area standalone ``reconcile`` grid (the halo makes edge kNN identical).
* **Two-budget byte-identity** -- the same run under a tiny vs a generous
  injected free-RAM budget produces byte-identical deliverables.
* **Resumability** -- a fault at a tile boundary, then a resume, reproduces an
  uninterrupted run's bytes with no partial survivors.
* **Bounded memory** -- no whole-AOI intermediate (``pointcloud.laz`` /
  ``reconciled.exr`` / an ortho mosaic) is ever written.

RAM/CPU/pool/fault are always injected, never live.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import numpy as np
import pytest

from ahn_cli.pipeline.errors import PipelineError
from ahn_cli.pipeline.manifest import TileStore
from ahn_cli.pipeline.model import TileKey
from ahn_cli.pipeline.planners import GridTilePlanner
from ahn_cli.pipeline.run import run_spec
from ahn_cli.pipeline.spec import parse_yaml
from ahn_cli.pipeline.stages.write import GRID_BLOB_NAME, decode_grid_blob
from ahn_cli.reconcile.method import IdwInterp
from ahn_cli.reconcile.reconcile import ReconcileRequest, reconcile
from ahn_cli.reconcile.writers import OutputFormat
from ahn_cli.tiles3d.build import build_tiles3d
from ahn_cli.tiles3d.profile import Profile
from tests.pipeline.scenes import build_site, linux_probe, spec_text

if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt

_WIDTH = 8
_HEIGHT = 6
_GENEROUS = 64 * 1024**3
_TINY = 1_000_000


def _hash_tree(root: Path) -> dict[str, str]:
    """Map every file under ``root`` to its SHA-256, keyed by relative path."""
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _tiles3d_spec(site: Path, tmp: Path, profile: str) -> str:
    return spec_text(
        site,
        tmp / "wd",
        tmp / "out",
        width=_WIDTH,
        height=_HEIGHT,
        tile_pixels=256,
        sink=f"{{ type: tiles3d, profile: {profile} }}",
    )


@pytest.mark.parametrize("profile", ["strict", "splat", "heightfield"])
def test_single_tile_fusion_is_byte_identical(
    tmp_path: Path, profile: str
) -> None:
    """A single-tile tiles3d run equals standalone reconcile -> tiles3d."""
    site, cloud, ortho = build_site(tmp_path, seed=0)
    std_recon = tmp_path / "recon"
    reconcile(
        ReconcileRequest(
            ortho,
            cloud,
            std_recon,
            IdwInterp(power=2.0, k=12),
            (OutputFormat.EXR,),
        )
    )
    std_out = tmp_path / "std"
    build_tiles3d(
        ortho,
        std_recon / "reconciled.exr",
        std_out,
        profile=Profile.parse(profile),
    )

    spec = parse_yaml(_tiles3d_spec(site, tmp_path, profile))
    result = run_spec(spec, probe=linux_probe(_GENEROUS), cpu_count=1)

    assert result.tile_count == 1
    assert _hash_tree(std_out) == _hash_tree(tmp_path / "out")


def _write_spec(site: Path, tmp: Path, out: Path, tile_pixels: int) -> str:
    return spec_text(
        site,
        tmp / "wd",
        out,
        width=_WIDTH,
        height=_HEIGHT,
        tile_pixels=tile_pixels,
        sink="{ type: write, path: cloud }",
    )


def _stitch(
    out: Path, spec_aoi: tuple[float, float, float, float]
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.uint8]]:
    """Place every write-sink grid tile at its pixel offset into one mosaic."""
    store = TileStore(out)
    planner = GridTilePlanner(tile_size_m=4.0)
    tiles = planner.plan(aoi_bbox=spec_aoi, halo_m=0.0, workdir=out)
    heights = np.zeros((_HEIGHT, _WIDTH), dtype=np.float32)
    colour = np.zeros((_HEIGHT, _WIDTH, 3), dtype=np.uint8)
    for ctx in tiles:
        blob = (store.tile_dir(ctx.key) / GRID_BLOB_NAME).read_bytes()
        grid = decode_grid_blob(blob)
        minx, miny, maxx, maxy = ctx.bbox
        col0 = round(minx)
        row0 = round(_HEIGHT - maxy)
        h = round(maxy - miny)
        w = round(maxx - minx)
        heights[row0 : row0 + h, col0 : col0 + w] = grid.heights
        colour[row0 : row0 + h, col0 : col0 + w, 0] = grid.red
        colour[row0 : row0 + h, col0 : col0 + w, 1] = grid.green
        colour[row0 : row0 + h, col0 : col0 + w, 2] = grid.blue
    return heights, colour


def test_multi_tile_with_halo_matches_whole_area(tmp_path: Path) -> None:
    """A 2x2 write run stitches back to the whole-area standalone reconcile."""
    site, cloud, ortho = build_site(tmp_path, seed=1)
    out = tmp_path / "out"
    spec = parse_yaml(_write_spec(site, tmp_path, out, tile_pixels=4))
    result = run_spec(spec, probe=linux_probe(_GENEROUS), cpu_count=1)
    assert result.tile_count == 4

    std = tmp_path / "recon"
    reconcile(
        ReconcileRequest(
            ortho, cloud, std, IdwInterp(power=2.0, k=12), (OutputFormat.PT,)
        )
    )
    whole = np.fromfile(std / "reconciled.pt", dtype="<f4").reshape(
        _HEIGHT, _WIDTH, 6
    )
    heights, colour = _stitch(out, (0.0, 0.0, float(_WIDTH), float(_HEIGHT)))
    assert heights.tobytes() == whole[:, :, 2].tobytes()
    assert np.array_equal(colour[:, :, 0], whole[:, :, 3].astype(np.uint8))
    assert np.array_equal(colour[:, :, 1], whole[:, :, 4].astype(np.uint8))
    assert np.array_equal(colour[:, :, 2], whole[:, :, 5].astype(np.uint8))


def _grid_blob_hashes(out: Path) -> dict[str, str]:
    """Hash every write-sink ``grid`` deliverable blob, keyed by tile dir path.

    The resume markers/manifest embed the output path (via ``spec_hash``), so
    they legitimately differ between two runs to different directories; the
    deliverable *content* -- the per-tile grid blobs -- must not.
    """
    return {
        path.parent.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(out.rglob(GRID_BLOB_NAME))
    }


def test_two_ram_budgets_are_byte_identical(tmp_path: Path) -> None:
    """Tiny vs generous free-RAM budgets yield byte-identical deliverables."""
    site, _cloud, _ortho = build_site(tmp_path, seed=2)
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    spec_a = parse_yaml(_write_spec(site, tmp_path / "ta", out_a, 4))
    spec_b = parse_yaml(_write_spec(site, tmp_path / "tb", out_b, 4))
    run_spec(spec_a, probe=linux_probe(_TINY), cpu_count=1)
    run_spec(spec_b, probe=linux_probe(_GENEROUS), cpu_count=1)
    hashes_a = _grid_blob_hashes(out_a)
    assert hashes_a  # the run produced tiles
    assert hashes_a == _grid_blob_hashes(out_b)


class _FaultAt:
    """A fault hook that raises on its ``n``-th invocation (0-based)."""

    def __init__(self, n: int) -> None:
        self._n = n
        self._count = 0

    def __call__(self, _point: str) -> None:
        current = self._count
        self._count += 1
        if current == self._n:
            msg = "injected kill"
            raise RuntimeError(msg)


def test_resume_after_fault_is_byte_identical(tmp_path: Path) -> None:
    """A fault at a tile boundary then a resume reproduces a clean run's bytes."""
    site, _cloud, _ortho = build_site(tmp_path, seed=3)
    clean_out = tmp_path / "clean"
    clean = parse_yaml(_write_spec(site, tmp_path / "tc", clean_out, 4))
    run_spec(clean, probe=linux_probe(_TINY), cpu_count=1)

    resumed_out = tmp_path / "resumed"
    spec = parse_yaml(_write_spec(site, tmp_path / "tr", resumed_out, 4))
    with pytest.raises(RuntimeError, match="injected kill"):
        run_spec(
            spec, probe=linux_probe(_TINY), cpu_count=1, fault=_FaultAt(3)
        )
    result = run_spec(spec, probe=linux_probe(_TINY), cpu_count=1)

    assert 0 < result.processed <= result.tile_count
    assert result.skipped == result.tile_count - result.processed
    assert _grid_blob_hashes(clean_out) == _grid_blob_hashes(resumed_out)


def test_no_whole_area_intermediate_is_written(tmp_path: Path) -> None:
    """No whole-AOI pointcloud/exr/mosaic ever lands on disk."""
    site, _cloud, _ortho = build_site(tmp_path, seed=4)
    out = tmp_path / "out"
    spec = parse_yaml(_tiles3d_spec(site, tmp_path, "strict"))
    run_spec(spec, probe=linux_probe(_GENEROUS), cpu_count=1)
    banned = {"pointcloud.laz", "reconciled.exr", "reconciled.laz"}
    for path in (tmp_path / "wd", out).__iter__():
        for child in path.rglob("*"):
            assert child.name not in banned


def test_fetch_source_is_a_clear_error(tmp_path: Path) -> None:
    """A fetch source is a deferred seam: run_spec rejects it clearly."""
    spec = parse_yaml(
        f"""
aoi: {{ bbox: "0,0,8,6" }}
workdir: {tmp_path / "wd"}
output: {tmp_path / "out"}
stages:
  - {{ type: fetch, source: pdok }}
  - {{ type: reconcile, method: idw }}
  - {{ type: tiles3d, profile: strict }}
"""
    )
    with pytest.raises(PipelineError, match="only a `read` source"):
        run_spec(spec)


def test_read_site_without_ortho_is_an_error(tmp_path: Path) -> None:
    """A read site missing its ortho is a clear error."""
    site = tmp_path / "bare"
    (site / "ahn").mkdir(parents=True)
    spec = parse_yaml(
        f"""
aoi: {{ bbox: "0,0,8,6" }}
workdir: {tmp_path / "wd"}
output: {tmp_path / "out"}
stages:
  - {{ type: read, path: {site} }}
  - {{ type: tiles3d, profile: strict }}
"""
    )
    with pytest.raises(PipelineError, match="no orthophoto"):
        run_spec(spec)


def test_full_chain_with_dedup_and_thin_runs(tmp_path: Path) -> None:
    """A read -> dedup -> thin -> reconcile -> write chain completes every tile.

    Exercises the dedup and thin middle-stage wiring; the voxel thin round-trips
    coordinates through a centimetre-scale scratch LAZ, so this asserts the run
    completes (not byte-identity to a raw-cloud reconcile).
    """
    site, _cloud, _ortho = build_site(tmp_path, seed=6)
    out = tmp_path / "out"
    spec = parse_yaml(
        f"""
aoi: {{ bbox: "0,0,{_WIDTH},{_HEIGHT}" }}
tiling: {{ tile_pixels: 4, halo: auto }}
workdir: {tmp_path / "wd"}
output: {out}
stages:
  - {{ type: read, path: {site} }}
  - {{ type: dedup, include_classes: [1, 2, 3, 4, 5, 6] }}
  - {{ type: thin, method: voxel, voxel_size_m: 0.0 }}
  - {{ type: reconcile, method: idw, idw: {{ power: 2, neighbors: 12 }} }}
  - {{ type: write, path: cloud }}
"""
    )
    result = run_spec(
        spec, probe=linux_probe(_GENEROUS), cpu_count=1, point_spacing_m=0.3
    )
    assert result.processed == result.tile_count == 4


def test_middle_stage_must_not_be_a_source(tmp_path: Path) -> None:
    """A read stage wedged between source and sink is a wiring error."""
    site, _cloud, _ortho = build_site(tmp_path, seed=7)
    spec = parse_yaml(
        f"""
aoi: {{ bbox: "0,0,{_WIDTH},{_HEIGHT}" }}
workdir: {tmp_path / "wd"}
output: {tmp_path / "out"}
stages:
  - {{ type: read, path: {site} }}
  - {{ type: read, path: {site} }}
  - {{ type: tiles3d, profile: strict }}
"""
    )
    with pytest.raises(PipelineError, match="cannot appear between"):
        run_spec(spec, probe=linux_probe(_GENEROUS), cpu_count=1)


def test_read_site_layout_variants(tmp_path: Path) -> None:
    """Sheets directly under the site and a nested ortho both resolve."""
    site = tmp_path / "flat"
    site.mkdir()
    # Cloud sheet directly under the site (no ahn/ subdir).
    build = build_site(tmp_path / "src", seed=8)
    src_cloud = build[1]
    (site / "cloud.laz").write_bytes(src_cloud.read_bytes())
    # Ortho nested under ortho/ortho.tif (the fallback location).
    (site / "ortho").mkdir()
    (site / "ortho" / "ortho.tif").write_bytes(build[2].read_bytes())
    spec = parse_yaml(
        f"""
aoi: {{ bbox: "0,0,{_WIDTH},{_HEIGHT}" }}
workdir: {tmp_path / "wd"}
output: {tmp_path / "out"}
stages:
  - {{ type: read, path: {site} }}
  - {{ type: reconcile, method: idw }}
  - {{ type: write, path: cloud }}
"""
    )
    result = run_spec(spec, probe=linux_probe(_GENEROUS), cpu_count=1)
    assert result.tile_count == 1


def test_read_path_that_is_a_file_is_an_error(tmp_path: Path) -> None:
    """A read path pointing at a file, not a site directory, is refused."""
    a_file = tmp_path / "spec.txt"
    a_file.write_text("x", encoding="utf-8")
    spec = parse_yaml(
        f"""
aoi: {{ bbox: "0,0,8,6" }}
workdir: {tmp_path / "wd"}
output: {tmp_path / "out"}
stages:
  - {{ type: read, path: {a_file} }}
  - {{ type: tiles3d, profile: strict }}
"""
    )
    with pytest.raises(PipelineError, match="not a directory"):
        run_spec(spec)


def test_key_offsets_are_placed(tmp_path: Path) -> None:
    """Sanity: every 2x2 write tile writes a distinct committed grid blob."""
    site, _cloud, _ortho = build_site(tmp_path, seed=5)
    out = tmp_path / "out"
    spec = parse_yaml(_write_spec(site, tmp_path, out, tile_pixels=4))
    run_spec(spec, probe=linux_probe(_GENEROUS), cpu_count=1)
    store = TileStore(out)
    for ty in range(2):
        for tx in range(2):
            key = TileKey(level=0, tx=tx, ty=ty)
            assert (store.tile_dir(key) / GRID_BLOB_NAME).is_file()
