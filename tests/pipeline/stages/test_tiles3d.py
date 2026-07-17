"""Tests for :class:`~ahn_cli.pipeline.stages.tiles3d.Tiles3dSink`.

The load-bearing claim is byte-identity to the standalone ``tiles3d`` verb:
for a single-tile AOI, the sink's encoded blob(s) must byte-match what
:func:`ahn_cli.tiles3d.build.build_tiles3d` writes for the same source pixels,
across every profile. A second group of tests checks the parent-LOD stride/
geometric-error arithmetic and the region-containment invariant the standalone
emitter relies on; a third assembles the sink's single-tile output into the
on-disk shape a real build would produce and runs it through the standalone
strict verifier.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from ahn_cli.pipeline.errors import PipelineError
from ahn_cli.pipeline.model import EncodedTile as PipelineEncodedTile
from ahn_cli.pipeline.model import GridTile, TileContext, TileKey
from ahn_cli.pipeline.stages.tiles3d import Tiles3dSink
from ahn_cli.tiles3d.build import build_tiles3d
from ahn_cli.tiles3d.pack import TileKey as PackTileKey
from ahn_cli.tiles3d.profile import Profile
from ahn_cli.tiles3d.quadtree import (
    geometric_error as tiles3d_geometric_error,
)
from ahn_cli.tiles3d.tileset import (
    tile_entry,
    tileset_document,
    write_tileset,
)
from ahn_cli.tiles3d.verify import verify_tiles3d
from tests.pipeline.harness import make_point_tile
from tests.tiles3d.conftest import (
    MAXY,
    MINX,
    RES,
    grid_for_ortho,
    make_ortho,
    pack_blob,
    synth_rgb,
    write_exr,
)

if TYPE_CHECKING:
    from pathlib import Path

    from ahn_cli.domain import BBox

_PROFILES = list(Profile)


def _grid_tile(width: int, height: int, seed: int) -> tuple[GridTile, BBox]:
    """Build a synthetic :class:`GridTile` and its world bbox.

    Anchored the same way :mod:`tests.tiles3d.conftest`'s ``make_ortho`` /
    ``grid_for_ortho`` anchor their raster (top-left at ``(MINX, MAXY)``), so
    a grid built here matches a standalone-built ortho+EXR pair pixel for
    pixel when given the same ``width``/``height``/``seed``.
    """
    rgb = synth_rgb(width, height, seed)
    grid = grid_for_ortho(rgb)
    heights = np.ascontiguousarray(grid[:, :, 2].astype(np.float32))
    red = np.ascontiguousarray(rgb[:, :, 0])
    green = np.ascontiguousarray(rgb[:, :, 1])
    blue = np.ascontiguousarray(rgb[:, :, 2])
    tile = GridTile(heights=heights, red=red, green=green, blue=blue)
    bbox: BBox = (MINX, MAXY - height * RES, MINX + width * RES, MAXY)
    return tile, bbox


def _blob(encoded: PipelineEncodedTile, name: str) -> bytes:
    """Return the bytes of the blob named ``name``, or fail the test."""
    for blob in encoded.blobs:
        if blob.name == name:
            return blob.data
    msg = f"no {name!r} blob among {[b.name for b in encoded.blobs]}"
    raise AssertionError(msg)


def _make_ctx(bbox: BBox, workdir: Path, *, level: int = 0) -> TileContext:
    return TileContext(
        key=TileKey(level=level, tx=0, ty=0),
        bbox=bbox,
        halo_m=0.0,
        workdir=workdir,
    )


# --- construction -----------------------------------------------------------


def test_construction_rejects_non_positive_pixel_size() -> None:
    """A zero or negative native pixel size is a configuration error."""
    with pytest.raises(ValueError, match="native_pixel_size_m"):
        Tiles3dSink(profile=Profile.STRICT, native_pixel_size_m=0.0)


def test_construction_rejects_non_finite_pixel_size() -> None:
    """A NaN native pixel size is a configuration error."""
    with pytest.raises(ValueError, match="native_pixel_size_m"):
        Tiles3dSink(profile=Profile.STRICT, native_pixel_size_m=float("nan"))


def test_construction_rejects_negative_levels() -> None:
    """A negative tree depth is a configuration error."""
    with pytest.raises(ValueError, match="levels"):
        Tiles3dSink(
            profile=Profile.STRICT, native_pixel_size_m=0.5, levels=-1
        )


def test_halo_m_is_always_zero() -> None:
    """The sink needs no source overlap: its grid is already sampled."""
    sink = Tiles3dSink(profile=Profile.GAME, native_pixel_size_m=0.5)
    assert sink.halo_m() == 0.0


# --- input validation ---------------------------------------------------


def test_run_rejects_non_grid_tile(tmp_path: Path) -> None:
    """A stage upstream of the sampling stage is a pipeline wiring error."""
    sink = Tiles3dSink(profile=Profile.STRICT, native_pixel_size_m=0.5)
    ctx = _make_ctx((0.0, 0.0, 1.0, 1.0), tmp_path)
    with pytest.raises(PipelineError, match="not a GridTile"):
        sink.run(make_point_tile(), ctx)


def test_payload_rejects_empty_width(tmp_path: Path) -> None:
    """A zero-width grid has no pixels to mesh."""
    tile = GridTile(
        heights=np.zeros((2, 0), dtype=np.float32),
        red=np.zeros((2, 0), dtype=np.uint8),
        green=np.zeros((2, 0), dtype=np.uint8),
        blue=np.zeros((2, 0), dtype=np.uint8),
    )
    ctx = _make_ctx((0.0, 0.0, 1.0, 1.0), tmp_path)
    sink = Tiles3dSink(profile=Profile.STRICT, native_pixel_size_m=0.5)
    with pytest.raises(PipelineError, match="empty grid"):
        sink.run(tile, ctx)


def test_payload_rejects_empty_height(tmp_path: Path) -> None:
    """A zero-height grid has no pixels to mesh."""
    tile = GridTile(
        heights=np.zeros((0, 2), dtype=np.float32),
        red=np.zeros((0, 2), dtype=np.uint8),
        green=np.zeros((0, 2), dtype=np.uint8),
        blue=np.zeros((0, 2), dtype=np.uint8),
    )
    ctx = _make_ctx((0.0, 0.0, 1.0, 1.0), tmp_path)
    sink = Tiles3dSink(profile=Profile.STRICT, native_pixel_size_m=0.5)
    with pytest.raises(PipelineError, match="empty grid"):
        sink.run(tile, ctx)


def test_geometric_error_of_rejects_level_beyond_depth(
    tmp_path: Path,
) -> None:
    """A tile deeper than the configured tree depth is a wiring error."""
    sink = Tiles3dSink(
        profile=Profile.STRICT, native_pixel_size_m=0.5, levels=1
    )
    ctx = _make_ctx((0.0, 0.0, 1.0, 1.0), tmp_path, level=2)
    with pytest.raises(PipelineError, match="deeper than"):
        sink.geometric_error_of(ctx)


# --- byte-identity to the standalone verb -----------------------------------


@pytest.mark.parametrize(
    "profile", _PROFILES, ids=[p.value for p in _PROFILES]
)
def test_run_is_byte_identical_to_standalone_build(
    tmp_path: Path, profile: Profile
) -> None:
    """A single-tile AOI's sink output matches ``build_tiles3d`` byte for byte.

    Also cross-checks :meth:`Tiles3dSink.region_of` /
    :meth:`Tiles3dSink.geometric_error_of` against the standalone build's
    ``tileset.json`` root entry -- for a single leaf tile that entry's own
    region/error *is* the sink's per-tile region/error, no union needed.
    """
    width, height, seed = 6, 6, 11
    rgb = synth_rgb(width, height, seed)
    ortho = make_ortho(tmp_path / "ortho.tif", rgb)
    heights_path = write_exr(tmp_path / "r.exr", grid_for_ortho(rgb))
    out = tmp_path / "out"
    build_tiles3d(ortho, heights_path, out, profile=profile)

    grid, bbox = _grid_tile(width, height, seed)
    ctx = _make_ctx(bbox, tmp_path)
    sink = Tiles3dSink(profile=profile, native_pixel_size_m=RES, levels=0)
    encoded = sink.run(grid, ctx)
    assert isinstance(encoded, PipelineEncodedTile)
    assert encoded.key == ctx.key

    if profile is Profile.STRICT:
        oracle_content = (out / "tiles" / "0-0-0.glb").read_bytes()
        assert _blob(encoded, "geometry") == oracle_content
        assert not any(b.name == "texture" for b in encoded.blobs)
    else:
        primary, texture = pack_blob(
            out / "tiles.hfp", PackTileKey(0, 0, 0, 0)
        )
        assert _blob(encoded, "geometry") == primary
        if texture is None:
            assert not any(b.name == "texture" for b in encoded.blobs)
        else:
            assert _blob(encoded, "texture") == texture

    document = json.loads((out / "tileset.json").read_text())
    root = document["root"]
    assert sink.geometric_error_of(ctx) == root["geometricError"]
    assert list(sink.region_of(grid, ctx)) == root["boundingVolume"]["region"]


def test_run_is_deterministic(tmp_path: Path) -> None:
    """Running the same tile twice yields byte-identical blobs."""
    grid, bbox = _grid_tile(4, 4, seed=7)
    ctx = _make_ctx(bbox, tmp_path)
    sink = Tiles3dSink(profile=Profile.GAME, native_pixel_size_m=RES)
    first = sink.run(grid, ctx)
    second = sink.run(grid, ctx)
    assert isinstance(first, PipelineEncodedTile)
    assert isinstance(second, PipelineEncodedTile)
    assert first.blobs == second.blobs


# --- parent-LOD stride / geometric error / region containment --------------


@pytest.mark.parametrize(
    ("levels", "level", "expected_stride"),
    [(0, 0, 1), (2, 0, 4), (2, 1, 2), (2, 2, 1)],
)
def test_geometric_error_of_matches_quadtree_stride_formula(
    tmp_path: Path, levels: int, level: int, expected_stride: int
) -> None:
    """A tile's stride is ``2 ** (levels - level)``, the quadtree convention."""
    sink = Tiles3dSink(
        profile=Profile.STRICT, native_pixel_size_m=0.5, levels=levels
    )
    ctx = _make_ctx((0.0, 0.0, 1.0, 1.0), tmp_path, level=level)
    expected = tiles3d_geometric_error(expected_stride, 0.5)
    assert sink.geometric_error_of(ctx) == expected


def test_region_of_parent_contains_child_horizontally(tmp_path: Path) -> None:
    """A coarser parent tile's region bounds a nested finer child's, in X/Y.

    The parent and child grids share the same top-left anchor (both built by
    :func:`_grid_tile`), so the child's bbox is a genuine sub-box of the
    parent's; every mesh vertex is a pixel centre strictly inside its own
    bbox, so this containment holds regardless of either tile's sampling
    resolution. Vertical (height) containment is *not* guaranteed by a single
    tile's own region -- that is exactly why the standalone emitter
    (:mod:`ahn_cli.tiles3d.emit`) explicitly unions each child's region into
    its parent's tileset entry (:func:`ahn_cli.tiles3d.tileset.union_region`);
    performing that union across tiles is a later assembly stage's job, not
    this per-tile sink's.
    """
    parent_grid, parent_bbox = _grid_tile(4, 4, seed=21)
    child_grid, child_bbox = _grid_tile(2, 2, seed=22)
    sink = Tiles3dSink(
        profile=Profile.STRICT, native_pixel_size_m=RES, levels=1
    )
    parent_ctx = _make_ctx(parent_bbox, tmp_path, level=0)
    child_ctx = _make_ctx(child_bbox, tmp_path, level=1)
    parent_region = sink.region_of(parent_grid, parent_ctx)
    child_region = sink.region_of(child_grid, child_ctx)
    assert parent_region[0] <= child_region[0]  # west
    assert parent_region[1] <= child_region[1]  # south
    assert parent_region[2] >= child_region[2]  # east
    assert parent_region[3] >= child_region[3]  # north


# --- standalone verifier byte-rebuild on the windowed/per-tile path --------


def test_verifier_accepts_a_windowed_single_tile_strict_build(
    tmp_path: Path,
) -> None:
    """Assembling the sink's output into the standard shape passes ``verify``.

    ``verify_tiles3d`` independently rebuilds every tile from the ortho/EXR
    sources and demands the on-disk bytes match exactly; since the sink's
    blob is already proven byte-identical to that rebuild (see
    ``test_run_is_byte_identical_to_standalone_build``), writing it through
    the sink instead of ``build_tiles3d`` must pass the same verifier.
    """
    width, height, seed = 6, 6, 11
    rgb = synth_rgb(width, height, seed)
    ortho = make_ortho(tmp_path / "ortho.tif", rgb)
    heights_path = write_exr(tmp_path / "r.exr", grid_for_ortho(rgb))

    grid, bbox = _grid_tile(width, height, seed)
    ctx = _make_ctx(bbox, tmp_path)
    sink = Tiles3dSink(
        profile=Profile.STRICT, native_pixel_size_m=RES, levels=0
    )
    encoded = sink.run(grid, ctx)
    assert isinstance(encoded, PipelineEncodedTile)
    region = sink.region_of(grid, ctx)
    error = sink.geometric_error_of(ctx)

    out = tmp_path / "windowed"
    (out / "tiles").mkdir(parents=True)
    (out / "tiles" / "0-0-0.glb").write_bytes(_blob(encoded, "geometry"))
    entry = tile_entry(region, error, "tiles/0-0-0.glb", [], root=True)
    document = tileset_document(entry, RES * 4.0)
    write_tileset(document, out / "tileset.json")

    verify_tiles3d(out, ortho, heights_path, profile=Profile.STRICT)
