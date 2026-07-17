"""Tests for the cloud/:class:`GridWriteSink` and its grid-blob codec."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from ahn_cli.pipeline.errors import PipelineError
from ahn_cli.pipeline.model import EncodedTile, TileContext, TileKey
from ahn_cli.pipeline.stages.write import (
    GRID_BLOB_NAME,
    GridWriteSink,
    decode_grid_blob,
)
from tests.pipeline.harness import make_grid_tile, make_point_tile

if TYPE_CHECKING:
    from pathlib import Path


def _ctx(workdir: Path) -> TileContext:
    return TileContext(
        key=TileKey(level=0, tx=1, ty=2),
        bbox=(0.0, 0.0, 4.0, 4.0),
        halo_m=0.0,
        workdir=workdir,
    )


def test_halo_is_zero() -> None:
    """The grid is already sampled; no source overlap needed."""
    assert GridWriteSink().halo_m() == 0.0


def test_round_trip(tmp_path: Path) -> None:
    """A grid encoded by the sink decodes back to equal planes."""
    grid = make_grid_tile(height=3, width=5, seed=1)
    encoded = GridWriteSink().run(grid, _ctx(tmp_path))
    assert isinstance(encoded, EncodedTile)
    assert encoded.key == TileKey(level=0, tx=1, ty=2)
    assert [b.name for b in encoded.blobs] == [GRID_BLOB_NAME]
    back = decode_grid_blob(encoded.blobs[0].data)
    assert back.heights.tobytes() == grid.heights.tobytes()
    assert back.red.tobytes() == grid.red.tobytes()
    assert back.green.tobytes() == grid.green.tobytes()
    assert back.blue.tobytes() == grid.blue.tobytes()


def test_deterministic(tmp_path: Path) -> None:
    """Two encodings of the same grid are byte-identical."""
    grid = make_grid_tile(seed=2)
    first = GridWriteSink().run(grid, _ctx(tmp_path))
    second = GridWriteSink().run(grid, _ctx(tmp_path))
    assert isinstance(first, EncodedTile)
    assert isinstance(second, EncodedTile)
    assert first.blobs == second.blobs


def test_non_grid_tile_is_an_error(tmp_path: Path) -> None:
    """A point payload before the reconcile stage is a wiring error."""
    with pytest.raises(PipelineError, match="not a GridTile"):
        GridWriteSink().run(make_point_tile(), _ctx(tmp_path))


def test_decode_rejects_bad_magic() -> None:
    """A blob without the magic prefix is refused."""
    with pytest.raises(PipelineError, match="bad magic"):
        decode_grid_blob(b"XXXX" + b"\x00" * 8)


def test_decode_rejects_length_mismatch(tmp_path: Path) -> None:
    """A truncated blob is refused."""
    grid = make_grid_tile(seed=3)
    encoded = GridWriteSink().run(grid, _ctx(tmp_path))
    assert isinstance(encoded, EncodedTile)
    with pytest.raises(PipelineError, match="does not match"):
        decode_grid_blob(encoded.blobs[0].data[:-1])
