"""Tests for the pipeline streaming contracts (`ahn_cli.pipeline.model`).

Locks the validated shapes every workstream builds against: the unified
:class:`TileKey`, the per-tile :class:`TileContext`, the structure-of-arrays
:class:`PointTile`/:class:`GridTile` payloads, the :class:`EncodedTile` sink
output, and the :class:`Stage` protocol's conformance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from ahn_cli.pipeline.model import (
    EncodedBlob,
    EncodedTile,
    GridTile,
    PointTile,
    Stage,
    TileContext,
    TileKey,
)
from tests.pipeline.harness import (
    IdentityStage,
    make_grid_tile,
    make_point_tile,
    make_tile_key,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_tile_key_defaults_tz_to_zero() -> None:
    """`tz` defaults to 0, the 2.5D depth used by today's tilings."""
    key = TileKey(level=2, tx=1, ty=3)
    assert (key.level, key.tx, key.ty, key.tz) == (2, 1, 3, 0)


def test_tile_key_is_hashable_and_value_equal() -> None:
    """Two keys with equal fields are equal and share a hash (manifest key)."""
    assert TileKey(1, 2, 3, 4) == TileKey(1, 2, 3, 4)
    assert hash(TileKey(1, 2, 3, 4)) == hash(TileKey(1, 2, 3, 4))


@pytest.mark.parametrize(
    ("level", "tx", "ty", "tz"),
    [(-1, 0, 0, 0), (0, -1, 0, 0), (0, 0, -1, 0), (0, 0, 0, -1)],
)
def test_tile_key_rejects_negative_components(
    level: int, tx: int, ty: int, tz: int
) -> None:
    """Any negative level or tile index is rejected."""
    with pytest.raises(ValueError, match="non-negative"):
        TileKey(level=level, tx=tx, ty=ty, tz=tz)


def test_tile_context_accepts_a_valid_tile(tmp_path: Path) -> None:
    """A finite non-negative halo over a valid bbox constructs cleanly."""
    ctx = TileContext(
        key=make_tile_key(),
        bbox=(0.0, 0.0, 10.0, 10.0),
        halo_m=5.0,
        workdir=tmp_path,
    )
    assert ctx.halo_m == 5.0
    assert ctx.workdir == tmp_path


def test_tile_context_allows_zero_halo(tmp_path: Path) -> None:
    """A stage that needs no source overlap may use a zero halo."""
    ctx = TileContext(
        key=make_tile_key(),
        bbox=(0.0, 0.0, 1.0, 1.0),
        halo_m=0.0,
        workdir=tmp_path,
    )
    assert ctx.halo_m == 0.0


def test_tile_context_rejects_degenerate_bbox(tmp_path: Path) -> None:
    """A zero-area bbox is rejected via the domain's bbox validator."""
    with pytest.raises(ValueError, match="minx < maxx"):
        TileContext(
            key=make_tile_key(),
            bbox=(0.0, 0.0, 0.0, 10.0),
            halo_m=1.0,
            workdir=tmp_path,
        )


@pytest.mark.parametrize("halo", [-1.0, float("nan"), float("inf")])
def test_tile_context_rejects_bad_halo(tmp_path: Path, halo: float) -> None:
    """A negative or non-finite halo is rejected."""
    with pytest.raises(ValueError, match="halo_m must be finite"):
        TileContext(
            key=make_tile_key(),
            bbox=(0.0, 0.0, 10.0, 10.0),
            halo_m=halo,
            workdir=tmp_path,
        )


def test_point_tile_accepts_valid_planes_without_rgb() -> None:
    """A contiguous equal-length plane set constructs (rgb optional)."""
    tile = make_point_tile(count=6)
    assert tile.rgb is None
    assert tile.x.shape == (6,)


def test_point_tile_accepts_optional_rgb() -> None:
    """The optional `(n, 3)` colour plane is accepted."""
    tile = make_point_tile(count=6, with_rgb=True)
    assert tile.rgb is not None
    assert tile.rgb.shape == (6, 3)


def test_point_tile_rejects_non_one_dimensional_plane() -> None:
    """A 2-D coordinate plane is rejected."""
    good = make_point_tile(count=4)
    with pytest.raises(ValueError, match="must be 1-D"):
        PointTile(
            x=np.ascontiguousarray(good.x.reshape(2, 2)),
            y=good.y,
            z=good.z,
            gps_time=good.gps_time,
            classification=good.classification,
        )


def test_point_tile_rejects_non_contiguous_plane() -> None:
    """A strided (non-C-contiguous) plane is rejected."""
    strided = np.arange(8, dtype=np.float64)[::2]
    good = make_point_tile(count=4)
    with pytest.raises(ValueError, match="C-contiguous"):
        PointTile(
            x=strided,
            y=good.y,
            z=good.z,
            gps_time=good.gps_time,
            classification=good.classification,
        )


def test_point_tile_rejects_length_mismatch() -> None:
    """Planes of differing length are rejected."""
    good = make_point_tile(count=4)
    with pytest.raises(ValueError, match="does not match the point count"):
        PointTile(
            x=good.x,
            y=np.ascontiguousarray(good.y[:3]),
            z=good.z,
            gps_time=good.gps_time,
            classification=good.classification,
        )


def test_point_tile_rejects_bad_rgb_shape() -> None:
    """An rgb plane that is not `(n, 3)` is rejected."""
    good = make_point_tile(count=4)
    with pytest.raises(ValueError, match="must have shape"):
        PointTile(
            x=good.x,
            y=good.y,
            z=good.z,
            gps_time=good.gps_time,
            classification=good.classification,
            rgb=np.zeros((4, 4), dtype=np.uint16),
        )


def test_point_tile_rejects_non_contiguous_rgb() -> None:
    """A non-contiguous rgb plane is rejected."""
    good = make_point_tile(count=4)
    rgb = np.zeros((4, 6), dtype=np.uint16)[:, ::2]
    with pytest.raises(ValueError, match="rgb must be C-contiguous"):
        PointTile(
            x=good.x,
            y=good.y,
            z=good.z,
            gps_time=good.gps_time,
            classification=good.classification,
            rgb=rgb,
        )


def test_point_tile_uses_identity_equality() -> None:
    """Array-carrying payloads compare by identity, not by value."""
    tile = make_point_tile(count=4)
    assert tile == tile  # noqa: PLR0124 -- identity equality is the contract
    assert tile != make_point_tile(count=4)


def test_grid_tile_accepts_matching_planes() -> None:
    """Equal-shape 2-D contiguous planes construct cleanly."""
    grid = make_grid_tile(height=3, width=5)
    assert grid.heights.shape == (3, 5)


def test_grid_tile_rejects_non_two_dimensional_heights() -> None:
    """A 1-D heights plane is rejected."""
    good = make_grid_tile(height=2, width=2)
    with pytest.raises(ValueError, match="heights must be 2-D"):
        GridTile(
            heights=np.ascontiguousarray(good.heights.reshape(-1)),
            red=good.red,
            green=good.green,
            blue=good.blue,
        )


def test_grid_tile_rejects_non_two_dimensional_colour() -> None:
    """A 1-D colour plane is rejected."""
    good = make_grid_tile(height=2, width=2)
    with pytest.raises(ValueError, match="must be 2-D"):
        GridTile(
            heights=good.heights,
            red=np.zeros(4, dtype=np.uint8),
            green=good.green,
            blue=good.blue,
        )


def test_grid_tile_rejects_shape_mismatch() -> None:
    """A colour plane whose shape differs from heights is rejected."""
    good = make_grid_tile(height=2, width=2)
    with pytest.raises(ValueError, match="does not match the grid shape"):
        GridTile(
            heights=good.heights,
            red=np.zeros((2, 3), dtype=np.uint8),
            green=good.green,
            blue=good.blue,
        )


def test_grid_tile_rejects_non_contiguous_colour() -> None:
    """A non-contiguous colour plane of the right shape is rejected."""
    good = make_grid_tile(height=2, width=2)
    red = np.zeros((2, 4), dtype=np.uint8)[:, ::2]
    with pytest.raises(ValueError, match="must be C-contiguous"):
        GridTile(
            heights=good.heights,
            red=red,
            green=good.green,
            blue=good.blue,
        )


def test_encoded_blob_rejects_blank_name() -> None:
    """An empty or whitespace blob name is rejected."""
    with pytest.raises(ValueError, match="non-blank"):
        EncodedBlob(name="  ", data=b"x")


def test_encoded_tile_requires_a_blob() -> None:
    """An encoded tile must carry at least one blob."""
    with pytest.raises(ValueError, match="at least one blob"):
        EncodedTile(key=make_tile_key(), blobs=())


def test_encoded_tile_is_value_equal() -> None:
    """Encoded tiles compare and hash by value (no numpy planes)."""
    blob = EncodedBlob(name="geometry", data=b"g")
    left = EncodedTile(key=make_tile_key(), blobs=(blob,))
    right = EncodedTile(key=make_tile_key(), blobs=(blob,))
    assert left == right
    assert hash(left) == hash(right)


def test_identity_stage_conforms_to_the_stage_protocol() -> None:
    """A class with `halo_m`/`run` is a `Stage` (runtime-checkable protocol)."""
    assert isinstance(IdentityStage(), Stage)


def test_plain_object_is_not_a_stage() -> None:
    """An object missing the stage methods is not a `Stage`."""
    assert not isinstance(object(), Stage)
