"""Self-test of the byte-identity harness (`tests.pipeline.harness`).

Proves the shared machinery every later stage test depends on actually works:
an identity stage chain is byte-identical to its input, a mutating stage is
detectably different, payload/file/tree hashing is deterministic, and the
synthetic-AOI builders round-trip through laspy/rasterio. Entirely offline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import laspy
import numpy as np
import rasterio

from ahn_cli.pipeline.model import PointTile, TilePayload
from tests.pipeline.harness import (
    IdentityStage,
    hash_payload,
    hash_tree,
    make_encoded_tile,
    make_grid_tile,
    make_point_tile,
    make_tile_context,
    run_stages,
    serialize_payload,
    sha256_file,
    write_synthetic_laz,
    write_synthetic_ortho,
)

if TYPE_CHECKING:
    from pathlib import Path

    from ahn_cli.pipeline.model import TileContext


@dataclass(frozen=True)
class _AddOneZ:
    """A stage that shifts every point's Z by one metre (a detectable change)."""

    def halo_m(self) -> float:
        """Return the zero halo this stage needs."""
        return 0.0

    def run(
        self,
        tile: TilePayload,
        ctx: TileContext,  # noqa: ARG002 -- transform ignores context
    ) -> TilePayload:
        """Return a new `PointTile` with Z raised by one."""
        assert isinstance(tile, PointTile)
        return PointTile(
            x=tile.x,
            y=tile.y,
            z=np.ascontiguousarray(tile.z + 1.0),
            gps_time=tile.gps_time,
            classification=tile.classification,
            rgb=tile.rgb,
        )


def test_identity_stage_chain_is_byte_identical(tmp_path: Path) -> None:
    """An identity chain leaves the payload hash unchanged (the self-test)."""
    tile = make_point_tile(count=8, with_rgb=True)
    ctx = make_tile_context(tmp_path)
    result = run_stages(tile, ctx, [IdentityStage(), IdentityStage(halo=2.0)])
    assert hash_payload(result) == hash_payload(tile)


def test_mutating_stage_changes_the_hash(tmp_path: Path) -> None:
    """A real transform diverges, so the harness cannot pass by luck."""
    tile = make_point_tile(count=8)
    ctx = make_tile_context(tmp_path)
    result = run_stages(tile, ctx, [_AddOneZ()])
    assert hash_payload(result) != hash_payload(tile)


def test_identity_stage_reports_its_halo() -> None:
    """`IdentityStage.halo_m` returns its configured halo."""
    assert IdentityStage(halo=3.5).halo_m() == 3.5


def test_serialize_is_deterministic_and_distinguishing() -> None:
    """Identical payloads serialize equally; rgb presence is distinguished."""
    assert serialize_payload(make_point_tile(count=5)) == serialize_payload(
        make_point_tile(count=5)
    )
    assert serialize_payload(
        make_point_tile(count=5, with_rgb=True)
    ) != serialize_payload(make_point_tile(count=5))


def test_hash_payload_covers_every_payload_kind() -> None:
    """Grid and encoded payloads hash without error and differ from points."""
    hashes = {
        hash_payload(make_point_tile(count=4)),
        hash_payload(make_grid_tile(height=2, width=2)),
        hash_payload(make_encoded_tile()),
    }
    assert len(hashes) == 3


def test_sha256_file_and_hash_tree(tmp_path: Path) -> None:
    """File hashing is streamed and the tree map is keyed by relative path."""
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.bin").write_bytes(b"alpha")
    (tmp_path / "sub" / "b.bin").write_bytes(b"beta")
    tree = hash_tree(tmp_path)
    assert set(tree) == {"a.bin", "sub/b.bin"}
    assert tree["a.bin"] == sha256_file(tmp_path / "a.bin")


def test_write_synthetic_laz_round_trips(tmp_path: Path) -> None:
    """The synthetic PDRF-6 LAZ writer produces a readable point cloud."""
    points = np.array(
        [[1.0, 2.0, 3.0, 0.5, 2], [4.0, 5.0, 6.0, 0.6, 6]], dtype=np.float64
    )
    path = tmp_path / "cloud.laz"
    write_synthetic_laz(path, points)
    with laspy.open(str(path)) as reader:
        data = reader.read()
    assert len(data.x) == 2


def test_write_synthetic_ortho_round_trips(tmp_path: Path) -> None:
    """The synthetic ortho writer produces a readable EPSG:28992 GeoTIFF."""
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, (3, 4, 5)).astype(np.uint8)
    path = tmp_path / "ortho.tif"
    write_synthetic_ortho(path, rgb, (0.0, 0.0, 5.0, 4.0))
    with rasterio.open(path) as src:
        assert src.count == 3
        assert (src.width, src.height) == (5, 4)
        assert src.crs is not None
