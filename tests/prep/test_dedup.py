"""Tests for the prep-context tile-deduplication transform (WP10).

The fixtures are small, synthetic LAZ tiles built in-process with laspy -- no
network, no large files. They exercise the two mandatory stages (crop before
merge, then the exact XYZ+GPS-time sweep), the no-overlap passthrough, the
offset-harmonizing merge, and byte-level determinism.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import laspy
import numpy as np
import numpy.typing as npt
import pytest

from ahn_cli.prep.dedup import CanonicalTile, DedupStats, deduplicate_tiles

if TYPE_CHECKING:
    from pathlib import Path

    from ahn_cli.domain import BBox

Point = tuple[float, float, float, float]  # (x, y, z, gps_time)


def _write_tile(
    path: Path,
    points: list[Point],
    *,
    offsets: tuple[float, float, float] = (0.0, 0.0, 0.0),
    scales: tuple[float, float, float] = (0.01, 0.01, 0.01),
) -> None:
    """Write a synthetic format-6 (gps_time-bearing) tile to ``path``."""
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.offsets = np.array(offsets, dtype=float)
    header.scales = np.array(scales, dtype=float)
    las = laspy.LasData(header)
    arr = np.array(points, dtype=float)
    las.x = arr[:, 0]
    las.y = arr[:, 1]
    las.z = arr[:, 2]
    las.gps_time = arr[:, 3]
    las.write(str(path))


def _read_points(path: Path) -> laspy.LasData:
    """Read a LAZ/LAS file fully into memory."""
    with laspy.open(str(path)) as reader:
        return reader.read()


def _xyzt_keys(las: laspy.LasData) -> npt.NDArray[np.void]:
    """Return the exact (X, Y, Z, gps_time) dedup keys of every point."""
    return np.rec.fromarrays([las.X, las.Y, las.Z, las.gps_time])


def _exact_duplicate_count(las: laspy.LasData) -> int:
    """Return how many points share an exact XYZ+GPS-time key with another."""
    keys = _xyzt_keys(las)
    return len(keys) - len(np.unique(keys))


def _world_xy(las: laspy.LasData) -> set[tuple[float, float]]:
    """Return the rounded world (x, y) coordinates present in ``las``."""
    return {
        (round(float(x), 3), round(float(y), 3))
        for x, y in zip(las.x, las.y, strict=True)
    }


# --------------------------------------------------------------------------
# Value objects
# --------------------------------------------------------------------------


def test_canonical_tile_is_a_frozen_value_object(tmp_path: Path) -> None:
    """CanonicalTile is hashable and equal by field value."""
    extent: BBox = (0.0, 0.0, 10.0, 10.0)
    first = CanonicalTile(path=tmp_path / "a.laz", extent=extent)
    second = CanonicalTile(path=tmp_path / "a.laz", extent=extent)

    assert first == second
    assert len({first, second}) == 1
    assert first.extent == extent


def test_canonical_tile_rejects_a_degenerate_extent(tmp_path: Path) -> None:
    """A zero-area extent is refused at construction."""
    with pytest.raises(ValueError, match="minx < maxx"):
        CanonicalTile(path=tmp_path / "a.laz", extent=(10.0, 0.0, 10.0, 10.0))


def test_dedup_stats_is_a_frozen_value_object() -> None:
    """DedupStats is equal by field value."""
    stats = DedupStats(
        input_points=7,
        cropped_points=7,
        duplicates_removed=0,
        output_points=7,
    )

    assert stats == DedupStats(7, 7, 0, 7)


# --------------------------------------------------------------------------
# Guard
# --------------------------------------------------------------------------


def test_deduplicate_tiles_rejects_an_empty_tile_list(tmp_path: Path) -> None:
    """With no tiles there is nothing to merge, so the call is refused."""
    with pytest.raises(ValueError, match="at least one tile"):
        deduplicate_tiles([], tmp_path / "out.laz")


# --------------------------------------------------------------------------
# Crop before merge (spec integration test)
# --------------------------------------------------------------------------


def test_crop_before_merge_drops_the_seam_and_preserves_interior(
    tmp_path: Path,
) -> None:
    """A seam band is cropped; interior points survive; no double-count.

    Tile A owns ``[0, 10)`` in x, tile B owns ``[10, 20)``. Each file also holds
    points from the shared overlap band. The half-open crop must drop every
    band point (including the shared ``x == 10`` point from A's side, which B
    owns), leave every interior point intact, and produce exactly the analytic
    merged count with zero exact duplicates.
    """
    tile_a = tmp_path / "a.laz"
    tile_b = tmp_path / "b.laz"
    # A interior: (1,1),(5,5),(9,9); A seam band (cropped): (10,5),(11,5).
    _write_tile(
        tile_a,
        [
            (1.0, 1.0, 0.0, 1.0),
            (5.0, 5.0, 0.0, 2.0),
            (9.0, 9.0, 0.0, 3.0),
            (10.0, 5.0, 0.0, 10.0),
            (11.0, 5.0, 0.0, 11.0),
        ],
    )
    # B interior/edge: (10,5),(12,5),(15,5),(19,9); B seam band (cropped): (9,5).
    _write_tile(
        tile_b,
        [
            (10.0, 5.0, 0.0, 10.0),
            (12.0, 5.0, 0.0, 12.0),
            (15.0, 5.0, 0.0, 15.0),
            (19.0, 9.0, 0.0, 19.0),
            (9.0, 5.0, 0.0, 9.0),
        ],
    )
    tiles = [
        CanonicalTile(path=tile_a, extent=(0.0, 0.0, 10.0, 10.0)),
        CanonicalTile(path=tile_b, extent=(10.0, 0.0, 20.0, 10.0)),
    ]
    out = tmp_path / "out.laz"

    stats = deduplicate_tiles(tiles, out)

    # A keeps 3 interior; B keeps 4 (10,12,15,19); analytic merged = 7.
    assert stats.input_points == 10
    assert stats.cropped_points == 7
    assert stats.duplicates_removed == 0
    assert stats.output_points == 7

    result = _read_points(out)
    assert len(result.points) == 7
    assert _exact_duplicate_count(result) == 0
    # Interior fully preserved (no over-crop), seam band fully gone.
    world = _world_xy(result)
    assert {(1.0, 1.0), (5.0, 5.0), (9.0, 9.0)} <= world
    assert {(12.0, 5.0), (15.0, 5.0), (19.0, 9.0), (10.0, 5.0)} <= world
    assert (11.0, 5.0) not in world
    assert (9.0, 5.0) not in world


# --------------------------------------------------------------------------
# Post-merge exact-duplicate sweep
# --------------------------------------------------------------------------


def test_sweep_removes_exact_duplicates_surviving_the_crop(
    tmp_path: Path,
) -> None:
    """A point present in two overlapping tiles is kept exactly once.

    This is the sweep's ``insurance`` role: two tiles whose canonical extents
    both claim the same point (e.g. the same tile ingested twice) share an exact
    XYZ+GPS-time duplicate that the crop cannot remove; the sweep must.
    """
    tile_a = tmp_path / "a.laz"
    tile_b = tmp_path / "b.laz"
    _write_tile(
        tile_a,
        [(2.0, 2.0, 0.0, 1.0), (5.0, 5.0, 0.0, 5.0)],
    )
    _write_tile(
        tile_b,
        [(5.0, 5.0, 0.0, 5.0), (8.0, 8.0, 0.0, 8.0)],
    )
    extent: BBox = (0.0, 0.0, 10.0, 10.0)
    tiles = [
        CanonicalTile(path=tile_a, extent=extent),
        CanonicalTile(path=tile_b, extent=extent),
    ]
    out = tmp_path / "out.laz"

    stats = deduplicate_tiles(tiles, out)

    assert stats.cropped_points == 4
    assert stats.duplicates_removed == 1
    assert stats.output_points == 3

    result = _read_points(out)
    assert _exact_duplicate_count(result) == 0
    # First occurrence kept in input order: A(2,2), A(5,5), B(8,8).
    assert [round(float(v), 3) for v in result.x] == [2.0, 5.0, 8.0]


# --------------------------------------------------------------------------
# No-overlap passthrough
# --------------------------------------------------------------------------


def test_disjoint_tiles_pass_through_untouched(tmp_path: Path) -> None:
    """Disjoint tiles lose no points and gain no duplicate removals."""
    tile_a = tmp_path / "a.laz"
    tile_b = tmp_path / "b.laz"
    _write_tile(
        tile_a,
        [(1.0, 1.0, 0.0, 1.0), (2.0, 2.0, 0.0, 2.0)],
    )
    _write_tile(
        tile_b,
        [(12.0, 5.0, 0.0, 12.0), (15.0, 5.0, 0.0, 15.0)],
    )
    tiles = [
        CanonicalTile(path=tile_a, extent=(0.0, 0.0, 10.0, 10.0)),
        CanonicalTile(path=tile_b, extent=(10.0, 0.0, 20.0, 10.0)),
    ]
    out = tmp_path / "out.laz"

    stats = deduplicate_tiles(tiles, out)

    assert stats == DedupStats(
        input_points=4,
        cropped_points=4,
        duplicates_removed=0,
        output_points=4,
    )
    result = _read_points(out)
    assert len(result.points) == 4
    assert _world_xy(result) == {
        (1.0, 1.0),
        (2.0, 2.0),
        (12.0, 5.0),
        (15.0, 5.0),
    }


def test_deduplicate_tiles_reports_progress_per_tile(tmp_path: Path) -> None:
    """Each cropped tile reports (tiles_done, total_tiles) to progress."""
    tile_a = tmp_path / "a.laz"
    tile_b = tmp_path / "b.laz"
    _write_tile(tile_a, [(1.0, 1.0, 0.0, 1.0)])
    _write_tile(tile_b, [(12.0, 5.0, 0.0, 12.0)])
    tiles = [
        CanonicalTile(path=tile_a, extent=(0.0, 0.0, 10.0, 10.0)),
        CanonicalTile(path=tile_b, extent=(10.0, 0.0, 20.0, 10.0)),
    ]
    out = tmp_path / "out.laz"
    calls: list[tuple[int, int]] = []

    deduplicate_tiles(
        tiles, out, progress=lambda done, total: calls.append((done, total))
    )

    assert calls == [(1, 2), (2, 2)]


# --------------------------------------------------------------------------
# Offset-harmonizing merge
# --------------------------------------------------------------------------


def test_merge_harmonizes_differing_tile_offsets(tmp_path: Path) -> None:
    """Tiles with different header offsets merge onto one integer grid.

    The same world point stored under two different LAS offsets must reduce to a
    single point after the sweep -- proving the reused offset correction lands
    both tiles on the harmonized grid before keys are compared.
    """
    tile_a = tmp_path / "a.laz"
    tile_b = tmp_path / "b.laz"
    _write_tile(
        tile_a,
        [(5.0, 5.0, 0.0, 5.0), (2.0, 2.0, 0.0, 2.0)],
        offsets=(0.0, 0.0, 0.0),
    )
    _write_tile(
        tile_b,
        [(5.0, 5.0, 0.0, 5.0), (8.0, 8.0, 0.0, 8.0)],
        offsets=(1000.0, 1000.0, 0.0),
    )
    extent: BBox = (0.0, 0.0, 10.0, 10.0)
    tiles = [
        CanonicalTile(path=tile_a, extent=extent),
        CanonicalTile(path=tile_b, extent=extent),
    ]
    out = tmp_path / "out.laz"

    stats = deduplicate_tiles(tiles, out)

    assert stats.duplicates_removed == 1
    result = _read_points(out)
    assert _exact_duplicate_count(result) == 0
    assert _world_xy(result) == {(5.0, 5.0), (2.0, 2.0), (8.0, 8.0)}


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------


def test_deduplication_is_byte_and_order_deterministic(
    tmp_path: Path,
) -> None:
    """Identical input yields byte-identical output and identical ordering."""
    tile_a = tmp_path / "a.laz"
    tile_b = tmp_path / "b.laz"
    _write_tile(
        tile_a,
        [(2.0, 2.0, 0.0, 1.0), (5.0, 5.0, 0.0, 5.0)],
    )
    _write_tile(
        tile_b,
        [(5.0, 5.0, 0.0, 5.0), (8.0, 8.0, 0.0, 8.0)],
    )
    extent: BBox = (0.0, 0.0, 10.0, 10.0)
    tiles = [
        CanonicalTile(path=tile_a, extent=extent),
        CanonicalTile(path=tile_b, extent=extent),
    ]
    out_first = tmp_path / "first.laz"
    out_second = tmp_path / "second.laz"

    stats_first = deduplicate_tiles(tiles, out_first)
    stats_second = deduplicate_tiles(tiles, out_second)

    assert stats_first == stats_second
    assert out_first.read_bytes() == out_second.read_bytes()

    first = _read_points(out_first)
    second = _read_points(out_second)
    assert np.array_equal(first.X, second.X)
    assert np.array_equal(first.Y, second.Y)
    assert np.array_equal(first.gps_time, second.gps_time)
