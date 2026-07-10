"""Tests for the shared fetch-source value objects and CRS helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from ahn_cli.fetch.source import (
    RemoteTile,
    ResolvedFeed,
    SourceKind,
    UnknownSourceError,
    boxes_intersect,
    resolve_source_token,
    source_kind_tokens,
    to_rd,
    to_wgs84,
)

if TYPE_CHECKING:
    from ahn_cli.domain import BBox

# A Delft-area EPSG:28992 box and its rough WGS84 longitude/latitude envelope.
_DELFT_RD: BBox = (84000.0, 447000.0, 85000.0, 448000.0)


def test_source_kind_tokens_lists_pdok_first() -> None:
    """The token list is derived from the enum with PDOK (default) first."""
    assert source_kind_tokens() == ("pdok", "geotiles")


def test_resolve_source_token_maps_known_tokens() -> None:
    """A known token resolves to its SourceKind member."""
    assert resolve_source_token("pdok") is SourceKind.PDOK
    assert resolve_source_token("geotiles") is SourceKind.GEOTILES


def test_resolve_source_token_rejects_unknown() -> None:
    """An unknown token raises the typed UnknownSourceError."""
    with pytest.raises(UnknownSourceError):
        resolve_source_token("wms")


def test_remote_tile_is_value_typed() -> None:
    """A RemoteTile is a frozen, hashable value object equal by field."""
    first = RemoteTile("C_37EN1", (0.0, 0.0, 1.0, 1.0), "https://x/a.LAZ")
    second = RemoteTile("C_37EN1", (0.0, 0.0, 1.0, 1.0), "https://x/a.LAZ")

    assert first == second
    assert len({first, second}) == 1


def test_remote_tile_rejects_blank_id() -> None:
    """A blank sheet id is rejected."""
    with pytest.raises(ValueError, match="tile_id"):
        RemoteTile("  ", (0.0, 0.0, 1.0, 1.0), "https://x/a.LAZ")


def test_remote_tile_rejects_blank_url() -> None:
    """A blank download URL is rejected."""
    with pytest.raises(ValueError, match="download_url"):
        RemoteTile("C_37EN1", (0.0, 0.0, 1.0, 1.0), "   ")


def test_remote_tile_rejects_degenerate_bbox() -> None:
    """A degenerate extent is rejected via the shared validator."""
    with pytest.raises(ValueError, match="bbox"):
        RemoteTile("C_37EN1", (1.0, 0.0, 0.0, 1.0), "https://x/a.LAZ")


def test_resolved_feed_is_value_typed() -> None:
    """A ResolvedFeed is a frozen value object carrying licence and tiles."""
    tile = RemoteTile("C_37EN1", (0.0, 0.0, 1.0, 1.0), "https://x/a.LAZ")
    feed = ResolvedFeed(licence="CC0", attribution="RWS", tiles=(tile,))

    assert feed.tiles == (tile,)
    assert feed == ResolvedFeed(
        licence="CC0", attribution="RWS", tiles=(tile,)
    )


def test_to_wgs84_projects_into_the_dutch_longitude_band() -> None:
    """A Delft RD box projects into plausible WGS84 lon/lat ranges."""
    minlon, minlat, maxlon, maxlat = to_wgs84(_DELFT_RD)

    assert 4.0 < minlon < maxlon < 5.0
    assert 51.5 < minlat < maxlat < 52.5


def test_to_rd_round_trips_with_to_wgs84() -> None:
    """Projecting to WGS84 and back recovers the RD box within a metre."""
    round_tripped = to_rd(to_wgs84(_DELFT_RD))

    for actual, expected in zip(round_tripped, _DELFT_RD, strict=True):
        assert abs(actual - expected) <= 1.0


def test_to_wgs84_rejects_degenerate_box() -> None:
    """A degenerate RD box has no well-defined WGS84 extent."""
    with pytest.raises(ValueError, match="bbox"):
        to_wgs84((1.0, 1.0, 0.0, 0.0))


def test_to_rd_rejects_degenerate_box() -> None:
    """A degenerate WGS84 box has no well-defined RD extent."""
    with pytest.raises(ValueError, match="bbox"):
        to_rd((1.0, 1.0, 0.0, 0.0))


def test_boxes_intersect_true_when_overlapping() -> None:
    """Overlapping boxes intersect."""
    assert boxes_intersect((0.0, 0.0, 2.0, 2.0), (1.0, 1.0, 3.0, 3.0))


def test_boxes_intersect_true_when_touching_on_an_edge() -> None:
    """Edge-touching boxes count as intersecting (touch-inclusive)."""
    assert boxes_intersect((0.0, 0.0, 1.0, 1.0), (1.0, 0.0, 2.0, 1.0))


def test_boxes_intersect_false_when_disjoint_in_x() -> None:
    """Boxes separated along x do not intersect."""
    assert not boxes_intersect((0.0, 0.0, 1.0, 1.0), (2.0, 0.0, 3.0, 1.0))


def test_boxes_intersect_false_when_disjoint_in_y() -> None:
    """Boxes separated along y do not intersect."""
    assert not boxes_intersect((0.0, 0.0, 1.0, 1.0), (0.0, 2.0, 1.0, 3.0))
