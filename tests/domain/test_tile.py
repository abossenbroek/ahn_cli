"""Tests for the :class:`Tile` value object and the shared bbox validator."""

import pytest

from ahn_cli.domain import (
    Generation,
    Product,
    Tile,
    Vintage,
    ensure_valid_bbox,
)

_BBOX = (0.0, 0.0, 10.0, 10.0)


def test_ensure_valid_bbox_accepts_positive_area_box() -> None:
    """A box with positive area passes validation."""
    assert ensure_valid_bbox(_BBOX) is None


def test_ensure_valid_bbox_rejects_inverted_x() -> None:
    """A box whose x extent is empty or inverted is rejected."""
    with pytest.raises(ValueError, match="minx < maxx"):
        ensure_valid_bbox((10.0, 0.0, 10.0, 10.0))


def test_ensure_valid_bbox_rejects_inverted_y() -> None:
    """A box whose y extent is empty or inverted is rejected."""
    with pytest.raises(ValueError, match="minx < maxx"):
        ensure_valid_bbox((0.0, 10.0, 10.0, 10.0))


def test_tile_with_generation_axis_is_valid() -> None:
    """An AHN-family tile is pinned by a generation."""
    tile = Tile(
        tile_id="37FN2",
        product=Product.AHN_POINT_CLOUD,
        bbox=_BBOX,
        generation=Generation(4),
    )
    assert tile.generation == Generation(4)
    assert tile.vintage is None


def test_tile_with_vintage_axis_is_valid() -> None:
    """A dated-imagery tile is pinned by a vintage."""
    tile = Tile(
        tile_id="2023_37FN2",
        product=Product.ORTHO,
        bbox=_BBOX,
        vintage=Vintage(2023),
    )
    assert tile.vintage == Vintage(2023)
    assert tile.generation is None


def test_tile_rejects_blank_identifier() -> None:
    """A whitespace-only tile id is not an identity."""
    with pytest.raises(ValueError, match="non-blank identifier"):
        Tile(
            tile_id="   ",
            product=Product.AHN_POINT_CLOUD,
            bbox=_BBOX,
            generation=Generation(4),
        )


def test_tile_rejects_degenerate_bbox() -> None:
    """A degenerate extent is rejected via the shared validator."""
    with pytest.raises(ValueError, match="minx < maxx"):
        Tile(
            tile_id="37FN2",
            product=Product.AHN_POINT_CLOUD,
            bbox=(10.0, 0.0, 0.0, 10.0),
            generation=Generation(4),
        )


def test_tile_rejects_missing_temporal_axis() -> None:
    """A tile with neither generation nor vintage has no temporal axis."""
    with pytest.raises(ValueError, match="exactly one temporal axis"):
        Tile(tile_id="37FN2", product=Product.DSM, bbox=_BBOX)


def test_tile_rejects_both_temporal_axes() -> None:
    """A tile cannot be pinned to both a generation and a vintage."""
    with pytest.raises(ValueError, match="exactly one temporal axis"):
        Tile(
            tile_id="37FN2",
            product=Product.DSM,
            bbox=_BBOX,
            generation=Generation(4),
            vintage=Vintage(2023),
        )


def test_tile_equality_and_hash_are_value_based() -> None:
    """Structurally identical tiles are equal, hash equal, and dedupe in sets."""
    first = Tile(
        tile_id="37FN2",
        product=Product.AHN_POINT_CLOUD,
        bbox=_BBOX,
        generation=Generation(4),
    )
    second = Tile(
        tile_id="37FN2",
        product=Product.AHN_POINT_CLOUD,
        bbox=_BBOX,
        generation=Generation(4),
    )
    assert first == second
    assert len({first, second}) == 1
