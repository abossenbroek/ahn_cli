"""Tests for :class:`ahn_cli.cache.CacheKey` derivation and digest stability."""

import pytest

from ahn_cli.cache import CacheKey
from ahn_cli.domain import Generation, Product, Tile, Vintage

_BBOX = (0.0, 0.0, 10.0, 10.0)

# Pinned digests of the canonical key encoding. Hard-coded (not recomputed from
# the implementation) so the test fails if the canonical byte encoding ever
# drifts -- these are the guarantee that a key hashes identically across
# processes and releases.
_GEN_DIGEST = (
    "b3003f0577cad3dc916d105f5890a28d8fa476eb69f0bff664269f7c3544927e"
)
_VIN_DIGEST = (
    "03ee24f61ace7ef121e7cd8023147ffdf8af19dc37a80c63eedee26afa264fb5"
)


def _generation_tile() -> Tile:
    """Return a canonical AHN-family (generation-pinned) tile."""
    return Tile(
        tile_id="37FN2",
        product=Product.AHN_POINT_CLOUD,
        bbox=_BBOX,
        generation=Generation(4),
    )


def _vintage_tile() -> Tile:
    """Return a canonical dated-imagery (vintage-pinned) tile."""
    return Tile(
        tile_id="2023_37FN2",
        product=Product.ORTHO,
        bbox=_BBOX,
        vintage=Vintage(2023),
    )


def test_from_tile_preserves_identity_fields() -> None:
    """The derived key carries the tile's product, id, and temporal axis."""
    tile = _generation_tile()
    key = CacheKey.from_tile(tile)
    assert key.product == Product.AHN_POINT_CLOUD
    assert key.tile_id == "37FN2"
    assert key.generation == Generation(4)
    assert key.vintage is None


def test_from_tile_preserves_vintage_axis() -> None:
    """A dated-imagery tile derives a vintage-pinned key."""
    key = CacheKey.from_tile(_vintage_tile())
    assert key.vintage == Vintage(2023)
    assert key.generation is None


def test_generation_key_digest_is_pinned() -> None:
    """A generation-pinned key hashes to its stable, cross-process digest."""
    assert CacheKey.from_tile(_generation_tile()).digest() == _GEN_DIGEST


def test_vintage_key_digest_is_pinned() -> None:
    """A vintage-pinned key hashes to its stable, cross-process digest."""
    assert CacheKey.from_tile(_vintage_tile()).digest() == _VIN_DIGEST


def test_digest_is_stable_across_repeated_derivation() -> None:
    """Deriving and hashing the same tile twice yields the identical digest."""
    first = CacheKey.from_tile(_generation_tile()).digest()
    second = CacheKey.from_tile(_generation_tile()).digest()
    assert first == second == _GEN_DIGEST


def test_equal_tiles_produce_equal_digests() -> None:
    """Two structurally equal tiles derive equal keys and equal digests."""
    key_a = CacheKey.from_tile(_generation_tile())
    key_b = CacheKey.from_tile(_generation_tile())
    assert key_a == key_b
    assert key_a.digest() == key_b.digest() == _GEN_DIGEST


def test_generation_and_vintage_axes_do_not_collide() -> None:
    """Generation- and vintage-pinned keys never share a digest."""
    gen_digest = CacheKey.from_tile(_generation_tile()).digest()
    vin_digest = CacheKey.from_tile(_vintage_tile()).digest()
    assert gen_digest != vin_digest


def test_different_tile_id_yields_different_digest() -> None:
    """Changing only the tile id changes the digest."""
    base = CacheKey.from_tile(_generation_tile()).digest()
    other = CacheKey(
        product=Product.AHN_POINT_CLOUD,
        tile_id="37FN3",
        generation=Generation(4),
    ).digest()
    assert base != other


def test_different_product_yields_different_digest() -> None:
    """Changing only the product changes the digest."""
    ahn = CacheKey(
        product=Product.AHN_POINT_CLOUD,
        tile_id="37FN2",
        generation=Generation(4),
    ).digest()
    dsm = CacheKey(
        product=Product.DSM,
        tile_id="37FN2",
        generation=Generation(4),
    ).digest()
    assert ahn != dsm


def test_cachekey_rejects_blank_tile_id() -> None:
    """A whitespace-only tile id is not a valid key identity."""
    with pytest.raises(ValueError, match="non-blank"):
        CacheKey(
            product=Product.AHN_POINT_CLOUD,
            tile_id="   ",
            generation=Generation(4),
        )


def test_cachekey_rejects_missing_temporal_axis() -> None:
    """A key with neither generation nor vintage has no temporal axis."""
    with pytest.raises(ValueError, match="exactly one temporal axis"):
        CacheKey(product=Product.DSM, tile_id="37FN2")


def test_cachekey_rejects_both_temporal_axes() -> None:
    """A key cannot be pinned to both a generation and a vintage."""
    with pytest.raises(ValueError, match="exactly one temporal axis"):
        CacheKey(
            product=Product.DSM,
            tile_id="37FN2",
            generation=Generation(4),
            vintage=Vintage(2023),
        )
