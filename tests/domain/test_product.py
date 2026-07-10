"""Tests for the :class:`Product` value object."""

from ahn_cli.domain import Product


def test_product_has_exactly_the_four_dataset_kinds() -> None:
    """The closed membership is exactly the four spec'd dataset kinds."""
    assert {member.name for member in Product} == {
        "AHN_POINT_CLOUD",
        "ORTHO",
        "DSM",
        "VIIRS",
    }


def test_product_values_are_stable_canonical_codes() -> None:
    """Each member exposes its canonical string code as ``value``."""
    assert Product.AHN_POINT_CLOUD.value == "ahn_point_cloud"
    assert Product.ORTHO.value == "ortho"
    assert Product.DSM.value == "dsm"
    assert Product.VIIRS.value == "viirs"


def test_product_members_are_hashable_and_distinct() -> None:
    """Members are usable as set keys and are mutually distinct."""
    assert len(set(Product)) == len(list(Product))
    assert Product.DSM != Product.AHN_POINT_CLOUD
