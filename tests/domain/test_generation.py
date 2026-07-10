"""Tests for the :class:`Generation` value object."""

import pytest

from ahn_cli.domain import Generation


def test_generation_exposes_number_and_family_code() -> None:
    """A generation carries its ordinal and derives its family code."""
    generation = Generation(4)
    assert generation.number == 4
    assert generation.code == "AHN4"


def test_generation_number_one_is_the_lowest_valid_ordinal() -> None:
    """The boundary ordinal ``1`` is accepted."""
    assert Generation(1).code == "AHN1"


def test_generation_rejects_non_positive_ordinals() -> None:
    """A generation below ``1`` is not a real AHN generation."""
    with pytest.raises(ValueError, match="positive integer"):
        Generation(0)


def test_generation_rejects_booleans() -> None:
    """A ``bool`` (an int subclass) is not a valid generation ordinal."""
    boolean_ordinal = True
    with pytest.raises(ValueError, match="not a bool"):
        Generation(boolean_ordinal)


def test_generation_equality_and_hash_are_value_based() -> None:
    """Equal ordinals compare equal and hash equal; different ones do not."""
    assert Generation(5) == Generation(5)
    assert hash(Generation(5)) == hash(Generation(5))
    assert Generation(4) != Generation(5)
