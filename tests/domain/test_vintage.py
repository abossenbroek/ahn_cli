"""Tests for the :class:`Vintage` value object."""

import pytest

from ahn_cli.domain import Vintage


def test_vintage_carries_its_acquisition_year() -> None:
    """A vintage records the four-digit acquisition year."""
    assert Vintage(2023).year == 2023


def test_vintage_year_1900_is_the_lowest_valid_year() -> None:
    """The boundary year ``1900`` is accepted."""
    assert Vintage(1900).year == 1900


def test_vintage_rejects_implausible_years() -> None:
    """A year before ``1900`` cannot be a real acquisition year."""
    with pytest.raises(ValueError, match="plausible acquisition year"):
        Vintage(1899)


def test_vintage_equality_and_hash_are_value_based() -> None:
    """Equal years compare equal and hash equal; different ones do not."""
    assert Vintage(2024) == Vintage(2024)
    assert hash(Vintage(2024)) == hash(Vintage(2024))
    assert Vintage(2023) != Vintage(2024)
