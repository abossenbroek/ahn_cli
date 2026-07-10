"""Tests for the fetch-context acquisition seam."""

from pathlib import Path

import pytest

from ahn_cli.domain import Generation
from ahn_cli.fetch.acquisition import (
    SITE_SUBDIRS,
    AcquisitionRequest,
    AreaSelectorKind,
    SourceNotWiredError,
    acquire,
    create_site_layout,
)


def test_site_subdirs_are_the_three_canonical_products() -> None:
    """The layout is fixed to the ahn/ortho/viirs product subdirectories."""
    assert SITE_SUBDIRS == ("ahn", "ortho", "viirs")


def test_create_site_layout_makes_every_subdir_in_order(
    tmp_path: Path,
) -> None:
    """The layout creates one directory per product, in canonical order."""
    site = tmp_path / "delft"

    created = create_site_layout(site)

    assert created == tuple(site / name for name in SITE_SUBDIRS)
    for subdir in created:
        assert subdir.is_dir()


def test_create_site_layout_is_idempotent(tmp_path: Path) -> None:
    """Re-running on an existing site leaves it intact and does not raise."""
    site = tmp_path / "delft"
    create_site_layout(site)

    created_again = create_site_layout(site)

    assert created_again == tuple(site / name for name in SITE_SUBDIRS)
    for subdir in created_again:
        assert subdir.is_dir()


def test_acquire_refuses_until_a_fetcher_is_wired(tmp_path: Path) -> None:
    """The seam raises SourceNotWiredError naming the site it refused."""
    request = AcquisitionRequest(
        site_dir=tmp_path / "delft",
        selector=AreaSelectorKind.CITY,
        area="delft",
    )

    with pytest.raises(SourceNotWiredError) as excinfo:
        acquire(request)

    assert str(tmp_path / "delft") in str(excinfo.value)


def test_source_not_wired_error_is_a_not_implemented_error() -> None:
    """The typed error is a NotImplementedError so callers can catch broadly."""
    assert issubclass(SourceNotWiredError, NotImplementedError)


def test_acquisition_request_is_hashable_and_value_typed(
    tmp_path: Path,
) -> None:
    """The request is a frozen value object: hashable and equal by value."""
    first = AcquisitionRequest(
        site_dir=tmp_path,
        selector=AreaSelectorKind.BBOX,
        area="0,0,1,1",
    )
    second = AcquisitionRequest(
        site_dir=tmp_path,
        selector=AreaSelectorKind.BBOX,
        area="0,0,1,1",
    )

    assert first == second
    assert len({first, second}) == 1


def test_acquisition_request_generation_defaults_to_none(
    tmp_path: Path,
) -> None:
    """The generation is optional: it defaults to None (auto at download)."""
    request = AcquisitionRequest(
        site_dir=tmp_path,
        selector=AreaSelectorKind.CITY,
        area="delft",
    )

    assert request.generation is None


def test_acquisition_request_carries_an_explicit_generation(
    tmp_path: Path,
) -> None:
    """An explicit generation is carried and keeps the request value-typed."""
    request = AcquisitionRequest(
        site_dir=tmp_path,
        selector=AreaSelectorKind.CITY,
        area="delft",
        generation=Generation(4),
    )

    assert request.generation == Generation(4)
    assert len({request}) == 1
