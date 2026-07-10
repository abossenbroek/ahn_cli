"""Tests for the prep-context transform/export seam."""

from pathlib import Path

import pytest

from ahn_cli.prep.transform import (
    PrepRequest,
    TransformNotWiredError,
    prepare,
)


def test_prep_request_defaults_to_no_filters_and_no_export(tmp_path: Path) -> None:
    """A bare request selects every class and exports nothing."""
    request = PrepRequest(data_dir=tmp_path)

    assert request.include_classes == ()
    assert request.exclude_classes == ()
    assert request.export_points is False


def test_prepare_refuses_until_a_transform_is_wired(tmp_path: Path) -> None:
    """The seam raises TransformNotWiredError naming the data dir it refused."""
    request = PrepRequest(data_dir=tmp_path / "delft")

    with pytest.raises(TransformNotWiredError) as excinfo:
        prepare(request)

    assert str(tmp_path / "delft") in str(excinfo.value)


def test_transform_not_wired_error_is_a_not_implemented_error() -> None:
    """The typed error is a NotImplementedError so callers can catch broadly."""
    assert issubclass(TransformNotWiredError, NotImplementedError)


def test_prep_request_is_hashable_and_value_typed(tmp_path: Path) -> None:
    """The request is a frozen value object: hashable and equal by value."""
    first = PrepRequest(data_dir=tmp_path, include_classes=(2, 6))
    second = PrepRequest(data_dir=tmp_path, include_classes=(2, 6))

    assert first == second
    assert len({first, second}) == 1
