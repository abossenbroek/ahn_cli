"""Tests for the pipeline context's single typed error."""

from __future__ import annotations

import pytest

from ahn_cli.pipeline import PipelineError


def test_pipeline_error_is_an_exception() -> None:
    """`PipelineError` subclasses `Exception` so callers can catch it plainly."""
    assert issubclass(PipelineError, Exception)


def test_pipeline_error_carries_its_message() -> None:
    """A raised `PipelineError` round-trips the message it was given."""
    message = "spec is invalid"
    with pytest.raises(PipelineError, match=message):
        raise PipelineError(message)
