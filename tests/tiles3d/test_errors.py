"""Tests for the tiles3d context's typed error."""

from ahn_cli import tiles3d
from ahn_cli.tiles3d import Tiles3dError
from ahn_cli.tiles3d.errors import Tiles3dError as ErrorsTiles3dError


def test_tiles3d_error_is_the_context_exception() -> None:
    """Tiles3dError is a plain Exception, re-exported at package level."""
    assert issubclass(Tiles3dError, Exception)
    assert Tiles3dError is ErrorsTiles3dError
    assert tiles3d.__all__ == ["Tiles3dError"]
