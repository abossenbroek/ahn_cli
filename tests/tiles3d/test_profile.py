"""Tests for the tiles3d export ``Profile`` value object."""

from __future__ import annotations

import pytest

from ahn_cli.tiles3d.encoders import (
    GameEncoder,
    HeightfieldEncoder,
    StrictEncoder,
)
from ahn_cli.tiles3d.errors import Tiles3dError
from ahn_cli.tiles3d.profile import Profile


def test_profile_values() -> None:
    """The members carry their CLI string values."""
    assert Profile.STRICT.value == "strict"
    assert Profile.GAME.value == "game"
    assert Profile.HEIGHTFIELD.value == "heightfield"


def test_encoder_maps_each_profile_to_its_encoder() -> None:
    """``encoder`` returns the matching encoder instance per member."""
    assert isinstance(Profile.STRICT.encoder(), StrictEncoder)
    assert isinstance(Profile.GAME.encoder(), GameEncoder)
    assert isinstance(Profile.HEIGHTFIELD.encoder(), HeightfieldEncoder)


def test_content_and_texture_suffixes() -> None:
    """Each profile names its content and (optional) texture extensions."""
    assert Profile.STRICT.content_suffix() == ".glb"
    assert Profile.GAME.content_suffix() == ".glb"
    assert Profile.HEIGHTFIELD.content_suffix() == ".hf"
    assert Profile.STRICT.texture_suffix() is None
    assert Profile.GAME.texture_suffix() is None
    assert Profile.HEIGHTFIELD.texture_suffix() == ".jpg"


def test_parse_returns_the_named_member() -> None:
    """``parse`` maps each valid string to its member."""
    assert Profile.parse("strict") is Profile.STRICT
    assert Profile.parse("game") is Profile.GAME
    assert Profile.parse("heightfield") is Profile.HEIGHTFIELD


def test_parse_rejects_an_unknown_profile() -> None:
    """An unknown name raises ``Tiles3dError`` naming the valid choices."""
    with pytest.raises(Tiles3dError, match="unknown tiles3d profile 'bogus'"):
        Profile.parse("bogus")
