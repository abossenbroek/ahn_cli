"""The tiles3d export profile: the value object that selects an encoder.

A :class:`Profile` names one on-disk representation and maps to its
:class:`~ahn_cli.tiles3d.payload.TileEncoder`. It is the single place the
pipeline turns a profile choice into a concrete encoder — everything
downstream stays agnostic to the packing. Stringly-typed input lives only
at the CLI boundary: :meth:`Profile.parse` turns a caller string into a
member, raising the context's typed :class:`~ahn_cli.tiles3d.errors.Tiles3dError`
on an unknown name so no raw ``ValueError`` escapes.
"""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING

from ahn_cli.tiles3d.encoders import (
    GameEncoder,
    HeightfieldEncoder,
    SplatEncoder,
    StrictEncoder,
)
from ahn_cli.tiles3d.errors import Tiles3dError

if TYPE_CHECKING:
    from ahn_cli.tiles3d.payload import TileEncoder

__all__ = ["Profile"]


class Profile(enum.Enum):
    """A tiles3d export profile — the encoder-selecting value object.

    Contract:
        - ``STRICT`` (`"strict"`) is the lossless float32 + PNG profile;
          ``GAME`` (`"game"`) is the quantized + meshopt + JPEG profile;
          ``HEIGHTFIELD`` (`"heightfield"`) is the vendor ``.hf`` height
          chunk + sibling JPEG profile; ``SPLAT`` (`"splat"`) is the 3D
          Gaussian Splatting ``.ply`` profile (no texture; colour lives in
          the gaussians).
        - :meth:`encoder` returns a fresh :class:`TileEncoder` for the
          member; :meth:`parse` turns a CLI string into a member.
        - :meth:`content_suffix` / :meth:`texture_suffix` name the
          on-disk file extensions a profile writes per tile (the texture
          suffix is ``None`` for the embedded-texture glTF profiles), so
          the emitter and verifier resolve tile filenames without
          branching on the profile themselves.

    Invariants:
        - The only mapping from a profile name to an encoder; the rest of
          the pipeline never branches on the profile itself.
    """

    STRICT = "strict"
    GAME = "game"
    HEIGHTFIELD = "heightfield"
    SPLAT = "splat"

    @classmethod
    def parse(cls, text: str) -> Profile:
        """Return the profile named ``text`` (CLI-boundary parsing).

        Failure modes:
            - Raises :class:`~ahn_cli.tiles3d.errors.Tiles3dError` naming
              the unknown value and the valid choices, so no raw
              ``ValueError`` escapes the context.
        """
        try:
            return cls(text)
        except ValueError as exc:
            choices = ", ".join(member.value for member in cls)
            msg = (
                f"unknown tiles3d profile {text!r}; choose one of: {choices}."
            )
            raise Tiles3dError(msg) from exc

    def encoder(self) -> TileEncoder:
        """Return a fresh :class:`TileEncoder` for this profile."""
        return _ENCODERS[self]()

    def content_suffix(self) -> str:
        """Return this profile's per-tile content-file extension."""
        return _CONTENT_SUFFIX[self]

    def texture_suffix(self) -> str | None:
        """Return the sibling texture extension, or ``None`` if embedded."""
        return _TEXTURE_SUFFIX[self]


_ENCODERS = {
    Profile.STRICT: StrictEncoder,
    Profile.GAME: GameEncoder,
    Profile.HEIGHTFIELD: HeightfieldEncoder,
    Profile.SPLAT: SplatEncoder,
}

_CONTENT_SUFFIX = {
    Profile.STRICT: ".glb",
    Profile.GAME: ".glb",
    Profile.HEIGHTFIELD: ".hf",
    Profile.SPLAT: ".ply",
}

_TEXTURE_SUFFIX: dict[Profile, str | None] = {
    Profile.STRICT: None,
    Profile.GAME: None,
    Profile.HEIGHTFIELD: ".jpg",
    Profile.SPLAT: None,
}
