"""Tests for the packed-profile deep checks in the post-write verifier.

These exercise the two checks that compare the pack against the demoted
``tileset.json`` / ``manifest.json`` sidecars — the **two-encodings
witness** (pack index vs tileset, bit-for-bit) and the **manifest
recompute** — independently of the container-level rejects
:func:`~ahn_cli.tiles3d.pack.read_pack` already enforces. Each negative
builds a valid packed deliverable, corrupts exactly the bytes one check
guards (repatching the pack's own CRCs/hashes via
:func:`~tests.tiles3d.conftest.rewrite_pack` where the pack itself is the
target, so only the targeted verifier check fires), and asserts its
message. Both content kinds (game ``.glb`` / heightfield ``.hf``) are run.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from typing import TYPE_CHECKING, Any, cast

import pytest

from ahn_cli.tiles3d.build import build_tiles3d
from ahn_cli.tiles3d.errors import Tiles3dError
from ahn_cli.tiles3d.manifest import parse_manifest
from ahn_cli.tiles3d.pack import read_pack
from ahn_cli.tiles3d.profile import Profile
from ahn_cli.tiles3d.verify import verify_tiles3d
from tests.tiles3d.conftest import (
    grid_for_ortho,
    make_ortho,
    rewrite_pack,
    synth_rgb,
    write_exr,
)

if TYPE_CHECKING:
    from pathlib import Path

_Site = tuple["Path", "Path", "Path", Profile]


@pytest.fixture(params=[Profile.GAME, Profile.HEIGHTFIELD])
def packed_site(request: pytest.FixtureRequest, tmp_path: Path) -> _Site:
    """Build a valid two-level packed tileset for each lossy profile."""
    profile = cast("Profile", request.param)
    rgb = synth_rgb(20, 14, seed=13)
    ortho = make_ortho(tmp_path / "ortho.tif", rgb)
    heights = write_exr(tmp_path / "r.exr", grid_for_ortho(rgb))
    out = tmp_path / "out"
    build_tiles3d(ortho, heights, out, tile_pixels=8, profile=profile)
    return out, ortho, heights, profile


def _verify(site: _Site) -> None:
    out, ortho, heights, profile = site
    verify_tiles3d(out, ortho, heights, tile_pixels=8, profile=profile)


def _load_tileset(out: Path) -> dict[str, Any]:
    return cast(
        "dict[str, Any]", json.loads((out / "tileset.json").read_text())
    )


def _dump_tileset(out: Path, document: object) -> None:
    (out / "tileset.json").write_text(
        json.dumps(document, sort_keys=True, indent=2) + "\n"
    )


def _leaf(document: dict[str, Any]) -> dict[str, Any]:
    return cast(
        "dict[str, Any]", document["root"]["children"][0]["children"][0]
    )


def test_pristine_packed_build_verifies(packed_site: _Site) -> None:
    """The verifier accepts what each packed builder just wrote."""
    _verify(packed_site)


# -- two-encodings witness -------------------------------------------------


def test_flipped_tileset_region_is_refused(packed_site: _Site) -> None:
    """A tileset region double that drifts from the pack index is refused."""
    out = packed_site[0]
    document = _load_tileset(out)
    # Widen the root west (no parent to contain it, children stay enclosed).
    document["root"]["boundingVolume"]["region"][0] -= 0.001
    _dump_tileset(out, document)
    with pytest.raises(
        Tiles3dError, match="tileset region does not bit-equal"
    ):
        _verify(packed_site)


def test_flipped_pack_region_is_refused(packed_site: _Site) -> None:
    """A pack index region double that drifts from the tileset is refused."""
    out = packed_site[0]

    def drop_root_min(entries: list[Any]) -> list[Any]:
        return [
            dataclasses.replace(
                entry,
                region=(
                    *entry.region[:4],
                    entry.region[4] - 1.0,
                    entry.region[5],
                ),
            )
            if entry.key.level == 0
            else entry
            for entry in entries
        ]

    rewrite_pack(out / "tiles.hfp", drop_root_min)
    with pytest.raises(
        Tiles3dError, match="tileset region does not bit-equal"
    ):
        _verify(packed_site)


def test_flipped_pack_geometric_error_is_refused(
    packed_site: _Site,
) -> None:
    """A pack index geometric_error that drifts from the tileset is refused."""
    out = packed_site[0]

    def bump_root_error(entries: list[Any]) -> list[Any]:
        return [
            dataclasses.replace(
                entry, geometric_error=entry.geometric_error + 1.0
            )
            if entry.key.level == 0
            else entry
            for entry in entries
        ]

    rewrite_pack(out / "tiles.hfp", bump_root_error)
    with pytest.raises(
        Tiles3dError, match="geometricError does not bit-equal"
    ):
        _verify(packed_site)


def test_flipped_root_geometric_error_is_refused(
    packed_site: _Site,
) -> None:
    """A pack header root_geometric_error drifting from the tileset is caught."""
    out = packed_site[0]
    original = read_pack(out / "tiles.hfp").header.root_geometric_error
    rewrite_pack(
        out / "tiles.hfp",
        lambda entries: entries,
        root_geometric_error=original + 1.0,
    )
    with pytest.raises(
        Tiles3dError, match="root_geometric_error does not bit-equal"
    ):
        _verify(packed_site)


def test_orphan_tileset_entry_is_refused(packed_site: _Site) -> None:
    """A tileset entry with no matching pack index entry is refused."""
    out, _, _, profile = packed_site
    document = _load_tileset(out)
    phantom = copy.deepcopy(_leaf(document))
    phantom["content"]["uri"] = f"tiles/2-99-0{profile.content_suffix()}"
    document["root"]["children"][0]["children"].append(phantom)
    _dump_tileset(out, document)
    with pytest.raises(
        Tiles3dError, match="has no matching pack index entry"
    ):
        _verify(packed_site)


def test_orphan_pack_entry_is_refused(packed_site: _Site) -> None:
    """A pack index entry with no matching tileset entry is refused."""
    out = packed_site[0]
    document = _load_tileset(out)
    del document["root"]["children"][0]["children"][0]
    _dump_tileset(out, document)
    with pytest.raises(
        Tiles3dError, match=r"has no matching tileset.json entry"
    ):
        _verify(packed_site)


def test_malformed_content_uri_is_refused(packed_site: _Site) -> None:
    """A content.uri that is not the canonical key form is refused."""
    out, _, _, profile = packed_site
    document = _load_tileset(out)
    # A leading zero in the level is not a base-10-no-leading-zeros integer.
    _leaf(document)["content"]["uri"] = (
        f"tiles/02-0-0{profile.content_suffix()}"
    )
    _dump_tileset(out, document)
    with pytest.raises(Tiles3dError, match="canonical"):
        _verify(packed_site)


def _repatch_manifest_file(out: Path, name: str) -> None:
    """Recompute one manifest file record's sha256/size from disk.

    Used after mutating a file the manifest already covers, so a
    corruption test's failure attributes to the single check it targets
    rather than a stale (and unrelated) manifest-recompute reject.
    """
    data = (out / name).read_bytes()
    digest = hashlib.sha256(data).hexdigest()

    def _patch(files: dict[str, Any], dataset_id: str) -> str:
        files[name] = {"sha256": digest, "size": len(data)}
        return dataset_id

    _rewrite_manifest(out, _patch)


def test_content_uri_extension_mismatch_is_refused(
    packed_site: _Site,
) -> None:
    """A uri whose extension is well-formed but wrong for this profile.

    Distinct from :func:`test_malformed_content_uri_is_refused` (which
    fails ``_URI_PATTERN.fullmatch`` outright): here the regex matches —
    the uri is otherwise canonical — and only ``match.group(4) !=
    expected_ext`` fires, e.g. a ``.glb`` uri in a heightfield build (whose
    canonical extension is ``.hf``) or vice versa. The manifest is
    repatched so only the witness parse fires.
    """
    out, _, _, profile = packed_site
    wrong_ext = ".hf" if profile is Profile.GAME else ".glb"
    document = _load_tileset(out)
    leaf = _leaf(document)
    leaf["content"]["uri"] = (
        leaf["content"]["uri"].rsplit(".", 1)[0] + wrong_ext
    )
    _dump_tileset(out, document)
    _repatch_manifest_file(out, "tileset.json")
    with pytest.raises(Tiles3dError, match="is not the canonical tiles/"):
        _verify(packed_site)


# -- manifest recompute ----------------------------------------------------


def _rewrite_manifest(out: Path, mutate: Any) -> None:  # noqa: ANN401
    manifest = parse_manifest((out / "manifest.json").read_text())
    files = {
        name: {"sha256": digest.sha256, "size": digest.size}
        for name, digest in manifest.files.items()
    }
    dataset_id = manifest.dataset_id
    dataset_id = mutate(files, dataset_id)
    document = {
        "algorithm": "sha256",
        "dataset_id": dataset_id,
        "files": files,
    }
    (out / "manifest.json").write_text(
        json.dumps(document, sort_keys=True, indent=2) + "\n"
    )


def test_manifest_sha_mismatch_is_refused(packed_site: _Site) -> None:
    """A manifest sha256 that does not match the on-disk file is refused."""
    out = packed_site[0]

    def flip_sha(files: dict[str, Any], dataset_id: str) -> str:
        files["tileset.json"]["sha256"] = "0" * 64
        return dataset_id

    _rewrite_manifest(out, flip_sha)
    with pytest.raises(Tiles3dError, match="does not match a recomputation"):
        _verify(packed_site)


def test_manifest_size_mismatch_is_refused(packed_site: _Site) -> None:
    """A manifest size that does not match the on-disk file is refused."""
    out = packed_site[0]

    def bump_size(files: dict[str, Any], dataset_id: str) -> str:
        files["tiles.hfp"]["size"] += 1
        return dataset_id

    _rewrite_manifest(out, bump_size)
    with pytest.raises(Tiles3dError, match="does not match a recomputation"):
        _verify(packed_site)


def test_manifest_dataset_id_mismatch_is_refused(
    packed_site: _Site,
) -> None:
    """A manifest dataset_id that does not match the pack is refused."""
    out = packed_site[0]

    def flip_dataset(_files: dict[str, Any], _dataset_id: str) -> str:
        return "f" * 64

    _rewrite_manifest(out, flip_dataset)
    with pytest.raises(Tiles3dError, match="does not match a recomputation"):
        _verify(packed_site)
