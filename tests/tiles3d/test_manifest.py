"""Tests for the deterministic ``manifest.json`` codec.

The codec is exercised for exact bytes (a golden document restated from
the normative spec, not from the codec's own output), the parse
round-trip, the LF-only guarantee, and every documented parse reject.
"""

from __future__ import annotations

import pytest

from ahn_cli.tiles3d.errors import Tiles3dError
from ahn_cli.tiles3d.manifest import (
    ALGORITHM,
    FileDigest,
    Manifest,
    parse_manifest,
    render_manifest,
)

_DATASET_ID = "a" * 64
_SHA_A = "b" * 64
_SHA_B = "c" * 64


def _files() -> dict[str, FileDigest]:
    """Two file records with keys deliberately out of sorted order."""
    return {
        "tiles.hfp": FileDigest(sha256=_SHA_B, size=725),
        "provenance.json": FileDigest(sha256=_SHA_A, size=42),
    }


def test_algorithm_constant_is_sha256() -> None:
    """The manifest names exactly the sha256 algorithm."""
    assert ALGORITHM == "sha256"


def test_render_matches_the_golden_bytes() -> None:
    """The rendered text is the spec's exact sorted-key, 2-space document."""
    rendered = render_manifest(_files(), _DATASET_ID)
    expected = (
        "{\n"
        '  "algorithm": "sha256",\n'
        f'  "dataset_id": "{_DATASET_ID}",\n'
        '  "files": {\n'
        '    "provenance.json": {\n'
        f'      "sha256": "{_SHA_A}",\n'
        '      "size": 42\n'
        "    },\n"
        '    "tiles.hfp": {\n'
        f'      "sha256": "{_SHA_B}",\n'
        '      "size": 725\n'
        "    }\n"
        "  }\n"
        "}\n"
    )
    assert rendered == expected


def test_render_is_lf_only_with_trailing_newline() -> None:
    """The bytes carry only LF newlines and end in one."""
    rendered = render_manifest(_files(), _DATASET_ID)
    assert "\r" not in rendered
    assert rendered.endswith("\n")
    assert b"\r\n" not in rendered.encode("utf-8")


def test_render_is_deterministic_regardless_of_insertion_order() -> None:
    """File insertion order does not change the rendered bytes."""
    reversed_files = dict(reversed(list(_files().items())))
    assert render_manifest(reversed_files, _DATASET_ID) == render_manifest(
        _files(), _DATASET_ID
    )


def test_parse_round_trips_render() -> None:
    """parse(render(...)) reproduces the dataset id and file digests."""
    manifest = parse_manifest(render_manifest(_files(), _DATASET_ID))
    assert manifest == Manifest(dataset_id=_DATASET_ID, files=_files())


def test_parse_rejects_non_json() -> None:
    """Malformed JSON is a typed error."""
    with pytest.raises(Tiles3dError, match="not valid JSON"):
        parse_manifest("{not json")


def test_parse_rejects_a_non_object() -> None:
    """A JSON array (not an object) is refused."""
    with pytest.raises(Tiles3dError, match="must be a JSON object"):
        parse_manifest("[]")


def test_parse_rejects_an_unexpected_top_key() -> None:
    """A key set other than {algorithm, dataset_id, files} is refused."""
    text = render_manifest(_files(), _DATASET_ID).replace(
        '"files"', '"extra"'
    )
    with pytest.raises(Tiles3dError, match="must be a JSON object"):
        parse_manifest(text)


def test_parse_rejects_a_wrong_algorithm() -> None:
    """Any algorithm other than sha256 is refused."""
    text = render_manifest(_files(), _DATASET_ID).replace(
        '"sha256"', '"blake3"', 1
    )
    with pytest.raises(Tiles3dError, match="algorithm"):
        parse_manifest(text)


def test_parse_rejects_a_bad_dataset_id() -> None:
    """A dataset id that is not 64 hex chars is refused."""
    with pytest.raises(Tiles3dError, match="dataset_id"):
        parse_manifest(render_manifest(_files(), "deadbeef"))


def test_parse_rejects_an_uppercase_dataset_id() -> None:
    """Uppercase hex is refused: digests are lowercase."""
    with pytest.raises(Tiles3dError, match="dataset_id"):
        parse_manifest(render_manifest(_files(), "A" * 64))


def test_parse_rejects_a_non_object_files() -> None:
    """A non-object 'files' value is refused."""
    text = f'{{"algorithm": "sha256", "dataset_id": "{_DATASET_ID}", "files": []}}'
    with pytest.raises(Tiles3dError, match="'files' must be a JSON object"):
        parse_manifest(text)


def test_parse_rejects_a_malformed_file_record() -> None:
    """A file record missing 'size' is refused."""
    text = render_manifest(_files(), _DATASET_ID).replace(
        '      "size": 42\n', '      "extra": 42\n'
    )
    with pytest.raises(Tiles3dError, match="keys"):
        parse_manifest(text)


def test_parse_rejects_a_bad_file_sha256() -> None:
    """A file digest that is not 64 hex chars is refused."""
    text = render_manifest(
        {"tiles.hfp": FileDigest(sha256="ff", size=1)}, _DATASET_ID
    )
    with pytest.raises(Tiles3dError, match="sha256"):
        parse_manifest(text)


def test_parse_rejects_a_negative_size() -> None:
    """A negative byte size is refused."""
    text = render_manifest(_files(), _DATASET_ID).replace("725", "-1")
    with pytest.raises(Tiles3dError, match="size"):
        parse_manifest(text)


def test_parse_rejects_a_boolean_size() -> None:
    """A JSON boolean is not an acceptable size (bool is an int subclass)."""
    text = render_manifest(_files(), _DATASET_ID).replace("725", "true")
    with pytest.raises(Tiles3dError, match="size"):
        parse_manifest(text)
