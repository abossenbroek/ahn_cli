"""Tests for the PDOK INSPIRE ATOM distribution source."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from pyproj import Transformer

from ahn_cli.fetch.pdok import (
    PdokFeedError,
    PdokSource,
    parse_atom_feed,
)

if TYPE_CHECKING:
    from ahn_cli.domain import BBox
    from ahn_cli.fetch.generation import GenerationSource

_FIXTURES = Path(__file__).parent / "fixtures"
_ATOM_BYTES = (_FIXTURES / "pdok_ahn_atom.xml").read_bytes()
_TO_RD = Transformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=True)


def _rd_bbox(
    minlon: float, minlat: float, maxlon: float, maxlat: float
) -> BBox:
    """Project a WGS84 lon/lat box to an EPSG:28992 AOI for the tests."""
    minx, miny = _TO_RD.transform(minlon, minlat)
    maxx, maxy = _TO_RD.transform(maxlon, maxlat)
    return (minx, miny, maxx, maxy)


# An AOI straddling the shared edge of C_37EN1 and C_37EN2.
_SHARED_EDGE_AOI = _rd_bbox(4.40, 51.99, 4.44, 52.02)
# An AOI over open country covered by no fixture sheet.
_UNCOVERED_AOI = _rd_bbox(6.40, 52.40, 6.50, 52.50)


def _feed_fetch(_url: str) -> bytes:
    """Return the fixture feed regardless of the requested URL."""
    return _ATOM_BYTES


def test_parse_atom_feed_reads_terms_and_tiles() -> None:
    """A valid feed yields its licence, attribution, and every section tile."""
    feed = parse_atom_feed(_ATOM_BYTES)

    assert feed.licence.startswith("https://creativecommons.org/publicdomain")
    assert feed.attribution == "RWS"
    assert tuple(tile.tile_id for tile in feed.tiles) == (
        "C_37EN1",
        "C_37EN2",
        "C_02DN1",
    )


def test_parse_atom_feed_rejects_non_xml() -> None:
    """Bytes that are not XML raise a typed feed error."""
    with pytest.raises(PdokFeedError, match="well-formed"):
        parse_atom_feed(b"not xml <<<")


def _feed(
    rights: str = "<rights>CC0</rights>",
    author: str = "<name>RWS</name>",
    link: str = '<link rel="section" href="https://x/C_A.LAZ" '
    'bbox="4.30 51.90 4.40 52.00" />',
) -> bytes:
    """Assemble a minimal single-entry ATOM feed from parts."""
    return (
        '<feed xmlns="http://www.w3.org/2005/Atom">'
        f"{rights}<author>{author}</author>"
        f"<entry>{link}</entry></feed>"
    ).encode()


def test_parse_atom_feed_requires_rights() -> None:
    """A feed with no rights element is rejected."""
    with pytest.raises(PdokFeedError, match="rights"):
        parse_atom_feed(_feed(rights=""))


def test_parse_atom_feed_rejects_blank_rights() -> None:
    """A feed whose rights element is whitespace-only is rejected."""
    with pytest.raises(PdokFeedError, match="rights"):
        parse_atom_feed(_feed(rights="<rights> </rights>"))


def test_parse_atom_feed_requires_author_name() -> None:
    """A feed with no author name is rejected."""
    with pytest.raises(PdokFeedError, match="author"):
        parse_atom_feed(_feed(author="<email>x@y</email>"))


def test_parse_atom_feed_requires_section_href() -> None:
    """A section link without a href is rejected."""
    with pytest.raises(PdokFeedError, match="href"):
        parse_atom_feed(_feed(link='<link rel="section" bbox="4 51 5 52" />'))


def test_parse_atom_feed_rejects_blank_section_href() -> None:
    """A section link with a blank href is rejected."""
    with pytest.raises(PdokFeedError, match="href"):
        parse_atom_feed(
            _feed(link='<link rel="section" href="  " bbox="4 51 5 52" />')
        )


def test_parse_atom_feed_requires_section_bbox() -> None:
    """A section link without a bbox is rejected."""
    with pytest.raises(PdokFeedError, match="bbox"):
        parse_atom_feed(
            _feed(link='<link rel="section" href="https://x/C_A.LAZ" />')
        )


def test_parse_atom_feed_rejects_wrong_bbox_length() -> None:
    """A bbox that is not four numbers is rejected."""
    with pytest.raises(PdokFeedError, match="four numbers"):
        parse_atom_feed(
            _feed(
                link='<link rel="section" href="https://x/C_A.LAZ" '
                'bbox="4 51 5" />'
            )
        )


def test_parse_atom_feed_rejects_non_numeric_bbox() -> None:
    """A bbox with a non-numeric coordinate is rejected."""
    with pytest.raises(PdokFeedError, match="non-numeric"):
        parse_atom_feed(
            _feed(
                link='<link rel="section" href="https://x/C_A.LAZ" '
                'bbox="4 51 5 north" />'
            )
        )


def test_parse_atom_feed_rejects_href_without_filename() -> None:
    """A section href with no filename stem is rejected."""
    with pytest.raises(PdokFeedError, match="no filename"):
        parse_atom_feed(
            _feed(
                link='<link rel="section" href="https://x/.LAZ" '
                'bbox="4 51 5 52" />'
            )
        )


def test_resolve_selects_covering_tiles_in_order() -> None:
    """Resolving the shared-edge AOI returns both edge sheets, id-ordered."""
    resolved = PdokSource().resolve(
        _generation_source(), _SHARED_EDGE_AOI, _feed_fetch
    )

    assert [tile.tile_id for tile in resolved.tiles] == ["C_37EN1", "C_37EN2"]
    assert resolved.licence.startswith("https://creativecommons.org")
    assert resolved.attribution == "RWS"


def test_resolve_bbox_is_projected_to_rd() -> None:
    """A resolved tile's extent is on the Dutch grid, not WGS84 lon/lat."""
    resolved = PdokSource().resolve(
        _generation_source(), _SHARED_EDGE_AOI, _feed_fetch
    )

    minx, miny, _maxx, _maxy = resolved.tiles[0].bbox
    assert minx > 1000.0
    assert miny > 1000.0


def test_resolve_returns_no_tiles_for_uncovered_aoi() -> None:
    """An AOI covered by no sheet resolves to an empty tile tuple."""
    resolved = PdokSource().resolve(
        _generation_source(), _UNCOVERED_AOI, _feed_fetch
    )

    assert resolved.tiles == ()


def test_generation_registry_offers_ahn5_and_ahn4() -> None:
    """The PDOK registry advertises AHN5 (newest) then AHN4."""
    registry = PdokSource().generation_registry(_feed_fetch)

    assert registry.tokens() == ("auto", "ahn5", "ahn4")


def test_generation_registry_probe_reports_real_coverage() -> None:
    """Each generation's probe reflects whether the feed covers the AOI."""
    registry = PdokSource().generation_registry(_feed_fetch)
    newest = registry.sources()[0]

    assert newest.probe(_SHARED_EDGE_AOI) is True
    assert newest.probe(_UNCOVERED_AOI) is False


def _generation_source() -> GenerationSource:
    """Return the newest PDOK generation source for resolve() tests."""
    return PdokSource().generation_registry(_feed_fetch).sources()[0]
