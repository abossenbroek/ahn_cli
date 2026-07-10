"""Nightly live portal-contract checks (excluded from the fast offline suite).

These are the epic's "portal contract (nightly, live)" tier: they hit the *real*
PDOK / Beeldmateriaal ATOM endpoints and assert the contract the offline
fetchers are pinned against still holds -- the feeds parse, carry their licence
terms, and expose section sheets. They are marked :func:`pytest.mark.nightly`, so
the default suite deselects them (``addopts = -m 'not nightly'``) and never
performs network I/O; the nightly CI job runs them with ``pytest -m nightly``.

They import only production parsing code the fast suite already covers, so
excluding them leaves the 100% coverage gate intact. When run without the
``AHN_CLI_NIGHTLY`` opt-in (e.g. an ad-hoc ``pytest -m nightly`` on a laptop
offline), each skips cleanly rather than failing on a network error.
"""

from __future__ import annotations

import os

import pytest
import requests

from ahn_cli.fetch.pdok import parse_atom_feed

_PDOK_AHN_FEEDS = (
    "https://service.pdok.nl/rws/ahn/atom/ahn5_laz.xml",
    "https://service.pdok.nl/rws/ahn/atom/ahn4_laz.xml",
)
_HTTP_TIMEOUT_SECONDS = 60

pytestmark = [
    pytest.mark.nightly,
    pytest.mark.skipif(
        os.environ.get("AHN_CLI_NIGHTLY") != "1",
        reason="live portal contract; set AHN_CLI_NIGHTLY=1 to run",
    ),
]


@pytest.mark.parametrize("feed_url", _PDOK_AHN_FEEDS)
def test_pdok_ahn_feed_parses_and_carries_terms(feed_url: str) -> None:
    """The live PDOK AHN ATOM feed parses and still exposes licence + sheets."""
    response = requests.get(feed_url, timeout=_HTTP_TIMEOUT_SECONDS)
    response.raise_for_status()

    feed = parse_atom_feed(response.content)

    assert feed.licence.strip()
    assert feed.attribution.strip()
    assert feed.tiles, "PDOK AHN feed must expose at least one section sheet"
    assert all(tile.download_url.strip() for tile in feed.tiles)
