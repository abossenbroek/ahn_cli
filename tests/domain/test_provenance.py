"""Tests for the :class:`Provenance` value object."""

import math
from datetime import datetime, timezone

import pytest

from ahn_cli.domain import (
    Generation,
    Product,
    Provenance,
    Vintage,
)

_BBOX = (0.0, 0.0, 10.0, 10.0)
_START = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
_FINISH = datetime(2024, 1, 1, 12, 5, tzinfo=timezone.utc)


def _minimal_provenance(**overrides: object) -> Provenance:
    """Build a valid provenance record, applying any field overrides."""
    fields: dict[str, object] = {
        "source_portal": "pdok",
        "product": Product.AHN_POINT_CLOUD,
        "licence": "CC-BY-4.0",
        "attribution": "© PDOK / Kadaster",
        "bbox": _BBOX,
        "download_started_at": _START,
        "download_finished_at": _FINISH,
        "input_checksum": "sha256:aaaa",
        "output_checksum": "sha256:bbbb",
        "tool_version": "0.3.5",
    }
    fields.update(overrides)
    return Provenance(**fields)  # type: ignore[arg-type]


def test_provenance_records_required_fields() -> None:
    """A minimal record stores the mandatory acquisition facts."""
    record = _minimal_provenance()
    assert record.source_portal == "pdok"
    assert record.attribution == "© PDOK / Kadaster"
    assert record.request_keys == ()
    assert record.generation is None


def test_provenance_records_optional_fields() -> None:
    """Optional axis, zone, resolution, and request keys round-trip."""
    record = _minimal_provenance(
        product=Product.ORTHO,
        vintage=Vintage(2023),
        zone="D20",
        resolution_tier="5cm",
        generation=Generation(5),
        request_keys=(("tile_id", "37FN2"), ("product", "ortho")),
    )
    assert record.vintage == Vintage(2023)
    assert record.zone == "D20"
    assert record.resolution_tier == "5cm"
    assert record.generation == Generation(5)
    assert record.request_keys == (("tile_id", "37FN2"), ("product", "ortho"))


def test_provenance_allows_zero_length_download_window() -> None:
    """A finish equal to the start is a valid (instantaneous) window."""
    record = _minimal_provenance(download_finished_at=_START)
    assert record.download_finished_at == record.download_started_at


def test_provenance_rejects_degenerate_bbox() -> None:
    """A degenerate extent is rejected via the shared validator."""
    with pytest.raises(ValueError, match="minx < maxx"):
        _minimal_provenance(bbox=(10.0, 0.0, 0.0, 10.0))


def test_provenance_rejects_non_finite_bbox() -> None:
    """A non-finite extent is rejected when constructing a Provenance."""
    with pytest.raises(ValueError, match="finite"):
        _minimal_provenance(bbox=(math.nan, math.nan, 10.0, 10.0))


def test_provenance_rejects_finish_before_start() -> None:
    """A download that finishes before it starts is inconsistent."""
    with pytest.raises(ValueError, match="must not precede"):
        _minimal_provenance(
            download_finished_at=_START, download_started_at=_FINISH
        )


def test_provenance_equality_and_hash_are_value_based() -> None:
    """Structurally identical records are equal and hash equal."""
    first = _minimal_provenance()
    second = _minimal_provenance()
    assert first == second
    assert hash(first) == hash(second)
