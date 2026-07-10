"""Tests for the deterministic ``provenance.json`` sidecar codec.

These exercise the WP3 writer/reader over the WP1 :class:`Provenance` value
object: byte-identical serialisation, write/read round-trips, and every
schema/content validation branch (missing/unknown/mistyped fields, empty
attribution, naive datetimes, bad bbox/vintage/generation/request keys).
"""

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from ahn_cli.domain import (
    BBox,
    Generation,
    Product,
    Provenance,
    Vintage,
)
from ahn_cli.provenance import (
    ProvenanceError,
    provenance_from_json_bytes,
    provenance_to_json_bytes,
    read_provenance,
    write_provenance,
)

_BBOX: BBox = (0.0, 0.0, 10.0, 10.0)
_START = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
_FINISH = datetime(2024, 1, 1, 12, 5, tzinfo=timezone.utc)

_GOLDEN = (
    "{\n"
    '  "attribution": "© PDOK / Kadaster",\n'
    '  "bbox": [\n'
    "    0.0,\n"
    "    0.0,\n"
    "    10.0,\n"
    "    10.0\n"
    "  ],\n"
    '  "download_finished_at": "2024-01-01T12:05:00+00:00",\n'
    '  "download_started_at": "2024-01-01T12:00:00+00:00",\n'
    '  "generation": null,\n'
    '  "input_checksum": "sha256:aaaa",\n'
    '  "licence": "CC-BY-4.0",\n'
    '  "output_checksum": "sha256:bbbb",\n'
    '  "product": "ahn_point_cloud",\n'
    '  "request_keys": {},\n'
    '  "resolution_tier": null,\n'
    '  "source_portal": "pdok",\n'
    '  "tool_version": "0.3.5",\n'
    '  "vintage": null,\n'
    '  "zone": null\n'
    "}\n"
).encode("utf-8")


def _minimal() -> Provenance:
    """Return a valid record with every optional field left unset."""
    return Provenance(
        source_portal="pdok",
        product=Product.AHN_POINT_CLOUD,
        licence="CC-BY-4.0",
        attribution="© PDOK / Kadaster",
        bbox=_BBOX,
        download_started_at=_START,
        download_finished_at=_FINISH,
        input_checksum="sha256:aaaa",
        output_checksum="sha256:bbbb",
        tool_version="0.3.5",
    )


def _full() -> Provenance:
    """Return a valid record with every optional field populated."""
    return replace(
        _minimal(),
        product=Product.ORTHO,
        vintage=Vintage(2023),
        zone="D20",
        resolution_tier="5cm",
        generation=Generation(5),
        request_keys=(("tile_id", "37FN2"), ("product", "ortho")),
    )


def _sidecar_dict() -> dict[str, object]:
    """Return the canonical full record decoded to a mutable JSON mapping."""
    data: dict[str, object] = json.loads(provenance_to_json_bytes(_full()))
    return data


def _dumps(data: dict[str, object]) -> bytes:
    """Serialise a raw mapping back to sidecar bytes for the reader."""
    return json.dumps(data).encode("utf-8")


def test_round_trip_minimal_is_equal() -> None:
    """A minimal record survives write then read unchanged."""
    record = _minimal()
    assert provenance_from_json_bytes(provenance_to_json_bytes(record)) == record


def test_round_trip_full_is_equal() -> None:
    """A fully populated record (all optionals) round-trips unchanged."""
    record = _full()
    assert provenance_from_json_bytes(provenance_to_json_bytes(record)) == record


def test_write_and_read_file_round_trip(tmp_path: Path) -> None:
    """The file writer/reader pair round-trips through disk."""
    path = tmp_path / "provenance.json"
    record = _full()
    write_provenance(record, path)
    assert read_provenance(path) == record


def test_repeat_serialisation_is_byte_identical() -> None:
    """Serialising the same record twice yields identical bytes."""
    record = _full()
    assert provenance_to_json_bytes(record) == provenance_to_json_bytes(record)


def test_serialisation_matches_golden_bytes() -> None:
    """The minimal record serialises to the exact expected bytes."""
    assert provenance_to_json_bytes(_minimal()) == _GOLDEN


def test_serialisation_has_sorted_keys_and_trailing_newline() -> None:
    """Top-level keys are sorted and the file ends with a newline."""
    raw = provenance_to_json_bytes(_full())
    assert raw.endswith(b"\n")
    parsed: dict[str, object] = json.loads(raw)
    assert list(parsed.keys()) == sorted(parsed.keys())


def test_attribution_carries_cc_by_string() -> None:
    """The CC-BY attribution string is present in the serialised bytes."""
    record = replace(_minimal(), attribution="CC BY 4.0 Kadaster/PDOK")
    assert b"CC BY 4.0 Kadaster/PDOK" in provenance_to_json_bytes(record)


def test_write_normalises_non_utc_to_utc() -> None:
    """Aware non-UTC timestamps are normalised to UTC and still round-trip."""
    plus_two = timezone(timedelta(hours=2))
    record = replace(
        _minimal(),
        download_started_at=datetime(2024, 1, 1, 14, 0, tzinfo=plus_two),
        download_finished_at=datetime(2024, 1, 1, 14, 5, tzinfo=plus_two),
    )
    raw = provenance_to_json_bytes(record)
    assert b"2024-01-01T12:00:00+00:00" in raw
    assert provenance_from_json_bytes(raw) == record


def test_write_rejects_empty_attribution() -> None:
    """A blank attribution is rejected at serialisation time."""
    record = replace(_minimal(), attribution="   ")
    with pytest.raises(ProvenanceError, match="attribution"):
        provenance_to_json_bytes(record)


def test_write_rejects_naive_datetime() -> None:
    """A timezone-naive timestamp is rejected at serialisation time."""
    naive_start = _START.replace(tzinfo=None)
    naive_finish = _FINISH.replace(tzinfo=None)
    record = replace(
        _minimal(),
        download_started_at=naive_start,
        download_finished_at=naive_finish,
    )
    with pytest.raises(ProvenanceError, match="timezone-aware"):
        provenance_to_json_bytes(record)


def test_write_rejects_duplicate_request_keys() -> None:
    """Duplicate request-key names cannot be losslessly serialised."""
    record = replace(_minimal(), request_keys=(("k", "1"), ("k", "2")))
    with pytest.raises(ProvenanceError, match="duplicate"):
        provenance_to_json_bytes(record)


def test_read_rejects_non_json() -> None:
    """Bytes that are not JSON at all are rejected."""
    with pytest.raises(ProvenanceError, match="valid JSON"):
        provenance_from_json_bytes(b"{not json")


def test_read_rejects_non_object() -> None:
    """A JSON value that is not an object is rejected."""
    with pytest.raises(ProvenanceError, match="JSON object"):
        provenance_from_json_bytes(b"[]")


def test_read_rejects_missing_field() -> None:
    """A sidecar missing a required field is rejected."""
    data = _sidecar_dict()
    del data["licence"]
    with pytest.raises(ProvenanceError, match="keys"):
        provenance_from_json_bytes(_dumps(data))


def test_read_rejects_unknown_field() -> None:
    """A sidecar with an unexpected extra field is rejected."""
    data = _sidecar_dict()
    data["surprise"] = 1
    with pytest.raises(ProvenanceError, match="keys"):
        provenance_from_json_bytes(_dumps(data))


def test_read_rejects_empty_attribution() -> None:
    """A blank attribution string on read is rejected."""
    data = _sidecar_dict()
    data["attribution"] = "   "
    with pytest.raises(ProvenanceError, match="attribution"):
        provenance_from_json_bytes(_dumps(data))


def test_read_rejects_wrong_type_string_field() -> None:
    """A required string field given a non-string value is rejected."""
    data = _sidecar_dict()
    data["source_portal"] = 123
    with pytest.raises(ProvenanceError, match="source_portal"):
        provenance_from_json_bytes(_dumps(data))


def test_read_rejects_wrong_type_optional_field() -> None:
    """An optional string field given a non-string value is rejected."""
    data = _sidecar_dict()
    data["zone"] = 5
    with pytest.raises(ProvenanceError, match="zone"):
        provenance_from_json_bytes(_dumps(data))


def test_read_rejects_unknown_product() -> None:
    """An unknown product code is rejected."""
    data = _sidecar_dict()
    data["product"] = "flux_capacitor"
    with pytest.raises(ProvenanceError, match="product"):
        provenance_from_json_bytes(_dumps(data))


def test_read_rejects_non_string_product() -> None:
    """A non-string product value is rejected."""
    data = _sidecar_dict()
    data["product"] = 42
    with pytest.raises(ProvenanceError, match="product"):
        provenance_from_json_bytes(_dumps(data))


def test_read_rejects_bbox_wrong_length() -> None:
    """A bbox that is not a four-element array is rejected."""
    data = _sidecar_dict()
    data["bbox"] = [1.0, 2.0, 3.0]
    with pytest.raises(ProvenanceError, match="bbox"):
        provenance_from_json_bytes(_dumps(data))


def test_read_rejects_bbox_non_number_coord() -> None:
    """A bbox with a non-numeric coordinate is rejected."""
    data = _sidecar_dict()
    data["bbox"] = ["a", "b", "c", "d"]
    with pytest.raises(ProvenanceError, match="bbox"):
        provenance_from_json_bytes(_dumps(data))


def test_read_rejects_bbox_bool_coord() -> None:
    """A bbox with a boolean coordinate is rejected."""
    data = _sidecar_dict()
    data["bbox"] = [True, 0.0, 10.0, 10.0]
    with pytest.raises(ProvenanceError, match="bbox"):
        provenance_from_json_bytes(_dumps(data))


def test_read_rejects_degenerate_bbox() -> None:
    """A structurally valid but degenerate bbox is rejected via the domain."""
    data = _sidecar_dict()
    data["bbox"] = [10.0, 0.0, 0.0, 10.0]
    with pytest.raises(ProvenanceError, match="invariant"):
        provenance_from_json_bytes(_dumps(data))


def test_read_rejects_non_string_datetime() -> None:
    """A non-string timestamp value is rejected."""
    data = _sidecar_dict()
    data["download_started_at"] = 123
    with pytest.raises(ProvenanceError, match="download_started_at"):
        provenance_from_json_bytes(_dumps(data))


def test_read_rejects_unparseable_datetime() -> None:
    """A timestamp string that is not ISO-8601 is rejected."""
    data = _sidecar_dict()
    data["download_started_at"] = "not-a-date"
    with pytest.raises(ProvenanceError, match="ISO-8601"):
        provenance_from_json_bytes(_dumps(data))


def test_read_rejects_naive_datetime() -> None:
    """A timezone-naive timestamp on read is rejected."""
    data = _sidecar_dict()
    data["download_started_at"] = "2024-01-01T12:00:00"
    with pytest.raises(ProvenanceError, match="timezone-aware"):
        provenance_from_json_bytes(_dumps(data))


def test_read_rejects_non_integer_vintage() -> None:
    """A non-integer vintage value is rejected."""
    data = _sidecar_dict()
    data["vintage"] = "2023"
    with pytest.raises(ProvenanceError, match="vintage"):
        provenance_from_json_bytes(_dumps(data))


def test_read_rejects_bool_vintage() -> None:
    """A boolean vintage value is rejected (bool is not a year)."""
    data = _sidecar_dict()
    data["vintage"] = True
    with pytest.raises(ProvenanceError, match="vintage"):
        provenance_from_json_bytes(_dumps(data))


def test_read_rejects_invalid_vintage_year() -> None:
    """A vintage year the domain rejects surfaces as a sidecar error."""
    data = _sidecar_dict()
    data["vintage"] = 1800
    with pytest.raises(ProvenanceError, match="vintage"):
        provenance_from_json_bytes(_dumps(data))


def test_read_rejects_non_integer_generation() -> None:
    """A non-integer generation value is rejected."""
    data = _sidecar_dict()
    data["generation"] = "5"
    with pytest.raises(ProvenanceError, match="generation"):
        provenance_from_json_bytes(_dumps(data))


def test_read_rejects_bool_generation() -> None:
    """A boolean generation value is rejected."""
    data = _sidecar_dict()
    data["generation"] = False
    with pytest.raises(ProvenanceError, match="generation"):
        provenance_from_json_bytes(_dumps(data))


def test_read_rejects_invalid_generation_number() -> None:
    """A generation ordinal the domain rejects surfaces as a sidecar error."""
    data = _sidecar_dict()
    data["generation"] = 0
    with pytest.raises(ProvenanceError, match="generation"):
        provenance_from_json_bytes(_dumps(data))


def test_read_rejects_non_object_request_keys() -> None:
    """A request_keys value that is not an object is rejected."""
    data = _sidecar_dict()
    data["request_keys"] = []
    with pytest.raises(ProvenanceError, match="request_keys"):
        provenance_from_json_bytes(_dumps(data))


def test_read_rejects_non_string_request_key_value() -> None:
    """A request_keys entry with a non-string value is rejected."""
    data = _sidecar_dict()
    data["request_keys"] = {"tile_id": 5}
    with pytest.raises(ProvenanceError, match="request_keys"):
        provenance_from_json_bytes(_dumps(data))


def test_read_rejects_finish_before_start() -> None:
    """A download window that finishes before it starts is rejected."""
    data = _sidecar_dict()
    data["download_started_at"] = "2024-01-01T12:05:00+00:00"
    data["download_finished_at"] = "2024-01-01T12:00:00+00:00"
    with pytest.raises(ProvenanceError, match="invariant"):
        provenance_from_json_bytes(_dumps(data))


def test_provenance_error_is_value_error() -> None:
    """``ProvenanceError`` is a ``ValueError`` for ergonomic handling."""
    assert issubclass(ProvenanceError, ValueError)
