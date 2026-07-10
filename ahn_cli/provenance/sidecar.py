"""Deterministic ``provenance.json`` sidecar codec for :class:`Provenance`.

This module serialises the domain :class:`~ahn_cli.domain.Provenance` value
object to a byte-identical JSON sidecar and reads it back. Determinism is
load-bearing: identical input always yields identical bytes -- top-level keys
are sorted, timestamps are ISO-8601 UTC, and the file ends with a single
newline. On read the schema is validated field by field, and the field content
WP1 deferred to WP3 (non-empty attribution, timezone-aware timestamps) is
enforced here so a malformed sidecar is rejected with a typed error rather than
silently accepted.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Final, cast

from ahn_cli.domain import Generation, Product, Provenance, Vintage

if TYPE_CHECKING:
    from pathlib import Path

    from ahn_cli.domain import BBox

_BBOX_LEN: Final = 4

_FIELDS: Final = frozenset(
    {
        "attribution",
        "bbox",
        "download_finished_at",
        "download_started_at",
        "generation",
        "input_checksum",
        "licence",
        "output_checksum",
        "product",
        "request_keys",
        "resolution_tier",
        "source_portal",
        "tool_version",
        "vintage",
        "zone",
    }
)


class ProvenanceError(ValueError):
    """Raised when a provenance sidecar cannot be serialised or parsed.

    Contract:
        - Signals a malformed sidecar on read (missing/unknown/mistyped field,
          bad datetime or bbox) or invalid content on write (blank attribution,
          naive timestamp, duplicate request key).
        - Subclasses :class:`ValueError`, so callers may catch either.
    """


def provenance_to_json_bytes(provenance: Provenance) -> bytes:
    """Serialise ``provenance`` to deterministic ``provenance.json`` bytes.

    Contract:
        - Returns UTF-8 bytes with top-level keys sorted, timestamps rendered
          as ISO-8601 UTC, and a single trailing newline; the same record
          always yields byte-identical output.
        - ``request_keys`` is written as a JSON object preserving pair order.

    Failure modes:
        - :class:`ProvenanceError` if ``attribution`` is blank, either
          timestamp is timezone-naive, or ``request_keys`` has a duplicate name.
    """
    payload = _encode(provenance)
    ordered = {key: payload[key] for key in sorted(payload)}
    text = json.dumps(ordered, ensure_ascii=False, indent=2)
    return (text + "\n").encode("utf-8")


def provenance_from_json_bytes(data: bytes) -> Provenance:
    """Parse and validate ``data`` back into a :class:`Provenance`.

    Contract:
        - Accepts bytes produced by :func:`provenance_to_json_bytes` and
          reconstructs an equal record (write then read is the identity).
        - Every field is type- and content-checked before the value object is
          built; the domain's own extent/window invariants are then enforced.

    Failure modes:
        - :class:`ProvenanceError` if the bytes are not a JSON object, the key
          set does not match exactly, any field has the wrong type, a datetime
          is unparsable or naive, or a domain invariant is violated.
    """
    try:
        parsed = json.loads(data)
    except json.JSONDecodeError as exc:
        msg = "provenance sidecar is not valid JSON."
        raise ProvenanceError(msg) from exc
    if not isinstance(parsed, dict):
        msg = "provenance sidecar must be a JSON object."
        raise ProvenanceError(msg)
    mapping = cast("dict[str, object]", parsed)
    if frozenset(mapping) != _FIELDS:
        missing = sorted(_FIELDS - frozenset(mapping))
        unknown = sorted(frozenset(mapping) - _FIELDS)
        msg = (
            "provenance sidecar keys mismatch; "
            f"missing={missing}, unknown={unknown}."
        )
        raise ProvenanceError(msg)

    attribution = _ensure_attribution(_require_str(mapping, "attribution"))
    try:
        return Provenance(
            source_portal=_require_str(mapping, "source_portal"),
            product=_decode_product(mapping["product"]),
            licence=_require_str(mapping, "licence"),
            attribution=attribution,
            bbox=_decode_bbox(mapping["bbox"]),
            download_started_at=_decode_datetime(
                mapping["download_started_at"], "download_started_at"
            ),
            download_finished_at=_decode_datetime(
                mapping["download_finished_at"], "download_finished_at"
            ),
            input_checksum=_require_str(mapping, "input_checksum"),
            output_checksum=_require_str(mapping, "output_checksum"),
            tool_version=_require_str(mapping, "tool_version"),
            vintage=_decode_vintage(mapping["vintage"]),
            zone=_require_opt_str(mapping, "zone"),
            resolution_tier=_require_opt_str(mapping, "resolution_tier"),
            generation=_decode_generation(mapping["generation"]),
            request_keys=_decode_request_keys(mapping["request_keys"]),
        )
    except ProvenanceError:
        raise
    except ValueError as exc:
        msg = f"provenance sidecar violates a domain invariant: {exc}"
        raise ProvenanceError(msg) from exc


def write_provenance(provenance: Provenance, path: Path) -> None:
    """Write ``provenance`` to ``path`` as a deterministic sidecar file.

    Contract:
        - Writes exactly the bytes of :func:`provenance_to_json_bytes`, so
          re-writing the same record leaves the file byte-identical.

    Failure modes:
        - Propagates :class:`ProvenanceError` from serialisation.
    """
    path.write_bytes(provenance_to_json_bytes(provenance))


def read_provenance(path: Path) -> Provenance:
    """Read and validate the provenance sidecar stored at ``path``.

    Contract:
        - Returns the :class:`Provenance` reconstructed from the file; equal to
          the record originally written.

    Failure modes:
        - Propagates :class:`ProvenanceError` from parsing/validation.
    """
    return provenance_from_json_bytes(path.read_bytes())


def _encode(provenance: Provenance) -> dict[str, object]:
    """Map a validated provenance to its JSON-serialisable field values."""
    generation = provenance.generation
    vintage = provenance.vintage
    return {
        "attribution": _ensure_attribution(provenance.attribution),
        "bbox": list(provenance.bbox),
        "download_finished_at": _encode_datetime(
            provenance.download_finished_at, "download_finished_at"
        ),
        "download_started_at": _encode_datetime(
            provenance.download_started_at, "download_started_at"
        ),
        "generation": None if generation is None else generation.number,
        "input_checksum": provenance.input_checksum,
        "licence": provenance.licence,
        "output_checksum": provenance.output_checksum,
        "product": provenance.product.value,
        "request_keys": _encode_request_keys(provenance.request_keys),
        "resolution_tier": provenance.resolution_tier,
        "source_portal": provenance.source_portal,
        "tool_version": provenance.tool_version,
        "vintage": None if vintage is None else vintage.year,
        "zone": provenance.zone,
    }


def _ensure_attribution(value: str) -> str:
    """Return ``value`` unchanged, or raise if it is blank."""
    if not value.strip():
        msg = "attribution must be a non-empty string."
        raise ProvenanceError(msg)
    return value


def _encode_request_keys(
    pairs: tuple[tuple[str, str], ...],
) -> dict[str, str]:
    """Encode ordered request-key pairs to an object, rejecting duplicates."""
    result: dict[str, str] = {}
    for name, value in pairs:
        if name in result:
            msg = f"request_keys has a duplicate name: {name!r}."
            raise ProvenanceError(msg)
        result[name] = value
    return result


def _encode_datetime(value: datetime, field: str) -> str:
    """Render a timezone-aware datetime as an ISO-8601 UTC string."""
    if value.tzinfo is None:
        msg = f"{field} must be timezone-aware."
        raise ProvenanceError(msg)
    return value.astimezone(timezone.utc).isoformat()


def _require_str(mapping: dict[str, object], key: str) -> str:
    """Return ``mapping[key]`` as a string, or raise on the wrong type."""
    value = mapping[key]
    if not isinstance(value, str):
        msg = f"{key} must be a string; got {type(value).__name__}."
        raise ProvenanceError(msg)
    return value


def _require_opt_str(mapping: dict[str, object], key: str) -> str | None:
    """Return ``mapping[key]`` as a string or ``None``, else raise."""
    value = mapping[key]
    if value is None:
        return None
    if not isinstance(value, str):
        msg = f"{key} must be a string or null; got {type(value).__name__}."
        raise ProvenanceError(msg)
    return value


def _decode_product(value: object) -> Product:
    """Decode a product code string into a :class:`Product` member."""
    if not isinstance(value, str):
        msg = "product must be a string."
        raise ProvenanceError(msg)
    try:
        return Product(value)
    except ValueError as exc:
        msg = f"unknown product code: {value!r}."
        raise ProvenanceError(msg) from exc


def _decode_bbox(value: object) -> BBox:
    """Decode a four-number JSON array into a bbox tuple."""
    if not isinstance(value, list):
        msg = "bbox must be a JSON array of four numbers."
        raise ProvenanceError(msg)
    coords = cast("list[object]", value)
    if len(coords) != _BBOX_LEN:
        msg = "bbox must be a JSON array of four numbers."
        raise ProvenanceError(msg)
    numbers: list[float] = []
    for coord in coords:
        if isinstance(coord, bool) or not isinstance(coord, (int, float)):
            msg = "bbox coordinates must all be numbers."
            raise ProvenanceError(msg)
        numbers.append(float(coord))
    return (numbers[0], numbers[1], numbers[2], numbers[3])


def _decode_datetime(value: object, field: str) -> datetime:
    """Decode an ISO-8601 string into a timezone-aware datetime."""
    if not isinstance(value, str):
        msg = f"{field} must be an ISO-8601 string."
        raise ProvenanceError(msg)
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        msg = f"{field} is not a valid ISO-8601 datetime: {value!r}."
        raise ProvenanceError(msg) from exc
    if parsed.tzinfo is None:
        msg = f"{field} must be timezone-aware."
        raise ProvenanceError(msg)
    return parsed


def _decode_vintage(value: object) -> Vintage | None:
    """Decode an integer year into a :class:`Vintage`, or ``None``."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        msg = "vintage must be an integer year or null."
        raise ProvenanceError(msg)
    try:
        return Vintage(value)
    except ValueError as exc:
        msg = f"invalid vintage year: {value}."
        raise ProvenanceError(msg) from exc


def _decode_generation(value: object) -> Generation | None:
    """Decode an integer ordinal into a :class:`Generation`, or ``None``."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        msg = "generation must be an integer ordinal or null."
        raise ProvenanceError(msg)
    try:
        return Generation(value)
    except ValueError as exc:
        msg = f"invalid generation ordinal: {value}."
        raise ProvenanceError(msg) from exc


def _decode_request_keys(value: object) -> tuple[tuple[str, str], ...]:
    """Decode a request-keys object into ordered ``(name, value)`` pairs."""
    if not isinstance(value, dict):
        msg = "request_keys must be a JSON object."
        raise ProvenanceError(msg)
    mapping = cast("dict[str, object]", value)
    pairs: list[tuple[str, str]] = []
    for name, item in mapping.items():
        if not isinstance(item, str):
            msg = f"request_keys value for {name!r} must be a string."
            raise ProvenanceError(msg)
        pairs.append((name, item))
    return tuple(pairs)
