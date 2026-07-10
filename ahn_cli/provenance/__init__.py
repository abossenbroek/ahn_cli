"""The provenance sidecar: deterministic ``provenance.json`` (de)serialisation.

This bounded concern serialises the domain :class:`~ahn_cli.domain.Provenance`
value object to a byte-identical ``provenance.json`` sidecar and reads it back,
validating the schema and the field content WP1 deferred (non-empty
attribution, timezone-aware UTC timestamps). It owns no acquisition or
transform logic; every fetcher imports the writer to record its provenance.
"""

from ahn_cli.provenance.sidecar import (
    ProvenanceError,
    provenance_from_json_bytes,
    provenance_to_json_bytes,
    read_provenance,
    write_provenance,
)

__all__ = [
    "ProvenanceError",
    "provenance_from_json_bytes",
    "provenance_to_json_bytes",
    "read_provenance",
    "write_provenance",
]
