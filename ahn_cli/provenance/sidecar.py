"""RED stub: importable, deliberately unimplemented WP3 sidecar codec.

The real implementation lands in the GREEN commit. These stubs exist so the
test suite imports cleanly and fails at assertion time (wrong values / no
error raised), not at collection time.
"""

from datetime import datetime, timezone
from pathlib import Path

from ahn_cli.domain import Product, Provenance


class ProvenanceError(ValueError):
    """Raised when a provenance sidecar cannot be (de)serialised."""


_EPOCH = datetime(2024, 1, 1, tzinfo=timezone.utc)
_DUMMY = Provenance(
    source_portal="stub",
    product=Product.AHN_POINT_CLOUD,
    licence="stub",
    attribution="stub",
    bbox=(0.0, 0.0, 1.0, 1.0),
    download_started_at=_EPOCH,
    download_finished_at=_EPOCH,
    input_checksum="stub",
    output_checksum="stub",
    tool_version="0",
)


def provenance_to_json_bytes(provenance: Provenance) -> bytes:  # noqa: ARG001
    """Stub: returns a constant so exact-bytes assertions fail."""
    return b"{}\n"


def provenance_from_json_bytes(data: bytes) -> Provenance:  # noqa: ARG001
    """Stub: returns a fixed record so round-trip assertions fail."""
    return _DUMMY


def write_provenance(provenance: Provenance, path: Path) -> None:
    """Write the serialised provenance bytes to ``path``."""
    path.write_bytes(provenance_to_json_bytes(provenance))


def read_provenance(path: Path) -> Provenance:
    """Read and deserialise a provenance sidecar from ``path``."""
    return provenance_from_json_bytes(path.read_bytes())
