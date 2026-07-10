"""The :class:`Provenance` value object: an in-memory acquisition record.

Every fetcher populates one :class:`Provenance` per dataset, capturing where
the data came from, under which licence, and how to reproduce it. This module
defines the *domain type only*. Serialisation to the ``provenance.json``
sidecar (schema, field encoding, checksum formatting) is WP3's responsibility
and is deliberately absent here.
"""

from dataclasses import dataclass
from datetime import datetime

from ahn_cli.domain.generation import Generation
from ahn_cli.domain.product import Product
from ahn_cli.domain.tile import BBox, ensure_valid_bbox
from ahn_cli.domain.vintage import Vintage


@dataclass(frozen=True)
class Provenance:
    """A reproducible record of one dataset acquisition.

    Contract (fields, matching the acquisition spec's exhaustive list):
        source_portal: The portal the data was fetched from (e.g. ``"pdok"``).
        product: The dataset kind acquired.
        licence: The licence identifier the data is distributed under.
        attribution: The required attribution string (e.g. the CC-BY credit).
        bbox: The acquired extent as :data:`~ahn_cli.domain.tile.BBox` in
            EPSG:28992.
        download_started_at / download_finished_at: The download time window.
        input_checksum: Checksum of the bytes as fetched from the portal.
        output_checksum: Checksum of the bytes written by the pipeline.
        tool_version: The ``ahn_cli`` version that produced the dataset.
        vintage: The acquisition vintage, when the product is dated.
        zone: The acquisition zone (e.g. an ortho D20 zone), when applicable.
        resolution_tier: The resolution tier obtained (e.g. ``"5cm"``).
        generation: The AHN generation used, when the product is AHN-family.
        request_keys: The ordered ``(name, value)`` request-key pairs that
            content-address the fetch; kept as an immutable tuple so the record
            stays hashable. WP3 serialises these to a JSON object.

    Invariants:
        - Immutable and hashable; two records are equal iff every field is
          equal.

    Failure modes:
        - ``ValueError`` if ``bbox`` is degenerate.
        - ``ValueError`` if ``download_finished_at`` precedes
          ``download_started_at``.

    Note:
        Field-content policy (non-empty attribution, checksum/licence formats,
        timezone-awareness of the timestamps) is validated by WP3 at
        serialisation time and is intentionally not enforced here.

    """

    source_portal: str
    product: Product
    licence: str
    attribution: str
    bbox: BBox
    download_started_at: datetime
    download_finished_at: datetime
    input_checksum: str
    output_checksum: str
    tool_version: str
    vintage: Vintage | None = None
    zone: str | None = None
    resolution_tier: str | None = None
    generation: Generation | None = None
    request_keys: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        """Validate the extent and the download time window."""
        ensure_valid_bbox(self.bbox)
        if self.download_finished_at < self.download_started_at:
            msg = (
                "download_finished_at must not precede download_started_at; "
                f"got start={self.download_started_at!r}, "
                f"finish={self.download_finished_at!r}."
            )
            raise ValueError(msg)
