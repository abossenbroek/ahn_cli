"""Pipeline context: fuse the standalone verbs into a tile-streaming pipeline.

This bounded context runs the full acquisition -> deliverable chain over an
arbitrarily large area of interest by streaming one **spatial tile** end-to-end
through fused stages *in RAM*, rather than processing the whole area at once and
round-tripping a full intermediate artifact through disk between verbs. A tile
(an AHN sheet plus a correctness halo) is small, so a tile-scoped stage reuses
the existing in-memory verb logic unchanged while peak memory stays bounded.

This module holds the contracts every other pipeline workstream builds against:
the streaming value objects and :class:`Stage` protocol (:mod:`.model`), the
single typed :class:`PipelineError` (:mod:`.errors`), and the machine-facts /
free-RAM sensing behind an injectable probe that backs the RAM-adaptive tiling
(:mod:`.machine`). The spec parser, executor, and stage adapters build on these.
"""

from __future__ import annotations

from ahn_cli.pipeline.errors import PipelineError
from ahn_cli.pipeline.machine import (
    MachineFacts,
    SystemProbe,
    free_ram_bytes,
    machine_facts,
)
from ahn_cli.pipeline.model import (
    EncodedBlob,
    EncodedTile,
    GridTile,
    PointTile,
    Stage,
    TileContext,
    TileKey,
    TilePayload,
)

__all__ = [
    "EncodedBlob",
    "EncodedTile",
    "GridTile",
    "MachineFacts",
    "PipelineError",
    "PointTile",
    "Stage",
    "SystemProbe",
    "TileContext",
    "TileKey",
    "TilePayload",
    "free_ram_bytes",
    "machine_facts",
]
