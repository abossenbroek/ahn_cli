"""Prep-context transform/export orchestration.

The ``prep`` bounded context turns cached source tiles into finished
deliverables. WP14 wires the pipeline end to end over the already-merged prep
transforms, preserving this module's public surface (:class:`PrepRequest`,
:func:`prepare`):

1. Read the cached AHN tiles a prior ``fetch`` wrote under ``<data_dir>/ahn/``,
   each paired with its provenance-recorded extent.
2. :func:`~ahn_cli.prep.dedup.deduplicate_tiles` -- crop-before-merge plus an
   exact XYZ+GPS-time sweep -- into ``<data_dir>/pointcloud.laz``.
3. Apply the classification ``include``/``exclude`` filter.
4. Apply the graded :data:`~ahn_cli.prep.decimate.Thinning` request (voxel-grid
   or Poisson-disk) via the CPU reference backend, when one is requested. This
   is additive to the legacy nth-point decimation, which is untouched.
5. Write the site-root ``provenance.json`` recording the prep lineage.
6. Export ``<data_dir>/pointcloud.ply`` when ``export_points`` is set.

The mirrored ordering is the documented "filter classes -> clip -> decimate"
with WP10's crop-before-merge folded into step 2 (the merge is where the crop
must happen). Every stage is deterministic, so identical inputs yield
byte-identical outputs. Expected failures (a missing site layout, no tiles, a
tile with no provenance sidecar) raise the typed :class:`PrepError` so the CLI
reports a tidy message rather than leaking a traceback.
"""

from dataclasses import dataclass
from pathlib import Path

from ahn_cli.prep.decimate import Thinning


class PrepError(RuntimeError):
    """Raised when a prep run cannot proceed for an expected reason.

    Signals a missing site layout (no ``<data_dir>/ahn/`` directory), no fetched
    AHN tiles to prepare, or a tile whose provenance sidecar -- the source of its
    crop extent -- is absent. Subclasses :class:`RuntimeError` so the CLI catches
    it and reports a clean error rather than a traceback.
    """


@dataclass(frozen=True)
class PrepRequest:
    """A validated intent to transform a fetched site.

    Contract:
        - ``data_dir`` is the site directory a prior ``fetch`` produced.
        - ``include_classes`` / ``exclude_classes`` are the parsed, validated
          classification filters; empty tuples mean "no filter on this side".
          The caller guarantees the two do not overlap.
        - ``export_points`` requests the point-cloud export.
        - ``thinning`` is the validated graded-thinning request (voxel-grid or
          Poisson-disk), or ``None`` for no additional thinning. It is additive
          to the legacy nth-point decimation, which is unaffected.

    Invariants:
        - Frozen: an immutable, hashable value object, equal by field value.
    """

    data_dir: Path
    include_classes: tuple[int, ...] = ()
    exclude_classes: tuple[int, ...] = ()
    export_points: bool = False
    thinning: Thinning | None = None


def prepare(request: PrepRequest) -> None:
    """Not yet wired (assertion-level RED stub, replaced by the GREEN body)."""
    del request
