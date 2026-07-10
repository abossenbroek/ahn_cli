"""Prep-context transform/export seam.

The ``prep`` bounded context turns cached source tiles into finished
deliverables (classification filtering, clipping, decimation, export, and the
provenance sidecar). WP2 ships only the *seam*: it records the requested
transform intent and raises :class:`TransformNotWiredError`. Real transforms
arrive in WP10-WP13 and replace :func:`prepare`'s body without changing this
module's public surface.
"""

from dataclasses import dataclass
from pathlib import Path

from ahn_cli.prep.decimate import Thinning


class TransformNotWiredError(NotImplementedError):
    """No real prep transform is wired yet (WP10-WP13).

    A :class:`NotImplementedError` subclass so a caller may catch it broadly
    while still distinguishing the deliberate "not yet built" state from an
    accidental one.
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
    """Prep-context seam: record intent, then refuse until logic is wired.

    Contract:
        - Accepts a fully validated :class:`PrepRequest`.
        - Performs no transform or export in WP2.

    Failure modes:
        - :class:`TransformNotWiredError`, unconditionally: WP2 wires no
          transform, so preparation cannot yet proceed. WP10-WP13 replace this
          body.
    """
    msg = (
        f"No prep transform is wired yet for {request.data_dir}; "
        "real transforms land in WP10-WP13."
    )
    raise TransformNotWiredError(msg)
