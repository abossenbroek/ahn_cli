"""The ``dedup`` pipeline stage: class filter, then exact duplicate sweep.

:class:`DedupStage` is the tile-scoped, in-memory adapter over the ``prep``
context's de-duplication contract (:mod:`ahn_cli.prep.dedup` /
:mod:`ahn_cli.prep.dedup_stream`). Within one tile the source has already
merged the overlapping AHN sheets into a single
:class:`~ahn_cli.pipeline.model.PointTile`, so the cross-sheet crop-before-merge
step is the source's job; the stage's remaining job is the exact-duplicate
sweep the oracle performs:

1. **Class filter** -- keep a point when its classification is in
   ``include_classes`` (or ``include_classes`` is empty) and not in
   ``exclude_classes``, matching :func:`ahn_cli.prep.transform._class_mask`.
2. **Exact-duplicate sweep** -- two points coincide when their ``x``/``y``/``z``
   and ``gps_time`` all match; of each such group the survivor is the one with
   the smallest index, and survivors are returned in ascending index order --
   exactly :func:`ahn_cli.prep.dedup.deduplicate_tiles`'s
   ``np.sort(np.unique(..., return_index=True))`` reduction, restricted to a
   single already-cropped tile.

A tile is small, so this stays in memory (a national-scale run is bounded by
the executor's *tiling*, not by an out-of-core sweep here). :meth:`halo_m`
returns ``0``: de-duplication only ever selects a subset of the tile's own
points, never reading past its bounds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ahn_cli.pipeline.model import PointTile

if TYPE_CHECKING:
    import numpy.typing as npt

    from ahn_cli.pipeline.model import TileContext, TilePayload

__all__ = ["DedupStage"]


def _class_keep(
    classification: npt.NDArray[np.uint8],
    include: tuple[int, ...],
    exclude: tuple[int, ...],
) -> npt.NDArray[np.bool_]:
    """Return the classification-filter keep-mask (empty/empty keeps all)."""
    keep = np.ones(classification.shape[0], dtype=np.bool_)
    if include:
        keep &= np.isin(classification, np.asarray(include))
    if exclude:
        keep &= ~np.isin(classification, np.asarray(exclude))
    return keep


def _select(tile: PointTile, indices: npt.NDArray[np.intp]) -> PointTile:
    """Return a new :class:`PointTile` holding only ``tile``'s rows at ``indices``."""
    rgb = (
        np.ascontiguousarray(tile.rgb[indices])
        if tile.rgb is not None
        else None
    )
    return PointTile(
        x=np.ascontiguousarray(tile.x[indices]),
        y=np.ascontiguousarray(tile.y[indices]),
        z=np.ascontiguousarray(tile.z[indices]),
        gps_time=np.ascontiguousarray(tile.gps_time[indices]),
        classification=np.ascontiguousarray(tile.classification[indices]),
        rgb=rgb,
    )


def _unique_first_indices(tile: PointTile) -> npt.NDArray[np.intp]:
    """Return the ascending smallest-index survivors of the exact-XYZ+gps sweep.

    Mirrors the oracle: group by exact ``(x, y, z, gps_time)`` equality, keep
    the smallest original index of each group, and return those indices sorted
    ascending. Built with a single lexsort + a run-start mask (rather than
    ``np.unique``'s hashing) so the smallest-index rule is explicit and the
    order is deterministic.
    """
    count = tile.x.shape[0]
    order = np.lexsort((tile.gps_time, tile.z, tile.y, tile.x))
    sx = tile.x[order]
    sy = tile.y[order]
    sz = tile.z[order]
    sg = tile.gps_time[order]
    is_start = np.ones(count, dtype=np.bool_)
    if count > 1:
        is_start[1:] = (
            (sx[1:] != sx[:-1])
            | (sy[1:] != sy[:-1])
            | (sz[1:] != sz[:-1])
            | (sg[1:] != sg[:-1])
        )
    # Within each equal-key run the smallest original index is the survivor;
    # np.minimum.reduceat over the (index-ascending within a key is not
    # guaranteed by lexsort) run needs the per-run minimum, so reduce it.
    starts = np.flatnonzero(is_start)
    group_min = np.minimum.reduceat(order, starts) if count else order
    return np.sort(group_min).astype(np.intp)


@dataclass(frozen=True)
class DedupStage:
    """The ``dedup`` pipeline stage: class filter, then exact duplicate sweep.

    Contract:
        - ``include_classes`` / ``exclude_classes`` are the classification
          filter applied first; empty tuples mean "no filter on that side".
        - :meth:`run` returns a :class:`~ahn_cli.pipeline.model.PointTile`
          holding the smallest-index survivor of each exact
          ``(x, y, z, gps_time)`` group, in ascending index order.

    Invariants:
        - Frozen value object, equal by field value.
        - :meth:`halo_m` is always ``0.0``: de-duplication is tile-local.

    Failure modes:
        - :class:`TypeError` if :meth:`run` is given a payload other than a
          :class:`~ahn_cli.pipeline.model.PointTile`.
    """

    include_classes: tuple[int, ...] = field(default_factory=tuple)
    exclude_classes: tuple[int, ...] = field(default_factory=tuple)

    def halo_m(self) -> float:
        """Return ``0.0``: de-duplication never reads beyond the tile."""
        return 0.0

    def run(self, tile: TilePayload, ctx: TileContext) -> TilePayload:  # noqa: ARG002 -- ctx unused; the sweep is tile-local
        """Class-filter, then exact-duplicate-sweep ``tile``.

        Failure modes:
            - :class:`TypeError` if ``tile`` is not a
              :class:`~ahn_cli.pipeline.model.PointTile`.
        """
        if not isinstance(tile, PointTile):
            msg = (
                "DedupStage requires a PointTile payload; got "
                f"{type(tile).__name__}."
            )
            raise TypeError(msg)
        filtered = self._filter(tile)
        survivors = _unique_first_indices(filtered)
        return _select(filtered, survivors)

    def _filter(self, tile: PointTile) -> PointTile:
        """Apply the classification filter only (no sweep)."""
        if not self.include_classes and not self.exclude_classes:
            return tile
        keep = _class_keep(
            tile.classification, self.include_classes, self.exclude_classes
        )
        return _select(tile, np.flatnonzero(keep).astype(np.intp))
