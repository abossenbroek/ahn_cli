"""The pipeline spec parser/validator: PDAL-style YAML/JSON, source of truth.

A pipeline spec names an area of interest, a tiling plan, scratch/output
locations, and an ordered chain of stages, e.g.::

    aoi:      { geojson: westland.geojson }
    tiling:   { grid: quadtree, tile_pixels: 256, halo: auto }
    workdir:  /workspace/scratch
    output:   data/westland/tiles3d
    stages:
      - { type: fetch,     ahn_generation: ahn5, source: geotiles }
      - { type: dedup }
      - { type: thin,      method: voxel, voxel_size_m: 1.0 }
      - { type: reconcile, method: idw, idw: { power: 2, neighbors: 12 } }
      - { type: tiles3d,   profile: splat }

:func:`parse_yaml` and :func:`parse_json` are the two entry points; both
validate at parse time (no execution, no I/O beyond reading the given text)
and return a fully typed, immutable :class:`PipelineSpec`. Every failure --
malformed YAML/JSON, an unknown or mistyped key, a stage chain that does not
start with a source (``fetch``/``read``) and end with a sink
(``tiles3d``/``write``), or an out-of-range parameter -- raises the single
:class:`~ahn_cli.pipeline.errors.PipelineError`; no other exception escapes
either entry point.

The spec favours long, self-explanatory keys over the terse strings the
existing verbs' CLI options accept (``idw: {power, neighbors}`` rather than
``"2,12"``, ``voxel_size_m`` rather than a bare grade index), but reuses each
verb's own typed value objects wherever one already exists (``SourceKind``,
``ThinMethod``, ``InterpMethod``/``IdwInterp``/``KrigingInterp``, ``Profile``)
so the spec and the CLI can never disagree about what a parameter means.

:func:`canonical` renders a parsed spec back to a sorted-key,
JSON-serialisable ``dict`` and :func:`spec_hash` reduces that to a stable
SHA-256 hex digest for provenance -- YAML and JSON documents describing the
same spec parse to equal :class:`PipelineSpec` values and therefore hash
identically, regardless of source key order.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

import yaml

from ahn_cli.domain import BBox, Generation, ensure_valid_bbox
from ahn_cli.fetch.generation import (
    AUTO_CHOICE,
    UnknownGenerationError,
    default_registry,
)
from ahn_cli.fetch.source import (
    SourceKind,
    UnknownSourceError,
    resolve_source_token,
)
from ahn_cli.pipeline.errors import PipelineError
from ahn_cli.prep.decimate import (
    DEFAULT_SEED,
    ThinMethod,
    voxel_size_for_grade,
)
from ahn_cli.reconcile.method import (
    DEFAULT_IDW_K,
    DEFAULT_IDW_POWER,
    DEFAULT_KRIGING_K,
    IdwInterp,
    InterpMethod,
    KrigingInterp,
    LinearInterp,
    Variogram,
    VariogramModel,
)
from ahn_cli.tiles3d.errors import Tiles3dError
from ahn_cli.tiles3d.profile import Profile

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "HALO_AUTO",
    "AoiSpec",
    "DedupStage",
    "FetchStage",
    "PipelineSpec",
    "ReadStage",
    "ReconcileStage",
    "StageSpec",
    "ThinStage",
    "Tiles3dStage",
    "TilingSpec",
    "WriteStage",
    "canonical",
    "parse_json",
    "parse_yaml",
    "spec_hash",
    "stage_type",
]

HALO_AUTO: Final = "auto"
"""The ``tiling.halo`` token requesting RAM-adaptive halo sizing."""

_DEFAULT_TILE_PIXELS = 256
"""Default ``tiling.tile_pixels`` when the key is omitted."""

_DEFAULT_DOWNLOAD_JOBS = 1
"""Default ``fetch`` stage ``download_jobs`` when the key is omitted."""

_DEFAULT_KRIGING_MODEL = "spherical"
"""Default ``reconcile`` stage kriging variogram model."""

_DEFAULT_KRIGING_NUGGET = 0.0
"""Default ``reconcile`` stage kriging nugget."""

_DEFAULT_KRIGING_SILL = 1.0
"""Default ``reconcile`` stage kriging sill."""

_DEFAULT_KRIGING_RANGE_M = 50.0
"""Default ``reconcile`` stage kriging range, in metres."""

_BBOX_FIELD_COUNT = 4
"""An ``aoi.bbox`` string has exactly ``minx,miny,maxx,maxy``."""

_TOP_LEVEL_KEYS = frozenset({"aoi", "tiling", "workdir", "output", "stages"})
_REQUIRED_TOP_LEVEL_KEYS = frozenset({"aoi", "workdir", "output", "stages"})
_AOI_KEYS = frozenset({"geojson", "bbox"})
_TILING_KEYS = frozenset({"grid", "tile_pixels", "halo"})
_FETCH_KEYS = frozenset(
    {"type", "ahn_generation", "source", "ortho", "download_jobs"}
)
_READ_KEYS = frozenset({"type", "path"})
_DEDUP_KEYS = frozenset({"type", "include_classes", "exclude_classes"})
_THIN_KEYS = frozenset(
    {"type", "method", "voxel_size_m", "voxel_grade", "radius_m", "seed"}
)
_RECONCILE_KEYS = frozenset({"type", "method", "idw", "kriging"})
_TILES3D_KEYS = frozenset({"type", "profile"})
_WRITE_KEYS = frozenset({"type", "path"})
_IDW_KEYS = frozenset({"power", "neighbors"})
_KRIGING_KEYS = frozenset({"model", "nugget", "sill", "range_m", "neighbors"})

_GENERATION_REGISTRY = default_registry()
"""The AHN generation registry backing a ``fetch`` stage's ``ahn_generation``."""


@dataclass(frozen=True)
class AoiSpec:
    """The pipeline's area of interest: a GeoJSON file or an EPSG:28992 bbox.

    Contract:
        - Exactly one of ``geojson`` / ``bbox`` is set.
        - ``geojson`` is a filesystem path, as given (existence is an
          executor-time concern, not a parse-time one).
        - ``bbox`` is an EPSG:28992 ``(minx, miny, maxx, maxy)`` box.

    Invariants:
        - Frozen value object, equal by field value.

    Failure modes:
        - :class:`~ahn_cli.pipeline.errors.PipelineError` if neither or both
          of ``geojson``/``bbox`` are set, or ``bbox`` is degenerate.
    """

    geojson: str | None
    bbox: BBox | None

    def __post_init__(self) -> None:
        """Reject neither-or-both selectors and a degenerate ``bbox``."""
        if (self.geojson is None) == (self.bbox is None):
            msg = "aoi must specify exactly one of geojson or bbox."
            raise PipelineError(msg)
        if self.bbox is not None:
            try:
                ensure_valid_bbox(self.bbox)
            except ValueError as exc:
                raise PipelineError(str(exc)) from exc


@dataclass(frozen=True)
class TilingSpec:
    """The pipeline's sink-driven tiling parameters.

    Contract:
        - ``grid`` names the tiling scheme, or ``None`` to default from the
          sink stage at execution time.
        - ``tile_pixels`` is the tile edge length in pixels; a positive
          integer.
        - ``halo`` is :data:`HALO_AUTO` for RAM-adaptive sizing, or a
          non-negative metres float.

    Invariants:
        - Frozen value object, equal by field value.

    Failure modes:
        - :class:`~ahn_cli.pipeline.errors.PipelineError` if ``tile_pixels``
          is not positive, or ``halo`` is neither :data:`HALO_AUTO` nor a
          finite non-negative number.
    """

    grid: str | None
    tile_pixels: int
    halo: float | str

    def __post_init__(self) -> None:
        """Validate ``tile_pixels`` and ``halo``."""
        if self.tile_pixels < 1:
            msg = (
                "tiling tile_pixels must be a positive integer; got "
                f"{self.tile_pixels}."
            )
            raise PipelineError(msg)
        if isinstance(self.halo, str):
            if self.halo != HALO_AUTO:
                msg = (
                    f"tiling halo must be {HALO_AUTO!r} or a metres number; "
                    f"got {self.halo!r}."
                )
                raise PipelineError(msg)
            return
        if not math.isfinite(self.halo) or self.halo < 0.0:
            msg = f"tiling halo must be finite and non-negative; got {self.halo}."
            raise PipelineError(msg)


@dataclass(frozen=True)
class FetchStage:
    """A ``fetch`` stage: acquire the tile's AHN (+ optional ortho) sheets.

    Contract:
        - ``ahn_generation`` is an explicit :class:`~ahn_cli.domain.Generation`,
          or ``None`` for automatic newest-available selection (the spec's
          ``ahn_generation: auto`` or an omitted key).
        - ``source`` selects the distribution portal.
        - ``ortho`` requests the matching orthophoto window alongside AHN.
        - ``download_jobs`` is the per-tile download concurrency; a positive
          integer.

    Invariants:
        - Frozen value object, equal by field value.

    Failure modes:
        - :class:`~ahn_cli.pipeline.errors.PipelineError` if ``download_jobs``
          is not a positive integer.
    """

    ahn_generation: Generation | None
    source: SourceKind
    ortho: bool
    download_jobs: int

    def __post_init__(self) -> None:
        """Reject a non-positive ``download_jobs``."""
        if self.download_jobs < 1:
            msg = (
                "fetch stage download_jobs must be a positive integer; "
                f"got {self.download_jobs}."
            )
            raise PipelineError(msg)


@dataclass(frozen=True)
class ReadStage:
    """A ``read`` stage: seed the pipeline from an existing on-disk source.

    Contract:
        - ``path`` names the pre-populated location to read from in place of
          a network ``fetch``; non-blank.

    Invariants:
        - Frozen value object, equal by field value.

    Failure modes:
        - :class:`~ahn_cli.pipeline.errors.PipelineError` if ``path`` is blank.
    """

    path: str

    def __post_init__(self) -> None:
        """Reject a blank ``path``."""
        if not self.path.strip():
            msg = "read stage path must be a non-blank location."
            raise PipelineError(msg)


@dataclass(frozen=True)
class DedupStage:
    """A ``dedup`` stage: de-duplicate and optionally class-filter a tile.

    Contract:
        - ``include_classes`` keeps only these classification codes (empty
          tuple keeps every class).
        - ``exclude_classes`` drops these classification codes (empty tuple
          drops none).
        - A code must not appear in both lists.

    Invariants:
        - Frozen value object, equal by field value.

    Failure modes:
        - :class:`~ahn_cli.pipeline.errors.PipelineError` if any code is both
          included and excluded.
    """

    include_classes: tuple[int, ...]
    exclude_classes: tuple[int, ...]

    def __post_init__(self) -> None:
        """Reject a classification code that is both included and excluded."""
        overlap = sorted(
            set(self.include_classes) & set(self.exclude_classes)
        )
        if overlap:
            msg = (
                "dedup stage classes cannot be both included and excluded: "
                f"{overlap}."
            )
            raise PipelineError(msg)


@dataclass(frozen=True)
class ThinStage:
    """A ``thin`` stage: graded voxel-grid or Poisson-disk thinning.

    Contract:
        - ``method`` selects voxel-grid or Poisson-disk thinning.
        - Voxel: ``voxel_size_m`` is the resolved edge length in metres
          (from an explicit ``voxel_size_m`` or a ``voxel_grade`` shorthand
          mapped via
          :func:`~ahn_cli.prep.decimate.voxel_size_for_grade`); ``radius_m``
          is ``None``.
        - Poisson: ``radius_m`` is the minimum spacing in metres and
          ``seed`` the RNG seed; ``voxel_size_m`` is ``None``.

    Invariants:
        - Frozen value object, equal by field value.

    Failure modes:
        - :class:`~ahn_cli.pipeline.errors.PipelineError` if the method's
          parameter is missing, negative, or the other method's parameter is
          set.
    """

    method: ThinMethod
    voxel_size_m: float | None
    radius_m: float | None
    seed: int

    def __post_init__(self) -> None:
        """Validate the parameter set matches the chosen method."""
        if self.method is ThinMethod.VOXEL:
            if self.radius_m is not None:
                msg = "thin stage voxel method does not take radius_m."
                raise PipelineError(msg)
            if self.voxel_size_m is None or self.voxel_size_m < 0.0:
                msg = (
                    "thin stage voxel method requires a non-negative "
                    f"voxel_size_m; got {self.voxel_size_m!r}."
                )
                raise PipelineError(msg)
            return
        if self.voxel_size_m is not None:
            msg = "thin stage poisson method does not take voxel_size_m."
            raise PipelineError(msg)
        if self.radius_m is None or self.radius_m <= 0.0:
            msg = (
                "thin stage poisson method requires a positive radius_m; "
                f"got {self.radius_m!r}."
            )
            raise PipelineError(msg)


@dataclass(frozen=True)
class ReconcileStage:
    """A ``reconcile`` stage: interpolate the tile's ortho grid from points.

    Contract:
        - ``method`` is the reused
          :data:`~ahn_cli.reconcile.method.InterpMethod` value object
          (:class:`~ahn_cli.reconcile.method.LinearInterp`,
          :class:`~ahn_cli.reconcile.method.IdwInterp`, or
          :class:`~ahn_cli.reconcile.method.KrigingInterp`), already
          validated by its own type.

    Invariants:
        - Frozen value object, equal by field value.
    """

    method: InterpMethod


@dataclass(frozen=True)
class Tiles3dStage:
    """A ``tiles3d`` stage: encode the tile into the sink's export profile.

    Contract:
        - ``profile`` selects the tiles3d export profile (reused from
          :mod:`ahn_cli.tiles3d.profile`).

    Invariants:
        - Frozen value object, equal by field value.
    """

    profile: Profile


@dataclass(frozen=True)
class WriteStage:
    """A ``write`` stage: sink the pipeline to a raw on-disk location.

    Contract:
        - ``path`` names the destination the sink writes to; non-blank.

    Invariants:
        - Frozen value object, equal by field value.

    Failure modes:
        - :class:`~ahn_cli.pipeline.errors.PipelineError` if ``path`` is blank.
    """

    path: str

    def __post_init__(self) -> None:
        """Reject a blank ``path``."""
        if not self.path.strip():
            msg = "write stage path must be a non-blank location."
            raise PipelineError(msg)


StageSpec = (
    DedupStage
    | FetchStage
    | ReadStage
    | ReconcileStage
    | ThinStage
    | Tiles3dStage
    | WriteStage
)
"""One parsed, validated pipeline stage (the ``stages`` chain's element type)."""

_STAGE_TYPE_BY_CLASS: dict[type, str] = {
    FetchStage: "fetch",
    ReadStage: "read",
    DedupStage: "dedup",
    ThinStage: "thin",
    ReconcileStage: "reconcile",
    Tiles3dStage: "tiles3d",
    WriteStage: "write",
}

_SOURCE_TYPES = frozenset({"fetch", "read"})
_SINK_TYPES = frozenset({"tiles3d", "write"})


def stage_type(stage: StageSpec) -> str:
    """Return ``stage``'s canonical ``type`` token (e.g. ``"fetch"``)."""
    return _STAGE_TYPE_BY_CLASS[type(stage)]


@dataclass(frozen=True)
class PipelineSpec:
    """A fully parsed and validated pipeline spec.

    Contract:
        - ``aoi`` / ``tiling`` / ``workdir`` / ``output`` are the spec's
          top-level settings.
        - ``stages`` is the non-empty, ordered stage chain; the first stage
          must be a source (``fetch``/``read``) and the last a sink
          (``tiles3d``/``write``).

    Invariants:
        - Frozen value object, equal by field value.

    Failure modes:
        - :class:`~ahn_cli.pipeline.errors.PipelineError` if ``stages`` is
          empty, its first stage is not a source, or its last is not a sink.
    """

    aoi: AoiSpec
    tiling: TilingSpec
    workdir: Path
    output: Path
    stages: tuple[StageSpec, ...]

    def __post_init__(self) -> None:
        """Validate the stage chain is non-empty and source-first/sink-last."""
        if not self.stages:
            msg = "pipeline spec must declare at least one stage."
            raise PipelineError(msg)
        first = stage_type(self.stages[0])
        if first not in _SOURCE_TYPES:
            msg = (
                "the first stage must be a source (fetch/read); got "
                f"{first!r}."
            )
            raise PipelineError(msg)
        last = stage_type(self.stages[-1])
        if last not in _SINK_TYPES:
            msg = (
                "the last stage must be a sink (tiles3d/write); got "
                f"{last!r}."
            )
            raise PipelineError(msg)


# --------------------------------------------------------------------------
# Raw-value coercion helpers (shape-only; semantic validation lives on the
# value objects above so each concern is checked in exactly one place).
# --------------------------------------------------------------------------


def _require_mapping(value: object, context: str) -> dict[str, Any]:
    """Return ``value`` as a mapping, or raise if it is not one."""
    if not isinstance(value, dict):
        msg = f"{context} must be a mapping; got {type(value).__name__}."
        raise PipelineError(msg)
    return cast("dict[str, Any]", value)


def _reject_unknown_keys(
    raw: dict[str, Any], allowed: frozenset[str], context: str
) -> None:
    """Reject any key in ``raw`` outside ``allowed``."""
    unknown = sorted(set(raw) - allowed)
    if unknown:
        msg = (
            f"{context} has unknown key(s) {unknown}; allowed: "
            f"{sorted(allowed)}."
        )
        raise PipelineError(msg)


def _require_str(raw: dict[str, Any], key: str, context: str) -> str:
    """Return ``raw[key]`` as a non-blank string, or raise."""
    if key not in raw:
        msg = f"{context} is missing required key {key!r}."
        raise PipelineError(msg)
    value = raw[key]
    if not isinstance(value, str) or not value.strip():
        msg = f"{context} key {key!r} must be a non-blank string; got {value!r}."
        raise PipelineError(msg)
    return value


def _optional_int(
    raw: dict[str, Any], key: str, default: int, context: str
) -> int:
    """Return ``raw[key]`` as an ``int``, or ``default`` if absent."""
    if key not in raw:
        return default
    value = raw[key]
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"{context} key {key!r} must be an integer; got {value!r}."
        raise PipelineError(msg)
    return value


def _optional_number(
    raw: dict[str, Any], key: str, default: float, context: str
) -> float:
    """Return ``raw[key]`` as a ``float``, or ``default`` if absent."""
    if key not in raw:
        return default
    value = raw[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = f"{context} key {key!r} must be a number; got {value!r}."
        raise PipelineError(msg)
    return float(value)


def _optional_bool(
    raw: dict[str, Any], key: str, *, default: bool, context: str
) -> bool:
    """Return ``raw[key]`` as a ``bool``, or ``default`` if absent."""
    if key not in raw:
        return default
    value = raw[key]
    if not isinstance(value, bool):
        msg = f"{context} key {key!r} must be a boolean; got {value!r}."
        raise PipelineError(msg)
    return value


def _optional_int_list(
    raw: dict[str, Any], key: str, context: str
) -> tuple[int, ...]:
    """Return ``raw[key]`` as a tuple of ``int``, or ``()`` if absent."""
    if key not in raw:
        return ()
    value = raw[key]
    if not isinstance(value, list):
        msg = f"{context} key {key!r} must be a list of integers; got {value!r}."
        raise PipelineError(msg)
    codes: list[int] = []
    for item in cast("list[Any]", value):
        if isinstance(item, bool) or not isinstance(item, int):
            msg = (
                f"{context} key {key!r} must be a list of integers; got "
                f"{value!r}."
            )
            raise PipelineError(msg)
        codes.append(item)
    return tuple(codes)


# --------------------------------------------------------------------------
# aoi / tiling
# --------------------------------------------------------------------------


def _parse_bbox_string(text: str) -> BBox:
    """Parse a ``minx,miny,maxx,maxy`` string into a :data:`BBox` tuple."""
    parts = text.split(",")
    if len(parts) != _BBOX_FIELD_COUNT:
        msg = f"aoi bbox must be 'minx,miny,maxx,maxy'; got {text!r}."
        raise PipelineError(msg)
    try:
        minx, miny, maxx, maxy = (float(part) for part in parts)
    except ValueError as exc:
        msg = f"aoi bbox has a non-numeric coordinate: {text!r}."
        raise PipelineError(msg) from exc
    return (minx, miny, maxx, maxy)


def _parse_aoi(value: object) -> AoiSpec:
    """Parse the ``aoi`` mapping into an :class:`AoiSpec`."""
    raw = _require_mapping(value, "aoi")
    _reject_unknown_keys(raw, _AOI_KEYS, "aoi")
    has_geojson = "geojson" in raw
    has_bbox = "bbox" in raw
    if has_geojson == has_bbox:
        msg = "aoi must specify exactly one of geojson or bbox."
        raise PipelineError(msg)
    if has_geojson:
        return AoiSpec(geojson=_require_str(raw, "geojson", "aoi"), bbox=None)
    bbox_text = _require_str(raw, "bbox", "aoi")
    return AoiSpec(geojson=None, bbox=_parse_bbox_string(bbox_text))


def _parse_halo(value: object) -> float | str:
    """Coerce a raw ``tiling.halo`` value to a string or a number."""
    if isinstance(value, str):
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = (
            f"tiling key 'halo' must be {HALO_AUTO!r} or a number; got "
            f"{value!r}."
        )
        raise PipelineError(msg)
    return float(value)


def _parse_tiling(value: object | None) -> TilingSpec:
    """Parse the optional ``tiling`` mapping into a :class:`TilingSpec`."""
    if value is None:
        return TilingSpec(
            grid=None, tile_pixels=_DEFAULT_TILE_PIXELS, halo=HALO_AUTO
        )
    raw = _require_mapping(value, "tiling")
    _reject_unknown_keys(raw, _TILING_KEYS, "tiling")
    grid_value = raw.get("grid")
    if grid_value is not None and not isinstance(grid_value, str):
        msg = f"tiling key 'grid' must be a string; got {grid_value!r}."
        raise PipelineError(msg)
    tile_pixels = _optional_int(
        raw, "tile_pixels", _DEFAULT_TILE_PIXELS, "tiling"
    )
    halo = _parse_halo(raw.get("halo", HALO_AUTO))
    return TilingSpec(grid=grid_value, tile_pixels=tile_pixels, halo=halo)


# --------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------


def _parse_ahn_generation(value: object) -> Generation | None:
    """Resolve a raw ``fetch`` stage ``ahn_generation`` token."""
    if not isinstance(value, str):
        msg = (
            f"fetch stage key 'ahn_generation' must be a string; got "
            f"{value!r}."
        )
        raise PipelineError(msg)
    try:
        return _GENERATION_REGISTRY.resolve_token(value)
    except UnknownGenerationError as exc:
        raise PipelineError(str(exc)) from exc


def _parse_source(value: object) -> SourceKind:
    """Resolve a raw ``fetch`` stage ``source`` token."""
    if not isinstance(value, str):
        msg = f"fetch stage key 'source' must be a string; got {value!r}."
        raise PipelineError(msg)
    try:
        return resolve_source_token(value)
    except UnknownSourceError as exc:
        raise PipelineError(str(exc)) from exc


def _parse_fetch_stage(raw: dict[str, Any]) -> FetchStage:
    """Parse a ``{type: fetch, ...}`` mapping into a :class:`FetchStage`."""
    _reject_unknown_keys(raw, _FETCH_KEYS, "fetch stage")
    generation = _parse_ahn_generation(raw.get("ahn_generation", AUTO_CHOICE))
    source = _parse_source(raw.get("source", SourceKind.PDOK.value))
    ortho = _optional_bool(raw, "ortho", default=False, context="fetch stage")
    download_jobs = _optional_int(
        raw, "download_jobs", _DEFAULT_DOWNLOAD_JOBS, "fetch stage"
    )
    return FetchStage(
        ahn_generation=generation,
        source=source,
        ortho=ortho,
        download_jobs=download_jobs,
    )


def _parse_read_stage(raw: dict[str, Any]) -> ReadStage:
    """Parse a ``{type: read, ...}`` mapping into a :class:`ReadStage`."""
    _reject_unknown_keys(raw, _READ_KEYS, "read stage")
    return ReadStage(path=_require_str(raw, "path", "read stage"))


def _parse_write_stage(raw: dict[str, Any]) -> WriteStage:
    """Parse a ``{type: write, ...}`` mapping into a :class:`WriteStage`."""
    _reject_unknown_keys(raw, _WRITE_KEYS, "write stage")
    return WriteStage(path=_require_str(raw, "path", "write stage"))


def _parse_dedup_stage(raw: dict[str, Any]) -> DedupStage:
    """Parse a ``{type: dedup, ...}`` mapping into a :class:`DedupStage`."""
    _reject_unknown_keys(raw, _DEDUP_KEYS, "dedup stage")
    include = _optional_int_list(raw, "include_classes", "dedup stage")
    exclude = _optional_int_list(raw, "exclude_classes", "dedup stage")
    return DedupStage(include_classes=include, exclude_classes=exclude)


def _parse_thin_stage(raw: dict[str, Any]) -> ThinStage:
    """Parse a ``{type: thin, ...}`` mapping into a :class:`ThinStage`."""
    _reject_unknown_keys(raw, _THIN_KEYS, "thin stage")
    method_value = raw.get("method")
    if not isinstance(method_value, str):
        msg = "thin stage is missing a string 'method' (voxel or poisson)."
        raise PipelineError(msg)
    try:
        method = ThinMethod(method_value)
    except ValueError as exc:
        choices = [member.value for member in ThinMethod]
        msg = (
            f"thin stage has unknown method {method_value!r}; choose one of "
            f"{choices}."
        )
        raise PipelineError(msg) from exc
    if method is ThinMethod.VOXEL:
        return _parse_voxel_thin(raw)
    return _parse_poisson_thin(raw)


def _parse_voxel_thin(raw: dict[str, Any]) -> ThinStage:
    """Parse the voxel branch of a ``thin`` stage."""
    has_size = "voxel_size_m" in raw
    has_grade = "voxel_grade" in raw
    if "radius_m" in raw:
        msg = "thin stage voxel method does not take radius_m."
        raise PipelineError(msg)
    if has_size == has_grade:
        msg = (
            "thin stage voxel method needs exactly one of voxel_size_m or "
            "voxel_grade."
        )
        raise PipelineError(msg)
    if has_grade:
        grade_value = raw["voxel_grade"]
        if isinstance(grade_value, bool) or not isinstance(grade_value, int):
            msg = (
                f"thin stage key 'voxel_grade' must be an integer; got "
                f"{grade_value!r}."
            )
            raise PipelineError(msg)
        try:
            size = voxel_size_for_grade(grade_value)
        except ValueError as exc:
            raise PipelineError(str(exc)) from exc
    else:
        size_value = raw["voxel_size_m"]
        if isinstance(size_value, bool) or not isinstance(
            size_value, (int, float)
        ):
            msg = (
                f"thin stage key 'voxel_size_m' must be a number; got "
                f"{size_value!r}."
            )
            raise PipelineError(msg)
        size = float(size_value)
    return ThinStage(
        method=ThinMethod.VOXEL,
        voxel_size_m=size,
        radius_m=None,
        seed=DEFAULT_SEED,
    )


def _parse_poisson_thin(raw: dict[str, Any]) -> ThinStage:
    """Parse the poisson branch of a ``thin`` stage."""
    if "voxel_size_m" in raw or "voxel_grade" in raw:
        msg = (
            "thin stage poisson method does not take voxel_size_m/"
            "voxel_grade."
        )
        raise PipelineError(msg)
    if "radius_m" not in raw:
        msg = "thin stage poisson method requires radius_m."
        raise PipelineError(msg)
    radius_value = raw["radius_m"]
    if isinstance(radius_value, bool) or not isinstance(
        radius_value, (int, float)
    ):
        msg = (
            f"thin stage key 'radius_m' must be a number; got "
            f"{radius_value!r}."
        )
        raise PipelineError(msg)
    seed = _optional_int(raw, "seed", DEFAULT_SEED, "thin stage")
    return ThinStage(
        method=ThinMethod.POISSON,
        voxel_size_m=None,
        radius_m=float(radius_value),
        seed=seed,
    )


def _parse_idw(value: object) -> IdwInterp:
    """Parse a ``reconcile`` stage's ``idw`` sub-mapping."""
    raw = _require_mapping(value, "reconcile stage idw")
    _reject_unknown_keys(raw, _IDW_KEYS, "reconcile stage idw")
    power = _optional_number(
        raw, "power", DEFAULT_IDW_POWER, "reconcile stage idw"
    )
    neighbors = _optional_int(
        raw, "neighbors", DEFAULT_IDW_K, "reconcile stage idw"
    )
    try:
        return IdwInterp(power=power, k=neighbors)
    except ValueError as exc:
        raise PipelineError(str(exc)) from exc


def _parse_kriging(value: object) -> KrigingInterp:
    """Parse a ``reconcile`` stage's ``kriging`` sub-mapping."""
    raw = _require_mapping(value, "reconcile stage kriging")
    _reject_unknown_keys(raw, _KRIGING_KEYS, "reconcile stage kriging")
    model_value = raw.get("model", _DEFAULT_KRIGING_MODEL)
    if not isinstance(model_value, str):
        msg = (
            f"reconcile stage kriging key 'model' must be a string; got "
            f"{model_value!r}."
        )
        raise PipelineError(msg)
    try:
        model = VariogramModel(model_value)
    except ValueError as exc:
        choices = [member.value for member in VariogramModel]
        msg = (
            f"reconcile stage kriging has unknown model {model_value!r}; "
            f"choose one of {choices}."
        )
        raise PipelineError(msg) from exc
    nugget = _optional_number(
        raw, "nugget", _DEFAULT_KRIGING_NUGGET, "reconcile stage kriging"
    )
    sill = _optional_number(
        raw, "sill", _DEFAULT_KRIGING_SILL, "reconcile stage kriging"
    )
    vrange = _optional_number(
        raw, "range_m", _DEFAULT_KRIGING_RANGE_M, "reconcile stage kriging"
    )
    neighbors = _optional_int(
        raw, "neighbors", DEFAULT_KRIGING_K, "reconcile stage kriging"
    )
    try:
        variogram = Variogram(
            model=model, nugget=nugget, sill=sill, vrange=vrange
        )
        return KrigingInterp(variogram=variogram, k=neighbors)
    except ValueError as exc:
        raise PipelineError(str(exc)) from exc


def _parse_reconcile_stage(raw: dict[str, Any]) -> ReconcileStage:
    """Parse a ``{type: reconcile, ...}`` mapping into a :class:`ReconcileStage`."""
    _reject_unknown_keys(raw, _RECONCILE_KEYS, "reconcile stage")
    method_value = raw.get("method")
    if not isinstance(method_value, str):
        msg = "reconcile stage is missing a string 'method'."
        raise PipelineError(msg)
    if method_value == "linear":
        return ReconcileStage(method=LinearInterp())
    if method_value == "idw":
        return ReconcileStage(method=_parse_idw(raw.get("idw", {})))
    if method_value == "kriging":
        return ReconcileStage(method=_parse_kriging(raw.get("kriging", {})))
    msg = (
        f"reconcile stage has unknown method {method_value!r}; choose one of "
        "'linear', 'idw', 'kriging'."
    )
    raise PipelineError(msg)


def _parse_tiles3d_stage(raw: dict[str, Any]) -> Tiles3dStage:
    """Parse a ``{type: tiles3d, ...}`` mapping into a :class:`Tiles3dStage`."""
    _reject_unknown_keys(raw, _TILES3D_KEYS, "tiles3d stage")
    profile_value = raw.get("profile", Profile.STRICT.value)
    if not isinstance(profile_value, str):
        msg = (
            f"tiles3d stage key 'profile' must be a string; got "
            f"{profile_value!r}."
        )
        raise PipelineError(msg)
    try:
        profile = Profile.parse(profile_value)
    except Tiles3dError as exc:
        raise PipelineError(str(exc)) from exc
    return Tiles3dStage(profile=profile)


_STAGE_PARSERS: dict[str, Callable[[dict[str, Any]], StageSpec]] = {
    "fetch": _parse_fetch_stage,
    "read": _parse_read_stage,
    "dedup": _parse_dedup_stage,
    "thin": _parse_thin_stage,
    "reconcile": _parse_reconcile_stage,
    "tiles3d": _parse_tiles3d_stage,
    "write": _parse_write_stage,
}


def _parse_stage(value: object, index: int) -> StageSpec:
    """Parse one ``stages[index]`` mapping, dispatching on its ``type``."""
    raw = _require_mapping(value, f"stage[{index}]")
    type_value = raw.get("type")
    if not isinstance(type_value, str):
        msg = f"stage[{index}] is missing a string 'type'."
        raise PipelineError(msg)
    parser = _STAGE_PARSERS.get(type_value)
    if parser is None:
        msg = (
            f"stage[{index}] has unknown type {type_value!r}; choose one of "
            f"{sorted(_STAGE_PARSERS)}."
        )
        raise PipelineError(msg)
    return parser(raw)


# --------------------------------------------------------------------------
# top-level parse entry points
# --------------------------------------------------------------------------


def _parse_mapping(raw_value: object) -> PipelineSpec:
    """Parse a fully-decoded YAML/JSON mapping into a :class:`PipelineSpec`."""
    raw = _require_mapping(raw_value, "pipeline spec")
    _reject_unknown_keys(raw, _TOP_LEVEL_KEYS, "pipeline spec")
    missing = sorted(_REQUIRED_TOP_LEVEL_KEYS - set(raw))
    if missing:
        msg = f"pipeline spec is missing required key(s) {missing}."
        raise PipelineError(msg)
    aoi = _parse_aoi(raw["aoi"])
    tiling = _parse_tiling(raw.get("tiling"))
    workdir = Path(_require_str(raw, "workdir", "pipeline spec"))
    output = Path(_require_str(raw, "output", "pipeline spec"))
    stages_value = raw["stages"]
    if not isinstance(stages_value, list) or not stages_value:
        msg = "pipeline spec 'stages' must be a non-empty list."
        raise PipelineError(msg)
    stages = tuple(
        _parse_stage(item, index)
        for index, item in enumerate(cast("list[Any]", stages_value))
    )
    return PipelineSpec(
        aoi=aoi, tiling=tiling, workdir=workdir, output=output, stages=stages
    )


def parse_yaml(text: str) -> PipelineSpec:
    """Parse and validate a YAML pipeline spec into a :class:`PipelineSpec`.

    Failure modes:
        - :class:`~ahn_cli.pipeline.errors.PipelineError` if ``text`` is not
          valid YAML, or the decoded document fails any spec validation rule.
    """
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        msg = f"pipeline spec is not valid YAML: {exc}"
        raise PipelineError(msg) from exc
    return _parse_mapping(raw)


def parse_json(text: str) -> PipelineSpec:
    """Parse and validate a JSON pipeline spec into a :class:`PipelineSpec`.

    Failure modes:
        - :class:`~ahn_cli.pipeline.errors.PipelineError` if ``text`` is not
          valid JSON, or the decoded document fails any spec validation rule.
    """
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        msg = f"pipeline spec is not valid JSON: {exc}"
        raise PipelineError(msg) from exc
    return _parse_mapping(raw)


# --------------------------------------------------------------------------
# canonical form + hash
# --------------------------------------------------------------------------


def _aoi_canonical(aoi: AoiSpec) -> dict[str, object]:
    """Return ``aoi``'s canonical, JSON-serialisable form."""
    if aoi.bbox is not None:
        return {"bbox": list(aoi.bbox)}
    return {"geojson": aoi.geojson}


def _tiling_canonical(tiling: TilingSpec) -> dict[str, object]:
    """Return ``tiling``'s canonical, JSON-serialisable form."""
    return {
        "grid": tiling.grid,
        "tile_pixels": tiling.tile_pixels,
        "halo": tiling.halo,
    }


def _interp_canonical(method: InterpMethod) -> dict[str, object]:
    """Return an ``InterpMethod``'s canonical, JSON-serialisable form."""
    if isinstance(method, IdwInterp):
        return {"kind": "idw", "power": method.power, "neighbors": method.k}
    if isinstance(method, KrigingInterp):
        return {
            "kind": "kriging",
            "model": method.variogram.model.value,
            "nugget": method.variogram.nugget,
            "sill": method.variogram.sill,
            "range_m": method.variogram.vrange,
            "neighbors": method.k,
        }
    return {"kind": "linear"}


def _fetch_canonical(stage: FetchStage) -> dict[str, object]:
    """Return a :class:`FetchStage`'s canonical, JSON-serialisable form."""
    return {
        "type": "fetch",
        "ahn_generation": (
            "auto"
            if stage.ahn_generation is None
            else f"ahn{stage.ahn_generation.number}"
        ),
        "source": stage.source.value,
        "ortho": stage.ortho,
        "download_jobs": stage.download_jobs,
    }


def _read_canonical(stage: ReadStage) -> dict[str, object]:
    """Return a :class:`ReadStage`'s canonical, JSON-serialisable form."""
    return {"type": "read", "path": stage.path}


def _dedup_canonical(stage: DedupStage) -> dict[str, object]:
    """Return a :class:`DedupStage`'s canonical, JSON-serialisable form."""
    return {
        "type": "dedup",
        "include_classes": list(stage.include_classes),
        "exclude_classes": list(stage.exclude_classes),
    }


def _thin_canonical(stage: ThinStage) -> dict[str, object]:
    """Return a :class:`ThinStage`'s canonical, JSON-serialisable form."""
    return {
        "type": "thin",
        "method": stage.method.value,
        "voxel_size_m": stage.voxel_size_m,
        "radius_m": stage.radius_m,
        "seed": stage.seed,
    }


def _reconcile_canonical(stage: ReconcileStage) -> dict[str, object]:
    """Return a :class:`ReconcileStage`'s canonical, JSON-serialisable form."""
    return {"type": "reconcile", "method": _interp_canonical(stage.method)}


def _tiles3d_canonical(stage: Tiles3dStage) -> dict[str, object]:
    """Return a :class:`Tiles3dStage`'s canonical, JSON-serialisable form."""
    return {"type": "tiles3d", "profile": stage.profile.value}


def _write_canonical(stage: WriteStage) -> dict[str, object]:
    """Return a :class:`WriteStage`'s canonical, JSON-serialisable form."""
    return {"type": "write", "path": stage.path}


_CANONICAL_BUILDERS: dict[type, Callable[[Any], dict[str, object]]] = {
    FetchStage: _fetch_canonical,
    ReadStage: _read_canonical,
    DedupStage: _dedup_canonical,
    ThinStage: _thin_canonical,
    ReconcileStage: _reconcile_canonical,
    Tiles3dStage: _tiles3d_canonical,
    WriteStage: _write_canonical,
}


def _stage_canonical(stage: StageSpec) -> dict[str, object]:
    """Return one stage's canonical, JSON-serialisable form."""
    return _CANONICAL_BUILDERS[type(stage)](stage)


def canonical(spec: PipelineSpec) -> dict[str, object]:
    """Return ``spec``'s canonical, sorted-key, JSON-serialisable form.

    Contract:
        - Every field is rendered to a JSON primitive (``str``/``int``/
          ``float``/``bool``/``None``/``list``/``dict``), so
          ``json.dumps(canonical(spec), sort_keys=True)`` is stable
          regardless of the source dict's original key order.
        - Two :class:`PipelineSpec` values that compare equal produce equal
          canonical forms, and vice versa.
    """
    return {
        "aoi": _aoi_canonical(spec.aoi),
        "output": str(spec.output),
        "stages": [_stage_canonical(stage) for stage in spec.stages],
        "tiling": _tiling_canonical(spec.tiling),
        "workdir": str(spec.workdir),
    }


def spec_hash(spec: PipelineSpec) -> str:
    """Return the SHA-256 hex digest of ``spec``'s canonical form.

    Contract:
        - Deterministic: equal specs (however they were parsed -- YAML or
          JSON, any source key order) hash identically.
    """
    text = json.dumps(canonical(spec), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
