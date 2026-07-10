"""AHN generation selection: the fetch-context generation registry.

WP5 establishes the *generation registry*, the open-for-extension seam through
which every AHN survey generation (AHN4, AHN5, and a future AHN6, ...) is wired
into the fetch context. A generation is chosen -- explicitly, or by probing AOI
coverage newest-first -- purely by consulting the registry, so adding a
generation is a registry entry plus (later) its source module, never an edit to
a stringly-typed switch. WP6 (PDOK), WP8 (ortho) and WP9 (VIIRS) plug new
sources into this same mechanism.

The registry deliberately does not import the deprecated ``ahn_cli.config`` or
``ahn_cli.fetcher`` modules: importing them surfaces their module-level
``DeprecationWarning`` inside this gated module. The one datum it needs -- the
GeoTiles.nl base URL -- is duplicated here with a citation. Availability
probing is a per-generation callable carried in the registry entry, so tests
inject a deterministic fake and never touch the network; the default registry's
probes are un-wired placeholders until WP6 actuates real downloads.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from ahn_cli.domain import BBox, Generation, ensure_valid_bbox

AvailabilityProbe = Callable[[BBox], bool]
"""A per-generation AOI coverage probe.

Given an EPSG:28992 bounding box, reports whether this generation is available
for (covers) that area. Injected via the registry entry so tests supply a
deterministic fake and no network I/O occurs during selection.
"""

AUTO_CHOICE = "auto"
"""The ``--ahn`` token requesting automatic newest-available selection."""

# GeoTiles.nl AHN4 endpoint, duplicated from the deprecated ``ahn_cli.config``
# (``Config.geotiles_base_url``) so this gated module need not import that
# module and surface its module-level ``DeprecationWarning``. WP6 supersedes
# GeoTiles.nl with the PDOK ATOM distribution.
_AHN4_BASE_URL = "https://geotiles.citg.tudelft.nl/AHN4_T/"
# AHN5 GeoTiles.nl endpoint by the same naming convention. Not load-bearing
# until WP6 actuates downloads, and superseded by PDOK ATOM there.
_AHN5_BASE_URL = "https://geotiles.citg.tudelft.nl/AHN5_T/"

_AHN4_SEMANTICS = (
    "AHN4: surface model as an inverse-distance-weighted mean per cell; "
    "classification class 6 covers all buildings."
)
_AHN5_SEMANTICS = (
    "AHN5: surface model as the highest point per cell; classification class "
    "6 covers roofs only. Live for the western Netherlands."
)


class DuplicateGenerationError(ValueError):
    """Raised when registering a generation already in the registry.

    Signals a programming error in registry assembly: each generation may be
    registered at most once, so a duplicate would make selection ambiguous.
    """


class UnknownGenerationError(LookupError):
    """Raised when a generation (or ``--ahn`` token) is not in the registry.

    Signals that selection was asked for a generation no source is wired for;
    callers should offer only the registry's own
    :meth:`GenerationRegistry.tokens`.
    """


class GenerationUnavailableError(RuntimeError):
    """Raised when automatic selection finds no available generation.

    Signals that every registered generation's coverage probe reported the AOI
    uncovered, so no source can serve the request.
    """


class CoverageProbeNotWiredError(NotImplementedError):
    """Raised by the default registry's placeholder coverage probe.

    Real AOI coverage probing lands in WP6 (PDOK ATOM); until then the default
    registry records each generation's source but cannot actuate an ``auto``
    probe. Tests inject their own probe and never trigger this.
    """


@dataclass(frozen=True)
class GenerationSource:
    """A registry entry describing how to fetch one AHN generation.

    Contract:
        generation: The AHN generation this entry sources.
        base_url: The tile endpoint base URL; must be non-blank.
        probe: The AOI coverage probe consulted by ``auto`` selection.
        semantics: A human-readable note on the generation's surface-model and
            classification semantics, carried for provenance and operator
            context.

    Invariants:
        - Frozen: an immutable value object, equal by field value.

    Failure modes:
        - ``ValueError`` if ``base_url`` is blank.
    """

    generation: Generation
    base_url: str
    probe: AvailabilityProbe
    semantics: str

    def __post_init__(self) -> None:
        """Reject a blank base URL."""
        if not self.base_url.strip():
            msg = "base_url must be a non-blank endpoint URL."
            raise ValueError(msg)


def _empty_source_map() -> dict[Generation, GenerationSource]:
    """Return an empty generation-to-source map (a registry's initial state)."""
    return {}


@dataclass
class GenerationRegistry:
    """An open-for-extension registry of AHN generation sources.

    Contract:
        - Starts empty; :meth:`register` adds one source per generation.
        - :meth:`source_for`, :meth:`resolve_token`, :meth:`generations`,
          :meth:`sources` and :meth:`tokens` read the registry
          deterministically (newest generation first), never a hardcoded list.

    Invariants:
        - A generation is registered at most once.
        - Ordering is by descending generation number, so ``auto`` and the CLI
          token list are deterministic regardless of registration order.
    """

    _sources: dict[Generation, GenerationSource] = field(
        default_factory=_empty_source_map
    )

    def register(self, source: GenerationSource) -> None:
        """Add ``source`` to the registry.

        Failure modes:
            - :class:`DuplicateGenerationError` if the generation is already
              registered.
        """
        if source.generation in self._sources:
            msg = (
                f"generation {source.generation.code} is already registered."
            )
            raise DuplicateGenerationError(msg)
        self._sources[source.generation] = source

    def source_for(self, generation: Generation) -> GenerationSource:
        """Return the registered source for ``generation``.

        Failure modes:
            - :class:`UnknownGenerationError` if nothing is registered for it.
        """
        try:
            return self._sources[generation]
        except KeyError as exc:
            msg = f"no source registered for {generation.code}."
            raise UnknownGenerationError(msg) from exc

    def generations(self) -> tuple[Generation, ...]:
        """Return the registered generations, newest (highest number) first."""
        return tuple(
            sorted(self._sources, key=lambda gen: gen.number, reverse=True)
        )

    def sources(self) -> tuple[GenerationSource, ...]:
        """Return the registered sources, newest generation first."""
        return tuple(self._sources[gen] for gen in self.generations())

    def tokens(self) -> tuple[str, ...]:
        """Return the ``--ahn`` tokens: ``auto`` then one per generation.

        Contract:
            - Always begins with :data:`AUTO_CHOICE`, then ``ahn<N>`` for each
              registered generation newest-first, so the CLI choice list is
              derived from the registry rather than hardcoded.
        """
        return (
            AUTO_CHOICE,
            *(f"ahn{gen.number}" for gen in self.generations()),
        )

    def resolve_token(self, token: str) -> Generation | None:
        """Map an ``--ahn`` token to a generation, or ``None`` for ``auto``.

        Contract:
            - :data:`AUTO_CHOICE` maps to ``None`` (request automatic selection).
            - ``ahn<N>`` maps to the registered generation ``N``.

        Failure modes:
            - :class:`UnknownGenerationError` if ``token`` names no registered
              generation.
        """
        if token == AUTO_CHOICE:
            return None
        for generation in self.generations():
            if f"ahn{generation.number}" == token:
                return generation
        msg = f"unknown --ahn token: {token!r}."
        raise UnknownGenerationError(msg)


def select_source(
    requested: Generation | None,
    aoi: BBox,
    registry: GenerationRegistry,
) -> GenerationSource:
    """Select the generation source to fetch from.

    Contract:
        - ``requested`` is an explicit :class:`Generation`, or ``None`` to
          request automatic selection.
        - Explicit: returns that generation's registered source, unprobed.
        - Automatic: probes registered generations newest-first and returns the
          first whose coverage probe reports ``aoi`` covered.

    Failure modes:
        - ``ValueError`` if ``aoi`` is a degenerate bounding box.
        - :class:`UnknownGenerationError` if an explicit ``requested`` is not
          registered.
        - :class:`GenerationUnavailableError` if automatic selection finds no
          generation covering ``aoi``.
        - Propagates whatever an injected probe raises during automatic
          selection (e.g. the default registry's
          :class:`CoverageProbeNotWiredError`).
    """
    ensure_valid_bbox(aoi)
    if requested is not None:
        return registry.source_for(requested)
    for source in registry.sources():
        if source.probe(aoi):
            return source
    msg = (
        "no registered AHN generation covers the requested AOI "
        f"{aoi}; automatic selection found none available."
    )
    raise GenerationUnavailableError(msg)


def _unwired_probe(base_url: str) -> AvailabilityProbe:
    """Return a placeholder coverage probe that reports "not wired yet".

    Contract:
        - The returned probe validates the AOI, then raises
          :class:`CoverageProbeNotWiredError`: real coverage probing lands in
          WP6. It never returns a value and never performs network I/O.
    """

    def probe(aoi: BBox) -> bool:
        ensure_valid_bbox(aoi)
        msg = (
            f"coverage probing for {base_url} is not wired yet (WP6); "
            f"inject a probe to select automatically for AOI {aoi}."
        )
        raise CoverageProbeNotWiredError(msg)

    return probe


def default_registry() -> GenerationRegistry:
    """Return the registry wired with AHN5 and AHN4 GeoTiles.nl sources.

    Contract:
        - Registers AHN5 then AHN4 with their GeoTiles.nl base URLs and
          semantics notes; each carries an un-wired placeholder coverage probe
          (real probing lands in WP6).
        - Deterministic: the same sources every call, ordered newest-first.
    """
    registry = GenerationRegistry()
    registry.register(
        GenerationSource(
            generation=Generation(5),
            base_url=_AHN5_BASE_URL,
            probe=_unwired_probe(_AHN5_BASE_URL),
            semantics=_AHN5_SEMANTICS,
        )
    )
    registry.register(
        GenerationSource(
            generation=Generation(4),
            base_url=_AHN4_BASE_URL,
            probe=_unwired_probe(_AHN4_BASE_URL),
            semantics=_AHN4_SEMANTICS,
        )
    )
    return registry
