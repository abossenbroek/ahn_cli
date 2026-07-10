"""Tests for the fetch-context AHN generation registry and selection."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

import pytest

from ahn_cli.domain import BBox, Generation, Product, Provenance
from ahn_cli.fetch.generation import (
    AUTO_CHOICE,
    AvailabilityProbe,
    CoverageProbeNotWiredError,
    DuplicateGenerationError,
    GenerationRegistry,
    GenerationSource,
    GenerationUnavailableError,
    UnknownGenerationError,
    default_registry,
    select_source,
)
from ahn_cli.provenance import read_provenance, write_provenance

if TYPE_CHECKING:
    from pathlib import Path

_AOI: BBox = (194198.0, 443461.0, 194594.0, 443694.0)


def _constant_probe(*, available: bool) -> AvailabilityProbe:
    """Return a probe that ignores the AOI and reports a fixed availability."""

    def probe(aoi: BBox) -> bool:
        del aoi
        return available

    return probe


def _fail_probe(aoi: BBox) -> bool:
    """Fail if consulted, standing in for an explicit (unprobed) selection."""
    del aoi
    msg = "probe called on an explicit (unprobed) selection."
    raise AssertionError(msg)


def _source(
    number: int,
    *,
    available: bool,
    base_url: str = "https://example.test/AHN/",
) -> GenerationSource:
    """Build a source for generation ``number`` with a constant fake probe."""
    return GenerationSource(
        generation=Generation(number),
        base_url=base_url,
        probe=_constant_probe(available=available),
        semantics=f"AHN{number} test semantics.",
    )


def _registry_with_fakes() -> GenerationRegistry:
    """Return a registry like the default one but with deterministic probes."""
    registry = GenerationRegistry()
    registry.register(_source(5, available=True))
    registry.register(_source(4, available=True))
    return registry


def test_auto_token_is_the_literal_auto() -> None:
    """The automatic-selection token is the stable string ``auto``."""
    assert AUTO_CHOICE == "auto"


def test_generation_source_is_value_typed() -> None:
    """A source is a frozen value object, equal by field value."""
    probe = _constant_probe(available=True)
    first = GenerationSource(Generation(4), "u://a", probe, "note")
    second = GenerationSource(Generation(4), "u://a", probe, "note")

    assert first == second


def test_generation_source_rejects_blank_base_url() -> None:
    """A blank base URL is not a usable endpoint."""
    with pytest.raises(ValueError, match="base_url"):
        GenerationSource(
            Generation(4), "   ", _constant_probe(available=True), "note"
        )


def test_register_then_source_for_returns_the_source() -> None:
    """A registered source is retrievable by its generation."""
    registry = GenerationRegistry()
    source = _source(4, available=True)

    registry.register(source)

    assert registry.source_for(Generation(4)) is source


def test_register_rejects_a_duplicate_generation() -> None:
    """A generation may be registered at most once."""
    registry = GenerationRegistry()
    registry.register(_source(4, available=True))

    with pytest.raises(DuplicateGenerationError, match="AHN4"):
        registry.register(_source(4, available=False))


def test_source_for_unknown_generation_raises() -> None:
    """Asking for an unregistered generation is a typed lookup error."""
    registry = GenerationRegistry()

    with pytest.raises(UnknownGenerationError, match="AHN4"):
        registry.source_for(Generation(4))


def test_generations_and_sources_are_newest_first() -> None:
    """Ordering is by descending number regardless of registration order."""
    registry = GenerationRegistry()
    older = _source(4, available=True)
    newer = _source(5, available=True)
    registry.register(older)
    registry.register(newer)

    assert registry.generations() == (Generation(5), Generation(4))
    assert registry.sources() == (newer, older)


def test_tokens_are_auto_then_generations_newest_first() -> None:
    """The CLI token list is derived from the registry, newest-first."""
    registry = GenerationRegistry()
    registry.register(_source(4, available=True))
    registry.register(_source(5, available=True))

    assert registry.tokens() == ("auto", "ahn5", "ahn4")


def test_resolve_token_auto_maps_to_none() -> None:
    """The ``auto`` token requests automatic selection (no fixed generation)."""
    registry = GenerationRegistry()
    registry.register(_source(4, available=True))

    assert registry.resolve_token(AUTO_CHOICE) is None


def test_resolve_token_maps_each_generation() -> None:
    """Each ``ahn<N>`` token maps to its registered generation."""
    registry = GenerationRegistry()
    registry.register(_source(4, available=True))
    registry.register(_source(5, available=True))

    assert registry.resolve_token("ahn5") == Generation(5)
    assert registry.resolve_token("ahn4") == Generation(4)


def test_resolve_token_rejects_an_unregistered_token() -> None:
    """A token for no registered generation is a typed lookup error."""
    registry = GenerationRegistry()
    registry.register(_source(4, available=True))

    with pytest.raises(UnknownGenerationError, match="ahn9"):
        registry.resolve_token("ahn9")


def test_select_source_explicit_returns_that_generation_unprobed() -> None:
    """An explicit request returns the source without consulting the probe."""
    registry = GenerationRegistry()
    explicit = GenerationSource(Generation(4), "u://a", _fail_probe, "note")
    registry.register(explicit)

    assert select_source(Generation(4), _AOI, registry) is explicit


def test_select_source_explicit_unknown_generation_raises() -> None:
    """An explicit request for an unregistered generation raises."""
    registry = GenerationRegistry()
    registry.register(_source(4, available=True))

    with pytest.raises(UnknownGenerationError, match="AHN5"):
        select_source(Generation(5), _AOI, registry)


def test_select_source_rejects_a_degenerate_aoi() -> None:
    """Selection validates the AOI extent before anything else."""
    registry = GenerationRegistry()
    registry.register(_source(4, available=True))

    with pytest.raises(ValueError, match="bbox"):
        select_source(Generation(4), (1.0, 1.0, 0.0, 0.0), registry)


def test_select_source_auto_prefers_the_newest_available() -> None:
    """Auto returns the newest generation whose probe reports coverage."""
    registry = GenerationRegistry()
    newest = _source(5, available=True)
    registry.register(_source(4, available=True))
    registry.register(newest)

    assert select_source(None, _AOI, registry) is newest


def test_select_source_auto_falls_back_when_newest_uncovered() -> None:
    """Auto skips an uncovered newer generation and takes the next."""
    registry = GenerationRegistry()
    fallback = _source(4, available=True)
    registry.register(fallback)
    registry.register(_source(5, available=False))

    assert select_source(None, _AOI, registry) is fallback


def test_select_source_auto_raises_when_none_available() -> None:
    """Auto raises when no registered generation covers the AOI."""
    registry = GenerationRegistry()
    registry.register(_source(4, available=False))
    registry.register(_source(5, available=False))

    with pytest.raises(GenerationUnavailableError):
        select_source(None, _AOI, registry)


def test_default_registry_wires_ahn5_and_ahn4() -> None:
    """The default registry exposes AHN5 and AHN4 with the GeoTiles URLs."""
    registry = default_registry()

    assert registry.tokens() == ("auto", "ahn5", "ahn4")
    ahn5 = registry.source_for(Generation(5))
    ahn4 = registry.source_for(Generation(4))
    assert ahn5.base_url == "https://geotiles.citg.tudelft.nl/AHN5_T/"
    assert ahn4.base_url == "https://geotiles.citg.tudelft.nl/AHN4_T/"
    assert "roofs only" in ahn5.semantics
    assert "all buildings" in ahn4.semantics


def test_default_registry_probes_are_unwired_until_wp6() -> None:
    """The default probes report un-wired; real probing lands in WP6."""
    registry = default_registry()

    for source in registry.sources():
        with pytest.raises(CoverageProbeNotWiredError):
            source.probe(_AOI)


def test_a_new_generation_needs_only_a_registry_entry() -> None:
    """Extensibility: a new generation is selectable via registration only.

    Registering a fixture generation into a fresh registry -- touching zero
    production call sites, editing no switch -- makes it fully selectable both
    explicitly and, as the newest available, automatically.
    """
    registry = _registry_with_fakes()
    future = _source(6, available=True, base_url="https://example.test/AHN6/")

    registry.register(future)

    # The CLI token list extends itself from the registry.
    assert registry.tokens() == ("auto", "ahn6", "ahn5", "ahn4")
    # Explicit selection resolves the new generation.
    assert select_source(Generation(6), _AOI, registry) is future
    # Auto prefers it as the newest available, with no other code changed.
    assert select_source(None, _AOI, registry) is future


def test_selected_generation_is_recorded_in_the_provenance_sidecar(
    tmp_path: Path,
) -> None:
    """The selected generation round-trips through the provenance sidecar.

    Proves WP5's DoD: the generation chosen by :func:`select_source` is what a
    provenance sidecar records and reads back.
    """
    registry = _registry_with_fakes()
    chosen = select_source(None, _AOI, registry)

    moment = datetime(2024, 1, 1, tzinfo=timezone.utc)
    provenance = Provenance(
        source_portal="geotiles",
        product=Product.AHN_POINT_CLOUD,
        licence="CC-0",
        attribution="AHN, CC-0.",
        bbox=_AOI,
        download_started_at=moment,
        download_finished_at=moment,
        input_checksum="in",
        output_checksum="out",
        tool_version="0.3.5",
        generation=chosen.generation,
    )

    path = tmp_path / "provenance.json"
    write_provenance(provenance, path)

    assert read_provenance(path).generation == Generation(5)
