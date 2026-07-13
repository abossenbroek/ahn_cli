"""Regenerate the committed Rust-consumer tiles3d fixtures.

Run once whenever the fixtures legitimately need to change (a codec/format
revision) — never to paper over an unexpected drift, which is a real
regression the drift test in ``test_integration_profiles.py`` is meant to
catch::

    uv run python -m tests.tiles3d.regen_rust_fixtures

It pins geodesy *and* the provenance producer record to the same fixed
stand-ins the drift test uses (so the fixtures are machine-stable and
CI-checkable across the OS / Python matrix), rebuilds the ``game`` and
``heightfield`` packed tilesets into ``fixtures/rust-consumer/<profile>/``,
and leaves the sibling ``README.md`` in place.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from ahn_cli.tiles3d import provenance as provenance_module
from ahn_cli.tiles3d.geodesy import Geodesy
from tests.tiles3d.test_integration_profiles import (
    FIXTURE_PLATFORM,
    FIXTURE_PROFILES,
    FIXTURE_PYTHON,
    FIXTURE_ROOT,
    build_fixture,
    fake_to_ecef,
    fake_to_geodetic_from_ecef,
    fake_to_geodetic_radians,
)


def main() -> None:
    """Rebuild every committed Rust-consumer fixture in place."""
    Geodesy.to_ecef = fake_to_ecef  # type: ignore[method-assign]
    Geodesy.to_geodetic_radians = fake_to_geodetic_radians  # type: ignore[method-assign]
    Geodesy.to_geodetic_from_ecef = fake_to_geodetic_from_ecef  # type: ignore[method-assign]
    provenance_module.producer_platform = lambda: FIXTURE_PLATFORM
    provenance_module.producer_python = lambda: FIXTURE_PYTHON
    for name, profile in FIXTURE_PROFILES.items():
        out = FIXTURE_ROOT / name
        if out.exists():
            shutil.rmtree(out)
        with tempfile.TemporaryDirectory() as scratch:
            build_fixture(Path(scratch), out, profile)
        print(f"regenerated {out}")  # noqa: T201


if __name__ == "__main__":
    main()
