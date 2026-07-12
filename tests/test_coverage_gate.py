"""Contract tests for the WP0 coverage gate.

These tests protect the 100%-branch coverage policy that every later work
package in the 7rad data-acquisition epic relies on: legacy modules are
exempt, but any *new* module must land fully covered or the build fails.
"""

import subprocess
import sys
import textwrap
from pathlib import Path
from typing import cast

from coverage import Coverage

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"

# Modules that predate the epic and are intentionally exempt from the gate.
LEGACY_OMIT = (
    "ahn_cli/main.py",
    "ahn_cli/process.py",
    "ahn_cli/config.py",
    "ahn_cli/kwargs.py",
    "ahn_cli/validator.py",
    "ahn_cli/fetcher/*",
    "ahn_cli/manipulator/*",
)

# Packages later work packages introduce; these MUST stay gated (never omitted).
FUTURE_GATED = (
    "domain",
    "cli",
    "fetch",
    "prep",
    "provenance",
    "cache",
    "reconcile",
    "copc",
    "tiles3d",
)


def _load_config() -> Coverage:
    """Return a Coverage object loaded from the project's pyproject.toml.

    Contract: reflects the effective ``[tool.coverage.*]`` policy coverage.py
    applies during ``make test``; raises if pyproject cannot be read.
    """
    return Coverage(config_file=str(PYPROJECT))


def test_gate_policy_is_100_percent_branch() -> None:
    """The project config enforces 100% branch coverage over ``ahn_cli``.

    Contract: fails if branch measurement is off, the source is not exactly
    ``ahn_cli``, or the fail-under threshold is not 100 -- i.e. if the gate
    has been silently weakened.
    """
    cov = _load_config()
    assert cov.get_option("run:branch") is True
    assert cov.get_option("run:source") == ["ahn_cli"]
    assert cov.get_option("report:fail_under") == 100


def test_legacy_omitted_but_future_packages_gated() -> None:
    """Legacy files are exempt; future epic packages are not.

    Contract: every known legacy module stays in ``omit`` (so today's build is
    green) while none of the packages later WPs add appear there (so they are
    gated at 100% the instant they land).
    """
    # get_option widens to a config union; omit is always a list of globs.
    omit = cast("list[str]", _load_config().get_option("run:omit") or [])
    for legacy in LEGACY_OMIT:
        assert legacy in omit, f"legacy module dropped from omit: {legacy}"
    # Path-segment match so future ``fetch/`` is not confused with legacy
    # ``fetcher/`` (a plain substring test would collide on that prefix).
    for pkg in FUTURE_GATED:
        prefix = f"ahn_cli/{pkg}/"
        module = f"ahn_cli/{pkg}.py"
        for entry in omit:
            assert not entry.startswith(prefix), (
                f"future pkg omitted: {entry}"
            )
            assert entry != module, f"future module omitted: {entry}"


def test_gate_bites_on_uncovered_branch(tmp_path: Path) -> None:
    """A measured module with an uncovered branch fails the 100% gate.

    Contract: runs coverage (branch mode) over a synthetic module exercising
    only one side of a conditional and asserts the ``--fail-under=100`` report
    exits non-zero, proving a future under-covered module blocks CI. A
    fully-covered control run over the same module must pass, so the failure
    cannot be a false positive.
    """
    module = tmp_path / "sample_mod.py"
    module.write_text(
        textwrap.dedent(
            """\
            def branchy(flag):
                if flag:
                    return "yes"
                return "no"
            """
        )
    )

    def _run_report(driver_body: str) -> subprocess.CompletedProcess[str]:
        driver = tmp_path / "driver.py"
        driver.write_text(textwrap.dedent(driver_body))
        run = subprocess.run(
            [
                sys.executable,
                "-m",
                "coverage",
                "run",
                "--branch",
                "driver.py",
            ],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
        )
        assert run.returncode == 0, run.stderr
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "coverage",
                "report",
                "--include=sample_mod.py",
                "--fail-under=100",
                "--show-missing",
            ],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
        )

    # Only the True branch is taken -> gate must bite.
    uncovered = _run_report(
        """\
        import sample_mod

        sample_mod.branchy(True)
        """
    )
    assert uncovered.returncode != 0, uncovered.stdout + uncovered.stderr
    assert "sample_mod.py" in uncovered.stdout

    # Control: both branches taken -> the same gate passes.
    covered = _run_report(
        """\
        import sample_mod

        sample_mod.branchy(True)
        sample_mod.branchy(False)
        """
    )
    assert covered.returncode == 0, covered.stdout + covered.stderr
