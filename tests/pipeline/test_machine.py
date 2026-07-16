"""Tests for machine-facts and free-RAM sensing (`ahn_cli.pipeline.machine`).

Every reader is driven through an injected :class:`SystemProbe`, so the Linux
and macOS branches plus their error/fallback paths are all covered off any
single platform. No test touches the real system except the two default readers
(:func:`_run_command`, :func:`_slurp`), which are exercised directly.
"""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

from ahn_cli.pipeline import (
    MachineFacts,
    SystemProbe,
    free_ram_bytes,
    machine_facts,
)
from ahn_cli.pipeline.machine import (
    _DEFAULT_CACHE_LINE,  # pyright: ignore[reportPrivateUsage]
    _DEFAULT_FREE_RAM_BYTES,  # pyright: ignore[reportPrivateUsage]
    _DEFAULT_PAGE,  # pyright: ignore[reportPrivateUsage]
    _run_command,  # pyright: ignore[reportPrivateUsage]
    _slurp,  # pyright: ignore[reportPrivateUsage]
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from ahn_cli.pipeline.machine import SysconfReader

_MEMINFO = "MemTotal:       16384000 kB\nMemAvailable:    8192000 kB\n"
_MEMINFO_AVAILABLE_BYTES = 8192000 * 1024

_VM_STAT = (
    "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
    "Pages free:                          100000.\n"
    "Pages inactive:                       50000.\n"
)
_VM_STAT_FREE_BYTES = (100000 + 50000) * 16384

_SYSCTL: dict[str, str] = {
    "hw.cachelinesize": "128\n",
    "hw.pagesize": "16384\n",
}


def _unused_run(_args: Sequence[str]) -> str:
    """Raise, since the Linux path must not call ``run``."""
    raise AssertionError


def _unused_sysconf(_name: str) -> int:
    """Raise, since the macOS path must not call ``sysconf``."""
    raise AssertionError


def _unused_read(_path: str) -> str:
    """Raise, since only the Linux path calls ``read_text``."""
    raise AssertionError


def _raising_sysconf(_name: str) -> int:
    """Raise `OSError`, standing in for an unavailable ``sysconf`` name."""
    raise OSError


def _const_sysconf(value: int) -> SysconfReader:
    """Build a ``sysconf`` returning ``value`` for any name."""

    def reader(_name: str) -> int:
        return value

    return reader


def _map_sysconf(mapping: dict[str, int]) -> SysconfReader:
    """Build a ``sysconf`` resolving each name through ``mapping``."""

    def reader(name: str) -> int:
        return mapping[name]

    return reader


def _meminfo_reader(_path: str) -> str:
    """Return canned ``/proc/meminfo`` text with a `MemAvailable` line."""
    return _MEMINFO


def _macos_run(args: Sequence[str]) -> str:
    """Fake macOS ``sysctl``/``vm_stat`` reader keyed on the argv."""
    if args[0] == "vm_stat":
        return _VM_STAT
    return _SYSCTL[args[-1]]


def _linux_probe(sysconf: SysconfReader) -> SystemProbe:
    """Build a Linux probe with ``sysconf`` and canned meminfo."""
    return SystemProbe(
        platform="linux",
        sysconf=sysconf,
        run=_unused_run,
        read_text=_meminfo_reader,
    )


def _macos_probe() -> SystemProbe:
    """Build a macOS probe wired to the fake ``sysctl``/``vm_stat`` reader."""
    return SystemProbe(
        platform="darwin",
        sysconf=_unused_sysconf,
        run=_macos_run,
        read_text=_unused_read,
    )


def _other_probe() -> SystemProbe:
    """Build a probe for an unsupported platform (every reader raises)."""
    return SystemProbe(
        platform="sunos",
        sysconf=_unused_sysconf,
        run=_unused_run,
        read_text=_unused_read,
    )


def test_machine_facts_valid() -> None:
    """Positive sizes construct a `MachineFacts`."""
    facts = MachineFacts(cache_line_bytes=64, page_bytes=4096)
    assert facts.cache_line_bytes == 64
    assert facts.page_bytes == 4096


def test_machine_facts_rejects_non_positive_cache_line() -> None:
    """A non-positive cache-line size is rejected."""
    with pytest.raises(ValueError, match="cache_line_bytes must be positive"):
        MachineFacts(cache_line_bytes=0, page_bytes=4096)


def test_machine_facts_rejects_non_positive_page() -> None:
    """A non-positive page size is rejected."""
    with pytest.raises(ValueError, match="page_bytes must be positive"):
        MachineFacts(cache_line_bytes=64, page_bytes=-1)


def test_machine_facts_linux_branch() -> None:
    """The Linux path reads both sizes via `sysconf`."""
    sizes = {"SC_LEVEL1_DCACHE_LINESIZE": 64, "SC_PAGE_SIZE": 4096}
    facts = machine_facts(probe=_linux_probe(_map_sysconf(sizes)))
    assert facts == MachineFacts(cache_line_bytes=64, page_bytes=4096)


def test_machine_facts_macos_branch() -> None:
    """The macOS path reads both sizes via `sysctl`."""
    facts = machine_facts(probe=_macos_probe())
    assert facts == MachineFacts(cache_line_bytes=128, page_bytes=16384)


def test_machine_facts_unsupported_platform_falls_back() -> None:
    """An unsupported platform yields the default geometry."""
    facts = machine_facts(probe=_other_probe())
    assert facts == MachineFacts(
        cache_line_bytes=_DEFAULT_CACHE_LINE, page_bytes=_DEFAULT_PAGE
    )


def test_machine_facts_non_positive_reading_falls_back() -> None:
    """A non-positive `sysconf` reading falls back to the defaults."""
    facts = machine_facts(probe=_linux_probe(_const_sysconf(-1)))
    assert facts == MachineFacts(
        cache_line_bytes=_DEFAULT_CACHE_LINE, page_bytes=_DEFAULT_PAGE
    )


def test_machine_facts_raising_reading_falls_back() -> None:
    """A raising `sysconf` falls back to the defaults."""
    facts = machine_facts(probe=_linux_probe(_raising_sysconf))
    assert facts == MachineFacts(
        cache_line_bytes=_DEFAULT_CACHE_LINE, page_bytes=_DEFAULT_PAGE
    )


def test_free_ram_linux_branch() -> None:
    """The Linux path parses `MemAvailable` into bytes."""
    probe = _linux_probe(_const_sysconf(64))
    assert free_ram_bytes(probe=probe) == _MEMINFO_AVAILABLE_BYTES


def test_free_ram_macos_branch() -> None:
    """The macOS path sums free + inactive `vm_stat` pages into bytes."""
    assert free_ram_bytes(probe=_macos_probe()) == _VM_STAT_FREE_BYTES


def test_free_ram_unsupported_platform_falls_back() -> None:
    """An unsupported platform yields the conservative default budget."""
    assert free_ram_bytes(probe=_other_probe()) == _DEFAULT_FREE_RAM_BYTES


def test_free_ram_missing_mem_available_falls_back() -> None:
    """`/proc/meminfo` without `MemAvailable` falls back to the default."""

    def _bare_meminfo(_path: str) -> str:
        return "MemTotal: 1 kB\n"

    probe = SystemProbe(
        platform="linux",
        sysconf=_unused_sysconf,
        run=_unused_run,
        read_text=_bare_meminfo,
    )
    assert free_ram_bytes(probe=probe) == _DEFAULT_FREE_RAM_BYTES


def test_free_ram_unparsable_vm_stat_falls_back() -> None:
    """A `vm_stat` output missing a needed line falls back to the default.

    This also covers `_match_int`'s not-found branch through the public API.
    """

    def _empty_vm_stat(_args: Sequence[str]) -> str:
        return "no counters here"

    probe = SystemProbe(
        platform="darwin",
        sysconf=_unused_sysconf,
        run=_empty_vm_stat,
        read_text=_unused_read,
    )
    assert free_ram_bytes(probe=probe) == _DEFAULT_FREE_RAM_BYTES


def test_slurp_reads_file_text(tmp_path: Path) -> None:
    """The default text reader returns a file's full UTF-8 contents."""
    path = tmp_path / "meminfo"
    path.write_text(_MEMINFO, encoding="utf-8")
    assert _slurp(str(path)) == _MEMINFO


def test_run_command_captures_stdout() -> None:
    """The default command reader returns a subprocess's captured stdout."""
    out = _run_command(
        [sys.executable, "-c", "import sys; sys.stdout.write('ok')"]
    )
    assert out == "ok"


def test_run_command_propagates_failure() -> None:
    """A failing subprocess raises, so `_guarded` can fall back on it."""
    with pytest.raises(subprocess.CalledProcessError):
        _run_command([sys.executable, "-c", "import sys; sys.exit(1)"])
