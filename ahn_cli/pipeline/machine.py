"""Machine facts and free-RAM sensing for the RAM-adaptive tiling.

The ``halo: auto`` sizer and the §5 memory-layout work need two kinds of fact:
static machine geometry (cache-line and page sizes, to chunk and align buffers)
and the live free-RAM budget (to size the working set to a safe fraction). Both
are read from the real system by default, but through an **injected**
:class:`SystemProbe` so every pipeline caller can substitute a deterministic
fake -- and so the Linux and macOS reader branches, plus their error/fallback
paths, are all exercisable off any single platform without ``# pragma: no
cover``.

Platform readers:

* **Linux** -- cache line and page size via :func:`os.sysconf`
  (``SC_LEVEL1_DCACHE_LINESIZE`` / ``SC_PAGE_SIZE``); free RAM from
  ``/proc/meminfo``'s ``MemAvailable`` line.
* **macOS** -- cache line and page size via ``sysctl hw.cachelinesize`` /
  ``hw.pagesize``; free RAM from ``vm_stat`` (free + inactive pages, a
  conservative estimate -- under-counting reclaimable memory only shrinks the
  working set, never changes output).

When a probe fails (unsupported platform, a raising ``sysconf``/subprocess, or
an unparsable reading) the reading falls back to a sane constant
(:data:`_DEFAULT_CACHE_LINE`, :data:`_DEFAULT_PAGE`, or
:data:`_DEFAULT_FREE_RAM_BYTES`), so sensing never aborts a run.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

__all__ = [
    "MachineFacts",
    "SystemProbe",
    "free_ram_bytes",
    "machine_facts",
]

_DEFAULT_CACHE_LINE = 128
"""Fallback cache-line size (bytes) -- the Apple-silicon width, a safe default."""

_DEFAULT_PAGE = 4096
"""Fallback page size (bytes) when the page probe fails."""

_DEFAULT_FREE_RAM_BYTES = 2 * 1024**3
"""Conservative fallback free-RAM budget (2 GiB) when the RAM probe fails.

Deliberately small: an under-estimate only shrinks the working set (a
performance loss), never the deliverable."""

_LINUX = "linux"
"""``sys.platform`` prefix for Linux (matches ``linux``/``linux2``)."""

_MACOS = "darwin"
"""``sys.platform`` value for macOS."""

_SC_CACHE_LINE = "SC_LEVEL1_DCACHE_LINESIZE"
"""``os.sysconf`` name for the L1 data-cache line size (Linux)."""

_SC_PAGE_SIZE = "SC_PAGE_SIZE"
"""``os.sysconf`` name for the memory page size (Linux)."""

_PROC_MEMINFO = "/proc/meminfo"
"""Linux free-RAM source file."""

_MEM_AVAILABLE = "MemAvailable:"
"""The ``/proc/meminfo`` line prefix giving reclaimable memory, in kB."""

_KIB = 1024
"""Bytes per kibibyte (``/proc/meminfo`` reports MemAvailable in kB)."""

_VM_PAGE_SIZE_RE = re.compile(r"page size of (\d+) bytes")
"""Matches ``vm_stat``'s header page-size declaration."""

_VM_PAGES_FREE_RE = re.compile(r"Pages free:\s+(\d+)")
"""Matches ``vm_stat``'s free-page count."""

_VM_PAGES_INACTIVE_RE = re.compile(r"Pages inactive:\s+(\d+)")
"""Matches ``vm_stat``'s inactive-page count (reclaimable)."""

Command: TypeAlias = Callable[[Sequence[str]], str]
"""Runs an argv and returns captured stdout as text."""

SysconfReader: TypeAlias = Callable[[str], int]
"""Reads a named :func:`os.sysconf` value as an integer."""

TextReader: TypeAlias = Callable[[str], str]
"""Reads a file path and returns its full text."""


@dataclass(frozen=True)
class MachineFacts:
    """Static machine geometry used to chunk and align streaming buffers.

    Contract:
        - ``cache_line_bytes`` is the cache-line width used to pad per-worker
          buffers against false sharing.
        - ``page_bytes`` is the memory page size used to align mmap'd tile/spill
          buffers.
        - Both are strictly positive.

    Invariants:
        - Frozen value object, equal by field value.

    Failure modes:
        - :class:`ValueError` if either size is not positive.
    """

    cache_line_bytes: int
    page_bytes: int

    def __post_init__(self) -> None:
        """Reject a non-positive cache-line or page size."""
        if self.cache_line_bytes <= 0:
            msg = (
                "cache_line_bytes must be positive; "
                f"got {self.cache_line_bytes}."
            )
            raise ValueError(msg)
        if self.page_bytes <= 0:
            msg = f"page_bytes must be positive; got {self.page_bytes}."
            raise ValueError(msg)


def _run_command(args: Sequence[str]) -> str:
    """Run ``args`` and return captured stdout as text (the default reader)."""
    return subprocess.check_output(list(args), text=True)  # noqa: S603


def _slurp(path: str) -> str:
    """Read ``path`` and return its full UTF-8 text (the default reader)."""
    return Path(path).read_text(encoding="utf-8")


def _missing_sysconf(name: str) -> int:
    """Stand in for ``os.sysconf`` on platforms without it (Windows).

    Never actually invoked in practice: :func:`_read_cache_line` and
    :func:`_read_page` only call ``sysconf`` on the Linux branch, and
    ``sys.platform`` is never ``"linux*"`` on a platform lacking
    ``os.sysconf``. Exists so binding :data:`_SYSTEM_PROBE`'s ``sysconf``
    field never has to evaluate the missing ``os.sysconf`` attribute itself.
    """
    msg = f"os.sysconf is unavailable on this platform (requested {name!r})."
    raise OSError(msg)


_SYSCONF: SysconfReader = getattr(os, "sysconf", _missing_sysconf)
"""``os.sysconf`` where the platform defines it (POSIX); else
:func:`_missing_sysconf`, so importing this module never raises on Windows."""


@dataclass(frozen=True)
class SystemProbe:
    """The injectable system-access seam the readers depend on.

    Contract:
        - ``platform`` is a ``sys.platform``-style string selecting the reader
          branch (``linux*`` or ``darwin``).
        - ``sysconf`` reads a named :func:`os.sysconf` integer (Linux).
        - ``run`` runs an argv and returns its stdout text (macOS).
        - ``read_text`` reads a file path's text (Linux ``/proc/meminfo``).

    The default :data:`_SYSTEM_PROBE` wires the real system; every pipeline
    caller injects one so tests cover both platform branches deterministically.
    """

    platform: str
    sysconf: SysconfReader
    run: Command
    read_text: TextReader


_SYSTEM_PROBE = SystemProbe(
    platform=sys.platform,
    sysconf=_SYSCONF,
    run=_run_command,
    read_text=_slurp,
)
"""The default probe: reads the real machine via stdlib and subprocess."""


def _sysctl_int(key: str, *, run: Command) -> int:
    """Return the integer value of ``sysctl -n <key>``."""
    return int(run(["sysctl", "-n", key]).strip())


def _read_cache_line(probe: SystemProbe) -> int:
    """Read the cache-line size for ``probe``'s platform."""
    if probe.platform.startswith(_LINUX):
        return probe.sysconf(_SC_CACHE_LINE)
    if probe.platform == _MACOS:
        return _sysctl_int("hw.cachelinesize", run=probe.run)
    msg = f"unsupported platform for cache line: {probe.platform!r}."
    raise LookupError(msg)


def _read_page(probe: SystemProbe) -> int:
    """Read the memory page size for ``probe``'s platform."""
    if probe.platform.startswith(_LINUX):
        return probe.sysconf(_SC_PAGE_SIZE)
    if probe.platform == _MACOS:
        return _sysctl_int("hw.pagesize", run=probe.run)
    msg = f"unsupported platform for page size: {probe.platform!r}."
    raise LookupError(msg)


def _meminfo_available(text: str) -> int:
    """Return ``MemAvailable`` from ``/proc/meminfo`` text, in bytes."""
    for line in text.splitlines():
        if line.startswith(_MEM_AVAILABLE):
            return int(line.split()[1]) * _KIB
    msg = "MemAvailable not found in /proc/meminfo."
    raise LookupError(msg)


def _match_int(text: str, pattern: re.Pattern[str]) -> int:
    """Return the first capture group of ``pattern`` in ``text`` as an int."""
    match = pattern.search(text)
    if match is None:
        msg = f"pattern {pattern.pattern!r} not found."
        raise LookupError(msg)
    return int(match.group(1))


def _vm_stat_free(text: str) -> int:
    """Return free + inactive bytes from ``vm_stat`` text (conservative)."""
    page_size = _match_int(text, _VM_PAGE_SIZE_RE)
    free = _match_int(text, _VM_PAGES_FREE_RE)
    inactive = _match_int(text, _VM_PAGES_INACTIVE_RE)
    return (free + inactive) * page_size


def _read_free_ram(probe: SystemProbe) -> int:
    """Read the free-RAM estimate for ``probe``'s platform, in bytes."""
    if probe.platform.startswith(_LINUX):
        return _meminfo_available(probe.read_text(_PROC_MEMINFO))
    if probe.platform == _MACOS:
        return _vm_stat_free(probe.run(["vm_stat"]))
    msg = f"unsupported platform for free RAM: {probe.platform!r}."
    raise LookupError(msg)


def _guarded(reader: Callable[[], int], default: int) -> int:
    """Return ``reader()`` if it yields a positive int, else ``default``."""
    try:
        value = reader()
    except (LookupError, OSError, ValueError, subprocess.SubprocessError):
        return default
    return value if value > 0 else default


def machine_facts(*, probe: SystemProbe = _SYSTEM_PROBE) -> MachineFacts:
    """Return the machine's cache-line and page geometry.

    Contract:
        - Reads both sizes through ``probe``; a failed or non-positive reading
          falls back to :data:`_DEFAULT_CACHE_LINE` / :data:`_DEFAULT_PAGE`, so
          the result is always a valid :class:`MachineFacts`.

    ``probe`` is injectable so both platform branches and the fallback path are
    covered without depending on the host platform.
    """
    cache_line = _guarded(
        lambda: _read_cache_line(probe), _DEFAULT_CACHE_LINE
    )
    page = _guarded(lambda: _read_page(probe), _DEFAULT_PAGE)
    return MachineFacts(cache_line_bytes=cache_line, page_bytes=page)


def free_ram_bytes(*, probe: SystemProbe = _SYSTEM_PROBE) -> int:
    """Return an estimate of currently-available RAM in bytes.

    Contract:
        - Reads the estimate through ``probe``; a failed or non-positive
          reading falls back to :data:`_DEFAULT_FREE_RAM_BYTES`.

    ``probe`` is injectable so both platform branches and the fallback path are
    covered without depending on the host platform.
    """
    return _guarded(lambda: _read_free_ram(probe), _DEFAULT_FREE_RAM_BYTES)
