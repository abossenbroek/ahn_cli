"""Shared synthetic LAZ factory for the copc tests.

Tiny in-process LAZ files keep the unit tests fast and offline; the writer
supports the point formats the copc context must ingest (legacy PDRF 2 from
``reconcile``, PDRF 6/8-style formats from ``prep``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import laspy
import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt


class WriteLaz(Protocol):
    """Factory fixture protocol: write a LAZ, return its path."""

    def __call__(
        self,
        coords: list[tuple[float, float, float]],
        *,
        point_format: int = 7,
        rgb: list[tuple[int, int, int]] | None = None,
        gps_time: list[float] | None = None,
        classification: list[int] | None = None,
        returns: tuple[list[int], list[int]] | None = None,
        scan_angle_rank: list[int] | None = None,
        name: str = "cloud.laz",
    ) -> Path:
        """Write the LAZ and return its path."""
        ...


@pytest.fixture
def write_laz(tmp_path: Path) -> WriteLaz:
    """Return a factory writing a small deterministic LAZ from arrays."""

    def _write(
        coords: list[tuple[float, float, float]],
        *,
        point_format: int = 7,
        rgb: list[tuple[int, int, int]] | None = None,
        gps_time: list[float] | None = None,
        classification: list[int] | None = None,
        returns: tuple[list[int], list[int]] | None = None,
        scan_angle_rank: list[int] | None = None,
        name: str = "cloud.laz",
    ) -> Path:
        arr: npt.NDArray[np.float64] = np.asarray(coords, dtype=np.float64)
        version = "1.2" if point_format < 6 else "1.4"
        header = laspy.LasHeader(version=version, point_format=point_format)
        header.scales = np.asarray([0.001, 0.001, 0.001])
        header.offsets = np.floor(arr.min(axis=0))
        las = laspy.LasData(header)
        las.x = arr[:, 0]
        las.y = arr[:, 1]
        las.z = arr[:, 2]
        if rgb is not None:
            colours = np.asarray(rgb, dtype=np.uint16)
            las.red = colours[:, 0]
            las.green = colours[:, 1]
            las.blue = colours[:, 2]
        if gps_time is not None:
            las.gps_time = np.asarray(gps_time, dtype=np.float64)
        if classification is not None:
            las.classification = np.asarray(classification, dtype=np.uint8)
        if returns is not None:
            las.return_number = np.asarray(returns[0], dtype=np.uint8)
            las.number_of_returns = np.asarray(returns[1], dtype=np.uint8)
        if scan_angle_rank is not None:
            las.scan_angle_rank = np.asarray(scan_angle_rank, dtype=np.int8)
        path = tmp_path / name
        las.write(str(path))
        return path

    return _write
