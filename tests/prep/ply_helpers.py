"""Synthetic-cloud and PLY-reader helpers shared by the PLY export tests.

``make_cloud`` / ``write_cloud`` build a deterministic RD-magnitude LAZ with
every attribute set; ``read_ply`` parses a binary little-endian PLY without
using the writer, so a round trip is checked by an independent reader.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import laspy
import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from pathlib import Path

_PLY_DTYPES = {
    "double": "<f8",
    "float": "<f4",
    "uchar": "u1",
    "ushort": "<u2",
}
MM = 1e-3
ZERO = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class Cloud:
    """Per-point truth of a synthetic cloud (what the PLY must reproduce)."""

    x: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]
    z: npt.NDArray[np.float64]
    classification: npt.NDArray[np.uint8]
    intensity: npt.NDArray[np.uint16]
    return_number: npt.NDArray[np.uint8]
    number_of_returns: npt.NDArray[np.uint8]
    red: npt.NDArray[np.uint16]
    green: npt.NDArray[np.uint16]
    blue: npt.NDArray[np.uint16]
    overlap: npt.NDArray[np.uint8]


def make_cloud(
    count: int = 400, *, seed: int = 7, colour_max: int = 255
) -> Cloud:
    """Return a deterministic RD-magnitude cloud with every attribute set."""
    rng = np.random.default_rng(seed)
    return Cloud(
        x=np.round(194_000.0 + rng.random(count) * 80.0, 3),
        y=np.round(443_000.0 + rng.random(count) * 80.0, 3),
        z=np.round(-8.3 + rng.random(count) * 30.0, 3),
        classification=rng.choice([1, 2, 6, 9, 26], count).astype(np.uint8),
        intensity=rng.integers(0, 60_000, count).astype(np.uint16),
        return_number=rng.integers(1, 4, count).astype(np.uint8),
        number_of_returns=np.full(count, 3, dtype=np.uint8),
        red=rng.integers(0, colour_max + 1, count).astype(np.uint16),
        green=rng.integers(0, colour_max + 1, count).astype(np.uint16),
        blue=rng.integers(0, colour_max + 1, count).astype(np.uint16),
        overlap=rng.integers(0, 2, count).astype(np.uint8),
    )


def write_cloud(
    path: Path, cloud: Cloud, *, point_format: int = 8, scale: float = 0.001
) -> None:
    """Write ``cloud`` as a LAZ of ``point_format`` at ``path``."""
    header = laspy.LasHeader(
        point_format=point_format,
        version="1.4" if point_format >= 6 else "1.2",
    )
    header.offsets = np.array([190_000.0, 440_000.0, -10.0])
    header.scales = np.array([scale, scale, scale])
    las = laspy.LasData(header)
    las.x, las.y, las.z = cloud.x, cloud.y, cloud.z
    las.classification = cloud.classification
    las.intensity = cloud.intensity
    las.return_number = cloud.return_number
    las.number_of_returns = cloud.number_of_returns
    if point_format in {2, 3, 5, 7, 8, 10}:
        las.red, las.green, las.blue = cloud.red, cloud.green, cloud.blue
    if point_format >= 6:
        las.overlap = cloud.overlap
    las.write(str(path))


@dataclass(frozen=True)
class Ply:
    """A parsed PLY: header text, comments and the vertex records."""

    header: str
    comments: dict[str, list[float]]
    vertex: npt.NDArray[np.void]

    def world(
        self, offset: tuple[float, float, float]
    ) -> npt.NDArray[np.float64]:
        """Return ``vertex + offset`` as an ``(n, 3)`` float64 array."""
        return np.column_stack(
            [
                self.vertex[axis].astype(np.float64) + shift
                for axis, shift in zip("xyz", offset, strict=True)
            ]
        )


def read_ply(path: Path) -> Ply:
    """Parse a binary little-endian PLY independently of the writer."""
    data = path.read_bytes()
    marker = b"end_header\n"
    end = data.index(marker) + len(marker)
    header = data[:end].decode("ascii")
    fields = [
        (m.group(2), _PLY_DTYPES[m.group(1)])
        for m in re.finditer(r"^property (\w+) (\w+)$", header, re.MULTILINE)
    ]
    found = re.search(r"^element vertex (\d+)$", header, re.MULTILINE)
    assert found is not None
    comments = {
        m.group(1): [float(v) for v in m.group(2).split()]
        for m in re.finditer(
            r"^comment ahn_cli\.(\w+) ([-\d. e]+)$", header, re.MULTILINE
        )
    }
    vertex = np.frombuffer(
        data[end:], dtype=np.dtype(fields), count=int(found.group(1))
    )
    return Ply(header, comments, vertex)
