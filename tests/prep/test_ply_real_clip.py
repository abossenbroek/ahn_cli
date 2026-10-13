"""Real-data check: a small clip of the Moerkapelle COPC through the PLY export.

Skipped when ``assets/ahn_moerkapelle.copc.laz`` (2.8 GB, not in git) is absent.
The clip is 60 m square, read through the COPC octree, so it takes seconds.
The expected points come from a second, independent route (laspy's own COPC
query plus a numpy mask), not from the code under test.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from laspy.copc import Bounds, CopcReader

from ahn_cli.prep.ply import CoordMode, export_ply
from ahn_cli.prep.ply_product import PlyProductRequest, export_ply_product
from ahn_cli.provenance.product import read_product_sidecar, sidecar_path
from tests.prep.ply_helpers import read_ply

_ASSET = (
    Path(__file__).resolve().parents[2]
    / "assets"
    / "ahn_moerkapelle.copc.laz"
)
_CLIP = (100_958.0, 449_397.0, 101_018.0, 449_457.0)  # 60 m, greenhouse area
_MM = 1e-3

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not _ASSET.is_file(), reason="Moerkapelle COPC absent"
    ),
]


def _reference() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.uint8]]:
    """Return (xyz, classification) of the clip from laspy's own COPC query."""
    bounds = Bounds(mins=np.array(_CLIP[:2]), maxs=np.array(_CLIP[2:]))
    with CopcReader.open(str(_ASSET)) as reader:
        records = reader.query(bounds)
    x = np.asarray(records.x)
    y = np.asarray(records.y)
    inside = (
        (x >= _CLIP[0]) & (x <= _CLIP[2]) & (y >= _CLIP[1]) & (y <= _CLIP[3])
    )
    xyz = np.column_stack([np.asarray(records[a])[inside] for a in "xyz"])
    return xyz, np.asarray(records.classification)[inside]


def _by_position(
    xyz: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Return ``xyz`` rows in lexicographic order, to compare as sets."""
    return xyz[np.lexsort((xyz[:, 2], xyz[:, 1], xyz[:, 0]))]


def test_clip_loses_no_point_class_colour_or_datum(tmp_path: Path) -> None:
    """Every clip point, its class and its colour reach the PLY; the datum holds."""
    xyz_ref, cls_ref = _reference()
    out = tmp_path / "clip.ply"

    stats = export_ply(
        _ASSET, out, bbox=_CLIP, coords=CoordMode.FLOAT32_CENTRED
    )

    ply = read_ply(out)
    assert stats.point_count == len(xyz_ref) > 10_000
    world = ply.world(stats.offset_xyz)
    a, b = _by_position(world), _by_position(xyz_ref)
    assert float(np.abs(a - b).max()) < _MM  # datum recoverable to 1 mm
    hist_ref = {int(c): int((cls_ref == c).sum()) for c in np.unique(cls_ref)}
    assert dict(stats.classes_out) == hist_ref
    assert stats.colour is not None
    assert stats.colour.source_max <= 255
    assert not stats.colour.narrowed
    assert int(ply.vertex["red"].max()) == stats.colour.source_max


def test_clip_sidecar_hashes_the_real_input(tmp_path: Path) -> None:
    """The sidecar's input sha256 is the real COPC's, and counts match the PLY."""
    out = tmp_path / "clip.ply"

    export_ply_product(
        PlyProductRequest(source=_ASSET, output=out, bbox=_CLIP)
    )

    record = read_product_sidecar(sidecar_path(out))
    digest = hashlib.sha256()
    with _ASSET.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    source = record["input"]
    assert isinstance(source, dict)
    assert source["sha256"] == digest.hexdigest()
    assert source["is_copc"] is True
    counts = record["counts"]
    assert isinstance(counts, dict)
    assert counts["out"] == len(read_ply(out).vertex)
