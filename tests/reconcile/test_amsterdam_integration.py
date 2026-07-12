"""Integration test on the real Amsterdam AHN5 + orthophoto data (CC-BY 4.0).

A ~20 m urban window of the Dam, Amsterdam package -- the 8 cm orthophoto and the
AHN5 point cloud clipped to it -- is committed under ``fixtures/`` (git-LFS) and
reconciled end to end here. This is the epic's "test with the Amsterdam datasets"
check: it exercises all three methods and all four output formats on genuine
data, including the ill-conditioned kriging neighbourhoods that coincident LiDAR
returns produce, plus cross-run determinism of the numpy path.

Data: AHN5 (Het Waterschapshuis) & Beeldmateriaal / PDOK, licensed CC-BY 4.0.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from ahn_cli.reconcile.method import (
    IdwInterp,
    InterpMethod,
    KrigingInterp,
    LinearInterp,
    Variogram,
    VariogramModel,
)
from ahn_cli.reconcile.reconcile import (
    ReconcileError,
    ReconcileRequest,
    reconcile,
)
from ahn_cli.reconcile.writers import OutputFormat

_FIXTURES = Path(__file__).parent / "fixtures"
_ORTHO = _FIXTURES / "amsterdam_ortho.tif"
_CLOUD = _FIXTURES / "amsterdam_cloud.laz"
_SIZE = 256
_PIXELS = _SIZE * _SIZE
# The Amsterdam window spans NAP heights of roughly 0-11 m (canal water to roofs).
_Z_FLOOR = -10.0
_Z_CEIL = 100.0


def _request(
    out: Path,
    method: InterpMethod,
    formats: tuple[OutputFormat, ...],
) -> ReconcileRequest:
    return ReconcileRequest(
        ortho_path=_ORTHO,
        cloud_path=_CLOUD,
        output_dir=out,
        method=method,
        formats=formats,
    )


def _pt_points(out: Path) -> npt.NDArray[np.float32]:
    raw = (out / "reconciled.pt").read_bytes()
    return np.frombuffer(raw, dtype="<f4").reshape(-1, 6)


def test_amsterdam_idw_all_formats(tmp_path: Path) -> None:
    """IDW reconciles the full window and writes all four formats."""
    stats = reconcile(
        _request(tmp_path / "out", IdwInterp(k=12), tuple(OutputFormat))
    )
    assert (stats.width, stats.height) == (_SIZE, _SIZE)
    assert stats.valid_points == _PIXELS  # every cell has neighbours
    # Real AHN overlap leaves coincident returns; dedup removes them.
    assert 0 < stats.cleaned_points < stats.source_points
    assert {path.name for path in stats.outputs} == {
        "reconciled.laz",
        "reconciled.ply",
        "reconciled.pt",
        "reconciled.exr",
    }
    for path in stats.outputs:
        assert path.stat().st_size > 0
    points = _pt_points(tmp_path / "out")
    assert points.shape == (_PIXELS, 6)
    assert float(points[:, 2].min()) > _Z_FLOOR
    assert float(points[:, 2].max()) < _Z_CEIL
    assert (
        float(points[:, 3:6].max()) == 255.0
    )  # ortho colour carried through


def test_amsterdam_linear_hull_voids_are_refused(tmp_path: Path) -> None:
    """Linear's out-of-hull border cells are missing data: a hard error.

    The real window's border pixel centres fall outside the cloud's
    convex hull, so linear interpolation cannot produce a genuine
    estimate there. Reconcile refuses instead of writing a partial
    grid, and leaves no output files behind.
    """
    with pytest.raises(ReconcileError, match="no genuine elevation estimate"):
        reconcile(
            _request(tmp_path / "out", LinearInterp(), (OutputFormat.PT,))
        )
    assert list((tmp_path / "out").iterdir()) == []


def test_amsterdam_kriging_handles_coincident_returns(tmp_path: Path) -> None:
    """Kriging completes on real data despite ill-conditioned neighbourhoods."""
    method = KrigingInterp(
        variogram=Variogram(VariogramModel.SPHERICAL, 0.0, 1.0, 5.0), k=12
    )
    stats = reconcile(_request(tmp_path / "out", method, (OutputFormat.PT,)))
    assert stats.valid_points == _PIXELS
    points = _pt_points(tmp_path / "out")
    assert float(points[:, 2].min()) > _Z_FLOOR
    assert float(points[:, 2].max()) < _Z_CEIL


def test_amsterdam_reconcile_is_deterministic(tmp_path: Path) -> None:
    """Two numpy-backend runs on the real data are byte-identical."""
    formats = (
        OutputFormat.LAZ,
        OutputFormat.PLY,
        OutputFormat.PT,
        OutputFormat.EXR,
    )
    first = reconcile(_request(tmp_path / "a", IdwInterp(k=12), formats))
    second = reconcile(_request(tmp_path / "b", IdwInterp(k=12), formats))
    for left, right in zip(first.outputs, second.outputs, strict=True):
        assert left.read_bytes() == right.read_bytes()
