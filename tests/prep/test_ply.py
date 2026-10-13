"""Tests for the prep-context lossless LAZ/COPC -> PLY export.

Fixtures are small synthetic LAZ clouds built in-process with laspy -- no
network, no large files. They prove the contract that matters downstream: the
datum survives to 1 mm, class / colour / intensity / return attributes survive
exactly, thinning by spacing really separates points, the export is
deterministic and independent of the streaming window, and the cloud is read in
bounded windows.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import laspy
import numpy as np
import numpy.typing as npt
import pytest
from scipy.spatial import cKDTree
from typing_extensions import Self

from ahn_cli.copc.build import build_copc
from ahn_cli.prep import ply as ply_module
from ahn_cli.prep.decimate import PoissonThinning
from ahn_cli.prep.ply import (
    ColourRecord,
    CoordMode,
    PlyExportError,
    PlyExportStats,
    ShareThinning,
    export_ply,
)
from tests.prep.ply_helpers import (
    MM,
    ZERO,
    make_cloud,
    read_ply,
    write_cloud,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

# Real (unpatched) laspy.open, captured so the streaming spy can delegate to it
# after ``monkeypatch`` has replaced ``laspy.open`` with the spy factory.
_REAL_OPEN = laspy.open


def _order(values: npt.NDArray[np.float64]) -> npt.NDArray[np.intp]:
    """Lexicographic (x, y, z) order, to compare unordered point sets."""
    return np.lexsort((values[:, 2], values[:, 1], values[:, 0]))


# --------------------------------------------------------------------------
# Value objects and guards
# --------------------------------------------------------------------------


def test_stats_is_a_frozen_value_object() -> None:
    """PlyExportStats is hashable and equal by field value."""
    stats = PlyExportStats(point_count=5)

    assert stats == PlyExportStats(5)
    assert len({stats, PlyExportStats(5)}) == 1


@pytest.mark.parametrize("share", [0.0, -0.1, 1.5, float("nan")])
def test_share_thinning_rejects_out_of_range(share: float) -> None:
    """A share outside (0, 1] is refused."""
    with pytest.raises(ValueError, match="share"):
        ShareThinning(share=share)


def test_export_rejects_non_positive_chunk_size(tmp_path: Path) -> None:
    """A non-positive streaming window is refused before any I/O."""
    src = tmp_path / "src.laz"
    write_cloud(src, make_cloud(10))

    with pytest.raises(ValueError, match="positive"):
        export_ply(src, tmp_path / "out.ply", chunk_size=0)


def test_export_rejects_a_degenerate_bbox(tmp_path: Path) -> None:
    """A bbox with min >= max is refused."""
    src = tmp_path / "src.laz"
    write_cloud(src, make_cloud(10))

    with pytest.raises(ValueError, match="bbox|extent|min"):
        export_ply(src, tmp_path / "out.ply", bbox=(5.0, 5.0, 1.0, 9.0))


def test_empty_selection_is_refused(tmp_path: Path) -> None:
    """A filter that keeps nothing raises instead of writing an empty PLY."""
    src = tmp_path / "src.laz"
    write_cloud(src, make_cloud(50))

    with pytest.raises(PlyExportError, match="no points"):
        export_ply(src, tmp_path / "out.ply", include_classes=(18,))


# --------------------------------------------------------------------------
# Lossless round trip: datum, class, colour, intensity, returns, overlap
# --------------------------------------------------------------------------


def test_header_is_additive_after_xyz_and_rgb(tmp_path: Path) -> None:
    """X y z [rgb] come first, attributes follow, comments record the datum."""
    src = tmp_path / "src.laz"
    write_cloud(src, make_cloud(20))
    out = tmp_path / "out.ply"

    stats = export_ply(src, out)

    ply = read_ply(out)
    names = [name for name, _ in stats.properties]
    assert names == [
        "x",
        "y",
        "z",
        "red",
        "green",
        "blue",
        "classification",
        "intensity",
        "return_number",
        "number_of_returns",
        "overlap",
    ]
    assert ply.vertex.dtype.names == tuple(names)
    assert ply.header.startswith("ply\nformat binary_little_endian 1.0\n")
    assert (
        "property double x\nproperty double y\nproperty double z\n"
        "property uchar red\n"
    ) in ply.header
    assert ply.comments["offset_xyz"] == [0.0, 0.0, 0.0]
    assert ply.comments["las_scale"] == [0.001, 0.001, 0.001]
    assert ply.comments["las_offset"] == [190_000.0, 440_000.0, -10.0]


@pytest.mark.parametrize("chunk_size", [7, 1_000_000])
def test_round_trip_keeps_datum_class_colour_and_attributes(
    tmp_path: Path, chunk_size: int
) -> None:
    """LAS -> PLY -> read back loses no coordinate, class, colour or attribute."""
    cloud = make_cloud(400)
    src = tmp_path / "src.laz"
    write_cloud(src, cloud)
    out = tmp_path / "out.ply"

    stats = export_ply(src, out, chunk_size=chunk_size)

    ply = read_ply(out)
    assert stats.point_count == len(cloud.x) == len(ply.vertex)
    with laspy.open(str(src)) as reader:  # bit-exact against what LAS stores
        stored = reader.read()
    assert np.array_equal(ply.vertex["x"], np.asarray(stored.x))
    assert np.array_equal(ply.vertex["y"], np.asarray(stored.y))
    assert np.array_equal(ply.vertex["z"], np.asarray(stored.z))
    assert float(np.abs(ply.vertex["z"] - cloud.z).max()) < MM
    assert np.array_equal(ply.vertex["classification"], cloud.classification)
    assert np.array_equal(ply.vertex["intensity"], cloud.intensity)
    assert np.array_equal(ply.vertex["return_number"], cloud.return_number)
    assert np.array_equal(
        ply.vertex["number_of_returns"], cloud.number_of_returns
    )
    assert np.array_equal(ply.vertex["overlap"], cloud.overlap)
    assert np.array_equal(ply.vertex["red"], cloud.red)
    assert np.array_equal(ply.vertex["green"], cloud.green)
    assert np.array_equal(ply.vertex["blue"], cloud.blue)
    assert stats.colour == ColourRecord(
        source_max=int(
            max(cloud.red.max(), cloud.green.max(), cloud.blue.max())
        ),
        narrowed=False,
    )
    assert stats.classes_scanned == stats.classes_out
    assert sum(n for _, n in stats.classes_out) == len(cloud.x)


def test_float32_centred_offset_is_recoverable_to_one_millimetre(
    tmp_path: Path,
) -> None:
    """float32 vertices + the recorded offset give back the world position."""
    cloud = make_cloud(400)
    src = tmp_path / "src.laz"
    write_cloud(src, cloud)
    out = tmp_path / "out.ply"

    stats = export_ply(src, out, coords=CoordMode.FLOAT32_CENTRED)

    ply = read_ply(out)
    assert ply.vertex["x"].dtype == np.dtype("<f4")
    assert tuple(ply.comments["offset_xyz"]) == stats.offset_xyz
    assert stats.offset_xyz != ZERO
    for value in stats.offset_xyz:  # millimetre-exact offset
        assert round(value, 3) == value
    world = ply.world(stats.offset_xyz)
    truth = np.column_stack([cloud.x, cloud.y, cloud.z])
    assert float(np.abs(world - truth).max()) < MM
    assert float(np.abs(ply.vertex["x"]).max()) < 100.0  # actually centred
    assert float(ply.vertex["z"].min()) >= -MM  # z measured from its minimum


def test_float32_centred_refuses_an_extent_beyond_one_millimetre(
    tmp_path: Path,
) -> None:
    """A cloud too wide for float32 precision must use float64 instead."""
    wide = replace(
        make_cloud(4),
        x=np.array([194_000.0, 254_000.0, 194_100.0, 194_200.0]),
    )
    src = tmp_path / "wide.laz"
    write_cloud(src, wide)

    with pytest.raises(PlyExportError, match="float64"):
        export_ply(src, tmp_path / "o.ply", coords=CoordMode.FLOAT32_CENTRED)


def test_colour_stored_in_sixteen_bits_is_narrowed_and_recorded(
    tmp_path: Path,
) -> None:
    """Colour above 255 writes the high byte and says so; it never wraps."""
    cloud = make_cloud(100, colour_max=65_535)
    src = tmp_path / "src.laz"
    write_cloud(src, cloud)
    out = tmp_path / "out.ply"

    stats = export_ply(src, out)

    ply = read_ply(out)
    assert stats.colour is not None
    assert stats.colour.narrowed
    assert stats.colour.source_max > 255
    assert np.array_equal(
        ply.vertex["red"], (cloud.red >> 8).astype(np.uint8)
    )


def test_colour_depth_is_decided_from_all_points_not_a_sample(
    tmp_path: Path,
) -> None:
    """A bright value late in the file still triggers narrowing."""
    cloud = make_cloud(3000, colour_max=200)
    red = cloud.red.copy()
    red[-1] = 40_000
    src = tmp_path / "src.laz"
    write_cloud(src, replace(cloud, red=red))

    stats = export_ply(src, tmp_path / "out.ply")

    assert stats.colour is not None
    assert stats.colour.narrowed
    assert stats.colour.source_max == 40_000


def test_formats_without_colour_or_overlap_omit_those_properties(
    tmp_path: Path,
) -> None:
    """Point format 3 has colour but no overlap; format 6 has overlap, no colour."""
    cloud = make_cloud(30)
    fmt3 = tmp_path / "f3.laz"
    fmt6 = tmp_path / "f6.laz"
    write_cloud(fmt3, cloud, point_format=3)
    write_cloud(fmt6, cloud, point_format=6)

    stats3 = export_ply(fmt3, tmp_path / "f3.ply")
    stats6 = export_ply(fmt6, tmp_path / "f6.ply")

    names3 = [name for name, _ in stats3.properties]
    names6 = [name for name, _ in stats6.properties]
    assert "red" in names3
    assert "overlap" not in names3
    assert "red" not in names6
    assert "overlap" in names6
    assert stats6.colour is None


# --------------------------------------------------------------------------
# Selection: class filter, bbox, COPC node reads
# --------------------------------------------------------------------------


def test_class_filter_keeps_only_the_listed_classes(tmp_path: Path) -> None:
    """include_classes=(2, 6) keeps exactly those points, in source order."""
    cloud = make_cloud(400)
    src = tmp_path / "src.laz"
    write_cloud(src, cloud)
    out = tmp_path / "out.ply"

    stats = export_ply(src, out, include_classes=(2, 6), chunk_size=33)

    keep = np.isin(cloud.classification, [2, 6])
    ply = read_ply(out)
    assert stats.point_count == int(keep.sum()) == len(ply.vertex)
    assert stats.selected_points == stats.point_count
    assert stats.scanned_points == len(cloud.x)
    assert np.array_equal(ply.vertex["x"], cloud.x[keep])
    assert {c for c, _ in stats.classes_out} == {2, 6}
    assert {c for c, _ in stats.classes_scanned} == {1, 2, 6, 9, 26}


def test_bbox_clips_a_plain_laz(tmp_path: Path) -> None:
    """Bbox keeps only points inside it on a non-COPC source."""
    cloud = make_cloud(400)
    src = tmp_path / "src.laz"
    write_cloud(src, cloud)
    box = (194_020.0, 443_020.0, 194_050.0, 443_060.0)

    stats = export_ply(src, tmp_path / "out.ply", bbox=box)

    inside = (
        (cloud.x >= box[0])
        & (cloud.x <= box[2])
        & (cloud.y >= box[1])
        & (cloud.y <= box[3])
    )
    assert stats.point_count == int(inside.sum()) > 0
    assert stats.scanned_points == stats.point_count
    assert stats.source_points == len(cloud.x)


def test_bbox_on_a_copc_reads_nodes_and_matches_the_plain_clip(
    tmp_path: Path,
) -> None:
    """A COPC bbox clip yields the same points as clipping the plain LAZ."""
    side = 40
    xs, ys = np.meshgrid(
        np.arange(side, dtype=float), np.arange(side, dtype=float)
    )
    base = make_cloud(side * side)
    cloud = replace(
        base,
        x=194_000.0 + xs.ravel(),
        y=443_000.0 + ys.ravel(),
        z=1.0 + (xs.ravel() + ys.ravel()) % 5,
    )
    plain = tmp_path / "plain.laz"
    write_cloud(plain, cloud, point_format=7)
    copc = tmp_path / "plain.copc.laz"
    assert build_copc(plain, copc).written_points == side * side
    box = (194_010.0, 443_005.0, 194_025.0, 443_020.0)

    from_copc = export_ply(copc, tmp_path / "c.ply", bbox=box, chunk_size=50)
    from_plain = export_ply(plain, tmp_path / "p.ply", bbox=box)

    assert from_copc.point_count == from_plain.point_count == 16 * 16
    a = read_ply(tmp_path / "c.ply").world(ZERO)
    b = read_ply(tmp_path / "p.ply").world(ZERO)
    assert np.array_equal(a[_order(a)], b[_order(b)])
    assert from_copc.classes_out == from_plain.classes_out


# --------------------------------------------------------------------------
# Thinning: spacing (hard guarantee) and the legacy share
# --------------------------------------------------------------------------


def test_spacing_thinning_separates_points_on_a_synthetic_grid(
    tmp_path: Path,
) -> None:
    """Nearest-neighbour distance after Poisson thinning is >= 0.9 x spacing."""
    side = 40
    pitch = 0.25
    xs, ys = np.meshgrid(np.arange(side) * pitch, np.arange(side) * pitch)
    cloud = replace(
        make_cloud(side * side),
        x=194_000.0 + xs.ravel(),
        y=443_000.0 + ys.ravel(),
        z=np.zeros(side * side),
    )
    src = tmp_path / "grid.laz"
    write_cloud(src, cloud)
    out = tmp_path / "out.ply"
    spacing = 1.0

    stats = export_ply(
        src, out, thinning=PoissonThinning(radius=spacing), chunk_size=300
    )

    xyz = read_ply(out).world(ZERO)
    distances, _ = cKDTree(xyz).query(xyz, k=2)
    assert float(distances[:, 1].min()) >= 0.9 * spacing
    assert 0 < stats.point_count < stats.selected_points == side * side
    # still covers the 10 m x 10 m patch: at least a third of the ideal count
    assert stats.point_count > (side * pitch / spacing) ** 2 / 3


def test_spacing_thinning_is_deterministic_and_seeded(tmp_path: Path) -> None:
    """Same seed, same bytes across windows; another seed, another sample."""
    src = tmp_path / "src.laz"
    write_cloud(src, make_cloud(500))
    spec = PoissonThinning(radius=4.0, seed=3)

    export_ply(src, tmp_path / "a.ply", thinning=spec)
    export_ply(src, tmp_path / "b.ply", thinning=spec, chunk_size=64)
    export_ply(src, tmp_path / "c.ply", thinning=PoissonThinning(4.0, seed=4))

    first = (tmp_path / "a.ply").read_bytes()
    assert first == (tmp_path / "b.ply").read_bytes()
    assert first != (tmp_path / "c.ply").read_bytes()


def test_share_thinning_keeps_the_requested_share(tmp_path: Path) -> None:
    """The legacy share keeps round(share * n) points, seeded and ordered."""
    src = tmp_path / "src.laz"
    write_cloud(src, make_cloud(400))
    spec = ShareThinning(share=0.25, seed=1)

    stats = export_ply(src, tmp_path / "o.ply", thinning=spec)
    again = export_ply(src, tmp_path / "o2.ply", thinning=spec)

    assert stats.point_count == 100
    assert (tmp_path / "o.ply").read_bytes() == (
        tmp_path / "o2.ply"
    ).read_bytes()
    assert again == stats


def test_thinning_refuses_a_selection_above_the_memory_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Thinning keeps coordinates in memory, so a huge selection is refused."""
    src = tmp_path / "src.laz"
    write_cloud(src, make_cloud(100))
    monkeypatch.setattr(ply_module, "MAX_THINNING_POINTS", 50)

    with pytest.raises(PlyExportError, match="bbox"):
        export_ply(src, tmp_path / "o.ply", thinning=ShareThinning(share=0.5))


# --------------------------------------------------------------------------
# Determinism and streaming
# --------------------------------------------------------------------------


def test_export_is_byte_identical_on_repeat_and_across_chunk_sizes(
    tmp_path: Path,
) -> None:
    """Identical input yields identical bytes whatever the window."""
    src = tmp_path / "src.laz"
    write_cloud(src, make_cloud(300))

    stats_a = export_ply(src, tmp_path / "a.ply")
    stats_b = export_ply(src, tmp_path / "b.ply", chunk_size=11)

    assert stats_a == stats_b
    assert (tmp_path / "a.ply").read_bytes() == (
        tmp_path / "b.ply"
    ).read_bytes()


class _StreamLog:
    """A ledger of how an :class:`_SpyReader` was consumed."""

    def __init__(self) -> None:
        self.chunk_sizes: list[int] = []
        self.read_calls: int = 0


class _SpyReader:
    """A laspy reader wrapper recording chunk sizes and full-read attempts."""

    def __init__(self, path: str, log: _StreamLog) -> None:
        self._path = path
        self._log = log

    def __enter__(self) -> Self:
        self._cm = _REAL_OPEN(self._path)
        self._inner = self._cm.__enter__()
        return self

    def __exit__(self, *args: object) -> None:
        self._cm.__exit__(*args)

    @property
    def header(self) -> laspy.LasHeader:
        """Delegate the header the export reads its point count from."""
        return self._inner.header

    def read(self) -> laspy.ScaleAwarePointRecord:
        """Record and forward a full-record read (the anti-pattern to catch)."""
        self._log.read_calls += 1
        return self._inner.read()

    def chunk_iterator(
        self, points_per_iteration: int
    ) -> Iterator[laspy.ScaleAwarePointRecord]:
        """Yield the real reader's chunks, tallying each chunk's size."""
        for chunk in self._inner.chunk_iterator(points_per_iteration):
            self._log.chunk_sizes.append(len(chunk))
            yield chunk


def _spy_open_factory(log: _StreamLog) -> Callable[[str], _SpyReader]:
    """Return a drop-in ``laspy.open`` that produces logging readers."""

    def _spy_open(source: str) -> _SpyReader:
        return _SpyReader(source, log)

    return _spy_open


def test_export_streams_two_passes_in_bounded_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Scan then write: each pass reads windows of at most chunk_size."""
    src = tmp_path / "big.laz"
    write_cloud(src, make_cloud(25))
    log = _StreamLog()
    monkeypatch.setattr(laspy, "open", _spy_open_factory(log))

    stats = export_ply(src, tmp_path / "out.ply", chunk_size=10)

    assert log.read_calls == 0
    assert log.chunk_sizes == [10, 10, 5, 10, 10, 5]
    assert stats.point_count == 25


def test_export_reports_progress_per_written_chunk(tmp_path: Path) -> None:
    """Each written window reports (chunks_done, total_chunks)."""
    src = tmp_path / "big.laz"
    write_cloud(src, make_cloud(25))
    calls: list[tuple[int, int]] = []

    export_ply(
        src,
        tmp_path / "out.ply",
        chunk_size=10,
        progress=lambda done, total: calls.append((done, total)),
    )

    assert calls == [(1, 3), (2, 3), (3, 3)]
