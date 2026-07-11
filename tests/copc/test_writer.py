"""Tests for the copc-context COPC container writer (copclib façade)."""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING

import copclib
import laspy
import numpy as np
import pytest

from ahn_cli.copc.octree import BuildPlan, CopcError, NodeKey
from ahn_cli.copc.scatter import RECORD_DTYPE
from ahn_cli.copc.writer import (
    BARE_POINT_FORMAT,
    RGB_POINT_FORMAT,
    CopcNodeWriter,
    patch_gps_range,
    rd_new_wkt,
)

if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt


def _plan() -> BuildPlan:
    return BuildPlan(
        scale=0.001,
        anchor_m=(-1, -1, -10),
        side_m=64,
        bucket_level=0,
        max_depth=1,
        sample_grid=128,
        units_per_m=1000,
        voxel_units=500,
    )


def _records(
    rows: list[tuple[int, int, int]],
    *,
    gps: list[float] | None = None,
    rgb: tuple[int, int, int] = (300, 400, 500),
    return_number: int = 1,
) -> npt.NDArray[np.void]:
    records = np.zeros(len(rows), dtype=RECORD_DTYPE)
    records["x"] = [r[0] for r in rows]
    records["y"] = [r[1] for r in rows]
    records["z"] = [r[2] for r in rows]
    records["intensity"] = 700
    records["return_number"] = return_number
    records["number_of_returns"] = max(return_number, 1)
    records["classification"] = 2
    records["red"], records["green"], records["blue"] = rgb
    if gps is not None:
        records["gps_time"] = gps
    return records


def test_written_header_bounds_are_bit_exact_decodes(
    tmp_path: Path,
) -> None:
    """Header min/max equal ``int * scale + offset`` of the written extremes.

    This is the core fix: the header and the decoded points share one float64
    provenance path, so the PDAL-style sub-scale epsilon cannot exist.
    """
    plan = _plan()
    out = tmp_path / "out.copc.laz"
    writer = CopcNodeWriter(
        out, plan, point_format_id=RGB_POINT_FORMAT, wkt=rd_new_wkt()
    )
    writer.add_node(
        NodeKey(0, 0, 0, 0), _records([(0, 0, 0), (63_000, 21_000, 58_000)])
    )
    assert writer.finish() == 2
    with laspy.open(str(out)) as reader:
        header = reader.header
        las = reader.read()
    assert header.mins.tolist() == [-1.0, -1.0, -10.0]
    expected_max_z = 58_000 * 0.001 + -10.0
    assert header.maxs[2] == expected_max_z
    assert np.asarray(las.X).min() == 0
    assert np.asarray(las.Z).max() == 58_000
    # decoded doubles equal the header bounds bit-for-bit
    assert float(np.asarray(las.z).min()) == header.mins[2]
    assert float(np.asarray(las.z).max()) == header.maxs[2]


def test_rgb_attributes_roundtrip(tmp_path: Path) -> None:
    """RGB, intensity and classification survive the raw-byte packing."""
    plan = _plan()
    out = tmp_path / "rgb.copc.laz"
    writer = CopcNodeWriter(
        out, plan, point_format_id=RGB_POINT_FORMAT, wkt=rd_new_wkt()
    )
    writer.add_node(
        NodeKey(0, 0, 0, 0), _records([(1, 2, 3)], rgb=(1000, 2000, 3000))
    )
    writer.finish()
    with laspy.open(str(out)) as reader:
        las = reader.read()
    assert int(np.asarray(las.red)[0]) == 1000
    assert int(np.asarray(las.green)[0]) == 2000
    assert int(np.asarray(las.blue)[0]) == 3000
    assert int(np.asarray(las.intensity)[0]) == 700
    assert int(np.asarray(las.classification)[0]) == 2


def test_bare_point_format_omits_rgb(tmp_path: Path) -> None:
    """PDRF 6 output carries no RGB dimensions at all."""
    plan = _plan()
    out = tmp_path / "bare.copc.laz"
    writer = CopcNodeWriter(
        out, plan, point_format_id=BARE_POINT_FORMAT, wkt=rd_new_wkt()
    )
    writer.add_node(NodeKey(0, 0, 0, 0), _records([(1, 2, 3)]))
    writer.finish()
    with laspy.open(str(out)) as reader:
        names = set(reader.header.point_format.dimension_names)
    assert "red" not in names
    assert reader.header.point_format.id == BARE_POINT_FORMAT


def test_node_points_are_sorted_by_gps_time(tmp_path: Path) -> None:
    """Within a node, points are written GPS-ascending (validator warns)."""
    plan = _plan()
    out = tmp_path / "gps.copc.laz"
    writer = CopcNodeWriter(
        out, plan, point_format_id=RGB_POINT_FORMAT, wkt=rd_new_wkt()
    )
    writer.add_node(
        NodeKey(0, 0, 0, 0),
        _records([(1, 1, 1), (2, 2, 2), (3, 3, 3)], gps=[30.0, 10.0, 20.0]),
    )
    writer.finish()
    with laspy.open(str(out)) as reader:
        las = reader.read()
    gps = np.asarray(las.gps_time)
    assert gps.tolist() == sorted(gps.tolist())


def test_gps_range_is_patched_into_the_info_vlr(tmp_path: Path) -> None:
    """Non-zero GPS times land in the info VLR's gpstime min/max fields."""
    plan = _plan()
    out = tmp_path / "range.copc.laz"
    writer = CopcNodeWriter(
        out, plan, point_format_id=RGB_POINT_FORMAT, wkt=rd_new_wkt()
    )
    writer.add_node(
        NodeKey(0, 0, 0, 0), _records([(1, 1, 1), (2, 2, 2)], gps=[5.5, 9.5])
    )
    writer.finish()
    raw = out.read_bytes()
    low, high = struct.unpack_from("<dd", raw, 375 + 54 + 56)
    assert (low, high) == (5.5, 9.5)


def test_all_zero_gps_leaves_the_vlr_untouched(tmp_path: Path) -> None:
    """A GPS-less cloud (all zeros) needs no patch: range stays [0, 0]."""
    plan = _plan()
    out = tmp_path / "zero.copc.laz"
    writer = CopcNodeWriter(
        out, plan, point_format_id=RGB_POINT_FORMAT, wkt=rd_new_wkt()
    )
    writer.add_node(NodeKey(0, 0, 0, 0), _records([(1, 1, 1)]))
    writer.finish()
    raw = out.read_bytes()
    low, high = struct.unpack_from("<dd", raw, 375 + 54 + 56)
    assert (low, high) == (0.0, 0.0)


def test_points_by_return_histogram_is_written(tmp_path: Path) -> None:
    """The LAS 1.4 per-return histogram reflects the written points."""
    plan = _plan()
    out = tmp_path / "returns.copc.laz"
    writer = CopcNodeWriter(
        out, plan, point_format_id=RGB_POINT_FORMAT, wkt=rd_new_wkt()
    )
    writer.add_node(
        NodeKey(0, 0, 0, 0), _records([(1, 1, 1)], return_number=2)
    )
    writer.add_node(
        NodeKey(1, 0, 0, 0), _records([(2, 2, 2)], return_number=1)
    )
    writer.finish()
    with laspy.open(str(out)) as reader:
        by_return = list(reader.header.number_of_points_by_return)
    assert by_return[0] == 1
    assert by_return[1] == 1


def test_empty_node_is_rejected(tmp_path: Path) -> None:
    """Zero-point nodes are a builder bug and must never reach the file."""
    plan = _plan()
    writer = CopcNodeWriter(
        tmp_path / "e.copc.laz",
        plan,
        point_format_id=RGB_POINT_FORMAT,
        wkt=rd_new_wkt(),
    )
    with pytest.raises(CopcError, match="zero points"):
        writer.add_node(NodeKey(0, 0, 0, 0), np.zeros(0, dtype=RECORD_DTYPE))


def test_finishing_with_no_points_is_an_error(tmp_path: Path) -> None:
    """A file with no written nodes cannot be sealed."""
    writer = CopcNodeWriter(
        tmp_path / "n.copc.laz",
        _plan(),
        point_format_id=RGB_POINT_FORMAT,
        wkt=rd_new_wkt(),
    )
    with pytest.raises(CopcError, match="no points"):
        writer.finish()


def _exploding_add_node(
    self: copclib.FileWriter,
    key: copclib.VoxelKey,
    uncompressed_data: copclib.VectorChar,
) -> None:
    """Stand in for copclib's native AddNode, always failing."""
    del self, key, uncompressed_data
    msg = "native writer exploded"
    raise RuntimeError(msg)


def _exploding_close(self: copclib.FileWriter) -> None:
    """Stand in for copclib's native Close, always failing."""
    del self
    msg = "native close exploded"
    raise RuntimeError(msg)


def test_add_node_wraps_a_copclib_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A native copclib AddNode failure surfaces as the typed CopcError."""
    writer = CopcNodeWriter(
        tmp_path / "boom.copc.laz",
        _plan(),
        point_format_id=RGB_POINT_FORMAT,
        wkt=rd_new_wkt(),
    )
    monkeypatch.setattr(copclib.FileWriter, "AddNode", _exploding_add_node)

    with pytest.raises(CopcError, match="failed to write node"):
        writer.add_node(NodeKey(0, 0, 0, 0), _records([(1, 1, 1)]))


def test_finish_wraps_a_copclib_close_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A native copclib Close failure surfaces as the typed CopcError."""
    writer = CopcNodeWriter(
        tmp_path / "close.copc.laz",
        _plan(),
        point_format_id=RGB_POINT_FORMAT,
        wkt=rd_new_wkt(),
    )
    writer.add_node(NodeKey(0, 0, 0, 0), _records([(1, 1, 1)]))
    monkeypatch.setattr(copclib.FileWriter, "Close", _exploding_close)

    with pytest.raises(CopcError, match="failed to close"):
        writer.finish()


def test_unwritable_path_is_a_copc_error(tmp_path: Path) -> None:
    """An unwritable output path surfaces as the context's typed error."""
    with pytest.raises(CopcError, match="cannot open"):
        CopcNodeWriter(
            tmp_path / "missing_dir" / "out.copc.laz",
            _plan(),
            point_format_id=RGB_POINT_FORMAT,
            wkt=rd_new_wkt(),
        )


def test_patch_refuses_a_non_copc_file(tmp_path: Path) -> None:
    """The VLR patch verifies the info VLR before touching bytes."""
    bogus = tmp_path / "bogus.laz"
    bogus.write_bytes(b"\x00" * 600)
    with pytest.raises(CopcError, match="info VLR"):
        patch_gps_range(bogus, 1.0, 2.0)


def test_wkt_is_wkt1() -> None:
    """The SRS helper emits WKT1 (proj4js inside the validator needs it)."""
    wkt = rd_new_wkt()
    assert wkt.startswith("PROJCS[")  # WKT2 would start with PROJCRS
    assert "Amersfoort" in wkt


def test_header_carries_pinned_metadata(tmp_path: Path) -> None:
    """Creation date and system id are pinned constants (determinism)."""
    out = tmp_path / "meta.copc.laz"
    writer = CopcNodeWriter(
        out, _plan(), point_format_id=RGB_POINT_FORMAT, wkt=rd_new_wkt()
    )
    writer.add_node(NodeKey(0, 0, 0, 0), _records([(1, 1, 1)]))
    writer.finish()
    with laspy.open(str(out)) as reader:
        header = reader.header
    assert header.creation_date is not None
    assert (header.creation_date.year, header.creation_date.month) == (
        2020,
        1,
    )
    assert header.system_identifier.rstrip("\x00") == "ahn_cli"
    assert header.generating_software.rstrip("\x00") == "ahn_cli copc"
