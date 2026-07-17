"""Tests for the prep-context ``positions.exr`` export (WP12).

Fixtures are tiny valid EPSG:28992 GeoTIFFs built in-process with rasterio (a
stand-in for ``data/<site>/dsm.tif``) -- no network, no large files. They pin
the hand-written, uncompressed OpenEXR container byte-for-byte: the magic +
version prefix, the header attributes, the scanline offset table, and the
float32 R=easting / G=northing / B=elevation payload. A self-contained EXR
parser reconstructs the three channels so correctness is proven without any
external OpenEXR dependency (mirroring how ``test_ply`` parses its own PLY).

Both the nodata branch (a void pixel collapses to the Z=0.0 sentinel while its
X/Y stay set) and the valid-data branch are exercised, plus byte-identity
across two writes and the 1x1 / non-square / no-nodata edges.
"""

from __future__ import annotations

import hashlib
import struct
from typing import TYPE_CHECKING, cast

import numpy as np
import numpy.typing as npt
import pytest
import rasterio
from rasterio.transform import from_bounds
from rasterio.windows import Window

from ahn_cli.prep.positions import (
    PositionsExportError,
    PositionsExportStats,
    export_positions,
)

if TYPE_CHECKING:
    from pathlib import Path

# A north-up 0.5 m grid, top-left corner at (194000, 443020) on the Dutch grid.
_ORIGIN_X = 194000.0
_ORIGIN_Y = 443020.0
_RES = 0.5
_NODATA = -9999.0

_MAGIC = struct.pack("<I", 0x01312F76)
_VERSION = struct.pack("<I", 2)

# Golden: the exact magic + version + full header block for a 1x1 image. Every
# byte is a deterministic constant (the only size-dependent attribute is the
# data/display window, here 0,0,0,0). Pinning it locks the attribute order,
# each field width, and the channel count against any silent regression.
_GOLDEN_1X1_HEADER = bytes.fromhex(
    "762f3101020000006368616e6e656c730063686c697374003700000042000200"
    "0000000000000100000001000000470002000000000000000100000001000000"
    "52000200000000000000010000000100000000636f6d7072657373696f6e0063"
    "6f6d7072657373696f6e0001000000006461746157696e646f7700626f783269"
    "001000000000000000000000000000000000000000646973706c617957696e64"
    "6f7700626f7832690010000000000000000000000000000000000000006c696e"
    "654f72646572006c696e654f72646572000100000000706978656c4173706563"
    "74526174696f00666c6f617400040000000000803f73637265656e57696e646f"
    "7743656e746572007632660008000000000000000000000073637265656e5769"
    "6e646f77576964746800666c6f617400040000000000803f00"
)


def _write_dsm(
    path: Path,
    elevation: npt.NDArray[np.float32],
    *,
    nodata: float | None = _NODATA,
) -> None:
    """Write ``elevation`` as a single-band EPSG:28992 float32 GeoTIFF."""
    height, width = elevation.shape
    # North-up grid: top-left corner (_ORIGIN_X, _ORIGIN_Y), _RES m pixels.
    transform = from_bounds(
        _ORIGIN_X,
        _ORIGIN_Y - height * _RES,
        _ORIGIN_X + width * _RES,
        _ORIGIN_Y,
        width,
        height,
    )
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs="EPSG:28992",
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(elevation, 1)


def _read_cstr(data: bytes, pos: int) -> tuple[str, int]:
    """Read a NUL-terminated ASCII string, returning it and the next offset."""
    end = data.index(b"\x00", pos)
    return data[pos:end].decode("ascii"), end + 1


def _parse_exr(
    data: bytes,
) -> tuple[
    int,
    int,
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
]:
    """Parse the hand-written EXR back into ``(w, h, R, G, B)`` planes.

    Independently validates the magic/version, the header attributes, the
    scanline offset table (each offset is followed there), and the alphabetical
    B/G/R FLOAT channel interleave -- so a byte-layout regression is caught.
    """
    assert data[:4] == _MAGIC
    assert data[4:8] == _VERSION
    pos = 8
    attrs: dict[str, tuple[str, bytes]] = {}
    while data[pos] != 0:
        name, pos = _read_cstr(data, pos)
        type_name, pos = _read_cstr(data, pos)
        (size,) = struct.unpack_from("<I", data, pos)
        pos += 4
        attrs[name] = (type_name, data[pos : pos + size])
        pos += size
    pos += 1  # header-terminating NUL

    xmin, ymin, xmax, ymax = struct.unpack("<4i", attrs["dataWindow"][1])
    width = xmax - xmin + 1
    height = ymax - ymin + 1

    offsets = struct.unpack_from(f"<{height}Q", data, pos)
    channels: dict[str, npt.NDArray[np.float32]] = {
        name: np.zeros((height, width), dtype=np.float32)
        for name in ("R", "G", "B")
    }
    for off in offsets:
        (y,) = struct.unpack_from("<i", data, off)
        row = y - ymin
        cursor = off + 8  # skip y (4) + data-size (4)
        for name in ("B", "G", "R"):  # alphabetical channel interleave
            channels[name][row] = np.frombuffer(
                data, dtype="<f4", count=width, offset=cursor
            )
            cursor += width * 4
    return width, height, channels["R"], channels["G"], channels["B"]


# --------------------------------------------------------------------------
# Value object
# --------------------------------------------------------------------------


def test_positions_export_stats_is_a_frozen_value_object() -> None:
    """PositionsExportStats is hashable and equal by field value."""
    stats = PositionsExportStats(width=2, height=3, nodata_pixels=1)

    assert stats == PositionsExportStats(width=2, height=3, nodata_pixels=1)
    assert len({stats, PositionsExportStats(2, 3, 1)}) == 1


# --------------------------------------------------------------------------
# Container: magic, version, header attributes
# --------------------------------------------------------------------------


def test_exr_has_magic_version_and_controlled_header(tmp_path: Path) -> None:
    """The container opens with the EXR magic/version and the pinned header."""
    src = tmp_path / "dsm.tif"
    _write_dsm(src, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    out = tmp_path / "positions.exr"

    export_positions(src, out)

    data = out.read_bytes()
    assert data.startswith(_MAGIC + _VERSION)
    # Alphabetical FLOAT (pixel type 2) channels B, G, R, and NO_COMPRESSION.
    assert b"channels\x00chlist\x00" in data
    assert b"B\x00" in data
    assert b"compression\x00compression\x00\x01\x00\x00\x00\x00" in data
    # No non-deterministic attributes are emitted.
    assert b"capDate" not in data
    assert b"owner" not in data
    assert b"comments" not in data


def test_exr_datawindow_matches_raster_dimensions(tmp_path: Path) -> None:
    """The dataWindow/displayWindow describe the raster's pixel extent."""
    src = tmp_path / "dsm.tif"
    _write_dsm(src, np.arange(15, dtype=np.float32).reshape(3, 5))
    out = tmp_path / "positions.exr"

    export_positions(src, out)

    width, height, _, _, _ = _parse_exr(out.read_bytes())
    assert (width, height) == (5, 3)


# --------------------------------------------------------------------------
# Payload: valid-data branch
# --------------------------------------------------------------------------


def test_valid_pixels_carry_world_xy_and_elevation(tmp_path: Path) -> None:
    """R/G/B are the float32 pixel-centre easting/northing/elevation."""
    elevation = np.array([[10.0, 11.0], [12.0, 13.0]], dtype=np.float32)
    src = tmp_path / "dsm.tif"
    _write_dsm(src, elevation)
    out = tmp_path / "positions.exr"

    stats = export_positions(src, out)

    width, height, r, g, b = _parse_exr(out.read_bytes())
    assert (width, height) == (2, 2)
    assert stats == PositionsExportStats(width=2, height=2, nodata_pixels=0)
    # Pixel (row0, col0) centre: X = 194000.25, Y = 443019.75.
    assert r[0, 0] == np.float32(194000.25)
    assert g[0, 0] == np.float32(443019.75)
    assert r[0, 1] == np.float32(194000.75)
    assert g[1, 0] == np.float32(443019.25)
    assert np.array_equal(b, elevation)


# --------------------------------------------------------------------------
# Payload: nodata branch
# --------------------------------------------------------------------------


def test_nodata_pixels_collapse_to_zero_but_keep_xy(tmp_path: Path) -> None:
    """A void pixel's Z is the 0.0 sentinel; its easting/northing stay set."""
    elevation = np.array([[5.0, _NODATA], [7.0, 8.0]], dtype=np.float32)
    src = tmp_path / "dsm.tif"
    _write_dsm(src, elevation)
    out = tmp_path / "positions.exr"

    stats = export_positions(src, out)

    _, _, r, g, b = _parse_exr(out.read_bytes())
    assert stats.nodata_pixels == 1
    # The void pixel (row0, col1): Z sentinel 0.0, but X/Y still world coords.
    assert b[0, 1] == np.float32(0.0)
    assert r[0, 1] == np.float32(194000.75)
    assert g[0, 1] == np.float32(443019.75)
    # Valid neighbours keep their true elevation.
    assert b[0, 0] == np.float32(5.0)
    assert b[1, 1] == np.float32(8.0)


def test_raster_without_declared_nodata_keeps_all_pixels(
    tmp_path: Path,
) -> None:
    """When the raster declares no nodata, every pixel is valid (no masking)."""
    elevation = np.array([[_NODATA, 2.0]], dtype=np.float32)
    src = tmp_path / "dsm.tif"
    _write_dsm(src, elevation, nodata=None)
    out = tmp_path / "positions.exr"

    stats = export_positions(src, out)

    _, _, _, _, b = _parse_exr(out.read_bytes())
    assert stats.nodata_pixels == 0
    # -9999 is a real value here (no nodata declared), so it is NOT collapsed.
    assert np.array_equal(b, elevation)


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------


def test_export_is_byte_identical_on_repeat(tmp_path: Path) -> None:
    """Identical input yields byte-identical output (sha256) across writes."""
    elevation = np.array([[1.5, _NODATA], [3.5, 4.5]], dtype=np.float32)
    src = tmp_path / "dsm.tif"
    _write_dsm(src, elevation)
    first = tmp_path / "first.exr"
    second = tmp_path / "second.exr"

    stats_first = export_positions(src, first)
    stats_second = export_positions(src, second)

    assert stats_first == stats_second
    digest_first = hashlib.sha256(first.read_bytes()).hexdigest()
    digest_second = hashlib.sha256(second.read_bytes()).hexdigest()
    assert digest_first == digest_second


def test_export_reports_a_single_atomic_tick(tmp_path: Path) -> None:
    """The single-raster export reports (0, 1) then (1, 1) to progress."""
    elevation = np.array([[1.5, _NODATA], [3.5, 4.5]], dtype=np.float32)
    src = tmp_path / "dsm.tif"
    _write_dsm(src, elevation)
    out = tmp_path / "out.exr"
    calls: list[tuple[int, int]] = []

    export_positions(
        src, out, progress=lambda done, total: calls.append((done, total))
    )

    assert calls == [(0, 1), (1, 1)]


# --------------------------------------------------------------------------
# Edges: 1x1, non-square
# --------------------------------------------------------------------------


def test_single_pixel_raster(tmp_path: Path) -> None:
    """A 1x1 raster produces a one-pixel, one-scanline EXR."""
    src = tmp_path / "dsm.tif"
    _write_dsm(src, np.array([[42.0]], dtype=np.float32))
    out = tmp_path / "positions.exr"

    stats = export_positions(src, out)

    width, height, r, g, b = _parse_exr(out.read_bytes())
    assert (width, height) == (1, 1)
    assert stats == PositionsExportStats(width=1, height=1, nodata_pixels=0)
    assert r[0, 0] == np.float32(194000.25)
    assert g[0, 0] == np.float32(443019.75)
    assert b[0, 0] == np.float32(42.0)


def test_single_pixel_header_is_the_byte_exact_golden(tmp_path: Path) -> None:
    """The magic + version + header of a 1x1 EXR matches the pinned golden.

    A known-bytes prefix locks the attribute order, each field width, and the
    channel count -- an attribute reorder or a stray attribute would break it.
    """
    src = tmp_path / "dsm.tif"
    _write_dsm(src, np.array([[42.0]], dtype=np.float32))
    out = tmp_path / "positions.exr"

    export_positions(src, out)

    data = out.read_bytes()
    assert data[: len(_GOLDEN_1X1_HEADER)] == _GOLDEN_1X1_HEADER


def test_non_square_raster_round_trips(tmp_path: Path) -> None:
    """A wider-than-tall raster keeps its rows/cols straight through export."""
    elevation = np.arange(6, dtype=np.float32).reshape(2, 3)
    src = tmp_path / "dsm.tif"
    _write_dsm(src, elevation)
    out = tmp_path / "positions.exr"

    export_positions(src, out)

    width, height, _, _, b = _parse_exr(out.read_bytes())
    assert (width, height) == (3, 2)
    assert np.array_equal(b, elevation)


# --------------------------------------------------------------------------
# Streaming: bounded memory, and equality against a whole-load oracle
# --------------------------------------------------------------------------


def test_export_reads_the_dsm_one_scanline_at_a_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every DSM read is a single-row window -- never a whole-band read.

    Wraps :meth:`rasterio.DatasetReader.read` and asserts every call it
    intercepts carries a ``window`` bounded to exactly one scanline, proving
    the export genuinely streams rather than materialising the elevation
    plane whole under the hood.
    """
    elevation = np.arange(20, dtype=np.float32).reshape(4, 5)
    src = tmp_path / "dsm.tif"
    _write_dsm(src, elevation)
    out = tmp_path / "positions.exr"
    real_read = rasterio.DatasetReader.read

    def _guarded_read(
        self: rasterio.DatasetReader,
        indexes: int | None = None,
        *,
        window: Window | None = None,
    ) -> npt.NDArray[np.float32]:
        assert isinstance(window, Window), (
            "DSM read without a bounding window"
        )
        assert window.height == 1, (
            f"expected a single-scanline window, got height={window.height}"
        )
        return real_read(self, indexes, window=window)

    monkeypatch.setattr(rasterio.DatasetReader, "read", _guarded_read)

    export_positions(src, out)

    _, _, _, _, b = _parse_exr(out.read_bytes())
    assert np.array_equal(b, elevation)


def test_export_matches_a_whole_load_oracle(tmp_path: Path) -> None:
    """The streamed export matches an independent whole-load computation.

    Reads the DSM fully in the test itself (a from-scratch reimplementation
    of the retired whole-array algorithm: pixel-centre easting/northing via
    the transform, nodata collapsed to the ``0.0`` sentinel) and asserts the
    streamed :func:`export_positions` produces bit-identical R/G/B planes.
    """
    rng = np.random.default_rng(0)
    height, width = 6, 9
    elevation = rng.uniform(-5, 50, size=(height, width)).astype(np.float32)
    elevation[1, 2] = _NODATA
    elevation[4, 7] = _NODATA
    src = tmp_path / "dsm.tif"
    _write_dsm(src, elevation)
    out = tmp_path / "positions.exr"

    stats = export_positions(src, out)

    with rasterio.open(src) as dataset:
        # Same untyped-Affine-to-tuple cast the export itself uses (see its
        # ``affine = cast(...)`` comment): the stub exposes no typed members.
        transform = cast("tuple[float, ...]", dataset.transform)
    a, b_coef, c, d, e, f = transform[:6]
    cols = np.arange(width, dtype=np.float64) + 0.5
    rows = np.arange(height, dtype=np.float64) + 0.5
    col_grid, row_grid = np.meshgrid(cols, rows)
    expected_r = (a * col_grid + b_coef * row_grid + c).astype(np.float32)
    expected_g = (d * col_grid + e * row_grid + f).astype(np.float32)
    void = elevation == np.float32(_NODATA)
    expected_b = elevation.copy()
    expected_b[void] = np.float32(0.0)

    _, _, r, g, b = _parse_exr(out.read_bytes())
    assert np.array_equal(r, expected_r)
    assert np.array_equal(g, expected_g)
    assert np.array_equal(b, expected_b)
    assert stats.nodata_pixels == int(np.count_nonzero(void))


# --------------------------------------------------------------------------
# Failure mode
# --------------------------------------------------------------------------


def test_missing_dsm_raises_typed_error(tmp_path: Path) -> None:
    """An unreadable/absent DSM funnels to a typed PositionsExportError."""
    with pytest.raises(PositionsExportError, match="DSM"):
        export_positions(tmp_path / "absent.tif", tmp_path / "out.exr")


def test_constant_elevation_dsm_is_refused(tmp_path: Path) -> None:
    """A multi-pixel DSM at one constant elevation carries no genuine relief."""
    src = tmp_path / "dsm.tif"
    _write_dsm(src, np.full((2, 2), 5.0, dtype=np.float32))
    out = tmp_path / "out.exr"

    with pytest.raises(PositionsExportError, match="genuine relief"):
        export_positions(src, out)
    # The rejected run leaves no partial output or leftover temp file behind.
    assert not out.exists()
    assert not (tmp_path / "out.tmp.exr").exists()


def test_all_nodata_dsm_is_refused(tmp_path: Path) -> None:
    """A DSM whose every pixel is a void carries no genuine relief."""
    src = tmp_path / "dsm.tif"
    _write_dsm(src, np.full((2, 2), _NODATA, dtype=np.float32))

    with pytest.raises(PositionsExportError, match="genuine relief"):
        export_positions(src, tmp_path / "out.exr")
