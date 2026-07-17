"""Tests for the prep-context transform/export orchestration (WP14).

The fixtures are small, synthetic format-6 LAZ tiles built in-process with laspy,
each paired with a provenance sidecar (as a real ``fetch`` writes), so the whole
``prepare`` pipeline -- dedup, class filter, graded thinning, provenance, and PLY
export -- runs offline against real point clouds.
"""

from __future__ import annotations

import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path

import laspy
import numpy as np
import pytest

from ahn_cli.domain import BBox, Generation, Product, Provenance
from ahn_cli.prep import transform as transform_module
from ahn_cli.prep.decimate import PoissonThinning, VoxelThinning
from ahn_cli.prep.spill import DiskFloorError
from ahn_cli.prep.transform import PrepError, PrepRequest, prepare
from ahn_cli.provenance import read_provenance, write_provenance

_START = datetime(2026, 7, 10, 9, 0, tzinfo=timezone.utc)
_FINISH = datetime(2026, 7, 10, 9, 0, 5, tzinfo=timezone.utc)

# Two adjacent canonical extents on the Dutch grid, meeting at x == 10.
_EXTENT_A: BBox = (0.0, 0.0, 10.0, 10.0)
_EXTENT_B: BBox = (10.0, 0.0, 20.0, 10.0)

_GENERATION_5 = Generation(5)

Point = tuple[float, float, float, float, int]  # x, y, z, gps_time, class


def _write_tile(path: Path, points: list[Point]) -> None:
    """Write a synthetic format-6 (gps_time + classification) LAZ tile."""
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.offsets = np.array([0.0, 0.0, 0.0], dtype=float)
    header.scales = np.array([0.01, 0.01, 0.01], dtype=float)
    las = laspy.LasData(header)
    arr = np.array(points, dtype=float)
    las.x = arr[:, 0]
    las.y = arr[:, 1]
    las.z = arr[:, 2]
    las.gps_time = arr[:, 3]
    las.classification = arr[:, 4].astype(np.uint8)
    las.write(str(path))


def _write_sidecar(
    ahn_dir: Path,
    tile_id: str,
    extent: BBox,
    *,
    licence: str = "https://creativecommons.org/publicdomain/zero/1.0/",
    attribution: str = "AHN (Rijkswaterstaat).",
    generation: Generation | None = _GENERATION_5,
) -> None:
    """Write the provenance sidecar a fetch would leave beside a tile."""
    provenance = Provenance(
        source_portal="pdok",
        product=Product.AHN_POINT_CLOUD,
        licence=licence,
        attribution=attribution,
        bbox=extent,
        download_started_at=_START,
        download_finished_at=_FINISH,
        input_checksum="0" * 64,
        output_checksum="0" * 64,
        tool_version="wp14-test",
        generation=generation,
        request_keys=(("tile_id", tile_id),),
    )
    write_provenance(provenance, ahn_dir / f"{tile_id}.provenance.json")


def _fetched_site(
    data_dir: Path,
    *,
    with_sidecars: bool = True,
) -> Path:
    """Materialise a two-tile fetched site under ``data_dir`` and return it."""
    ahn_dir = data_dir / "ahn"
    ahn_dir.mkdir(parents=True, exist_ok=True)
    _write_tile(
        ahn_dir / "tile_a.LAZ",
        [
            (1.0, 1.0, 0.0, 1.0, 2),
            (2.0, 2.0, 0.5, 2.0, 6),
            (5.0, 5.0, 1.0, 3.0, 1),
        ],
    )
    _write_tile(
        ahn_dir / "tile_b.LAZ",
        [
            (12.0, 5.0, 0.0, 12.0, 2),
            (15.0, 5.0, 0.5, 15.0, 6),
            (18.0, 8.0, 1.0, 18.0, 9),
        ],
    )
    if with_sidecars:
        _write_sidecar(ahn_dir, "tile_a", _EXTENT_A)
        _write_sidecar(ahn_dir, "tile_b", _EXTENT_B)
    return data_dir


def _read_points(path: Path) -> laspy.LasData:
    """Read a LAZ/LAS file fully into memory."""
    with laspy.open(str(path)) as reader:
        return reader.read()


def _classes(path: Path) -> set[int]:
    """Return the set of classification codes present in a LAZ file."""
    return {int(c) for c in _read_points(path).classification}


# --------------------------------------------------------------------------- #
# Value objects
# --------------------------------------------------------------------------- #


def test_prep_request_defaults_to_no_filters_and_no_export(
    tmp_path: Path,
) -> None:
    """A bare request selects every class and exports nothing."""
    request = PrepRequest(data_dir=tmp_path)

    assert request.include_classes == ()
    assert request.exclude_classes == ()
    assert request.export_points is False
    assert request.thinning is None


def test_prep_request_is_hashable_and_value_typed(tmp_path: Path) -> None:
    """The request is a frozen value object: hashable and equal by value."""
    first = PrepRequest(data_dir=tmp_path, include_classes=(2, 6))
    second = PrepRequest(data_dir=tmp_path, include_classes=(2, 6))

    assert first == second
    assert len({first, second}) == 1


def test_prep_error_is_a_runtime_error() -> None:
    """The typed prep error is a RuntimeError the CLI can catch broadly."""
    assert issubclass(PrepError, RuntimeError)


# --------------------------------------------------------------------------- #
# Happy path: dedup -> provenance -> PLY
# --------------------------------------------------------------------------- #


def test_prepare_dedups_writes_provenance_and_exports_ply(
    tmp_path: Path,
) -> None:
    """A full prep run writes pointcloud.laz, provenance.json, and the PLY."""
    site = _fetched_site(tmp_path / "delft")

    prepare(PrepRequest(data_dir=site, export_points=True))

    laz = site / "pointcloud.laz"
    ply = site / "pointcloud.ply"
    provenance_path = site / "provenance.json"
    assert laz.is_file()
    assert ply.is_file()
    assert provenance_path.is_file()
    # No tiles overlap, so every point survives dedup (6 total).
    assert len(_read_points(laz).points) == 6
    assert ply.read_bytes().startswith(b"ply\n")

    provenance = read_provenance(provenance_path)
    assert provenance.product is Product.AHN_POINT_CLOUD
    assert provenance.licence.startswith("https://creativecommons.org")
    assert provenance.generation == Generation(5)
    keys = dict(provenance.request_keys)
    assert keys["stage"] == "prep"
    assert keys["thinning"] == "none"
    assert keys["points_exported"] == "true"
    assert keys["output_points"] == "6"


def test_prepare_without_export_skips_the_ply(tmp_path: Path) -> None:
    """Prep without --points still writes the cloud and provenance, no PLY."""
    site = _fetched_site(tmp_path / "delft")

    prepare(PrepRequest(data_dir=site, export_points=False))

    assert (site / "pointcloud.laz").is_file()
    assert (site / "provenance.json").is_file()
    assert not (site / "pointcloud.ply").exists()
    keys = dict(read_provenance(site / "provenance.json").request_keys)
    assert keys["points_exported"] == "false"


def test_prepare_is_byte_deterministic(tmp_path: Path) -> None:
    """Identical fetched inputs yield byte-identical prep outputs."""
    first = _fetched_site(tmp_path / "first")
    second = tmp_path / "second"
    shutil.copytree(first, second)

    prepare(PrepRequest(data_dir=first, export_points=True))
    prepare(PrepRequest(data_dir=second, export_points=True))

    for name in ("pointcloud.laz", "pointcloud.ply", "provenance.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()


# --------------------------------------------------------------------------- #
# Classification filtering
# --------------------------------------------------------------------------- #


def test_prepare_include_keeps_only_selected_classes(tmp_path: Path) -> None:
    """--include-class keeps only the requested classes."""
    site = _fetched_site(tmp_path / "delft")

    prepare(PrepRequest(data_dir=site, include_classes=(2, 6)))

    assert _classes(site / "pointcloud.laz") == {2, 6}


def test_prepare_exclude_drops_selected_classes(tmp_path: Path) -> None:
    """--exclude-class drops the requested classes and keeps the rest."""
    site = _fetched_site(tmp_path / "delft")

    prepare(PrepRequest(data_dir=site, exclude_classes=(6,)))

    present = _classes(site / "pointcloud.laz")
    assert 6 not in present
    assert 2 in present


def test_prepare_class_filter_only_streams_without_loading_whole_cloud(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A class-filter-only prep (no thinning) never reads the whole cloud.

    Routes the class-filter step through
    :func:`~ahn_cli.prep.voxel_stream.stream_voxel_thin` at grade 0 (its
    streamed class-filter identity) instead of the old whole-load
    ``reader.read()`` path. Runs the full public ``prepare()`` pipeline (so
    dedup's own legitimate whole-tile reads -- one per of the two source
    tiles ``_fetched_site`` writes -- are accounted for) and asserts no
    further :meth:`laspy.LasReader.read` call happens past those two.
    """
    site = _fetched_site(tmp_path / "delft")  # two source tiles
    # getattr, not `.read` directly: laspy's vendored stub doesn't declare it,
    # so direct attribute access fails pyright strict (reportAttributeAccessIssue).
    real_read = getattr(laspy.LasReader, "read")  # noqa: B009
    calls = 0

    def _guarded_read(self: laspy.LasReader) -> laspy.LasData:
        nonlocal calls
        calls += 1
        if calls > 2:  # dedup's legitimate per-tile reads (out of scope here)
            msg = "class-filter-only prep must not read the whole cloud"
            raise AssertionError(msg)
        return real_read(self)

    monkeypatch.setattr("laspy.lasreader.LasReader.read", _guarded_read)

    prepare(PrepRequest(data_dir=site, include_classes=(2, 6)))

    monkeypatch.undo()  # re-read with the real reader to verify the result
    assert _classes(site / "pointcloud.laz") == {2, 6}


def test_prepare_class_filter_only_matches_whole_load_oracle(
    tmp_path: Path,
) -> None:
    """The streamed class-filter-only path matches naive whole-load filtering.

    Runs an unfiltered prep to get the deduplicated baseline cloud, then an
    independent numpy boolean-mask filter over it stands in for the retired
    whole-load ``reader.read()`` path -- the oracle a filtered prep of the
    same site must match exactly, in order.
    """
    baseline_site = _fetched_site(tmp_path / "baseline")
    filtered_site = tmp_path / "filtered"
    shutil.copytree(baseline_site, filtered_site)

    prepare(PrepRequest(data_dir=baseline_site))
    prepare(PrepRequest(data_dir=filtered_site, include_classes=(2, 6)))

    baseline = _read_points(baseline_site / "pointcloud.laz")
    keep = np.isin(np.asarray(baseline.classification), (2, 6))
    expected_gps = np.asarray(baseline.gps_time)[keep].tolist()

    result = _read_points(filtered_site / "pointcloud.laz")
    assert result.gps_time.tolist() == expected_gps


# --------------------------------------------------------------------------- #
# Graded thinning (additive, via the CPU reference backend)
# --------------------------------------------------------------------------- #


def test_prepare_voxel_thinning_reduces_points(tmp_path: Path) -> None:
    """A coarse voxel grade collapses co-located points and is recorded."""
    site = tmp_path / "delft"
    ahn_dir = site / "ahn"
    ahn_dir.mkdir(parents=True)
    # Four points inside one coarse voxel (a 64 m grid at grade 9).
    _write_tile(
        ahn_dir / "tile_a.LAZ",
        [
            (1.0, 1.0, 0.0, 1.0, 2),
            (2.0, 2.0, 0.0, 2.0, 2),
            (3.0, 3.0, 0.0, 3.0, 2),
            (4.0, 4.0, 0.0, 4.0, 2),
        ],
    )
    _write_sidecar(ahn_dir, "tile_a", (0.0, 0.0, 100.0, 100.0))

    prepare(PrepRequest(data_dir=site, thinning=VoxelThinning(grade=9)))

    result = _read_points(site / "pointcloud.laz")
    assert len(result.points) == 1  # one voxel keeps one point
    keys = dict(read_provenance(site / "provenance.json").request_keys)
    assert keys["thinning"] == "voxel:9"
    assert keys["output_points"] == "1"


def test_prepare_poisson_thinning_is_recorded(tmp_path: Path) -> None:
    """A Poisson request thins to a minimum spacing and is recorded typed."""
    site = _fetched_site(tmp_path / "delft")

    prepare(
        PrepRequest(
            data_dir=site,
            thinning=PoissonThinning(radius=100.0, seed=7),
        ),
    )

    result = _read_points(site / "pointcloud.laz")
    # A 100 m radius over a 20 m-wide cloud keeps exactly one point.
    assert len(result.points) == 1


def test_prepare_poisson_thinning_applies_include_and_exclude_first(
    tmp_path: Path,
) -> None:
    """A Poisson request combined with a class filter filters before thinning.

    Exercises the in-memory ``_class_mask`` with both ``include`` and
    ``exclude`` set (the Poisson path is the only one left that still calls
    it, now that the class-filter-only and voxel paths stream instead).
    """
    site = _fetched_site(tmp_path / "delft")

    prepare(
        PrepRequest(
            data_dir=site,
            include_classes=(2, 6),
            exclude_classes=(6,),
            thinning=PoissonThinning(radius=100.0, seed=7),
        ),
    )

    result = _read_points(site / "pointcloud.laz")
    # Only class-2 points survive the filter; a 100 m radius over the
    # remaining 20 m-wide cloud keeps exactly one point.
    assert len(result.points) == 1
    assert result.classification.tolist() == [2]
    keys = dict(read_provenance(site / "provenance.json").request_keys)
    assert keys["thinning"] == "poisson:100.0:7"


def test_prepare_reports_progress_per_phase(tmp_path: Path) -> None:
    """dedup/thin/export progress callbacks each fire independently."""
    site = _fetched_site(tmp_path / "delft")
    dedup_calls: list[tuple[int, int]] = []
    thin_calls: list[tuple[int, int]] = []
    export_calls: list[tuple[int, int]] = []

    prepare(
        PrepRequest(
            data_dir=site,
            thinning=PoissonThinning(radius=100.0, seed=7),
            export_points=True,
        ),
        dedup_progress=lambda done, total: dedup_calls.append((done, total)),
        thin_progress=lambda done, total: thin_calls.append((done, total)),
        export_progress=lambda done, total: export_calls.append(
            (done, total)
        ),
    )

    assert dedup_calls == [(1, 2), (2, 2)]
    assert thin_calls == [(0, 1), (1, 1)]
    assert export_calls == [(1, 1)]


def test_prepare_skips_thin_progress_when_no_thinning_requested(
    tmp_path: Path,
) -> None:
    """No thinning is requested, so thin_progress is never called."""
    site = _fetched_site(tmp_path / "delft")
    thin_calls: list[tuple[int, int]] = []

    prepare(
        PrepRequest(data_dir=site),
        thin_progress=lambda done, total: thin_calls.append((done, total)),
    )

    assert thin_calls == []


# --------------------------------------------------------------------------- #
# Typed failure modes
# --------------------------------------------------------------------------- #


def test_prepare_without_ahn_directory_raises(tmp_path: Path) -> None:
    """A site directory with no ahn/ subdirectory is a tidy PrepError."""
    (tmp_path / "delft").mkdir()

    with pytest.raises(PrepError, match="no fetched AHN"):
        prepare(PrepRequest(data_dir=tmp_path / "delft"))


def test_prepare_with_no_tiles_raises(tmp_path: Path) -> None:
    """An ahn/ directory with no LAZ tiles is a tidy PrepError."""
    (tmp_path / "delft" / "ahn").mkdir(parents=True)

    with pytest.raises(PrepError, match="no AHN tiles"):
        prepare(PrepRequest(data_dir=tmp_path / "delft"))


def test_prepare_with_missing_sidecar_raises(tmp_path: Path) -> None:
    """A tile whose provenance sidecar is absent is a tidy PrepError."""
    site = _fetched_site(tmp_path / "delft", with_sidecars=False)

    with pytest.raises(PrepError, match="provenance sidecar"):
        prepare(PrepRequest(data_dir=site))


def test_prepare_rejects_a_filter_that_empties_the_cloud(
    tmp_path: Path,
) -> None:
    """An include filter matching no class yields a refused empty cloud."""
    site = _fetched_site(tmp_path / "delft")

    with pytest.raises(PrepError, match="no points survived"):
        prepare(
            PrepRequest(
                data_dir=site, include_classes=(26,), export_points=True
            )
        )
    # The rejected deliverable is removed and nothing past the gate ran:
    # no degenerate LAZ remains for a later `copc` run to pick up.
    assert not (site / "pointcloud.laz").exists()
    assert not (site / "pointcloud.ply").exists()
    assert not (site / "provenance.json").exists()


def test_prepare_maps_disk_floor_error_to_prep_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A DiskFloorError from the streaming voxel path surfaces as a PrepError."""
    site = _fetched_site(tmp_path / "delft")

    def _raise_disk_floor(*_args: object, **_kwargs: object) -> int:
        msg = (
            "writing under scratch would leave 1 bytes free, below the floor."
        )
        raise DiskFloorError(msg)

    monkeypatch.setattr(
        transform_module, "stream_voxel_thin", _raise_disk_floor
    )

    with pytest.raises(PrepError, match="below the floor"):
        prepare(PrepRequest(data_dir=site, thinning=VoxelThinning(grade=3)))


def test_prepare_maps_voxel_stream_os_error_to_prep_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A raw OSError from the streaming voxel path surfaces as a PrepError.

    Transient I/O failures mid-stream are environment conditions the CLI
    must report tidily, exactly like a disk-floor breach.
    """
    site = _fetched_site(tmp_path / "delft")

    def _raise_os_error(*_args: object, **_kwargs: object) -> int:
        msg = "disk I/O error mid-stream"
        raise OSError(msg)

    monkeypatch.setattr(
        transform_module, "stream_voxel_thin", _raise_os_error
    )

    with pytest.raises(PrepError, match="voxel thinning failed"):
        prepare(PrepRequest(data_dir=site, thinning=VoxelThinning(grade=3)))


def test_prepare_rejects_a_cloud_stacked_at_one_position(
    tmp_path: Path,
) -> None:
    """Points all at one identical XYZ are refused as fabricated output."""
    site = tmp_path / "delft"
    ahn_dir = site / "ahn"
    ahn_dir.mkdir(parents=True)
    # Two points at one XYZ with distinct gps_time survive the exact
    # XYZ+GPS-time dedup sweep, so the finished cloud is a degenerate stack.
    _write_tile(
        ahn_dir / "tile_a.LAZ",
        [
            (1.0, 1.0, 0.5, 1.0, 2),
            (1.0, 1.0, 0.5, 2.0, 2),
        ],
    )
    _write_sidecar(ahn_dir, "tile_a", _EXTENT_A)

    with pytest.raises(PrepError, match="identical position"):
        prepare(PrepRequest(data_dir=site, export_points=True))
    # The rejected deliverable is removed and nothing past the gate ran.
    assert not (site / "pointcloud.laz").exists()
    assert not (site / "pointcloud.ply").exists()
    assert not (site / "provenance.json").exists()


# --------------------------------------------------------------------------- #
# Output checksum: streamed, not a whole-cloud read_bytes()
# --------------------------------------------------------------------------- #


def test_output_checksum_matches_whole_load_sha256_without_read_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The streamed output checksum equals a whole-load SHA-256, no read_bytes.

    Patches :meth:`pathlib.Path.read_bytes` to explode for that one file --
    proving the recorded ``output_checksum`` comes from the chunked hasher,
    not a whole-file read -- while still independently re-hashing the
    finished file (via the unguarded real method) as the oracle the recorded
    digest must equal.
    """
    site = _fetched_site(tmp_path / "delft")
    real_read_bytes = Path.read_bytes

    def _guarded_read_bytes(self: Path) -> bytes:
        if self.name == "pointcloud.laz":
            msg = "pointcloud.laz must be hashed in bounded chunks"
            raise AssertionError(msg)
        return real_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", _guarded_read_bytes)

    prepare(PrepRequest(data_dir=site))

    monkeypatch.undo()  # re-hash with the real read_bytes for the oracle
    expected = hashlib.sha256(
        (site / "pointcloud.laz").read_bytes()
    ).hexdigest()
    provenance = read_provenance(site / "provenance.json")
    assert provenance.output_checksum == expected
