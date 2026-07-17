"""Tests for the pipeline spec parser/validator (`ahn_cli.pipeline.spec`).

Covers: YAML/JSON equivalence and canonical-hash determinism, every stage
type's param round-trip, and the negative matrix required by the workstream
(source-not-first, sink-not-last, unknown stage/param, class overlap,
malformed idw, both/neither voxel_size_m+voxel_grade, bad halo, malformed
bbox) -- each asserted to raise :class:`PipelineError` and nothing else.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from ahn_cli.domain import Generation
from ahn_cli.fetch.source import SourceKind
from ahn_cli.pipeline.errors import PipelineError
from ahn_cli.pipeline.spec import (
    HALO_AUTO,
    AoiSpec,
    DedupStage,
    FetchStage,
    PipelineSpec,
    ReadStage,
    ReconcileStage,
    ThinStage,
    Tiles3dStage,
    TilingSpec,
    WriteStage,
    canonical,
    parse_json,
    parse_yaml,
    spec_hash,
    stage_type,
)
from ahn_cli.prep.decimate import ThinMethod
from ahn_cli.reconcile.method import (
    IdwInterp,
    KrigingInterp,
    LinearInterp,
    Variogram,
    VariogramModel,
)
from ahn_cli.tiles3d.profile import Profile

_FULL_YAML = """
aoi:
  geojson: westland.geojson
tiling:
  grid: quadtree
  tile_pixels: 256
  halo: auto
workdir: /workspace/scratch
output: data/westland/tiles3d
stages:
  - type: fetch
    ahn_generation: ahn5
    source: geotiles
    ortho: true
    download_jobs: 8
  - type: dedup
  - type: thin
    method: voxel
    voxel_size_m: 1.0
  - type: reconcile
    method: idw
    idw:
      power: 2
      neighbors: 12
  - type: tiles3d
    profile: splat
"""

_FULL_DICT = {
    "aoi": {"geojson": "westland.geojson"},
    "tiling": {"grid": "quadtree", "tile_pixels": 256, "halo": "auto"},
    "workdir": "/workspace/scratch",
    "output": "data/westland/tiles3d",
    "stages": [
        {
            "type": "fetch",
            "ahn_generation": "ahn5",
            "source": "geotiles",
            "ortho": True,
            "download_jobs": 8,
        },
        {"type": "dedup"},
        {"type": "thin", "method": "voxel", "voxel_size_m": 1.0},
        {
            "type": "reconcile",
            "method": "idw",
            "idw": {"power": 2, "neighbors": 12},
        },
        {"type": "tiles3d", "profile": "splat"},
    ],
}

_MINIMAL_DICT = {
    "aoi": {"bbox": "0,0,10,10"},
    "workdir": "scratch",
    "output": "out",
    "stages": [
        {"type": "fetch"},
        {"type": "tiles3d"},
    ],
}


def _minimal_spec(**overrides: object) -> dict[str, object]:
    """Return a deep copy of :data:`_MINIMAL_DICT`, optionally patched."""
    spec = json.loads(json.dumps(_MINIMAL_DICT))
    spec.update(overrides)
    return spec


def _canonical_stage(spec: PipelineSpec, index: int) -> dict[str, Any]:
    """Return `canonical(spec)`'s stage dict at `index`, typed for assertions."""
    stages = canonical(spec)["stages"]
    assert isinstance(stages, list)
    stage = cast("list[Any]", stages)[index]
    assert isinstance(stage, dict)
    return cast("dict[str, Any]", stage)


# ---------------------------------------------------------------------
# YAML/JSON equivalence + canonical determinism
# ---------------------------------------------------------------------


def test_yaml_and_json_of_the_same_spec_parse_to_equal_canonical_forms() -> (
    None
):
    """The plan's example spec parses identically from YAML and JSON."""
    from_yaml = parse_yaml(_FULL_YAML)
    from_json = parse_json(json.dumps(_FULL_DICT))
    assert canonical(from_yaml) == canonical(from_json)
    assert spec_hash(from_yaml) == spec_hash(from_json)


def test_canonical_form_is_dict_order_independent() -> None:
    """Reordering source keys (top-level and nested) doesn't change the hash."""
    reordered = {
        "stages": _FULL_DICT["stages"],
        "output": _FULL_DICT["output"],
        "workdir": _FULL_DICT["workdir"],
        "tiling": {
            "halo": "auto",
            "tile_pixels": 256,
            "grid": "quadtree",
        },
        "aoi": {"geojson": "westland.geojson"},
    }
    original = parse_json(json.dumps(_FULL_DICT))
    shuffled = parse_json(json.dumps(reordered))
    assert canonical(original) == canonical(shuffled)
    assert spec_hash(original) == spec_hash(shuffled)


def test_canonical_form_has_sorted_keys_when_dumped() -> None:
    """`canonical()`'s JSON dump sorts keys at every nesting level."""
    spec = parse_json(json.dumps(_FULL_DICT))
    text = json.dumps(canonical(spec), sort_keys=True)
    parsed = json.loads(text)
    assert list(parsed.keys()) == sorted(parsed.keys())
    first_stage = parsed["stages"][0]
    assert list(first_stage.keys()) == sorted(first_stage.keys())


def test_spec_hash_differs_for_different_specs() -> None:
    """Two meaningfully different specs hash differently (sanity check)."""
    a = parse_json(json.dumps(_minimal_spec()))
    other = _minimal_spec()
    other["workdir"] = "elsewhere"
    b = parse_json(json.dumps(other))
    assert spec_hash(a) != spec_hash(b)


def test_canonical_form_of_read_and_write_stages() -> None:
    """`canonical()` renders `read`/`write` stages (not in the plan's example)."""
    spec = _minimal_spec(
        stages=[
            {"type": "read", "path": "data/site"},
            {"type": "write", "path": "data/out.laz"},
        ]
    )
    parsed = parse_json(json.dumps(spec))
    assert _canonical_stage(parsed, 0) == {
        "type": "read",
        "path": "data/site",
    }
    assert _canonical_stage(parsed, 1) == {
        "type": "write",
        "path": "data/out.laz",
    }


def test_canonical_form_of_kriging_reconcile_method() -> None:
    """`canonical()` renders a kriging `ReconcileStage` (not in the example)."""
    spec = _minimal_spec(
        stages=[
            {"type": "fetch"},
            {
                "type": "reconcile",
                "method": "kriging",
                "kriging": {"model": "gaussian", "neighbors": 4},
            },
            {"type": "tiles3d"},
        ]
    )
    parsed = parse_json(json.dumps(spec))
    assert _canonical_stage(parsed, 1) == {
        "type": "reconcile",
        "method": {
            "kind": "kriging",
            "model": "gaussian",
            "nugget": 0.0,
            "sill": 1.0,
            "range_m": 50.0,
            "neighbors": 4,
        },
    }


def test_full_spec_round_trips_expected_stage_values() -> None:
    """The plan's example spec parses to the expected typed stage chain."""
    spec = parse_yaml(_FULL_YAML)
    assert spec.aoi == AoiSpec(geojson="westland.geojson", bbox=None)
    assert spec.tiling == TilingSpec(
        grid="quadtree", tile_pixels=256, halo=HALO_AUTO
    )
    assert [stage_type(stage) for stage in spec.stages] == [
        "fetch",
        "dedup",
        "thin",
        "reconcile",
        "tiles3d",
    ]
    fetch = spec.stages[0]
    assert isinstance(fetch, FetchStage)
    assert fetch.ahn_generation == Generation(5)
    assert fetch.ortho is True
    assert fetch.download_jobs == 8
    thin = spec.stages[2]
    assert isinstance(thin, ThinStage)
    assert thin.method is ThinMethod.VOXEL
    assert thin.voxel_size_m == 1.0
    reconcile = spec.stages[3]
    assert isinstance(reconcile, ReconcileStage)
    assert isinstance(reconcile.method, IdwInterp)
    assert reconcile.method == IdwInterp(power=2.0, k=12)
    tiles3d = spec.stages[4]
    assert isinstance(tiles3d, Tiles3dStage)
    assert tiles3d.profile is Profile.SPLAT


# ---------------------------------------------------------------------
# aoi
# ---------------------------------------------------------------------


def test_aoi_accepts_geojson_only() -> None:
    """A geojson-only aoi constructs cleanly."""
    aoi = AoiSpec(geojson="area.geojson", bbox=None)
    assert aoi.geojson == "area.geojson"


def test_aoi_accepts_bbox_only() -> None:
    """A bbox-only aoi constructs cleanly."""
    aoi = AoiSpec(geojson=None, bbox=(0.0, 0.0, 10.0, 10.0))
    assert aoi.bbox == (0.0, 0.0, 10.0, 10.0)


def test_aoi_rejects_neither_geojson_nor_bbox() -> None:
    """Neither selector set is rejected."""
    with pytest.raises(PipelineError, match="exactly one"):
        AoiSpec(geojson=None, bbox=None)


def test_aoi_rejects_both_geojson_and_bbox() -> None:
    """Both selectors set is rejected."""
    with pytest.raises(PipelineError, match="exactly one"):
        AoiSpec(geojson="area.geojson", bbox=(0.0, 0.0, 10.0, 10.0))


def test_aoi_rejects_degenerate_bbox() -> None:
    """A structurally-valid but degenerate bbox is rejected."""
    with pytest.raises(PipelineError, match="minx"):
        AoiSpec(geojson=None, bbox=(10.0, 10.0, 5.0, 5.0))


def test_parse_spec_rejects_aoi_with_neither_selector() -> None:
    """A spec whose aoi has neither geojson nor bbox is rejected at parse time."""
    spec = _minimal_spec()
    spec["aoi"] = {}
    with pytest.raises(PipelineError, match="exactly one"):
        parse_json(json.dumps(spec))


def test_parse_spec_rejects_aoi_with_both_selectors() -> None:
    """A spec whose aoi has both geojson and bbox is rejected at parse time."""
    spec = _minimal_spec()
    spec["aoi"] = {"geojson": "a.geojson", "bbox": "0,0,10,10"}
    with pytest.raises(PipelineError, match="exactly one"):
        parse_json(json.dumps(spec))


def test_parse_spec_rejects_unknown_aoi_key() -> None:
    """An unrecognised aoi key is rejected."""
    spec = _minimal_spec()
    spec["aoi"] = {"bbox": "0,0,10,10", "crs": "EPSG:4326"}
    with pytest.raises(PipelineError, match="unknown key"):
        parse_json(json.dumps(spec))


def test_parse_spec_rejects_non_mapping_aoi() -> None:
    """A non-mapping aoi value is rejected."""
    spec = _minimal_spec()
    spec["aoi"] = "westland.geojson"
    with pytest.raises(PipelineError, match="must be a mapping"):
        parse_json(json.dumps(spec))


@pytest.mark.parametrize(
    "bbox_text",
    [
        "0,0,10",  # wrong field count
        "0,0,10,10,10",  # wrong field count
        "0,0,ten,10",  # non-numeric coordinate
    ],
)
def test_parse_spec_rejects_malformed_bbox_string(bbox_text: str) -> None:
    """A syntactically malformed aoi bbox string is rejected."""
    spec = _minimal_spec()
    spec["aoi"] = {"bbox": bbox_text}
    with pytest.raises(PipelineError):
        parse_json(json.dumps(spec))


def test_parse_spec_rejects_degenerate_bbox_string() -> None:
    """A syntactically valid but degenerate aoi bbox is rejected."""
    spec = _minimal_spec()
    spec["aoi"] = {"bbox": "10,10,0,0"}
    with pytest.raises(PipelineError, match="minx"):
        parse_json(json.dumps(spec))


# ---------------------------------------------------------------------
# tiling
# ---------------------------------------------------------------------


def test_tiling_defaults_when_omitted() -> None:
    """Omitting `tiling` entirely defaults grid=None, 256px, auto halo."""
    spec = _minimal_spec()
    assert "tiling" not in spec
    parsed = parse_json(json.dumps(spec))
    assert parsed.tiling == TilingSpec(
        grid=None, tile_pixels=256, halo=HALO_AUTO
    )


def test_tiling_accepts_an_explicit_numeric_halo() -> None:
    """A numeric halo in metres is accepted."""
    spec = _minimal_spec()
    spec["tiling"] = {"halo": 12.5}
    parsed = parse_json(json.dumps(spec))
    assert parsed.tiling.halo == 12.5


def test_tiling_spec_rejects_non_positive_tile_pixels() -> None:
    """A non-positive `tile_pixels` is rejected."""
    with pytest.raises(PipelineError, match="tile_pixels"):
        TilingSpec(grid=None, tile_pixels=0, halo=HALO_AUTO)


def test_tiling_spec_rejects_bad_halo_string() -> None:
    """A non-`auto` halo string is rejected."""
    with pytest.raises(PipelineError, match="halo"):
        TilingSpec(grid=None, tile_pixels=256, halo="sometimes")


def test_tiling_spec_rejects_negative_halo() -> None:
    """A negative numeric halo is rejected."""
    with pytest.raises(PipelineError, match="non-negative"):
        TilingSpec(grid=None, tile_pixels=256, halo=-1.0)


def test_tiling_spec_rejects_non_finite_halo() -> None:
    """A non-finite numeric halo is rejected."""
    with pytest.raises(PipelineError, match="finite"):
        TilingSpec(grid=None, tile_pixels=256, halo=float("nan"))


def test_parse_spec_rejects_non_positive_tile_pixels() -> None:
    """`tile_pixels: 0` is rejected at parse time."""
    spec = _minimal_spec()
    spec["tiling"] = {"tile_pixels": 0}
    with pytest.raises(PipelineError, match="tile_pixels"):
        parse_json(json.dumps(spec))


def test_parse_spec_rejects_bad_halo_type() -> None:
    """A `halo` value that is neither a string nor a number is rejected."""
    spec = _minimal_spec()
    spec["tiling"] = {"halo": [1, 2]}
    with pytest.raises(PipelineError, match="halo"):
        parse_json(json.dumps(spec))


def test_parse_spec_rejects_bad_halo_boolean() -> None:
    """A boolean `halo` value is rejected (bool is not a metres number)."""
    spec = _minimal_spec()
    spec["tiling"] = {"halo": True}
    with pytest.raises(PipelineError, match="halo"):
        parse_json(json.dumps(spec))


def test_parse_spec_rejects_non_string_grid() -> None:
    """A non-string `grid` value is rejected."""
    spec = _minimal_spec()
    spec["tiling"] = {"grid": 7}
    with pytest.raises(PipelineError, match="grid"):
        parse_json(json.dumps(spec))


def test_parse_spec_rejects_unknown_tiling_key() -> None:
    """An unrecognised tiling key is rejected."""
    spec = _minimal_spec()
    spec["tiling"] = {"gridd": "quadtree"}
    with pytest.raises(PipelineError, match="unknown key"):
        parse_json(json.dumps(spec))


def test_parse_spec_rejects_non_mapping_tiling() -> None:
    """A non-mapping `tiling` value is rejected."""
    spec = _minimal_spec()
    spec["tiling"] = "quadtree"
    with pytest.raises(PipelineError, match="must be a mapping"):
        parse_json(json.dumps(spec))


# ---------------------------------------------------------------------
# fetch stage
# ---------------------------------------------------------------------


def test_fetch_stage_defaults() -> None:
    """A bare `{type: fetch}` resolves to auto generation, pdok, no ortho."""
    spec = _minimal_spec()
    parsed = parse_json(json.dumps(spec))
    fetch = parsed.stages[0]
    assert isinstance(fetch, FetchStage)
    assert fetch.ahn_generation is None
    assert fetch.source.value == "pdok"
    assert fetch.ortho is False
    assert fetch.download_jobs == 1


def test_fetch_stage_rejects_non_positive_download_jobs() -> None:
    """A non-positive `download_jobs` is rejected."""
    with pytest.raises(PipelineError, match="download_jobs"):
        FetchStage(
            ahn_generation=None,
            source=SourceKind.PDOK,
            ortho=False,
            download_jobs=0,
        )


def test_parse_spec_rejects_non_positive_download_jobs() -> None:
    """`download_jobs: 0` is rejected at parse time."""
    spec = _minimal_spec(
        stages=[
            {"type": "fetch", "download_jobs": 0},
            {"type": "tiles3d"},
        ]
    )
    with pytest.raises(PipelineError, match="download_jobs"):
        parse_json(json.dumps(spec))


def test_parse_spec_rejects_non_integer_download_jobs() -> None:
    """A non-integer `download_jobs` is rejected."""
    spec = _minimal_spec(
        stages=[
            {"type": "fetch", "download_jobs": "eight"},
            {"type": "tiles3d"},
        ]
    )
    with pytest.raises(PipelineError, match="download_jobs"):
        parse_json(json.dumps(spec))


def test_parse_spec_rejects_non_string_ahn_generation() -> None:
    """A non-string `ahn_generation` is rejected."""
    spec = _minimal_spec(
        stages=[
            {"type": "fetch", "ahn_generation": 5},
            {"type": "tiles3d"},
        ]
    )
    with pytest.raises(PipelineError, match="ahn_generation"):
        parse_json(json.dumps(spec))


def test_parse_spec_rejects_unknown_ahn_generation() -> None:
    """An unregistered `ahn_generation` token is rejected."""
    spec = _minimal_spec(
        stages=[
            {"type": "fetch", "ahn_generation": "ahn99"},
            {"type": "tiles3d"},
        ]
    )
    with pytest.raises(PipelineError, match="ahn99"):
        parse_json(json.dumps(spec))


def test_parse_spec_rejects_non_string_source() -> None:
    """A non-string `source` is rejected."""
    spec = _minimal_spec(
        stages=[
            {"type": "fetch", "source": 1},
            {"type": "tiles3d"},
        ]
    )
    with pytest.raises(PipelineError, match="source"):
        parse_json(json.dumps(spec))


def test_parse_spec_rejects_unknown_source() -> None:
    """An unregistered `source` token is rejected."""
    spec = _minimal_spec(
        stages=[
            {"type": "fetch", "source": "bogus"},
            {"type": "tiles3d"},
        ]
    )
    with pytest.raises(PipelineError, match="bogus"):
        parse_json(json.dumps(spec))


def test_parse_spec_rejects_non_boolean_ortho() -> None:
    """A non-boolean `ortho` value is rejected."""
    spec = _minimal_spec(
        stages=[
            {"type": "fetch", "ortho": "yes"},
            {"type": "tiles3d"},
        ]
    )
    with pytest.raises(PipelineError, match="ortho"):
        parse_json(json.dumps(spec))


def test_parse_spec_rejects_unknown_fetch_key() -> None:
    """An unrecognised fetch stage key is rejected."""
    spec = _minimal_spec(
        stages=[
            {"type": "fetch", "sourcex": "pdok"},
            {"type": "tiles3d"},
        ]
    )
    with pytest.raises(PipelineError, match="unknown key"):
        parse_json(json.dumps(spec))


# ---------------------------------------------------------------------
# read / write stages
# ---------------------------------------------------------------------


def test_read_stage_round_trips() -> None:
    """A `read` source stage parses its `path`."""
    spec = _minimal_spec(
        stages=[
            {"type": "read", "path": "data/site"},
            {"type": "tiles3d"},
        ]
    )
    parsed = parse_json(json.dumps(spec))
    read = parsed.stages[0]
    assert isinstance(read, ReadStage)
    assert read.path == "data/site"


def test_write_stage_round_trips() -> None:
    """A `write` sink stage parses its `path`."""
    spec = _minimal_spec(
        stages=[
            {"type": "fetch"},
            {"type": "write", "path": "data/out.laz"},
        ]
    )
    parsed = parse_json(json.dumps(spec))
    write = parsed.stages[1]
    assert isinstance(write, WriteStage)
    assert write.path == "data/out.laz"


def test_read_stage_rejects_blank_path_directly() -> None:
    """Direct construction with a blank path is rejected (defense in depth)."""
    with pytest.raises(PipelineError, match="non-blank"):
        ReadStage(path="   ")


def test_write_stage_rejects_blank_path_directly() -> None:
    """Direct construction with a blank path is rejected (defense in depth)."""
    with pytest.raises(PipelineError, match="non-blank"):
        WriteStage(path="  ")


def test_parse_spec_rejects_read_stage_missing_path() -> None:
    """A `read` stage without `path` is rejected at parse time."""
    spec = _minimal_spec(
        stages=[
            {"type": "read"},
            {"type": "tiles3d"},
        ]
    )
    with pytest.raises(PipelineError, match="path"):
        parse_json(json.dumps(spec))


def test_parse_spec_rejects_write_stage_missing_path() -> None:
    """A `write` stage without `path` is rejected at parse time."""
    spec = _minimal_spec(
        stages=[
            {"type": "fetch"},
            {"type": "write"},
        ]
    )
    with pytest.raises(PipelineError, match="path"):
        parse_json(json.dumps(spec))


def test_parse_spec_rejects_unknown_read_key() -> None:
    """An unrecognised `read` stage key is rejected."""
    spec = _minimal_spec(
        stages=[
            {"type": "read", "path": "x", "extra": 1},
            {"type": "tiles3d"},
        ]
    )
    with pytest.raises(PipelineError, match="unknown key"):
        parse_json(json.dumps(spec))


def test_parse_spec_rejects_unknown_write_key() -> None:
    """An unrecognised `write` stage key is rejected."""
    spec = _minimal_spec(
        stages=[
            {"type": "fetch"},
            {"type": "write", "path": "x", "extra": 1},
        ]
    )
    with pytest.raises(PipelineError, match="unknown key"):
        parse_json(json.dumps(spec))


# ---------------------------------------------------------------------
# dedup stage
# ---------------------------------------------------------------------


def test_dedup_stage_defaults_to_no_class_filter() -> None:
    """A bare `{type: dedup}` keeps every class."""
    spec = _minimal_spec(
        stages=[
            {"type": "fetch"},
            {"type": "dedup"},
            {"type": "tiles3d"},
        ]
    )
    parsed = parse_json(json.dumps(spec))
    dedup = parsed.stages[1]
    assert isinstance(dedup, DedupStage)
    assert dedup.include_classes == ()
    assert dedup.exclude_classes == ()


def test_dedup_stage_round_trips_class_lists() -> None:
    """Disjoint include/exclude class lists round-trip."""
    spec = _minimal_spec(
        stages=[
            {"type": "fetch"},
            {
                "type": "dedup",
                "include_classes": [2, 6],
                "exclude_classes": [9],
            },
            {"type": "tiles3d"},
        ]
    )
    parsed = parse_json(json.dumps(spec))
    dedup = parsed.stages[1]
    assert isinstance(dedup, DedupStage)
    assert dedup.include_classes == (2, 6)
    assert dedup.exclude_classes == (9,)


def test_dedup_stage_rejects_class_overlap_directly() -> None:
    """A class in both lists is rejected (direct construction)."""
    with pytest.raises(PipelineError, match="both included and excluded"):
        DedupStage(include_classes=(2, 6), exclude_classes=(6, 9))


def test_parse_spec_rejects_dedup_class_overlap() -> None:
    """`include_classes` / `exclude_classes` overlap is rejected at parse time."""
    spec = _minimal_spec(
        stages=[
            {"type": "fetch"},
            {
                "type": "dedup",
                "include_classes": [2, 6],
                "exclude_classes": [6, 9],
            },
            {"type": "tiles3d"},
        ]
    )
    with pytest.raises(PipelineError, match="both included and excluded"):
        parse_json(json.dumps(spec))


def test_parse_spec_rejects_non_list_include_classes() -> None:
    """A non-list `include_classes` is rejected."""
    spec = _minimal_spec(
        stages=[
            {"type": "fetch"},
            {"type": "dedup", "include_classes": "2,6"},
            {"type": "tiles3d"},
        ]
    )
    with pytest.raises(PipelineError, match="include_classes"):
        parse_json(json.dumps(spec))


def test_parse_spec_rejects_non_integer_class_code() -> None:
    """A non-integer entry in `include_classes` is rejected."""
    spec = _minimal_spec(
        stages=[
            {"type": "fetch"},
            {"type": "dedup", "include_classes": [2, "six"]},
            {"type": "tiles3d"},
        ]
    )
    with pytest.raises(PipelineError, match="include_classes"):
        parse_json(json.dumps(spec))


def test_parse_spec_rejects_boolean_class_code() -> None:
    """A boolean entry in `include_classes` is rejected (bool guard)."""
    spec = _minimal_spec(
        stages=[
            {"type": "fetch"},
            {"type": "dedup", "include_classes": [2, True]},
            {"type": "tiles3d"},
        ]
    )
    with pytest.raises(PipelineError, match="include_classes"):
        parse_json(json.dumps(spec))


def test_parse_spec_rejects_unknown_dedup_key() -> None:
    """An unrecognised dedup stage key is rejected."""
    spec = _minimal_spec(
        stages=[
            {"type": "fetch"},
            {"type": "dedup", "classes": [2]},
            {"type": "tiles3d"},
        ]
    )
    with pytest.raises(PipelineError, match="unknown key"):
        parse_json(json.dumps(spec))


# ---------------------------------------------------------------------
# thin stage
# ---------------------------------------------------------------------


def test_thin_stage_voxel_size_m_round_trips() -> None:
    """An explicit `voxel_size_m` round-trips unchanged."""
    spec = _minimal_spec(
        stages=[
            {"type": "fetch"},
            {"type": "thin", "method": "voxel", "voxel_size_m": 0.5},
            {"type": "tiles3d"},
        ]
    )
    parsed = parse_json(json.dumps(spec))
    thin = parsed.stages[1]
    assert isinstance(thin, ThinStage)
    assert thin.method is ThinMethod.VOXEL
    assert thin.voxel_size_m == 0.5
    assert thin.radius_m is None


def test_thin_stage_voxel_grade_maps_to_voxel_size_m() -> None:
    """`voxel_grade: 3` resolves to the same size as `voxel_size_m: 1.0`."""
    spec = _minimal_spec(
        stages=[
            {"type": "fetch"},
            {"type": "thin", "method": "voxel", "voxel_grade": 3},
            {"type": "tiles3d"},
        ]
    )
    parsed = parse_json(json.dumps(spec))
    thin = parsed.stages[1]
    assert isinstance(thin, ThinStage)
    assert thin.voxel_size_m == 1.0


def test_thin_stage_voxel_grade_zero_is_the_identity() -> None:
    """`voxel_grade: 0` resolves to a size of 0.0 (keep every point)."""
    spec = _minimal_spec(
        stages=[
            {"type": "fetch"},
            {"type": "thin", "method": "voxel", "voxel_grade": 0},
            {"type": "tiles3d"},
        ]
    )
    parsed = parse_json(json.dumps(spec))
    thin = parsed.stages[1]
    assert isinstance(thin, ThinStage)
    assert thin.voxel_size_m == 0.0


def test_thin_stage_poisson_round_trips() -> None:
    """A poisson thin stage round-trips `radius_m` and `seed`."""
    spec = _minimal_spec(
        stages=[
            {"type": "fetch"},
            {
                "type": "thin",
                "method": "poisson",
                "radius_m": 1.5,
                "seed": 7,
            },
            {"type": "tiles3d"},
        ]
    )
    parsed = parse_json(json.dumps(spec))
    thin = parsed.stages[1]
    assert isinstance(thin, ThinStage)
    assert thin.method is ThinMethod.POISSON
    assert thin.radius_m == 1.5
    assert thin.seed == 7
    assert thin.voxel_size_m is None


def test_thin_stage_poisson_defaults_seed() -> None:
    """A poisson thin stage without `seed` uses the default seed."""
    spec = _minimal_spec(
        stages=[
            {"type": "fetch"},
            {"type": "thin", "method": "poisson", "radius_m": 1.0},
            {"type": "tiles3d"},
        ]
    )
    parsed = parse_json(json.dumps(spec))
    thin = parsed.stages[1]
    assert isinstance(thin, ThinStage)
    assert thin.seed == 0


def test_parse_spec_rejects_missing_thin_method() -> None:
    """A thin stage without `method` is rejected."""
    spec = _minimal_spec(
        stages=[
            {"type": "fetch"},
            {"type": "thin", "voxel_size_m": 1.0},
            {"type": "tiles3d"},
        ]
    )
    with pytest.raises(PipelineError, match="method"):
        parse_json(json.dumps(spec))


def test_parse_spec_rejects_unknown_thin_method() -> None:
    """An unknown thin `method` value is rejected."""
    spec = _minimal_spec(
        stages=[
            {"type": "fetch"},
            {"type": "thin", "method": "kmeans", "voxel_size_m": 1.0},
            {"type": "tiles3d"},
        ]
    )
    with pytest.raises(PipelineError, match="kmeans"):
        parse_json(json.dumps(spec))


def test_parse_spec_rejects_both_voxel_size_m_and_voxel_grade() -> None:
    """Giving both `voxel_size_m` and `voxel_grade` is rejected."""
    spec = _minimal_spec(
        stages=[
            {"type": "fetch"},
            {
                "type": "thin",
                "method": "voxel",
                "voxel_size_m": 1.0,
                "voxel_grade": 3,
            },
            {"type": "tiles3d"},
        ]
    )
    with pytest.raises(PipelineError, match="exactly one"):
        parse_json(json.dumps(spec))


def test_parse_spec_rejects_neither_voxel_size_m_nor_voxel_grade() -> None:
    """Giving neither `voxel_size_m` nor `voxel_grade` is rejected."""
    spec = _minimal_spec(
        stages=[
            {"type": "fetch"},
            {"type": "thin", "method": "voxel"},
            {"type": "tiles3d"},
        ]
    )
    with pytest.raises(PipelineError, match="exactly one"):
        parse_json(json.dumps(spec))


def test_parse_spec_rejects_radius_m_with_voxel_method() -> None:
    """`radius_m` alongside `method: voxel` is rejected."""
    spec = _minimal_spec(
        stages=[
            {"type": "fetch"},
            {
                "type": "thin",
                "method": "voxel",
                "voxel_size_m": 1.0,
                "radius_m": 1.0,
            },
            {"type": "tiles3d"},
        ]
    )
    with pytest.raises(PipelineError, match="radius_m"):
        parse_json(json.dumps(spec))


def test_parse_spec_rejects_non_integer_voxel_grade() -> None:
    """A non-integer `voxel_grade` is rejected."""
    spec = _minimal_spec(
        stages=[
            {"type": "fetch"},
            {"type": "thin", "method": "voxel", "voxel_grade": "three"},
            {"type": "tiles3d"},
        ]
    )
    with pytest.raises(PipelineError, match="voxel_grade"):
        parse_json(json.dumps(spec))


def test_parse_spec_rejects_out_of_range_voxel_grade() -> None:
    """A `voxel_grade` outside [0, 9] is rejected."""
    spec = _minimal_spec(
        stages=[
            {"type": "fetch"},
            {"type": "thin", "method": "voxel", "voxel_grade": 42},
            {"type": "tiles3d"},
        ]
    )
    with pytest.raises(PipelineError, match="grade"):
        parse_json(json.dumps(spec))


def test_parse_spec_rejects_non_numeric_voxel_size_m() -> None:
    """A non-numeric `voxel_size_m` is rejected."""
    spec = _minimal_spec(
        stages=[
            {"type": "fetch"},
            {"type": "thin", "method": "voxel", "voxel_size_m": "big"},
            {"type": "tiles3d"},
        ]
    )
    with pytest.raises(PipelineError, match="voxel_size_m"):
        parse_json(json.dumps(spec))


def test_parse_spec_rejects_voxel_size_m_with_poisson_method() -> None:
    """`voxel_size_m` alongside `method: poisson` is rejected."""
    spec = _minimal_spec(
        stages=[
            {"type": "fetch"},
            {
                "type": "thin",
                "method": "poisson",
                "voxel_size_m": 1.0,
                "radius_m": 1.0,
            },
            {"type": "tiles3d"},
        ]
    )
    with pytest.raises(PipelineError, match="voxel_size_m"):
        parse_json(json.dumps(spec))


def test_parse_spec_rejects_missing_radius_m_for_poisson() -> None:
    """A poisson thin stage without `radius_m` is rejected."""
    spec = _minimal_spec(
        stages=[
            {"type": "fetch"},
            {"type": "thin", "method": "poisson"},
            {"type": "tiles3d"},
        ]
    )
    with pytest.raises(PipelineError, match="radius_m"):
        parse_json(json.dumps(spec))


def test_parse_spec_rejects_non_numeric_radius_m() -> None:
    """A non-numeric `radius_m` is rejected."""
    spec = _minimal_spec(
        stages=[
            {"type": "fetch"},
            {"type": "thin", "method": "poisson", "radius_m": "far"},
            {"type": "tiles3d"},
        ]
    )
    with pytest.raises(PipelineError, match="radius_m"):
        parse_json(json.dumps(spec))


def test_parse_spec_rejects_non_positive_radius_m() -> None:
    """A non-positive `radius_m` is rejected via ThinStage's own validation."""
    spec = _minimal_spec(
        stages=[
            {"type": "fetch"},
            {"type": "thin", "method": "poisson", "radius_m": -1.0},
            {"type": "tiles3d"},
        ]
    )
    with pytest.raises(PipelineError, match="positive radius_m"):
        parse_json(json.dumps(spec))


def test_parse_spec_rejects_unknown_thin_key() -> None:
    """An unrecognised thin stage key is rejected."""
    spec = _minimal_spec(
        stages=[
            {"type": "fetch"},
            {"type": "thin", "method": "voxel", "voxel_sizes": 1.0},
            {"type": "tiles3d"},
        ]
    )
    with pytest.raises(PipelineError, match="unknown key"):
        parse_json(json.dumps(spec))


def test_thin_stage_rejects_radius_with_voxel_method_directly() -> None:
    """Direct construction: voxel method forbids `radius_m` set."""
    with pytest.raises(PipelineError, match="radius_m"):
        ThinStage(
            method=ThinMethod.VOXEL, voxel_size_m=1.0, radius_m=1.0, seed=0
        )


def test_thin_stage_rejects_missing_voxel_size_m_directly() -> None:
    """Direct construction: voxel method requires a set `voxel_size_m`."""
    with pytest.raises(PipelineError, match="voxel_size_m"):
        ThinStage(
            method=ThinMethod.VOXEL, voxel_size_m=None, radius_m=None, seed=0
        )


def test_thin_stage_rejects_negative_voxel_size_m_directly() -> None:
    """Direct construction: voxel method rejects a negative `voxel_size_m`."""
    with pytest.raises(PipelineError, match="voxel_size_m"):
        ThinStage(
            method=ThinMethod.VOXEL, voxel_size_m=-1.0, radius_m=None, seed=0
        )


def test_thin_stage_rejects_voxel_size_m_with_poisson_method_directly() -> (
    None
):
    """Direct construction: poisson method forbids `voxel_size_m` set."""
    with pytest.raises(PipelineError, match="voxel_size_m"):
        ThinStage(
            method=ThinMethod.POISSON,
            voxel_size_m=1.0,
            radius_m=1.0,
            seed=0,
        )


def test_thin_stage_rejects_missing_radius_m_directly() -> None:
    """Direct construction: poisson method requires a positive `radius_m`."""
    with pytest.raises(PipelineError, match="radius_m"):
        ThinStage(
            method=ThinMethod.POISSON,
            voxel_size_m=None,
            radius_m=None,
            seed=0,
        )


# ---------------------------------------------------------------------
# reconcile stage
# ---------------------------------------------------------------------


def test_reconcile_stage_linear() -> None:
    """`method: linear` needs no parameters."""
    spec = _minimal_spec(
        stages=[
            {"type": "fetch"},
            {"type": "reconcile", "method": "linear"},
            {"type": "tiles3d"},
        ]
    )
    parsed = parse_json(json.dumps(spec))
    reconcile = parsed.stages[1]
    assert isinstance(reconcile, ReconcileStage)
    assert reconcile.method == LinearInterp()
    assert _canonical_stage(parsed, 1) == {
        "type": "reconcile",
        "method": {"kind": "linear"},
    }


def test_reconcile_stage_idw_defaults() -> None:
    """`method: idw` without an `idw` block uses the IDW defaults."""
    spec = _minimal_spec(
        stages=[
            {"type": "fetch"},
            {"type": "reconcile", "method": "idw"},
            {"type": "tiles3d"},
        ]
    )
    parsed = parse_json(json.dumps(spec))
    reconcile = parsed.stages[1]
    assert isinstance(reconcile, ReconcileStage)
    assert reconcile.method == IdwInterp()


def test_reconcile_stage_kriging_round_trips() -> None:
    """A fully-specified kriging block round-trips exactly."""
    spec = _minimal_spec(
        stages=[
            {"type": "fetch"},
            {
                "type": "reconcile",
                "method": "kriging",
                "kriging": {
                    "model": "exponential",
                    "nugget": 0.1,
                    "sill": 2.0,
                    "range_m": 25.0,
                    "neighbors": 8,
                },
            },
            {"type": "tiles3d"},
        ]
    )
    parsed = parse_json(json.dumps(spec))
    reconcile = parsed.stages[1]
    assert isinstance(reconcile, ReconcileStage)
    assert reconcile.method == KrigingInterp(
        variogram=Variogram(
            model=VariogramModel.EXPONENTIAL,
            nugget=0.1,
            sill=2.0,
            vrange=25.0,
        ),
        k=8,
    )


def test_reconcile_stage_kriging_defaults() -> None:
    """`method: kriging` without a `kriging` block uses the kriging defaults."""
    spec = _minimal_spec(
        stages=[
            {"type": "fetch"},
            {"type": "reconcile", "method": "kriging"},
            {"type": "tiles3d"},
        ]
    )
    parsed = parse_json(json.dumps(spec))
    reconcile = parsed.stages[1]
    assert isinstance(reconcile, ReconcileStage)
    assert isinstance(reconcile.method, KrigingInterp)
    assert reconcile.method.k == 16


def test_parse_spec_rejects_missing_reconcile_method() -> None:
    """A reconcile stage without `method` is rejected."""
    spec = _minimal_spec(
        stages=[
            {"type": "fetch"},
            {"type": "reconcile"},
            {"type": "tiles3d"},
        ]
    )
    with pytest.raises(PipelineError, match="method"):
        parse_json(json.dumps(spec))


def test_parse_spec_rejects_unknown_reconcile_method() -> None:
    """An unknown reconcile `method` value is rejected."""
    spec = _minimal_spec(
        stages=[
            {"type": "fetch"},
            {"type": "reconcile", "method": "nearest"},
            {"type": "tiles3d"},
        ]
    )
    with pytest.raises(PipelineError, match="nearest"):
        parse_json(json.dumps(spec))


def test_parse_spec_rejects_unknown_idw_key() -> None:
    """An unrecognised `idw` sub-key is rejected (malformed idw)."""
    spec = _minimal_spec(
        stages=[
            {"type": "fetch"},
            {
                "type": "reconcile",
                "method": "idw",
                "idw": {"power": 2, "k": 12},
            },
            {"type": "tiles3d"},
        ]
    )
    with pytest.raises(PipelineError, match="unknown key"):
        parse_json(json.dumps(spec))


def test_parse_spec_rejects_idw_with_invalid_power() -> None:
    """An out-of-range `idw.power` is rejected (malformed idw)."""
    spec = _minimal_spec(
        stages=[
            {"type": "fetch"},
            {
                "type": "reconcile",
                "method": "idw",
                "idw": {"power": -1.0},
            },
            {"type": "tiles3d"},
        ]
    )
    with pytest.raises(PipelineError, match="power"):
        parse_json(json.dumps(spec))


def test_parse_spec_rejects_idw_with_non_numeric_power() -> None:
    """A non-numeric `idw.power` is rejected (malformed idw)."""
    spec = _minimal_spec(
        stages=[
            {"type": "fetch"},
            {
                "type": "reconcile",
                "method": "idw",
                "idw": {"power": "two"},
            },
            {"type": "tiles3d"},
        ]
    )
    with pytest.raises(PipelineError, match="power"):
        parse_json(json.dumps(spec))


def test_parse_spec_rejects_non_mapping_idw() -> None:
    """A non-mapping `idw` value is rejected."""
    spec = _minimal_spec(
        stages=[
            {"type": "fetch"},
            {"type": "reconcile", "method": "idw", "idw": "2,12"},
            {"type": "tiles3d"},
        ]
    )
    with pytest.raises(PipelineError, match="must be a mapping"):
        parse_json(json.dumps(spec))


def test_parse_spec_rejects_unknown_kriging_model() -> None:
    """An unknown kriging `model` value is rejected."""
    spec = _minimal_spec(
        stages=[
            {"type": "fetch"},
            {
                "type": "reconcile",
                "method": "kriging",
                "kriging": {"model": "cubic"},
            },
            {"type": "tiles3d"},
        ]
    )
    with pytest.raises(PipelineError, match="cubic"):
        parse_json(json.dumps(spec))


def test_parse_spec_rejects_non_string_kriging_model() -> None:
    """A non-string kriging `model` value is rejected."""
    spec = _minimal_spec(
        stages=[
            {"type": "fetch"},
            {
                "type": "reconcile",
                "method": "kriging",
                "kriging": {"model": 1},
            },
            {"type": "tiles3d"},
        ]
    )
    with pytest.raises(PipelineError, match="model"):
        parse_json(json.dumps(spec))


def test_parse_spec_rejects_invalid_kriging_variogram() -> None:
    """A sill below the nugget is rejected via `Variogram`'s own validation."""
    spec = _minimal_spec(
        stages=[
            {"type": "fetch"},
            {
                "type": "reconcile",
                "method": "kriging",
                "kriging": {"nugget": 5.0, "sill": 1.0},
            },
            {"type": "tiles3d"},
        ]
    )
    with pytest.raises(PipelineError, match="sill"):
        parse_json(json.dumps(spec))


def test_parse_spec_rejects_unknown_kriging_key() -> None:
    """An unrecognised `kriging` sub-key is rejected."""
    spec = _minimal_spec(
        stages=[
            {"type": "fetch"},
            {
                "type": "reconcile",
                "method": "kriging",
                "kriging": {"vrange": 5.0},
            },
            {"type": "tiles3d"},
        ]
    )
    with pytest.raises(PipelineError, match="unknown key"):
        parse_json(json.dumps(spec))


def test_parse_spec_rejects_unknown_reconcile_key() -> None:
    """An unrecognised reconcile stage key is rejected."""
    spec = _minimal_spec(
        stages=[
            {"type": "fetch"},
            {"type": "reconcile", "method": "linear", "classes": [2]},
            {"type": "tiles3d"},
        ]
    )
    with pytest.raises(PipelineError, match="unknown key"):
        parse_json(json.dumps(spec))


# ---------------------------------------------------------------------
# tiles3d stage
# ---------------------------------------------------------------------


def test_tiles3d_stage_defaults_to_strict() -> None:
    """A bare `{type: tiles3d}` defaults to the strict profile."""
    parsed = parse_json(json.dumps(_minimal_spec()))
    tiles3d = parsed.stages[-1]
    assert isinstance(tiles3d, Tiles3dStage)
    assert tiles3d.profile is Profile.STRICT


def test_tiles3d_stage_round_trips_profile() -> None:
    """An explicit `profile` round-trips."""
    spec = _minimal_spec(
        stages=[{"type": "fetch"}, {"type": "tiles3d", "profile": "game"}]
    )
    parsed = parse_json(json.dumps(spec))
    tiles3d = parsed.stages[-1]
    assert isinstance(tiles3d, Tiles3dStage)
    assert tiles3d.profile is Profile.GAME


def test_parse_spec_rejects_unknown_tiles3d_profile() -> None:
    """An unknown tiles3d `profile` value is rejected."""
    spec = _minimal_spec(
        stages=[{"type": "fetch"}, {"type": "tiles3d", "profile": "ultra"}]
    )
    with pytest.raises(PipelineError, match="ultra"):
        parse_json(json.dumps(spec))


def test_parse_spec_rejects_non_string_tiles3d_profile() -> None:
    """A non-string tiles3d `profile` value is rejected."""
    spec = _minimal_spec(
        stages=[{"type": "fetch"}, {"type": "tiles3d", "profile": 1}]
    )
    with pytest.raises(PipelineError, match="profile"):
        parse_json(json.dumps(spec))


def test_parse_spec_rejects_unknown_tiles3d_key() -> None:
    """An unrecognised tiles3d stage key is rejected."""
    spec = _minimal_spec(
        stages=[{"type": "fetch"}, {"type": "tiles3d", "profil": "game"}]
    )
    with pytest.raises(PipelineError, match="unknown key"):
        parse_json(json.dumps(spec))


# ---------------------------------------------------------------------
# stage dispatch negatives
# ---------------------------------------------------------------------


def test_parse_spec_rejects_unknown_stage_type() -> None:
    """An unrecognised stage `type` is rejected."""
    spec = _minimal_spec(stages=[{"type": "sparkle"}])
    with pytest.raises(PipelineError, match="sparkle"):
        parse_json(json.dumps(spec))


def test_parse_spec_rejects_stage_missing_type() -> None:
    """A stage mapping without a `type` key is rejected."""
    spec = _minimal_spec(stages=[{"ahn_generation": "ahn5"}])
    with pytest.raises(PipelineError, match="type"):
        parse_json(json.dumps(spec))


def test_parse_spec_rejects_non_mapping_stage() -> None:
    """A non-mapping stage entry is rejected."""
    spec = _minimal_spec(stages=["fetch"])
    with pytest.raises(PipelineError, match="must be a mapping"):
        parse_json(json.dumps(spec))


def test_parse_spec_rejects_non_string_stage_type() -> None:
    """A non-string stage `type` is rejected."""
    spec = _minimal_spec(stages=[{"type": 1}])
    with pytest.raises(PipelineError, match="type"):
        parse_json(json.dumps(spec))


# ---------------------------------------------------------------------
# top-level / stage-chain validation
# ---------------------------------------------------------------------


def test_parse_spec_rejects_source_not_first() -> None:
    """A chain whose first stage is not a source is rejected."""
    spec = _minimal_spec(
        stages=[
            {"type": "dedup"},
            {"type": "fetch"},
            {"type": "tiles3d"},
        ]
    )
    with pytest.raises(PipelineError, match="first stage must be a source"):
        parse_json(json.dumps(spec))


def test_parse_spec_rejects_sink_not_last() -> None:
    """A chain whose last stage is not a sink is rejected."""
    spec = _minimal_spec(
        stages=[
            {"type": "fetch"},
            {"type": "tiles3d"},
            {"type": "dedup"},
        ]
    )
    with pytest.raises(PipelineError, match="last stage must be a sink"):
        parse_json(json.dumps(spec))


def test_parse_spec_rejects_a_middle_only_chain() -> None:
    """A chain with no source and no sink is rejected."""
    spec = _minimal_spec(stages=[{"type": "dedup"}])
    with pytest.raises(PipelineError, match="first stage must be a source"):
        parse_json(json.dumps(spec))


def test_pipeline_spec_rejects_empty_stages_directly() -> None:
    """Direct construction with an empty stage tuple is rejected."""
    with pytest.raises(PipelineError, match="at least one stage"):
        PipelineSpec(
            aoi=AoiSpec(geojson=None, bbox=(0.0, 0.0, 1.0, 1.0)),
            tiling=TilingSpec(grid=None, tile_pixels=256, halo=HALO_AUTO),
            workdir=Path("scratch"),
            output=Path("out"),
            stages=(),
        )


def test_parse_spec_rejects_empty_stages_list() -> None:
    """An empty `stages` list is rejected at parse time."""
    spec = _minimal_spec(stages=[])
    with pytest.raises(PipelineError, match="non-empty"):
        parse_json(json.dumps(spec))


def test_parse_spec_rejects_non_list_stages() -> None:
    """A non-list `stages` value is rejected."""
    spec = _minimal_spec(stages={"type": "fetch"})
    with pytest.raises(PipelineError, match="non-empty"):
        parse_json(json.dumps(spec))


def test_parse_spec_rejects_unknown_top_level_key() -> None:
    """An unrecognised top-level key is rejected."""
    spec = _minimal_spec()
    spec["extra"] = 1
    with pytest.raises(PipelineError, match="unknown key"):
        parse_json(json.dumps(spec))


@pytest.mark.parametrize("missing", ["aoi", "workdir", "output", "stages"])
def test_parse_spec_rejects_missing_required_key(missing: str) -> None:
    """Any missing required top-level key is rejected."""
    spec = _minimal_spec()
    del spec[missing]
    with pytest.raises(PipelineError, match="missing required"):
        parse_json(json.dumps(spec))


def test_parse_spec_rejects_non_mapping_top_level() -> None:
    """A top-level document that is not a mapping is rejected."""
    with pytest.raises(PipelineError, match="must be a mapping"):
        parse_json(json.dumps(["not", "a", "mapping"]))


def test_parse_spec_rejects_non_string_workdir() -> None:
    """A non-string `workdir` is rejected."""
    spec = _minimal_spec()
    spec["workdir"] = 5
    with pytest.raises(PipelineError, match="workdir"):
        parse_json(json.dumps(spec))


def test_parse_spec_rejects_blank_output() -> None:
    """A blank `output` string is rejected."""
    spec = _minimal_spec()
    spec["output"] = "   "
    with pytest.raises(PipelineError, match="output"):
        parse_json(json.dumps(spec))


# ---------------------------------------------------------------------
# parse_yaml / parse_json malformed text
# ---------------------------------------------------------------------


def test_parse_yaml_rejects_malformed_yaml() -> None:
    """Unparsable YAML text is rejected."""
    with pytest.raises(PipelineError, match="not valid YAML"):
        parse_yaml("aoi: [unterminated")


def test_parse_json_rejects_malformed_json() -> None:
    """Unparsable JSON text is rejected."""
    with pytest.raises(PipelineError, match="not valid JSON"):
        parse_json("{not json")


def test_parse_yaml_accepts_a_minimal_spec() -> None:
    """A minimal valid YAML document parses cleanly."""
    text = """
aoi:
  bbox: "0,0,10,10"
workdir: scratch
output: out
stages:
  - type: fetch
  - type: tiles3d
"""
    parsed = parse_yaml(text)
    assert [stage_type(stage) for stage in parsed.stages] == [
        "fetch",
        "tiles3d",
    ]


# ---------------------------------------------------------------------
# stage_type helper
# ---------------------------------------------------------------------


def test_stage_type_names_every_stage_kind() -> None:
    """`stage_type` returns the expected token for every stage class."""
    fetch = FetchStage(
        ahn_generation=None,
        source=SourceKind.PDOK,
        ortho=False,
        download_jobs=1,
    )
    assert stage_type(fetch) == "fetch"
    assert stage_type(ReadStage(path="x")) == "read"
    assert stage_type(DedupStage((), ())) == "dedup"
    assert (
        stage_type(
            ThinStage(
                method=ThinMethod.VOXEL,
                voxel_size_m=1.0,
                radius_m=None,
                seed=0,
            )
        )
        == "thin"
    )
    assert stage_type(ReconcileStage(method=LinearInterp())) == "reconcile"
    assert stage_type(Tiles3dStage(profile=Profile.STRICT)) == "tiles3d"
    assert stage_type(WriteStage(path="x")) == "write"
