# PDAL `writers.copc` produces a COPC file that fails `copc-validator`'s `xyz` (bounds) check on geographically flat/elongated data — even at the root node

**Status: RESOLVED (2026-07-11)** — solved by giving `ahn_cli` its own COPC export stage instead of relying on PDAL: the `copc` bounded context (`ahn_cli/copc/`) and the `ahn_cli copc` CLI command.

**Resolution summary.** The root cause hypothesized below (§"Suggested next steps", items 1–2) was confirmed by construction: the epsilon disappears when the LAS header bounds and the octree cube share a *single* float64 provenance path. The new writer (a) computes header min/max as `int32 * scale + offset` from the quantized point store — bit-identical to what any reader (copc.js included) decodes; (b) anchors the cube on whole metres ≥ 1 m outside the data on every axis, so no point can sit on an outer cube face at all (the geometry that made Dutch terrain fail); (c) assigns points to octree nodes by descending through the exact `min + (max − min) / 2` double midpoints `copc.js`'s `Bounds.stepTo` uses, so validator-recomputed node bounds contain their points by construction (its comparisons are inclusive); and (d) hands copclib raw pre-packed int32 records, eliminating any second quantization/rounding path. Verified on this exact dataset: `ahn_cli copc --cloud data/moerkapelle/reconciled/reconciled.laz --out …/reconciled_ahn.copc.laz` (46,338,950 points, 4,560 nodes, ~92 s) → `npx copc-validator -d` **24/24 checks pass, zero warnings** — including `xyz` on the root node and all 281 previously-failing Z-face keys, `gpsTime`, `rgb`/`rgbi`, and `pointCountByReturn`. Design notes: `docs/superpowers/specs/2026-07-11-copc-design.md`. Item 5 (filing the two-provenance-paths bug upstream with PDAL) remains open and worthwhile but is no longer blocking `ahn_cli`.

**Original report follows.**

**Status (original):** unresolved, needs an octree/COPC internals expert (likely a PDAL upstream issue, not an `ahn_cli` bug).
**Affects:** `pdal 2.10.2` (Homebrew, `git-version: Release`), validated with `copc-validator` (npm, latest as of 2026-07-11).
**Not an `ahn_cli` defect**: reproduced with a bare `pdal pipeline` invocation, no `ahn_cli` code in the loop for the write/validate step.

## One-line summary

For a real-world Dutch elevation dataset (huge horizontal extent, tiny vertical extent — i.e. *the shape of the Netherlands*), `writers.copc`'s declared octree cube bounds and the LAS header's own declared point-bounds disagree by a sub-quantization floating-point epsilon (~9.24e-14 m against a 0.001 m scale) at the cube's Z-minimum face. `copc-validator`'s `xyz` check treats this epsilon as "points outside the octree cube" and fails — on **281 nodes**, including **the root node itself** (key `0-0-0-0`), which by construction should always trivially contain 100% of the data.

## Why "the Netherlands" specifically

COPC's octree requires a **cubic** bounding volume (equal X/Y/Z extent), auto-fit by `writers.copc` to the data's bounding box, padding the shortest axis. Dutch elevation data is extremely flat relative to its horizontal extent:

- This dataset's real extent: X≈3762 m, Y≈3762 m, **Z≈58.8 m** (from -8.57 m to +50.27 m NAP — genuinely one of the lowest-lying areas in the Netherlands, near Moerkapelle/Zuidplaspolder).
- Forced cube side: **3762.0 m** (matching the horizontal extent) — meaning the real Z range occupies only **1.56%** of the cube's Z axis.
- Consequence: essentially **every point in the dataset sits pinned against the cube's Z-minimum face**, at every octree level (all 281 failing node keys end in Z-index `0`). This is not a coincidental edge case for this one dataset — it is the *generic* shape of any nationwide/regional Dutch (or any comparably flat-terrain) LiDAR/DEM dataset, so this bug is expected to reproduce on essentially all such data, not just this one AOI.

A COPC writer whose cube-bounds and header-bounds computations aren't bit-reconcilable at the boundary will therefore fail validation far more often — and far more visibly (many nodes at once, including the root) — on flat, horizontally-stretched terrain than on more geometrically "cube-like" real-world extents (e.g. a single city block with tall buildings), which is presumably why the documented reference recipe (see "What we verified — and its limits" below) worked on a small, roughly-cubic test fixture but not here.

## Concrete evidence

Two independent runs (default `writers.copc` offset, and an explicit `offset_x/y/z` near the data's min corner) produce **bit-identical** discrepancies, confirming this is a deterministic computation artifact of the writer, not sensitive to offset/scale choices made in the pipeline:

```
LAS header min : [98874.9356930746, 448127.4962907126, -8.569092667231963]
LAS header max : [102636.9356930746, 451205.9962907126, 50.26617813493812]
COPC info cube : [98874.9356930746, 448127.4962907126, -8.569092667231871,
                   102636.9356930746, 451889.4962907126, 3753.430907332768]

header.minX - cube.minX =  0.0                     (bit-identical)
header.minY - cube.minY =  0.0                     (bit-identical)
header.minZ - cube.minZ = -9.237055564881302e-14    (!!)
```

`-9.237e-14` is **~11 orders of magnitude smaller** than the pipeline's declared `scale_z = 0.001` — i.e. it is pure `double`-precision floating-point noise, not a real difference in any point's physical elevation. But because it points in the "outside the cube" direction (header says a point exists 9.24e-14 m *below* the cube's declared floor), `copc-validator`'s strict boundary check (`point.z >= cube.minZ`) fails for the point(s) responsible, and by transitivity for **every ancestor/sibling node whose local bounds include that boundary** — which, because virtually all points sit at that same Z-minimum face, is most of the populated tree.

Full `xyz` check failure list (281 node keys, format `level-x-y-z`) is reproducible with the pipeline below; a representative sample: `0-0-0-0` (root!), `1-0-0-0`, `2-0-0-0`, `3-0-0-0`, `4-0-0-0`, `5-0-0-0`, `5-1-0-0`, `5-2-0-0`, ... — every single flagged key ends in Z-index `0`, consistent with the Z-minimum-face hypothesis above.

## Provenance of the input data (full chain, for exact reproducibility)

The dataset was produced by `ahn_cli`'s `fetch`/`prep`/`reconcile` pipeline against a real `--geojson` area-of-interest — a GEE `ee.Geometry.LinearRing` polygon around Moerkapelle, South Holland (5 vertices, EPSG:4326):

```
[[4.571357720881086, 52.02911541329606],
 [4.5695552764230785, 52.043688121142814],
 [4.609294885187727, 52.04669711031145],
 [4.6238861022287425, 52.03439592866906],
 [4.601999276667219, 52.01865815299909]]
```

`ahn_cli fetch --geojson <that polygon> --ahn ahn5 --source geotiles` derived an EPSG:28992 AOI bbox of `(98874.6857, 448127.3895, 102636.7710, 451206.2463)` — a ~3.76 km × 3.11 km rectangle — and fetched 16 real AHN5 tiles (462.6M raw points). `prep` deduped to 433.9M points; those were voxel-thinned (via `pdal filters.voxeldownsize`, cell=1.0 m, since `ahn_cli`'s own in-process dedup/thinning could not fit this AOI's point count in available RAM — a separate, unrelated finding) to 30,762,667 points, then `ahn_cli reconcile` (IDW, k=12) interpolated that cloud onto a synthetic 0.5 m target grid covering the same bbox (real Beeldmateriaal orthophoto imagery was unavailable — external outage, unrelated to this bug), producing the 46,338,950-point `reconciled.ply`/`reconciled.laz` that feeds the COPC pipeline below. **The critical shape-defining fact is the AOI bbox's real-world footprint**: ~3.76 km × 3.11 km horizontally against a genuine ~58.8 m elevation range (Moerkapelle sits in the Zuidplaspolder, one of the lowest-lying areas in the Netherlands) — any `--geojson`/`--bbox` AOI of comparable or larger horizontal extent over similarly flat Dutch terrain should reproduce this bug regardless of point count, thinning, or the placeholder-grid substitution described above (none of those are implicated — see the isolated bare-PDAL reproduction below).

## Reproduction

Input: a LAZ/PLY point cloud with real XYZ + 16-bit RGB, covering a large, flat, horizontally-elongated area (any AHN/Dutch-elevation-shaped dataset should reproduce this; ours is 46,338,950 points over a 3.76 km × 3.76 km × 58.8 m volume — a 46,338,950-point, ~1.3 GB PLY).

```jsonc
// reconciled_to_copc.json
{
  "pipeline": [
    { "type": "readers.ply", "filename": "reconciled_rgb16.ply" },
    { "type": "filters.assign", "value": ["ReturnNumber = 1", "NumberOfReturns = 1"] },
    {
      "type": "writers.copc",
      "filename": "reconciled.copc.laz",
      "a_srs": "EPSG:28992",
      "scale_x": 0.001,
      "scale_y": 0.001,
      "scale_z": 0.001
    }
  ]
}
```

```bash
pdal pipeline reconciled_to_copc.json
npx copc-validator -d reconciled.copc.laz   # xyz check fails, 281 nodes, incl. root "0-0-0-0"
```

Second run, only difference is an explicit offset near the data's min corner instead of PDAL's default:

```jsonc
{ "type": "writers.copc", ..., "offset_x": 98874.94, "offset_y": 448127.5, "offset_z": -8.57 }
```

→ **identical** `header.minZ - cube.minZ = -9.237055564881302e-14`, confirming offset choice is not the variable that matters.

## What we verified — and its limits

`ahn_cli`'s own README documents a working recipe for `reconcile`'s LAZ/PLY output → COPC, verified on a small (1,048,576-point, ~1024×1024 pixel) fixture at 24/24 `copc-validator` checks passing. That fixture's real-world extent was presumably close to square/cubic (a small tile at fixed resolution), which — per the hypothesis above — would keep points away from the cube's Z-minimum face and avoid triggering this epsilon. This bug only manifests (or manifests this severely) once the horizontal:vertical aspect ratio becomes large, which happens naturally at "whole region of the Netherlands" scale rather than "single small tile" scale.

We did **not** find a pipeline-level workaround: neither explicit `offset_x/y/z` nor (implicitly, since it made no difference) any coordinate-shift trick changes the underlying epsilon, because the discrepancy is generated *inside* `writers.copc`'s own bounds bookkeeping (comparing a raw double-precision running bbox against a value reconstructed from the int32-quantized, scaled+offset LAS point store), not by anything expressible in pipeline JSON.

## Suggested next steps for an octree/COPC expert

1. **Locate the two bounds computations in `writers.copc`** (PDAL source, `io/CopcWriter.cpp` or equivalent): one path produces the `info.cube` bounds written to the COPC VLR; a separate path produces the LAS header `min`/`max` (typically derived from the actual scaled/offset int32 point coordinates as points are written, or from a running float64 accumulator — need to check which). Confirm whether these two paths use the same accumulation order/precision, or whether one goes through the int32 quantization round-trip (`round((z - offset) / scale) * scale + offset`) while the other stays in raw float64. A round-trip through int32 quantization for a value sitting extremely close to a bound is a classic source of a sub-scale epsilon exactly like the one observed here.
2. **Check whether the cube bounds are computed from the *pre-quantization* float64 min, while the header stats are computed from the *post-quantization* (round-tripped) point value** — if so, the header can legitimately report a Z value the tiniest bit outside the "true" cube because quantization can round either up or down at the boundary, and nothing currently guarantees the cube is grown to conservatively contain the quantized (not just raw) point set.
3. **Test whether the bug reproduces on a synthetic minimal case**: a flat plane of points (Z within a few tens of metres) spread over several kilometres in X/Y, at scale `0.001`, offset either explicit or `"auto"`. If it reproduces there without any real terrain data, that isolates the bug to the aspect-ratio/quantization interaction and rules out anything dataset-specific (e.g. a stray outlier point).
4. **Confirm against `copc-validator`'s own bounds-check implementation** too — it's possible the fix belongs there instead (e.g. the check should tolerate an epsilon on the order of the scale factor, or should grow the reference cube by one quantization unit before comparing, rather than doing an exact `>=`/`<=` comparison against `double` values pulled from two different provenance paths).
5. If a genuine PDAL bug is confirmed, this is very likely worth filing upstream at `https://github.com/PDAL/PDAL/issues` — it will affect **any** country/region with a similarly flat, horizontally-large terrain profile (much of the Netherlands, but also e.g. river deltas, coastal plains, agricultural basins elsewhere) whenever someone tries to build a COPC covering more than a small tile.

## Files referenced (local, not committed — for the investigator's own repro)

- `data/moerkapelle/reconciled/reconciled_rgb16.ply` — 16-bit-RGB-widened input (46,338,950 pts)
- `data/moerkapelle/reconciled/reconciled.copc.laz` — default-offset output (fails `xyz`, 281 nodes)
- `data/moerkapelle/reconciled/reconciled_v2.copc.laz` — explicit-offset output (fails `xyz`, identical epsilon)
- `/tmp/copc_validate.log`, `/tmp/copc_validate_v2.log` — full `copc-validator` JSON output for both runs
