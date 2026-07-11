# COPC export bounded context (`ahn_cli/copc/`) — design

**Date:** 2026-07-11
**Motivation:** `docs/bugs/2026-07-11-pdal-copc-xyz-bounds-flat-terrain.md` — PDAL's
`writers.copc` declares COPC cube bounds and LAS header bounds through two different
float64 provenance paths; on flat, horizontally-huge Dutch terrain every point sits on
the cube's Z-minimum face and a ~9e-14 epsilon fails `copc-validator`'s `xyz` check on
281 nodes including the root. Rather than patch around an external writer, `ahn_cli`
gains its own COPC export stage whose bounds are consistent *by construction*.

## Requirements (from the goal)

1. **50 cm native coarseness preserved.** No thinning coarser than the AHN native
   0.5 m. De-duplication happens only when multiple points fall in the same 0.5 m
   voxel, and the survivor is chosen by robust outlier reasoning (median/MAD), never
   by synthesis of new coordinates.
2. **Streaming.** Bounded memory regardless of input size (434M-point site clouds
   must work). Chunked LAZ reads, disk scatter into spatial buckets, per-bucket
   processing, incremental node writes.
3. **Octree fit for the Netherlands.** Cube side is forced by the horizontal extent
   (kilometres) while Z occupies a sliver near the cube floor, possibly below NAP 0.
   Node geometry must be exact at the Z-minimum face and correct for negative Z.
4. **`copc-validator -d` fully green** (zero fails, zero warns) on real site data.

## Non-goals

- No PDAL involvement; no patching of PDAL output.
- No reprojection (input is EPSG:28992 / EPSG:7415 like the rest of the pipeline).
- No PLY input; the stage consumes LAZ (from `prep` or `reconcile`).

## Bounds correctness — the actual bug fix

All declared geometry derives from **one** provenance path: the quantized int32
point store.

- Output scale is fixed (default 0.001); offset is chosen as whole metres at/below
  the data minimum (exact in float64; handles negative Z trivially).
- During the scatter pass we track min/max of the *quantized* int32 coordinates.
- LAS header min/max are computed as `int * scale + offset` in float64 — the exact
  expression every reader (incl. copc.js inside copc-validator) uses to decode the
  points, so header bounds equal decoded point extremes **bit-for-bit**.
- The COPC cube is anchored at the header min padded down by one scale unit per
  axis, with side = max padded extent; node bounds replicate copc.js's own
  double-precision midpoint-halving math, and point→node assignment uses those same
  doubles, so no point can land outside its node's validator-computed bounds.
- A writer-side self-check asserts, per node, that all node points lie within the
  validator-style node bounds before the node is written (fail-fast instead of
  producing an invalid file).

(Exact copc.js comparison semantics — inclusive vs strict, `Bounds.stepTo` formula —
confirmed from the validator source; see the implementation of `copc/octree.py`.)

## Pipeline (two passes over the data, bounded memory)

```
pass 1 (scatter)  : laspy chunked read → quantize to output int32 grid →
                    running int min/max + count → append packed records to
                    2^k × 2^k XY column-bucket temp files (k chosen from count)
pass 2 (build)    : for each bucket (sorted key order, deterministic):
                      load bucket → 0.5 m voxel dedup (outlier-aware survivor) →
                      LOD grid-sample into octree nodes (levels ≥ k local to the
                      column; levels < k spill to shared in-memory ancestor grids) →
                      write finished level-≥k nodes via copclib AddNode, free memory
finalize          : write ancestor (level < k) nodes, hierarchy page(s),
                    header min/max + COPC info cube, WKT SRS, provenance
```

- Bucket = the XY footprint of a level-k octree node (all Z — Dutch data occupies
  few Z indices per column). Bucket files are packed little-endian records of the
  quantized coords + carried attributes; SSD-friendly append writes.
- k targets ≲ a few million points per bucket; buckets stream one at a time.

## De-duplication (0.5 m voxel, outlier-aware)

Within each occupied 0.5 m³ voxel (integer voxel key = quantized coord // 500 for
scale 0.001):

- 1 point → kept as-is (native coarseness preserved; nothing is ever moved).
- N > 1 → robust stats over the voxel's points: component-wise median and MAD of Z;
  points with |z − median| > 3.5·MAD(scaled) are outliers and excluded; the survivor
  is the remaining point nearest the (x̃, ỹ, z̃) median (deterministic tie-break on
  input order). Only original points survive; walls/vegetation stay intact because
  the voxel grid is 3-D (a façade spans many vertical voxels).

## LOD (why every level has points)

Classic COPC top-down grid sampling: node at level ℓ samples its cube cell on a
G×G×G grid (G = 128); a point is accepted at the shallowest level whose grid cell is
free, otherwise pushed down; at the capacity-derived max depth all remainders are
accepted. Flat terrain simply means Z grid indices are almost always 0 — the
occupancy sets are XY-dominated, which the design assumes rather than fights.

## Package layout

```
ahn_cli/copc/
  __init__.py      # public surface: build_copc(), CopcError
  octree.py        # cube fit, copc.js-exact node bounds, VoxelKey, LOD sampler
  dedup.py         # 0.5 m voxel outlier-aware dedup (numpy, vectorized)
  scatter.py       # pass-1 streaming scatter to bucket temp files
  writer.py        # thin typed façade over copclib (stubs; untyped lib contained)
  build.py         # orchestrator: scatter → per-bucket build → finalize, progress
```

CLI: `ahn_cli copc --cloud <in.laz> --out <out.copc.laz> [--cell 0.5]
[--srs EPSG:7415]` — thin adapter in `cli/app.py`, `CopcError` →
`click.ClickException`, progress callback like `reconcile`'s.

Dependency: `copclib` (BSD, wheels for cp310–cp313 on macOS arm64 + manylinux
x86_64/aarch64) as a **required** runtime dep of the copc stage.

## Testing

- `tests/copc/` mirrors the package; 100 % branch coverage, strict pyright, no
  network. Small synthetic LAZ fixtures built with laspy in-test (flat 2-D grids
  with negative Z, duplicate stacks, outlier spikes, boundary-pinned points).
- Unit-level: octree math cross-checked against hand-computed copc.js formulas,
  incl. points exactly on the cube Z-min face and on internal midplanes.
- Integration: build a small COPC end-to-end, read back with `laspy.CopcReader` +
  copclib, assert header/cube bit-consistency and per-node containment.
- Acceptance (manual/nightly, needs node): `npx copc-validator -d` on the built
  file → zero fails/warns; final check runs on the real Moerkapelle site.
