# Reconcile verb — design

*2026-07-10*

## Problem

`fetch` produces two rasters on **different native grids**: the Beeldmateriaal
orthophoto at **8 cm** (EPSG:28992, uint8 RGB) and the AHN DSM / point cloud at
**~50 cm** (EPSG:7415 = RD New + NAP height). Nothing today reconciles them onto
a common grid, so a downstream consumer cannot read one array carrying both a
surface height and its colour per cell.

`reconcile` closes that gap: it interpolates the AHN point cloud's elevation onto
the **ortho's 8 cm grid** and emits a dense, coloured point cloud — one point per
ortho pixel, `X,Y` = pixel centre, `Z` = interpolated elevation, `R,G,B` = that
ortho pixel's own colour (a *direct pick*, not a colour resample, because the
target grid *is* the ortho grid).

## CRS

The ortho is EPSG:28992; the DSM/LAZ are EPSG:7415. 7415 is 28992 horizontally
plus a NAP vertical datum, so **the horizontal grids coincide exactly** — XY
reconciles with no reprojection. The 7415 vertical component is simply the `Z`
we interpolate. (Verified against the Amsterdam package: both rasters share the
bounds `121000,487000 → 122000,488000`.)

## Output model — dense grid + validity mask

`reconcile` streams the ortho grid **one row-block at a time** as a
`(rows, W, 6)` float array (X,Y,Z,R,G,B) plus a `(rows, W)` validity mask
(`False` where no elevation could be interpolated). Writers consume each block:

- **laz / ply / pt** — point-list formats: emit **only valid cells** per block,
  in row-major order, so the concatenation is the whole cloud.
- **exr** — a dense image: header + scanline offset table written up front, then
  each row's scanline appended, with a `Z = 0.0` sentinel for void cells
  (mirroring `positions.exr`).

**Flat memory at any scale.** The interpolator (Delaunay / `cKDTree`) is built
**once** over the source cloud; the grid is never materialised whole. Peak memory
is bounded by the source cloud + its tree and one row-block — independent of the
output area — so a 12500² tile (or a 50 km sheet) streams like a 50 m window. The
block schedule is a deterministic function of the width (`_BLOCK_CELLS // W`
rows), so a blocked run is identical to a whole-grid one (byte-identical for the
uncompressed formats; point-identical for chunk-compressed LAZ). The only cost
that scales with area is loading the cloud + building its tree; a continental run
tiles the *source* cloud spatially (out of this verb's scope).

## Interpolation methods

All operate on the AHN points `(x_i, y_i, z_i)` and evaluate `Z` at each ortho
pixel centre.

| Method | Semantics | Neighbours |
|---|---|---|
| `linear` | Delaunay-barycentric linear interpolation (scipy `LinearNDInterpolator`); an *interpolant* that passes through the data. Cells outside the convex hull are void. | scipy Delaunay |
| `idw` | Inverse-distance weighting over the `k` nearest points: `z = Σ w_i z_i / Σ w_i`, `w_i = 1/d_i^p`. | scipy `cKDTree` |
| `kriging` | Ordinary kriging over the `k` nearest points with a **fixed** (parameterised) variogram model; per-cell `(k+1)²` solve. | scipy `cKDTree` |

`idw` and `kriging` share one `cKDTree` kNN primitive; `linear` uses scipy's
Delaunay directly. (A Metal-kernel kNN was built and benchmarked but removed —
see below.)

Typed dispatch on frozen value objects (`LinearInterp`, `IdwInterp`,
`KrigingInterp`) — no stringly-typed method switch.

### Kriging determinism

The variogram is **fixed/parameterised** (model + nugget + sill + range), never
auto-fitted (auto-fit is non-deterministic). Singular per-cell systems (e.g.
co-located neighbours) are caught host-side and fall back deterministically to
the IDW estimate — a covered branch.

## kNN and the Metal-kernel evaluation (numpy/scipy is the shipped path)

The interpolation algorithms — with **every branch** (empty input, `k > n`,
singular-system fallback, void cells) — obtain their neighbours from one
primitive:

```
knn(target_xy, source_xy, k) -> (sq_dist[Q,k], idx[Q,k])   # ascending, tie-break by source index
```

`knn` is scipy `cKDTree` — a fast, indexed, C-backed search — with a stable
`(distance, index)` tie-break. It produces every emitted value, so output is
**byte-identical** across runs and machines (determinism guardrail intact).

**A real `mlx.fast.metal_kernel` was built and benchmarked, then removed.** On
the Amsterdam fixture (65 536 queries × 9 491 pts) the raw float32 GPU kNN was
~10× faster at raw neighbour search, **but**: (a) its float32 distances select a
different neighbour *set* at the k-boundary on dense data, swinging Z by up to
4.5 m (IDW) / 8.1 m (kriging) — correctness needs a float64 host refinement that
made the *correct* mlx path (~113 ms) **slower than numpy (~86 ms)**; and (b)
brute-force `O(q·n)` cannot scale to a full 12500² tile against 23 M points,
whereas indexed `cKDTree` can. Per the guidance *"if a fast C implementation
already exists (numpy/numba/…), keep it — no point redoing it in Metal"*, the
Metal backend was dropped rather than shipped as dead-weight complexity. A
binned/grid Metal kernel could win at full-tile scale but its benefit is
unproven for this workload and it was not built speculatively.

## Writers (`.laz / .ply / .pt / .exr`)

Typed dispatch on an `OutputFormat` enum. All deterministic (no timestamps).

- **laz** — laspy point-format-2 cloud; RGB scaled uint8→uint16 (`×257`).
- **ply** — binary little-endian; `double x,y,z` + `uchar red,green,blue`.
- **pt** — raw little-endian `float32 [N,6]` (x,y,z,r,g,b), RGB as 0–255 floats;
  loadable via `torch.frombuffer`/`np.fromfile`. **No torch dependency.**
- **exr** — dense grid; generalised N-channel build of the `positions.py`
  hand-written uncompressed OpenEXR (X,Y,Z + R,G,B FLOAT channels), Z-sentinel
  voids.

## CLI

`reconcile` Click command: `--ortho ortho.tif`, `--cloud cloud.laz`,
`--out DIR`, `--method {linear,idw,kriging}`, composite method params
(`--idw "power,k"`, `--kriging "model,nugget,sill,range,k"`), and `--format`
(repeatable; default all four). Typed `ReconcileError` → `ClickException`.

## Testing

- Unit: value objects, kNN determinism/tie-break, each method vs hand-computed
  oracle, all fallback branches, each writer's bytes. 100 % line+branch.
- Integration (real Amsterdam data, CC-BY 4.0): a small clipped window from
  `~/Downloads/Dam_Amsterdam_AHN5_ortho_package` committed via git-LFS under
  `tests/reconcile/fixtures/`; end-to-end reconcile asserting all four outputs +
  cross-run determinism of the numpy path.

## New dependency

`scipy>=1.11` — Delaunay-linear (`LinearNDInterpolator`) and `cKDTree` kNN.
Justified by two concrete needs; the standard tool for spatial interpolation.
