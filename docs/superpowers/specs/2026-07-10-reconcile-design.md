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

`reconcile` computes a dense `(H, W, 6)` float array (X,Y,Z,R,G,B) over the ortho
grid plus a boolean `(H, W)` validity mask (`False` where no elevation could be
interpolated — e.g. a cell with no source point in range). Writers consume
`(grid, mask)`:

- **laz / ply / pt** — point-list formats: flatten and emit **only valid cells**.
- **exr** — a dense image: emit the **full grid** with a `Z = 0.0` sentinel for
  void cells, mirroring `positions.exr`'s nodata policy.

Full-tile scale note: a 12500² ortho ⇒ 156 M cells ⇒ ~3.75 GB for `(H,W,6)`
float32. Fine for the windowed tests; documented as the CLI's scale ceiling
(binned, windowed processing is the perf follow-up).

## Interpolation methods

All operate on the AHN points `(x_i, y_i, z_i)` and evaluate `Z` at each ortho
pixel centre.

| Method | Semantics | Backend |
|---|---|---|
| `linear` | Delaunay-barycentric linear interpolation (scipy `LinearNDInterpolator`); an *interpolant* that passes through the data. Cells outside the convex hull are void. | CPU (scipy) |
| `idw` | Inverse-distance weighting over the `k` nearest points: `z = Σ w_i z_i / Σ w_i`, `w_i = 1/d_i^p`. | kNN via backend |
| `kriging` | Ordinary kriging over the `k` nearest points with a **fixed** (parameterised) variogram model; per-cell `(k+1)²` solve. | kNN via backend |

`linear` is CPU-only by deliberate choice — the user singled out *kriging* for
Metal, and Delaunay linear is not a kNN-shaped problem. `idw` and `kriging` share
one kNN primitive, which is the Metal-accelerated part.

Typed dispatch on frozen value objects (`LinearInterp`, `IdwInterp`,
`KrigingInterp`) — no stringly-typed method switch.

### Kriging determinism

The variogram is **fixed/parameterised** (model + nugget + sill + range), never
auto-fitted (auto-fit is non-deterministic). Singular per-cell systems (e.g.
co-located neighbours) are caught host-side and fall back deterministically to
the IDW estimate — a covered branch.

## Backends and the Metal kernel

Mirrors `prep/decimate.py`. The interpolation algorithms — with **every branch**
(empty input, `k > n`, singular-system fallback, void cells) — live in host-side
Python written once against a narrow `InterpBackend` primitive:

```
knn(target_xy, source_xy, k) -> (sq_dist[Q,k], idx[Q,k])   # ascending, tie-break by source index
```

- **`NumpyBackend`** — the correctness oracle and the **default**. kNN via
  scipy `cKDTree` with a stable `(distance, index)` tie-break. Produces every
  *emitted* value, so the shipped default output is **byte-identical** across
  runs and machines (determinism guardrail intact).
- **`MlxBackend`** — opt-in accelerator routing through an injected `mlx.core`
  handle. The kNN runs as a real `mlx.fast.metal_kernel`: the host builds a
  deterministic uniform-grid bin index (bin offsets + sorted point indices);
  the kernel searches the query cell's bin neighbourhood ring-by-ring up to a
  fixed max ring — straight-line, data-parallel, no host-observable branch only
  the device takes.

**Determinism contract across backends:** GPU float reductions differ from CPU
in the last ULPs, so `MlxBackend` is **`np.allclose`-equivalent** to
`NumpyBackend`, *not* byte-identical. The default (numpy) path is byte-identical;
Metal is documented as tolerance-equivalent and opt-in (`--backend mlx`).
Equivalence tests use `allclose`, never `array_equal`.

**Coverage strategy (100 % on Linux CI, no `mlx`):** both the `mlx` handle *and*
`import_module` are injectable (as in `decimate.py`), so a numpy-backed fake
satisfying the narrow `MlxModule`/kernel surface exercises every host-side line
and both the "mlx present / absent" branches without `mlx` installed. The real
Metal kernel's *correctness* is proven by an **Apple-only** on-device
equivalence test (skipped on CI — skipped tests do not lower coverage because the
same host lines run under the fake). This build was verified by actually running
the kernel under `uv sync --extra mlx` on Apple Silicon.

### Performance (measured on the Amsterdam fixture, 65 536 queries × 9 491 pts)

Honest finding from benchmarking the real fixture:

- The raw float32 GPU kNN is ~10× faster than scipy `cKDTree` at *raw* neighbour
  search (7 ms vs 86 ms).
- **But** the raw float32 result selects a different neighbour *set* at the
  k-boundary on dense data, swinging interpolated Z by up to 4.5 m (IDW) / 8.1 m
  (kriging) versus the reference. Correctness requires the float64 host
  refinement, which costs ~30 ms — so the *correct* mlx path (~113 ms) is
  **slower than numpy (~86 ms) at this scale**.
- Brute-force is `O(q·n)`; a full 12500² ortho tile against 23 M points is
  astronomically large for it, whereas indexed `cKDTree` handles it. So the
  metal path is **not** a robust at-scale speedup.

Conclusion: the metal kernel is real, correct, and genuinely accelerates raw
kNN, but the numpy default is the faster *and* more scalable choice for the
sizes this tool sees. A **binned/grid metal kernel** (host-built spatial bins,
kernel searches only nearby cells) is the route to an at-scale GPU win and is a
scoped follow-up — deliberately not built speculatively, since the numpy path
already meets the need and its benefit is unproven for this workload.

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
`--out DIR`, `--method {linear,idw,kriging}`, method params (`--idw-power`,
`--idw-k`, `--kriging-*`), `--backend {numpy,mlx}` (default numpy), and
`--format` (repeatable; default all four). Typed `ReconcileError` → `ClickException`.

## Testing

- Unit: value objects, kNN determinism/tie-break, each method vs hand-computed
  oracle, all fallback branches, each writer's bytes, backend fake equivalence.
  100 % line+branch.
- Apple-only: real Metal kNN vs numpy kNN (`allclose`).
- Integration (real Amsterdam data, CC-BY 4.0): a small clipped window from
  `~/Downloads/Dam_Amsterdam_AHN5_ortho_package` committed via git-LFS under
  `tests/reconcile/fixtures/`; end-to-end reconcile asserting all four outputs +
  cross-run determinism of the numpy path.

## New dependency

`scipy>=1.11` — Delaunay-linear (`LinearNDInterpolator`) and `cKDTree` kNN.
Justified by two concrete needs; the standard tool for spatial interpolation.
