# Changelog — `tiles3d` compression profiles & Bevy renderer

This branch (PR #26) introduces a complete pipeline for producing, packaging,
and rendering high-performance 3D terrain tiles from Dutch elevation data
(AHN): three lossy compression profiles for the `tiles3d` exporter, a new
self-describing binary pack format (`.hfp`), a Rust decoder crate, and a Bevy
renderer to visualize the results — an end-to-end workflow from data processing
to interactive viewing, complementing the existing lossless `strict` profile.

## Added

### `tiles3d` exporter profiles (`--profile`)

Three new output formats, selected via `ahn_cli tiles3d --profile`. Each is a
deterministic, byte-for-byte reproducible transform of the source terrain. The
lossless `strict` profile (uncompressed glTF + PNG) remains the default.

- **`game`** — quantized glTF using `KHR_mesh_quantization` for positions/UVs
  and `EXT_meshopt_compression` on all streams, textured with a baseline JPEG.
- **`heightfield`** — a compact 2.5D heightmap in the custom `.hf` format
  (12-bit quantization, ≤25 mm error, ZSTD), paired with a JPEG texture.
- **`splat`** — a 3D Gaussian Splatting representation (`.ply`, ZSTD-wrapped)
  with colour stored as a degree-0 spherical-harmonic coefficient. A geometric
  encoding, not a trained radiance field.

### AHNP pack container (`.hfp`)

The `game`, `heightfield`, and `splat` profiles bundle every tile's blobs into
a single `tiles.hfp` **AHNP pack** — a self-describing binary archive holding a
quadtree scene index, designed to be the renderer's sole input for geometry and
textures.

- **Integrity-checked**: a Merkle-rooted `dataset_id`, per-blob SHA-256 hashes,
  and CRC32-protected header and index.
- **Self-contained sidecars**: `tileset.json` (interoperability),
  `manifest.json` (integrity), and `provenance.json` (build reproducibility).
- Normative spec: `docs/specs/2026-07-12-hfp-pack-format.md`.

### `ahn-heightfield` Rust crate

A new `no-unsafe` Rust crate for decoding the `.hf` chunk and `.hfp` archive
formats — with an optional `encode` feature for the chunk layer — implemented
directly against the normative specifications rather than the Python source.

- **Quality-assured**: lint/clippy/format/`cargo-deny`, a 3-OS × {stable, 1.77}
  test matrix, and a Python↔Rust round-trip against committed fixtures.
- **MSRV**: Rust `1.77`.

### `bevy_ahnp_ortho` renderer crate (Bevy 0.18)

A reusable crate that streams and renders AHNP packs by screen-space error,
draping tiles with their ortho texture.

- **Multi-profile**: renders all three packed profiles (`heightfield` grid
  mesh, `game` glTF, `splat` gaussian cloud), plus optional COPC point clouds.
- **Simple API**: `AhnpOrthoPlugin`, `AhnpPack`, `Framing` (camera fit), and
  consumer-tunable `SplatSettings`.
- **Performant**: zero-cost feature flags (`splat`, `points`, `gpu_textures`)
  and asynchronous tile decoding on a background thread pool.
- Dual MIT/Apache-2.0 licensed (crate-scoped), porting LOD/geodesy/meshopt from
  `bevy_3d_tiles`.

### Interactive demo

`examples/demo.rs` is a runnable app and a top-to-bottom integration tutorial:
a live FPS counter, sliders for lighting and level of detail, live splat
scale/opacity controls, and run-time switching between heightfield, game, and
splat packs.

## Changed

- **`.hf` format upgraded to v3 (NAP-native).** The `heightfield` format and
  its bounding volumes (the `.hf` header, `tileset.json`, and pack index) now
  store all vertical measurements in the NAP datum (EPSG:5709), tagged in a new
  `vertical_datum` header field. This corrects a latent inconsistency in which a
  NAP height plane was described by an ellipsoidal bounding volume, so
  reconstructed vertices are now correctly contained within their bounds.
  *Trade-off*: heightfield tilesets are vertically offset (~43 m) from the
  WGS84 ellipsoid and do not co-register with the `game`, `splat`, or `strict`
  profiles.
- **KTX2 texture generation moved to the renderer.** The producer now emits
  only JPEG textures; GPU-native compression (BC1) happens at load time in the
  renderer via the optional `gpu_textures` feature. This was required because no
  available KTX2 encoder produced deterministic, byte-identical output across
  all CI platforms.
- **Splat appearance is a render-time choice.** The `splat` profile is a
  faithful, unopinionated geometric encoding; visual styling (scale, opacity,
  draw mode) is controlled at render time via `SplatSettings` rather than baked
  into the pack.

## Known limitations

- **Vertical "wall smearing" in `game` and `heightfield`.** AHN provides a
  single height per grid cell (a 2.5D representation) and the ortho is a nadir
  photo, so vertical surfaces such as building faces render as stretched
  triangles. This is inherent to the source data and left uncorrected by design;
  genuinely resolving it needs side-facing (stereo/oblique) imagery. The `splat`
  profile is not affected.
- **`points` feature pins `copc-rs 0.3.0`.** Later versions fail to compile (an
  upstream `las`/`laz` conflict); `0.3.0` predates the regression. The feature
  is opt-in and off by default.

## Verification & quality

- **Reproducibility**: every producer output is deterministic and byte-frozen —
  the build's verifier re-reads each artifact from disk and byte-compares it
  against an independent rebuild from the sources.
- **Code quality**: the Python producer holds 100% branch coverage and passes
  `pyright --strict` and `ruff`; the Rust crates pass `clippy -D warnings`,
  `cargo fmt`, and `cargo-deny`.
- **Peer review**: the final commit was reviewed by four independent domain
  experts (Python, Rust, 3D graphics, and geodesy).
