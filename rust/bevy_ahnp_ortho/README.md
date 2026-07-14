# bevy_ahnp_ortho

A Bevy 0.18 renderer for AHNP (`tiles.hfp`) terrain packs — the streaming
quadtree pack format produced by [`ahn_cli`'s `tiles3d`
command](../../ahn_cli/tiles3d/) (`--profile game` / `--profile heightfield` /
`--profile splat`). Given an opened pack, the plugin streams the LOD cut
selected each frame by screen-space error, draping tiles with their ortho
texture.

**Fastest way to see it:** `cargo run --example demo -- path/to/tiles.hfp` —
a runnable app *and* a top-to-bottom integration tutorial (live FPS, lighting/
LOD/splat sliders, runtime pack switching). See [`demo`](#demo--the-recommended-starting-point) below.

## Status

- **`content_kind = 0` (heightfield):** implemented. Decodes the tile's `.hf`
  height grid + JPEG via `ahn-heightfield`, builds a grid mesh, drapes it
  unlit (`Tonemapping::None`, `Rgba8UnormSrgb`, no filmic curve — ortho
  colours render 1:1).
- **`content_kind = 1` (game — quantized glTF + `EXT_meshopt_compression`):**
  implemented (`ahnp::glb`). A targeted binary-glTF reader for the profile's
  fixed shape (not a generic glTF crate), reusing the ported `meshopt`
  decoder and dequantizing via the glTF node's `translation`/`scale`.
- **`content_kind = 2` (splat, `splat` feature):** implemented. Decodes the
  zstd-wrapped binary 3DGS `.ply` (proven to load through
  `bevy_gaussian_splatting 0.7`'s `io_ply` reader — see the Track C report's
  C-0 result) and renders it as a gaussian-splat cloud.
- **`points` (COPC `.copc.laz`, `points` feature):** implemented via
  `copc-rs`, pinned to `0.3.0` — the latest published version (`0.5.0`)
  doesn't compile at all (an upstream `las`/`laz` version conflict in its own
  Cargo.toml); `0.3.0` predates that regression and its own `las`/`laz` pair
  resolves cleanly. A one-shot load (not per-frame LOD-streamed like the
  AHNP tile path — see `points::load_points`'s doc comment), rendered as a
  `PrimitiveTopology::PointList` mesh.
- **`gpu_textures` (BC1/DXT1 transcode at load, `gpu_textures` feature):**
  implemented (`render::gpu_texture`). Decoded tile JPEGs transcode to BC1 via
  `intel_tex_2` instead of uploading plain `Rgba8UnormSrgb`.

## Usage

```rust,ignore
use bevy::prelude::*;
use bevy_ahnp_ortho::{render::AhnpPack, AhnpOrthoPlugin};

App::new()
    .add_plugins(DefaultPlugins)
    .add_plugins(AhnpOrthoPlugin)
    .add_systems(Startup, |mut commands: Commands| {
        commands.spawn(AhnpPack::open("path/to/tiles.hfp").unwrap());
    })
    .run();
```

With the `splat` feature, also add `bevy_gaussian_splatting::GaussianSplattingPlugin`
yourself (this crate doesn't add it implicitly, so an app that never opens a
splat pack never pays for it) — see `examples/viewer_splat.rs`.

### `demo` — the recommended starting point

`examples/demo.rs` is both a runnable app and an integration tutorial (read
top-to-bottom, it walks through the same four steps as the snippet above,
with pointers to exactly where each one lives in the file). It adds, on top
of the minimal integration: a live FPS readout, sliders for lighting (sun
azimuth/elevation/illuminance, ambient brightness, an unlit-vs-lit toggle)
and level-of-detail (`render::SseThreshold`), splat `global_scale`/
`global_opacity` sliders that restyle already-spawned tiles live, and a
runtime file loader (a text field plus any paths passed on the command line
as one-click buttons) that can switch between heightfield/game/splat packs
without restarting. `viewer`/`viewer_splat`/`viewer_points` remain as
narrower, single-purpose references.

```bash
cargo run --example demo
cargo run --example demo -- path/to/tiles.hfp
cargo run --example demo --features splat -- path/to/splat_tiles.hfp
cargo run --example demo --features "splat gpu_textures" -- a.hfp b.hfp c.hfp
```

The demo's UI is built with [`bevy_egui`](https://github.com/vladbat00/bevy_egui)
(a `dev-dependency`, so it never affects the library's own build), pinned to
`0.39.x` specifically — bevy_egui tracks Bevy release-for-release, and 0.39.x
is the last line whose own `bevy_ecs`/`bevy_app` are `"0.18.0"` (0.40+ already
moved to Bevy 0.19).

### The other example viewers

```bash
cargo run --example viewer -- path/to/tiles.hfp
cargo run --example viewer_splat --features splat -- path/to/splat_tiles.hfp
```

**Splat render settings are a consumer choice, not baked into the pack.** The
splat producer is a faithful, opinion-free encoding — one isotropic gaussian
per cell — so *how* those gaussians are drawn is exposed via the
`splat::SplatSettings(CloudSettings)` resource: insert it before tiles spawn
to set `global_scale`, `global_opacity`, draw/rasterize mode, sort, etc. This
is where the roof-vs-wall tradeoff lives: on 2.5D building walls the nadir AHN
has no samples, so the raw encoding shows sparse round gaussians with the
background between them; a larger `global_scale` overlaps them into a filled,
smear-like face at the cost of crispness on the roofs. `examples/viewer_splat`
wires this to `AHNP_SPLAT_SCALE` / `AHNP_SPLAT_OPACITY` env vars so you can
dial it live:

```bash
AHNP_SPLAT_SCALE=3 AHNP_SPLAT_OPACITY=0.6 \
  cargo run --example viewer_splat --features splat -- path/to/splat_tiles.hfp
```

## Known limitations

- **2.5D wall smearing (heightfield & game profiles) — intentionally not
  corrected.** The AHN source carries one height per cell — a 2.5D surface
  with no vertical wall geometry — and the ortho is a nadir (straight-down)
  photo with no side-of-building pixels. The renderer builds a *continuous*
  grid mesh, so at every large height discontinuity (a roof edge dropping to
  ground) it bridges the two cells with a near-vertical triangle onto which
  the roof/ground texel is stretched, producing vertical "curtain" smears
  down building edges. This is a faithful consequence of the input data, and
  it is **left uncorrected by design**: the renderer shows the raw data and
  its true artifacts rather than beautifying over a data limitation (no skirt
  culling, no discontinuity thresholding, no seam hiding). It is only
  genuinely resolvable with input that carries side-facing appearance — e.g.
  stereo or oblique imagery yielding real wall pixels and geometry — and this
  note is deliberately left open for whoever introduces that. The `splat`
  profile avoids the smear by construction (discrete gaussians, no bridging
  triangles) but is not a substitute for actual wall data.

## Workspace / CI

This crate is its **own self-rooted `[workspace]`**, excluded from
`rust/Cargo.toml` (mirroring the `ahn-heightfield/fuzz` precedent) — Bevy's
dependency tree never touches `ahn-heightfield`'s MSRV 1.77 / cargo-deny /
`rust.yml` gates. It depends on `ahn-heightfield` by path, one-directionally.
Built/tested by its own non-blocking CI job,
`.github/workflows/bevy_ahnp_ortho.yml`.

## License

Licensed under either of

- MIT license ([LICENSE-MIT](LICENSE-MIT))
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))

at your option. This dual license is scoped to this crate only — the
`ahn_cli` repository root stays TU Delft MIT, unchanged. See [NOTICE](NOTICE)
for third-party attribution (this crate ports several modules from
[`bevy_3d_tiles`](https://github.com/Arvikasoft/bevy_3d_tiles), Copyright (c)
2024 Arvikasoft AB, dual MIT/Apache-2.0).
