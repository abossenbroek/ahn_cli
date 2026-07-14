# bevy_ahnp_ortho

A Bevy 0.18 renderer for AHNP (`tiles.hfp`) terrain packs — the streaming
quadtree pack format produced by [`ahn_cli`'s `tiles3d`
command](../../ahn_cli/tiles3d/) (`--profile game` / `--profile heightfield` /
`--profile splat`). Given an opened pack, the plugin streams the LOD cut
selected each frame by screen-space error, draping tiles with their ortho
texture.

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

```bash
cargo run --example viewer -- path/to/tiles.hfp
cargo run --example viewer_splat --features splat -- path/to/splat_tiles.hfp
```

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
