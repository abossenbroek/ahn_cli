# Consuming the tiles in a Bevy game (`ahn-heightfield` + `bevy_pointcloud`)

This is a hand-off guide for a Rust/Bevy developer who has a generated tile set on
local disk and wants terrain + lidar in a [Bevy](https://bevyengine.org) game. It shows
exactly what the [`ahn-heightfield`](../rust/ahn-heightfield) crate gives you, how to turn
a decoded tile into a `bevy::render::mesh::Mesh`, how to stream tiles with the pack's
binary index, and how to load the matching point cloud via `bevy_pointcloud`.

Code that calls **`ahn-heightfield` is API-accurate** against this repo. Code that calls
**external crates (`bevy_pointcloud`, a COPC reader, `proj`) is illustrative** — check
each crate's current docs for exact signatures; their APIs are not controlled here.

## What you're integrating

`ahn_cli`'s `tiles3d --profile heightfield` produces, per site, a self-describing pack:

```
tiles3d_heightfield/
  tiles.hfp        # AHNP pack: binary scene index + every tile's .hf + .jpg blob
  tileset.json     # debug/interop sidecar — the game does NOT need it
  provenance.json  # encoder settings + dataset_id
  manifest.json    # sha256 integrity of the loose files + the pack
```

`ahn-heightfield` decodes `tiles.hfp` (and standalone `.hf` chunks). It is **decode-only,
`#![forbid(unsafe_code)]`**, and does **not** decode the JPEG textures or the point cloud —
those you hand to `image` and a COPC reader respectively. The normative byte formats live
in [`docs/specs/`](specs/); the crate codes against those specs, not the other way round.

The example artifacts referenced below are the ones produced from the `amsterdam_reconciled`
set:

- terrain pack: `tiles3d_heightfield/tiles.hfp`
- lidar: `reconciled.copc.laz` (a Cloud-Optimized Point Cloud, EPSG:7415)

## 1. Add the dependency

The crate is not on crates.io yet, so depend on it by git (Cargo finds the package by name
inside the repo's `rust/` workspace):

```toml
[dependencies]
ahn-heightfield = { git = "https://github.com/abossenbroek/ahn_cli" }
# or, for local hacking against a checkout:
# ahn-heightfield = { path = "../ahn_cli/rust/ahn-heightfield" }

bevy   = "0.14"          # match your game's Bevy version; see the note in §3
image  = "0.25"          # to decode the sibling JPEG textures
```

Default features are decode-only. The optional `encode` feature is for tooling/tests and
is not needed in a consumer.

## 2. Open the pack and read its index

`Archive::open` reads and validates the header + level directory + full index **once**
(the per-tile blobs stay on disk until you ask for them). It works over anything that
implements `ReadAt` — a `&[u8]` (mmap or `std::fs::read`), or `std::fs::File` directly for
positioned reads without loading the whole file.

```rust
use ahn_heightfield::{Archive, Entry};

// Load the whole pack into memory (fine for a city-sized pack — a few MB).
let bytes = std::fs::read("assets/tiles3d_heightfield/tiles.hfp")?;
let archive = Archive::open(&bytes[..])?;

let h = archive.header();
assert_eq!(h.content_kind, 0);          // 0 = heightfield (.hf + .jpg), 1 = game (.glb)
println!("{} tiles, {} levels", h.tile_count, h.level_count);

// Each Entry carries everything scheduling needs — no chunk is touched here:
for e in archive.entries() {
    // e.level, e.tx, e.ty
    // e.region = [west, south, east, north, minH, maxH]  (radians / metres, EPSG:4979)
    // e.geometric_error  (leaves are 0.0)
    // e.primary_size / e.texture_size
}
```

For streaming from `File` instead of a full read (bounds RAM on huge packs):

```rust
let archive = Archive::open(std::fs::File::open("assets/.../tiles.hfp")?)?;
```

`archive.dataset_id()` is a content hash — cache it and compare 32 bytes to detect a
changed data set without re-reading anything. `archive.verify_blobs()` checks every blob's
SHA-256 (an install/repair-time check; skip it on the hot path).

## 3. Decode one tile into a Bevy `Mesh`

`Archive::decode_tile(entry)` does the ranged read + zstd decode + validation and returns a
`Heightfield`: a `width × height` grid of `u16` levels plus the header. Geometry is
**implicit** — you rebuild the vertex grid from the tile's region exactly as the spec
describes ([`heightfield-chunk-format.md`](specs/heightfield-chunk-format.md),
"Grid-reconstruction contract").

Per vertex `(r, c)` (row `r` from the **top/north**, column `c` from the **west**):

- longitude `lon = west + (east - west) * c / (width - 1)`
- latitude  `lat = north - (north - south) * r / (height - 1)`
- height    `h   = hf.dequantize_at(r, c)`  (NAP metres)
- UV        `u = (c + 0.5) / width`, `v = (r + 0.5) / height`

Triangles per cell (matching the strict/game mesh winding exactly): with
`a = r*width + c`, `b = a+1`, `cc = a+width`, `d = cc+1`, emit `(a, cc, d)` and `(a, d, b)`.

A game wants **metres**, not radians. For a single site a local East-North-Up (ENU)
tangent-plane map is sub-metre accurate and dependency-free — pick a scene origin once
(e.g. the centre of the root tile's region) and place every vertex relative to it:

```rust
use ahn_heightfield::Heightfield;
use bevy::prelude::*;
use bevy::render::mesh::{Indices, PrimitiveTopology};
use bevy::render::render_asset::RenderAssetUsages;

const R_EARTH: f64 = 6_378_137.0; // WGS84 equatorial radius, good enough for ENU

/// (lon0, lat0) radians = your chosen scene origin. Reuse the SAME origin for every
/// tile and for the point cloud so the whole world lines up.
fn tile_to_mesh(hf: &Heightfield, lon0: f64, lat0: f64) -> Mesh {
    let hdr = hf.header();
    let (w, h) = (hf.width() as usize, hf.height() as usize);
    let [west, south, east, north, ..] = hdr.region;
    let cos_lat0 = lat0.cos();

    let mut pos = Vec::with_capacity(w * h);
    let mut uv = Vec::with_capacity(w * h);
    for r in 0..h {
        let lat = north - (north - south) * r as f64 / (h - 1) as f64;
        for c in 0..w {
            let lon = west + (east - west) * c as f64 / (w - 1) as f64;
            let east_m = (lon - lon0) * R_EARTH * cos_lat0; // ENU east, metres
            let north_m = (lat - lat0) * R_EARTH;           // ENU north, metres
            let up_m = hf.dequantize_at(r as u32, c as u32); // NAP height, metres
            // Bevy is y-up: (east, up, -north) puts north into -Z (right-handed).
            pos.push([east_m as f32, up_m as f32, -north_m as f32]);
            uv.push([(c as f32 + 0.5) / w as f32, (r as f32 + 0.5) / h as f32]);
        }
    }

    let mut idx = Vec::with_capacity((w - 1) * (h - 1) * 6);
    for r in 0..h - 1 {
        for c in 0..w - 1 {
            let a = (r * w + c) as u32;
            let (b, cc, d) = (a + 1, a + w as u32, a + w as u32 + 1);
            idx.extend_from_slice(&[a, cc, d, a, d, b]);
        }
    }

    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, pos.iter().map(|p| Vec3::from_array(*p)).collect::<Vec<_>>());
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uv);
    mesh.insert_indices(Indices::U32(idx));
    mesh.compute_normals(); // flat/smooth normals from the geometry
    mesh
}
```

> **Bevy version note.** `Mesh::new`, `insert_indices`, `compute_normals`, and
> `RenderAssetUsages` are Bevy 0.13/0.14-era APIs and move between releases. Adjust to your
> Bevy version (older Bevy used `Mesh::set_indices` / `Mesh::from`, etc.).

### Drape the sibling JPEG

`read_texture(entry)` returns the tile's baseline-JPEG bytes (`None` only for the `game`
profile, which embeds its texture in the glTF). Decode with `image`, upload as a Bevy
`Image`, and hang it on a `StandardMaterial`:

```rust
use image::GenericImageView;

fn tile_texture(bytes: &[u8], images: &mut Assets<Image>) -> Handle<Image> {
    let img = image::load_from_memory(bytes).expect("baseline jpeg");
    let (w, h) = img.dimensions();
    let image = Image::new(
        bevy::render::render_resource::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        bevy::render::render_resource::TextureDimension::D2,
        img.to_rgba8().into_raw(),
        bevy::render::render_resource::TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );
    images.add(image)
}
```

Then spawn the tile:

```rust
let hf = archive.decode_tile(entry)?;
let mesh = meshes.add(tile_to_mesh(&hf, lon0, lat0));
let tex  = archive.read_texture(entry)?.map(|jpg| tile_texture(&jpg, &mut images));
commands.spawn(PbrBundle {
    mesh,
    material: materials.add(StandardMaterial {
        base_color_texture: tex,
        perceptual_roughness: 1.0,
        ..default()
    }),
    ..default()
});
```

## 4. Stream by LOD (the point of the binary index)

The pack index is the scene graph — no JSON at runtime. Everything you need to decide
*which* tiles to load lives in the resident `Entry` array:

- `archive.level(n)` → the contiguous slice of entries at LOD `n` (coarse `0` … fine).
- `entry.region` → an axis-aligned bound (with `minH`/`maxH`) for frustum/distance culling
  **without decoding the chunk**.
- `entry.geometric_error` → feed a standard screen-space-error test; refine (load children)
  when the tile's error exceeds your pixel threshold at its current distance.
- `archive.children(entry)` → the up-to-four finer tiles under `entry` (REPLACE refinement:
  show a tile *or* its children, never both).
- `archive.header().root_geometric_error` seeds the traversal from the root.

A flight-sim-style loop: each frame, walk from the root, pick the LOD whose
`geometric_error` meets your SSE threshold at the camera distance, cull by `region`, and
`decode_tile` only the entries newly entering the resident set (they're small — a 256-px
tile is ~130 KB decompressed). Cache decoded meshes by `TileKey`; evict by distance.

## 5. The point cloud — COPC + `bevy_pointcloud`

`ahn-heightfield` does **not** touch the point cloud; that's a separate file
(`reconciled.copc.laz`). The path is: a COPC/LAZ reader → points → `bevy_pointcloud`.

1. **Read the COPC.** Use a Rust COPC/LAZ reader (e.g. the `copc-rs` crate, or `las` + `laz`).
   COPC is itself an octree-streaming format, so you can query only the region/detail you
   need — a natural match for the terrain streaming above. Pull `(x, y, z)` and, if present,
   `(r, g, b)` per point. *(Check the reader crate's current API for the exact call — this
   is an external dependency.)*
2. **Hand the points to [`bevy_pointcloud`](https://crates.io/crates/bevy_pointcloud).**
   It provides the Bevy plugin, point-cloud asset/component, and GPU rendering. Build its
   point-cloud asset from your `(position, colour)` arrays and spawn it. *(Follow
   `bevy_pointcloud`'s own examples for the exact asset type and spawn API — it evolves with
   Bevy.)*

Conceptually:

```rust
// Pseudocode — adapt to copc-rs + bevy_pointcloud's real APIs.
let points = copc_reader::read("assets/data/reconciled.copc.laz")?; // xyz (RD/NAP) + rgb
let positions: Vec<Vec3> = points.iter().map(|p| rd_nap_to_enu(p, origin)).collect();
let colors:    Vec<[f32;4]> = points.iter().map(|p| p.rgb_normalised()).collect();
// let cloud = PointCloud::from(positions, colors);   // bevy_pointcloud asset
// commands.spawn(PointCloudBundle { cloud: clouds.add(cloud), ..default() });
```

## 6. Making terrain and point cloud share one world

This is the one real gotcha, because the two products use **different horizontal CRSs**:

| product | horizontal | vertical |
|---------|-----------|----------|
| terrain pack (`Entry.region`) | EPSG:4979 geodetic (lon/lat radians) | NAP metres |
| COPC (`reconciled.copc.laz`) | EPSG:28992 "RD New" (metres) | NAP metres |

The **vertical datum is shared (NAP)**, so heights already line up — no Z conversion. Only
the horizontal frames differ. Pick one world frame and bring the other into it:

- **Terrain** reconstructs into ENU metres straight from its geodetic region (§3).
- **Point cloud** is in RD-New metres; convert its `(x, y)` to the *same* ENU frame. The
  clean way is the [`proj`](https://crates.io/crates/proj) crate (bindings to PROJ):
  transform EPSG:28992 → EPSG:4979 (lon/lat), then through the same equirectangular ENU map
  and origin `(lon0, lat0)` you used for the terrain. Do it once at load.

Use the **same scene origin `(lon0, lat0)`** everywhere and everything registers to within
centimetres over a city — which is the whole reason the pipeline keeps AHN and the ortho on
a shared horizontal grid to begin with (see [`docs/overview.md`](overview.md), "Coordinate
systems").

## 7. Integrity and gotchas

- **Trust boundary.** If you ever stream packs over a network, the decoder rejects
  malformed input safely (bounded allocation, CRC/`file_size`/`dataset_id` checks) — but CRC
  is not a signature. For untrusted transport, verify `manifest.json`/`dataset_id` against a
  trusted source, and run `verify_blobs()` at install time.
- **Winding / culling.** The triangle order `(a, cc, d), (a, d, b)` matches the strict and
  game profiles, so back-face culling and normals agree across profiles — don't reorder it.
- **Heights can exceed `[minH, maxH]`** by up to `z_scale/2` (the quantization bound); don't
  assert exact containment of dequantized values.
- **Textures are baseline JPEG** at a pinned quality — any JPEG decoder handles them; the
  crate hands them to you undecoded on purpose.

## Pointers

- Crate: [`rust/ahn-heightfield`](../rust/ahn-heightfield) — README, examples
  (`list_archive`, `dump_header`, `decode_to_pgm`), and the `Archive` / `Heightfield` API.
- Normative byte formats: [`docs/specs/hfp-pack-format.md`](specs/hfp-pack-format.md)
  and [`docs/specs/heightfield-chunk-format.md`](specs/heightfield-chunk-format.md).
- Domain background: [`docs/overview.md`](overview.md).
- Bevy point clouds: [`bevy_pointcloud`](https://crates.io/crates/bevy_pointcloud).
- Reprojection: [`proj`](https://crates.io/crates/proj); COPC reading: `copc-rs` / `las` + `laz`.
