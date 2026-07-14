//! Builds a Bevy grid `Mesh` from a decoded `content_kind = 0` (heightfield)
//! tile: one vertex per height-grid sample, ECEF -> the source's anchored
//! world frame, texel-centre UVs (matching `ahn_cli.tiles3d.mesh`'s
//! `_texel_centre_uvs` convention: vertex `(j, i)` maps to the *centre* of
//! texel `(j, i)` in the same-size `tw x th` texture, `((j+0.5)/tw,
//! (i+0.5)/th)` — not the naive corner-to-corner `0..1` span, which would
//! place the outermost vertices exactly on the texture's edge instead of half
//! a texel inside it).
//!
//! # Known limitation: 2.5D wall smearing (uncorrected by design)
//!
//! This builds a *continuous* grid — every adjacent cell pair is joined by
//! two triangles, unconditionally. Where the height jumps sharply (a roof
//! edge dropping to ground) those joining triangles stand nearly vertical,
//! and since the ortho is a nadir photo with no side-of-building pixels, the
//! roof/ground texel gets stretched down them — the vertical "curtain"
//! smears visible along building edges. That is a faithful rendering of the
//! source: AHN is one height per cell (2.5D, no wall geometry) draped with a
//! straight-down photo. We deliberately do **not** cull skirts or threshold
//! discontinuities to hide it — the renderer shows the raw data and its true
//! artifacts. Genuinely fixing it needs input with side-facing appearance
//! (stereo/oblique imagery → real wall pixels + geometry); this note is left
//! open for that. See the crate README's "Known limitations". (The `splat`
//! profile sidesteps it structurally — discrete gaussians, no bridging
//! triangles.)

use ahn_heightfield::Heightfield;
use bevy::asset::RenderAssetUsages;
use bevy::math::DVec3;
use bevy::mesh::{Indices, Mesh, PrimitiveTopology};

use crate::ahnp::source::AhnpSource;
use crate::engine::geodesy::geodetic_to_ecef;

/// Build the tile's grid mesh: `width * height` vertices, a flat-shaded
/// (`(0, 1, 0)`) normal (the material is unlit, so normals are never shaded —
/// they're present only because `Mesh`'s standard vertex layout expects one),
/// and UVs spanning the tile 0..1.
pub fn build_mesh(source: &AhnpSource, heightfield: &Heightfield) -> Mesh {
    let tw = heightfield.width() as usize;
    let th = heightfield.height() as usize;
    let [west, south, east, north, ..] = heightfield.header().region;

    let mut positions = Vec::with_capacity(tw * th);
    let mut normals = Vec::with_capacity(tw * th);
    let mut uvs = Vec::with_capacity(tw * th);
    for r in 0..th {
        let lat = north - (north - south) * (r as f64) / (th - 1) as f64;
        for c in 0..tw {
            let lon = west + (east - west) * (c as f64) / (tw - 1) as f64;
            let h = heightfield.dequantize_at(r as u32, c as u32);
            let (x, y, z) = geodetic_to_ecef(lat.to_degrees(), lon.to_degrees(), h);
            let world = source.world_pos(DVec3::new(x, y, z));
            positions.push(world.to_array());
            normals.push([0.0f32, 1.0, 0.0]);
            uvs.push([(c as f32 + 0.5) / tw as f32, (r as f32 + 0.5) / th as f32]);
        }
    }

    let mut indices = Vec::with_capacity((tw - 1) * (th - 1) * 6);
    for r in 0..th - 1 {
        for c in 0..tw - 1 {
            let i0 = (r * tw + c) as u32;
            let i1 = i0 + 1;
            let i2 = i0 + tw as u32;
            let i3 = i2 + 1;
            indices.extend_from_slice(&[i0, i2, i1, i1, i2, i3]);
        }
    }

    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
    .with_inserted_indices(Indices::U32(indices))
}
