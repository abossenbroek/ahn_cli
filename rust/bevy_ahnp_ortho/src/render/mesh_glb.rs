//! Builds a Bevy indexed `Mesh` from a decoded `content_kind = 1` (`game`)
//! tile: `ahnp::glb::GlbTile`'s ECEF positions -> the source's anchored
//! world frame, its UVs and triangle indices passed through as-is (already
//! dequantized to the unit square / a plain `uint32` triangle list).

use bevy::asset::RenderAssetUsages;
use bevy::math::DVec3;
use bevy::mesh::{Indices, Mesh, PrimitiveTopology};

use crate::ahnp::glb::GlbTile;
use crate::ahnp::source::AhnpSource;

/// Build the tile's mesh: one vertex per `tile.ecef_positions` entry, a
/// flat-shaded (`(0, 1, 0)`) normal (unlit material, never shaded — present
/// only because `Mesh`'s standard vertex layout expects one), UVs and
/// indices passed through unchanged.
pub fn build_mesh(source: &AhnpSource, tile: &GlbTile) -> Mesh {
    let positions: Vec<[f32; 3]> = tile
        .ecef_positions
        .iter()
        .map(|&[x, y, z]| source.world_pos(DVec3::new(x, y, z)).to_array())
        .collect();
    let normals = vec![[0.0f32, 1.0, 0.0]; positions.len()];

    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, tile.uvs.clone())
    .with_inserted_indices(Indices::U32(tile.indices.clone()))
}
