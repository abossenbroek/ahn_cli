//! Builds a Bevy `PointList` `Mesh` from a decoded [`crate::points::PointCloud`].
//!
//! A plain `PointList` mesh renders each point as a single GPU point-list
//! vertex (1 pixel, no size/billboard control) — the simplest of the plan's
//! sanctioned options ("`Mesh` with `PrimitiveTopology::PointList`, or
//! instanced"); a nicer instanced-quad/billboard point-sprite renderer with
//! controllable point size is a follow-up (see the Track C report).

use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Mesh, PrimitiveTopology};

use crate::points::PointCloud;

/// Build a `PointList` mesh: one vertex per point, position + vertex colour
/// (`Mesh::ATTRIBUTE_COLOR`, which Bevy's PBR pipeline multiplies into the
/// material's base colour automatically when present).
pub fn build_mesh(cloud: &PointCloud) -> Mesh {
    let colors: Vec<[f32; 4]> = cloud
        .colors
        .iter()
        .map(|&[r, g, b]| [r, g, b, 1.0])
        .collect();
    Mesh::new(
        PrimitiveTopology::PointList,
        RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, cloud.positions.clone())
    .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, colors)
}
