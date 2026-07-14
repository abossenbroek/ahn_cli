//! Decodes one tile's content, dispatching on the pack's `content_kind`.
//!
//! Synchronous for now (called from a Bevy `Update` system as tiles enter the
//! selection); handing this off to `AsyncComputeTaskPool` is a follow-up (see
//! the Track C report) rather than a design constraint of this module.

use ahn_heightfield::Heightfield;

use crate::ahnp::source::AhnpSource;
use crate::engine::tree::TileNode;
use crate::errors::AhnpError;

/// One tile's decoded content, dispatched by the pack's `content_kind`.
pub enum DecodedContent {
    /// `content_kind = 0`: a dequantized height grid + its sibling JPEG.
    Heightfield {
        heightfield: Heightfield,
        texture: Option<Vec<u8>>,
    },
    /// `content_kind = 2` (`splat` feature): a parsed gaussian cloud.
    #[cfg(feature = "splat")]
    Splat(bevy_gaussian_splatting::PlanarGaussian3d),
}

/// Decode `node`'s content from `source`.
///
/// # Errors
/// - [`AhnpError::DecodeTile`] / [`AhnpError::DecodeTexture`]: the pack's
///   bytes for this tile failed to decode.
/// - [`AhnpError::GameProfileNotYetSupported`]: `content_kind == 1` (the
///   `game` profile's quantized glTF + `EXT_meshopt_compression`) — not yet
///   implemented in this crate.
/// - [`AhnpError::SplatFeatureDisabled`]: `content_kind == 2` but this crate
///   was built without the `splat` feature.
pub fn decode_tile(source: &AhnpSource, node: &TileNode) -> Result<DecodedContent, AhnpError> {
    let entry = source
        .archive
        .find(node.key)
        .expect("every TileTree node key comes from the same archive's entries()");

    match source.archive.header().content_kind {
        0 => {
            let heightfield =
                source
                    .archive
                    .decode_tile(entry)
                    .map_err(|source_err| AhnpError::DecodeTile {
                        level: node.key.level,
                        tx: node.key.tx,
                        ty: node.key.ty,
                        source: source_err,
                    })?;
            let texture =
                source
                    .archive
                    .read_texture(entry)
                    .map_err(|source_err| AhnpError::DecodeTile {
                        level: node.key.level,
                        tx: node.key.tx,
                        ty: node.key.ty,
                        source: source_err,
                    })?;
            Ok(DecodedContent::Heightfield {
                heightfield,
                texture,
            })
        }
        1 => Err(AhnpError::GameProfileNotYetSupported {
            level: node.key.level,
            tx: node.key.tx,
            ty: node.key.ty,
        }),
        2 => decode_splat(source, node),
        other => {
            unreachable!("Archive::open only accepts content_kind in {{0, 1, 2}}, got {other}")
        }
    }
}

/// `ahn_cli.tiles3d.mesh`'s producer stores each tile's vertices (and,
/// bit-identically, each splat tile's gaussian positions) **relative to the
/// tile's own ECEF vertex-AABB centre**, glTF-y-up-swizzled
/// (`(x, y, z)_ecef -> (x, z, -y)_gltf`) — meaningful together with the
/// `game` profile's glTF node `translation` (the exact centre) and the 3D
/// Tiles-mandated y-up -> z-up rotation, neither of which an AHNP pack
/// carries on its own (no glTF wrapper, no tileset.json node transform).
///
/// This crate doesn't have that exact centre (only the pack's own
/// [`ahn_heightfield::Entry::region`], the tile's *enclosing* geodetic
/// envelope) — so it re-anchors every splat gaussian at the **region
/// midpoint's** ECEF point instead of the producer's exact vertex-AABB
/// midpoint. For a genuine tile (region == vertex envelope; see
/// `mesh.py`/`quadtree.py`) the two centres coincide to within the tile's own
/// sub-metre sampling error — an approximation, not a bit-exact replay of the
/// producer's placement, but exact enough to render coherently alongside
/// this crate's heightfield tiles in the same anchored world frame.
#[cfg(feature = "splat")]
fn decode_splat(source: &AhnpSource, node: &TileNode) -> Result<DecodedContent, AhnpError> {
    use std::io::Cursor;

    use bevy::math::DVec3;
    use bevy_interleave::prelude::Planar;

    use crate::engine::geodesy::geodetic_to_ecef;

    let entry = source.archive.find(node.key).expect("checked by caller");
    let primary =
        source
            .archive
            .read_primary(entry)
            .map_err(|source_err| AhnpError::DecodeTile {
                level: node.key.level,
                tx: node.key.tx,
                ty: node.key.ty,
                source: source_err,
            })?;
    let raw = zstd::decode_all(Cursor::new(primary)).map_err(|e| AhnpError::Splat {
        level: node.key.level,
        tx: node.key.tx,
        ty: node.key.ty,
        reason: e.to_string(),
    })?;
    let mut reader = Cursor::new(raw);
    let cloud = bevy_gaussian_splatting::io::ply::parse_ply_3d(&mut reader).map_err(|e| {
        AhnpError::Splat {
            level: node.key.level,
            tx: node.key.tx,
            ty: node.key.ty,
            reason: e.to_string(),
        }
    })?;

    let [west, south, east, north, min_h, max_h] = entry.region;
    let (lat, lon, h) = (
        (south + north) * 0.5,
        (west + east) * 0.5,
        (min_h + max_h) * 0.5,
    );
    let (cx, cy, cz) = geodetic_to_ecef(lat.to_degrees(), lon.to_degrees(), h);
    let center_ecef = DVec3::new(cx, cy, cz);

    let mut gaussians = cloud.to_interleaved();
    for g in &mut gaussians {
        let [px, py, pz] = g.position_visibility.position;
        // Invert mesh.py's `(rel_x, rel_z, -rel_y) -> (x, y, z)_gltf` swizzle.
        let rel_ecef = DVec3::new(f64::from(px), -f64::from(pz), f64::from(py));
        g.position_visibility.position = source.world_pos(center_ecef + rel_ecef).to_array();
    }
    Ok(DecodedContent::Splat(
        bevy_gaussian_splatting::PlanarGaussian3d::from_interleaved(gaussians),
    ))
}

#[cfg(not(feature = "splat"))]
fn decode_splat(_source: &AhnpSource, node: &TileNode) -> Result<DecodedContent, AhnpError> {
    Err(AhnpError::SplatFeatureDisabled {
        level: node.key.level,
        tx: node.key.tx,
        ty: node.key.ty,
    })
}
