//! Decodes one tile's content, dispatching on the pack's `content_kind`.
//!
//! Synchronous for now (called from a Bevy `Update` system as tiles enter the
//! selection); handing this off to `AsyncComputeTaskPool` is a follow-up (see
//! the Track C report) rather than a design constraint of this module.

use ahn_heightfield::{Entry, Heightfield};

use crate::ahnp::glb::{self, GlbTile};
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
    /// `content_kind = 1`: a dequantized glb mesh + its embedded JPEG.
    Game(GlbTile),
    /// `content_kind = 2` (`splat` feature): a parsed gaussian cloud.
    #[cfg(feature = "splat")]
    Splat(bevy_gaussian_splatting::PlanarGaussian3d),
}

/// Decode `node`'s content from `source`.
///
/// # Errors
/// - [`AhnpError::DecodeTile`]: the pack's heightfield/texture bytes for this
///   tile failed to decode.
/// - [`AhnpError::Glb`]: `content_kind == 1`'s glb container/meshopt/dequant
///   failed.
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
        1 => decode_game(source, node, entry),
        2 => decode_splat(source, node, entry),
        other => {
            unreachable!("Archive::open only accepts content_kind in {{0, 1, 2}}, got {other}")
        }
    }
}

/// `content_kind = 1`: decode the glb blob (JSON + meshopt + KHR dequant —
/// see `ahnp::glb`'s doc comment for the exact wire contract). The glb
/// already carries the tile's own RTC placement in its node
/// `translation`/`scale`, so `glb::decode_glb` returns positions already
/// un-swizzled to raw ECEF — no separate centre lookup needed here (unlike
/// the splat path below, which has to reconstruct one).
fn decode_game(
    source: &AhnpSource,
    node: &TileNode,
    entry: &Entry,
) -> Result<DecodedContent, AhnpError> {
    let bytes = source
        .archive
        .read_primary(entry)
        .map_err(|source_err| AhnpError::DecodeTile {
            level: node.key.level,
            tx: node.key.tx,
            ty: node.key.ty,
            source: source_err,
        })?;
    let tile = glb::decode_glb(&bytes).map_err(|source_err| AhnpError::Glb {
        level: node.key.level,
        tx: node.key.tx,
        ty: node.key.ty,
        source: source_err,
    })?;
    Ok(DecodedContent::Game(tile))
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
/// midpoint. How close an approximation that is depends on the tile's own
/// depth in the quadtree:
/// - **Leaves** (no descendants): `region` *is* the tile's own vertex
///   envelope (`mesh.py`/`quadtree.py`), so the two centres coincide to
///   within genuine chord-sag / sampling error — sub-millimetre in practice.
/// - **Coarse parents**: `region` is the *union* of the tile's own (coarser)
///   mesh envelope with every descendant's, so its height range can be
///   materially wider than the parent's own mesh ever reaches (e.g. a
///   building-top parent tile whose descendants include the ground beside
///   it) — the region midpoint's height can then sit tens of metres away
///   from the parent's true vertex-AABB centre. This wobble is bounded (never
///   worse than the parent's own enclosing region) and transient: `select()`
///   REPLACEs a coarse parent with its children on zoom-in, so a
///   mis-anchored coarse splat is never the final, close-up view — but it is
///   a real, visible placement error while that parent alone is on screen.
///
/// A fully bit-exact fix would carry the tile's own vertex-AABB centre in the
/// pack entry itself — a pack-format change, not attempted here.
#[cfg(feature = "splat")]
fn decode_splat(
    source: &AhnpSource,
    node: &TileNode,
    entry: &Entry,
) -> Result<DecodedContent, AhnpError> {
    use std::io::Cursor;

    use bevy::math::DVec3;
    use bevy_interleave::prelude::Planar;

    use crate::engine::geodesy::geodetic_to_ecef;

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
fn decode_splat(
    _source: &AhnpSource,
    node: &TileNode,
    _entry: &Entry,
) -> Result<DecodedContent, AhnpError> {
    Err(AhnpError::SplatFeatureDisabled {
        level: node.key.level,
        tx: node.key.tx,
        ty: node.key.ty,
    })
}
