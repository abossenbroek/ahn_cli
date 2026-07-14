//! Opens an AHNP pack file and builds its [`TileTree`] + world anchor.

use std::path::{Path, PathBuf};

use ahn_heightfield::Archive;
use bevy::math::{DMat4, DVec3};

use crate::engine::geodesy::world_from_ecef;
use crate::engine::tree::TileTree;
use crate::errors::AhnpError;

/// An opened AHNP pack: the archive (for on-demand blob decode), the
/// flattened [`TileTree`] built from its entries, and the rigid ECEF -> Bevy
/// world transform anchored at the root tile's region centre (so tile
/// coordinates land near the origin instead of at planetary ECEF magnitude).
pub struct AhnpSource {
    pub archive: Archive<std::fs::File>,
    pub tree: TileTree,
    /// ECEF (metres, f64) -> Bevy world (metres, f64; cast to f32 once
    /// composed with a tile's content).
    pub world_from_ecef: DMat4,
}

impl AhnpSource {
    /// Open `path`, build the tile tree, and anchor the world transform at
    /// the root tile's region centroid (`(west+east)/2`, `(south+north)/2`,
    /// ellipsoidal height 0).
    pub fn open(path: impl AsRef<Path>) -> Result<Self, AhnpError> {
        let path: PathBuf = path.as_ref().to_path_buf();
        let file = std::fs::File::open(&path).map_err(|e| AhnpError::Open {
            path: path.clone(),
            source: ahn_heightfield::HfError::Io(e),
        })?;
        let archive = Archive::open(file).map_err(|source| AhnpError::Open {
            path: path.clone(),
            source,
        })?;

        let tree =
            TileTree::build_from_entries(archive.entries()).map_err(|reason| AhnpError::Tree {
                path: path.clone(),
                reason,
            })?;

        let root = &archive.entries()[0];
        let [west, south, east, north, ..] = root.region;
        let (lat0, lon0) = ((south + north) * 0.5, (west + east) * 0.5);
        let anchor = world_from_ecef(lat0.to_degrees(), lon0.to_degrees(), 0.0);

        Ok(Self {
            archive,
            tree,
            world_from_ecef: anchor,
        })
    }

    /// Project an ECEF point (metres) into this source's anchored world
    /// frame, returning an f32 Bevy `Vec3` (planetary magnitudes cancel in
    /// the f64 matrix multiply, so casting only after composing is safe).
    pub fn world_pos(&self, ecef: DVec3) -> bevy::math::Vec3 {
        self.world_from_ecef.transform_point3(ecef).as_vec3()
    }

    /// This pack's world-space `(min, max)` axis-aligned bounding box, for
    /// framing a camera around the whole thing without assuming any
    /// particular pack size.
    ///
    /// The root entry's region alone is enough: per `engine::tree`'s doc
    /// comment, a tile's region is its *enclosing* envelope — its own mesh
    /// region unioned with every descendant's — so the root (index 0; every
    /// AHNP pack's entries are sorted `(level, tz, ty, tx)` with the root
    /// first) already encloses every other tile's region. Samples a 3x3
    /// lon/lat grid at both height extremes (18 points, not just the 8
    /// corners) — the same grid `engine::geo::region_to_ecef_volume` samples
    /// for its OBB — since the ECEF projection of a lon/lat/height box bulges
    /// ellipsoidally along its edges; corners alone under-cover that bulge
    /// and can clip a sliver of a large root out of the framed view.
    pub fn world_aabb(&self) -> (bevy::math::Vec3, bevy::math::Vec3) {
        use crate::engine::geodesy::geodetic_to_ecef;

        let [west, south, east, north, min_h, max_h] = self.archive.entries()[0].region;
        let span_lon = east - west;
        let span_lat = north - south;
        let mut lo = bevy::math::Vec3::splat(f32::INFINITY);
        let mut hi = bevy::math::Vec3::splat(f32::NEG_INFINITY);
        for i in 0..3 {
            for j in 0..3 {
                let lat = south + span_lat * f64::from(i) / 2.0;
                let lon = west + span_lon * f64::from(j) / 2.0;
                for h in [min_h, max_h] {
                    let (x, y, z) = geodetic_to_ecef(lat.to_degrees(), lon.to_degrees(), h);
                    let p = self.world_pos(DVec3::new(x, y, z));
                    lo = lo.min(p);
                    hi = hi.max(p);
                }
            }
        }
        (lo, hi)
    }
}
