//! COPC point-cloud rendering (`.copc.laz` -> point-cloud entities), via
//! `copc-rs` — pinned to `0.3` (see this crate's `Cargo.toml`: `0.5`, the
//! latest published version, doesn't compile at all — its own direct `laz`
//! dependency requirement and its `las` dependency's transitive `laz`
//! requirement are two different, semver-incompatible ranges, an upstream
//! Cargo.toml defect we can't fix from here; `0.3` predates the regression
//! and its own `las`/`laz` pair is internally consistent).
//!
//! Unlike the AHNP tile-streaming path (`ahnp`/`render`), a COPC file is a
//! wholly separate artifact — not part of a `tiles.hfp` pack at all (it's
//! `ahn_cli copc`'s own output, in whatever CRS the source cloud used;
//! `ahn_cli`'s own COPC files are EPSG:28992 (RD New) X/Y + NAP-height Z, not
//! ECEF) — so this loader is a standalone one-shot load, not integrated into
//! `AhnpPack`'s per-frame LOD selection. It reads every point at one fixed
//! octree level/resolution up front; per-frame LOD streaming for point
//! clouds is a follow-up (see the Track C report).

use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use bevy::math::Vec3;
pub use copc_rs::LodSelection;
use copc_rs::{BoundsSelection, CopcReader};
use las::point::Classification;

/// Everything that can go wrong opening or reading a COPC file.
#[derive(Debug, thiserror::Error)]
pub enum PointsError {
    #[error("opening COPC file {path}: {source}")]
    Open {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("reading COPC points from {path}: {reason}")]
    Read { path: PathBuf, reason: String },
}

/// A decoded point cloud: positions already re-centred to a local,
/// Bevy-y-up metric frame (see [`load_points`]'s doc comment), and one RGB
/// colour per point (from the file's own colour channel if present, else a
/// fixed per-classification palette).
#[derive(Debug)]
pub struct PointCloud {
    pub positions: Vec<[f32; 3]>,
    pub colors: Vec<[f32; 3]>,
}

/// Load every point at `level` from `path`, re-centred at the COPC octree's
/// own centre (`CopcInfo.center_{x,y,z}`) so coordinates land near the Bevy
/// origin instead of at the source CRS's native magnitude (RD New X/Y are
/// ~1e5-1e6). `(x, y, z)_source -> (x, z, -y)_bevy` (source "north", +Y,
/// maps to -Z — the same convention `engine::geodesy`'s ENU frame uses for
/// the AHNP tile path).
///
/// # Errors
/// [`PointsError::Open`] if `path` doesn't open as a COPC file;
/// [`PointsError::Read`] if the point stream itself fails mid-read.
pub fn load_points(path: impl AsRef<Path>, level: LodSelection) -> Result<PointCloud, PointsError> {
    let path = path.as_ref().to_path_buf();
    let file = File::open(&path).map_err(|source| PointsError::Open {
        path: path.clone(),
        source,
    })?;
    let mut reader =
        CopcReader::open(BufReader::new(file)).map_err(|source| PointsError::Open {
            path: path.clone(),
            source,
        })?;

    let info = reader.copc_info();
    let center = Vec3::new(
        info.center_x as f32,
        info.center_z as f32,
        -(info.center_y as f32),
    );

    let points = reader
        .points(level, BoundsSelection::All)
        .map_err(|e| PointsError::Read {
            path: path.clone(),
            reason: e.to_string(),
        })?;

    let mut positions = Vec::new();
    let mut colors = Vec::new();
    for point in points {
        let world = Vec3::new(point.x as f32, point.z as f32, -(point.y as f32));
        positions.push((world - center).to_array());
        colors.push(point_color(point.color, point.classification));
    }
    Ok(PointCloud { positions, colors })
}

/// The point's own RGB if present (normalized from the file's `u16`
/// channels), else a fixed colour by ASPRS classification (the common AHN
/// classes; anything else falls back to a neutral grey).
fn point_color(color: Option<las::Color>, classification: Classification) -> [f32; 3] {
    if let Some(c) = color {
        return [
            f32::from(c.red) / 65535.0,
            f32::from(c.green) / 65535.0,
            f32::from(c.blue) / 65535.0,
        ];
    }
    match classification {
        Classification::Ground => [0.55, 0.45, 0.30],
        Classification::Building => [0.75, 0.75, 0.75],
        Classification::Water => [0.20, 0.40, 0.80],
        Classification::LowVegetation
        | Classification::MediumVegetation
        | Classification::HighVegetation => [0.25, 0.60, 0.25],
        _ => [0.65, 0.65, 0.65],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A real `.copc.laz` fixture, produced by `ahn_cli copc` from
    /// `tests/reconcile/fixtures/amsterdam_cloud.laz` (9491 source points,
    /// deduplicated to 2170 across one octree node, point format 6 — no
    /// colour channel, so every point falls back to the classification
    /// palette).
    const FIXTURE: &str = "tests/data/points.copc.laz";

    #[test]
    fn loads_a_real_copc_file() {
        let cloud = load_points(FIXTURE, LodSelection::All).expect("load");
        assert_eq!(cloud.positions.len(), 2170);
        assert_eq!(cloud.colors.len(), 2170);
        // Re-centred: every position should be within the octree's own
        // halfsize (~11.5 m here) of the origin, not at RD-New magnitude
        // (~1e5-1e6).
        for p in &cloud.positions {
            assert!(p.iter().all(|c| c.is_finite()));
            assert!(
                p.iter().all(|c| c.abs() < 50.0),
                "position {p:?} not re-centred"
            );
        }
        // No colour channel in this fixture -> every colour comes from the
        // classification palette (a small fixed set of RGB triples).
        let palette: [[f32; 3]; 5] = [
            [0.55, 0.45, 0.30],
            [0.75, 0.75, 0.75],
            [0.20, 0.40, 0.80],
            [0.25, 0.60, 0.25],
            [0.65, 0.65, 0.65],
        ];
        for c in &cloud.colors {
            assert!(
                palette.contains(c),
                "colour {c:?} not in the classification palette"
            );
        }
    }

    #[test]
    fn missing_file_is_a_typed_error() {
        let err = load_points("tests/data/does-not-exist.copc.laz", LodSelection::All).unwrap_err();
        assert!(matches!(err, PointsError::Open { .. }));
    }
}
