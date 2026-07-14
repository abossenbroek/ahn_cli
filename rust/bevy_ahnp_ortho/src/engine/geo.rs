//! Region bounding-volume geometry (EPSG:4979 -> ECEF).
//!
//! Ported from github.com/Arvikasoft/bevy_3d_tiles (dual MIT/Apache-2.0),
//! `src/geo.rs` — trimmed to the one thing every AHNP pack needs: every tile
//! region is EPSG:4979 (radians), so the tree is always built in the ECEF
//! frame ([`super::tree::WorldVolume`]); the georeference-detection helper
//! (`tileset_is_georeferenced`, for telling local-metres tilesets from ECEF
//! ones) is dropped — an AHNP pack is unconditionally georeferenced.

use bevy::math::DVec3;

use super::geodesy::{WGS84_EQUATORIAL_RADIUS_M, geodetic_to_ecef};
use super::tree::WorldVolume;

/// Regions spanning more than this many radians of latitude or longitude get
/// the conservative whole-globe sphere instead of a sampled OBB (the grid
/// sampling under-covers strongly curved patches). ~28°.
const REGION_OBB_MAX_SPAN_RAD: f64 = 0.5;

/// ENU basis unit vectors (east, north, up) in ECEF at geodetic
/// `(lat, lon)` **radians**.
pub fn enu_basis_rad(lat: f64, lon: f64) -> (DVec3, DVec3, DVec3) {
    let (sin_lat, cos_lat) = lat.sin_cos();
    let (sin_lon, cos_lon) = lon.sin_cos();
    let east = DVec3::new(-sin_lon, cos_lon, 0.0);
    let north = DVec3::new(-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat);
    let up = DVec3::new(cos_lat * cos_lon, cos_lat * sin_lon, sin_lat);
    (east, north, up)
}

/// A tile's `region` (`[west, south, east, north (rad), minH, maxH]`,
/// EPSG:4979) -> an ECEF [`WorldVolume`].
///
/// Small/medium regions: an OBB in the ENU frame at the region's centre,
/// sized by projecting a 3×3 geodetic grid at both height bounds (the grid
/// captures the ellipsoidal bulge of the patch interior that corners alone
/// miss). Oversized or degenerate (antimeridian-crossing) regions degrade to
/// the conservative globe-bounding sphere — never culled, always refined
/// through, exactly what a planet-spanning root wants.
pub fn region_to_ecef_volume(region: &[f64; 6]) -> WorldVolume {
    let [west, south, east, north, min_h, max_h] = *region;
    let span_lon = east - west;
    let span_lat = north - south;
    let spans_usable = span_lon.is_finite()
        && span_lat.is_finite()
        && span_lon > 0.0 // antimeridian wrap arrives as west > east
        && span_lat > 0.0
        && span_lon <= REGION_OBB_MAX_SPAN_RAD
        && span_lat <= REGION_OBB_MAX_SPAN_RAD;
    if !spans_usable {
        return WorldVolume::Sphere {
            center: DVec3::ZERO,
            radius: WGS84_EQUATORIAL_RADIUS_M + max_h.max(0.0),
        };
    }

    let (clat, clon) = ((south + north) * 0.5, (west + east) * 0.5);
    let (e, n, u) = enu_basis_rad(clat, clon);
    let (cx, cy, cz) = geodetic_to_ecef(clat.to_degrees(), clon.to_degrees(), 0.0);
    let center0 = DVec3::new(cx, cy, cz);

    let mut lo = DVec3::splat(f64::INFINITY);
    let mut hi = DVec3::splat(f64::NEG_INFINITY);
    for i in 0..3 {
        for j in 0..3 {
            let lat = south + span_lat * (i as f64) / 2.0;
            let lon = west + span_lon * (j as f64) / 2.0;
            for h in [min_h, max_h] {
                let (x, y, z) = geodetic_to_ecef(lat.to_degrees(), lon.to_degrees(), h);
                let d = DVec3::new(x, y, z) - center0;
                let p = DVec3::new(d.dot(e), d.dot(n), d.dot(u));
                lo = lo.min(p);
                hi = hi.max(p);
            }
        }
    }
    let half = (hi - lo) * 0.5;
    let mid = (hi + lo) * 0.5;
    WorldVolume::Obb {
        center: center0 + e * mid.x + n * mid.y + u * mid.z,
        half_axes: [e * half.x, n * half.y, u * half.z],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Eugene, OR-ish region (the autzen neighbourhood): ~0.01 rad across.
    const SMALL_REGION: [f64; 6] = [-2.1496, 0.7686, -2.1478, 0.7696, 0.0, 120.0];

    #[test]
    fn small_region_obb_contains_its_corners_and_centre() {
        let vol = region_to_ecef_volume(&SMALL_REGION);
        let WorldVolume::Obb { .. } = vol else {
            panic!("expected OBB")
        };
        let [west, south, east, north, min_h, max_h] = SMALL_REGION;
        for (lat, lon) in [
            (south, west),
            (south, east),
            (north, west),
            (north, east),
            ((south + north) / 2.0, (west + east) / 2.0),
        ] {
            for h in [min_h, max_h, (min_h + max_h) / 2.0] {
                let (x, y, z) = geodetic_to_ecef(lat.to_degrees(), lon.to_degrees(), h);
                let d = vol.distance_to(DVec3::new(x, y, z));
                assert!(d < 1.0, "({lat}, {lon}, {h}) outside by {d} m");
            }
        }
    }

    #[test]
    fn small_region_obb_is_tight() {
        let vol = region_to_ecef_volume(&SMALL_REGION);
        let (_, r) = vol.bounding_sphere();
        assert!(r < 10_000.0, "radius {r} m not tight");
        let (clat, clon) = (
            (SMALL_REGION[1] + SMALL_REGION[3]) / 2.0,
            (SMALL_REGION[0] + SMALL_REGION[2]) / 2.0,
        );
        let (x, y, z) = geodetic_to_ecef(clat.to_degrees(), clon.to_degrees(), 1_000_000.0);
        let d = vol.distance_to(DVec3::new(x, y, z));
        assert!((d - 1_000_000.0).abs() < 10_000.0, "d = {d}");
    }

    #[test]
    fn huge_region_degrades_to_globe_sphere() {
        let vol = region_to_ecef_volume(&[
            -std::f64::consts::PI,
            -std::f64::consts::FRAC_PI_2,
            std::f64::consts::PI,
            std::f64::consts::FRAC_PI_2,
            0.0,
            9000.0,
        ]);
        let WorldVolume::Sphere { center, radius } = vol else {
            panic!("expected sphere")
        };
        assert_eq!(center, DVec3::ZERO);
        assert!(radius >= WGS84_EQUATORIAL_RADIUS_M);
        let (x, y, z) = geodetic_to_ecef(45.0, 10.0, 500.0);
        assert_eq!(vol.distance_to(DVec3::new(x, y, z)), 0.0);
    }
}
