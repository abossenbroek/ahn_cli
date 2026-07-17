//! Camera framing helpers: turn a world-space AABB into everything a host
//! needs to place a camera that sees the whole extent — whether it orbits
//! (a viewer) or picks a fixed establishing vantage (a game/app).
//!
//! These are engine-agnostic geometry: they take and return `bevy::math`
//! types but touch no ECS, so a host can call them at load time, on a
//! resize, or when the streamed extent grows, independent of render glue.
//! Get the AABB from [`crate::render::AhnpPack::world_aabb`] or
//! [`crate::points::PointCloud::world_aabb`].

use bevy::prelude::{Transform, Vec3};

/// Bevy's default perspective vertical field of view (radians) —
/// `PerspectiveProjection::default().fov`, 45°. The right value to pass to
/// [`Framing::fit`] for a camera left at the default projection.
pub const DEFAULT_FOV_Y: f32 = std::f32::consts::FRAC_PI_4;

/// Camera framing derived from a world-space AABB.
#[derive(Clone, Copy, Debug)]
pub struct Framing {
    /// Centre of the AABB — the point a camera should look at.
    pub center: Vec3,
    /// Radius of the AABB's bounding sphere.
    pub radius: f32,
    /// Distance from `center` at which a camera of the given vertical FOV
    /// frames the whole bounding sphere.
    pub distance: f32,
}

impl Framing {
    /// Fit `aabb` (a `(min, max)` pair) into a camera of vertical field of
    /// view `fov_y_radians`. The distance solves `d = r / sin(fov_y / 2)`
    /// (the camera-to-sphere-edge tangent), plus a one-metre floor so a
    /// degenerate (zero-extent) AABB still yields a positive, usable
    /// distance rather than a camera sitting on the geometry.
    ///
    /// Simplification: this only solves the *vertical* tangent, not the
    /// horizontal one — on a portrait/narrow-aspect viewport (where the
    /// horizontal FOV, not the vertical one, is the tighter constraint) the
    /// bounding sphere can still clip left/right at this distance. A fully
    /// aspect-aware fit would also take the viewport's width/height ratio
    /// and solve both tangents. Fine for a roughly-square/landscape viewer
    /// window; harden with an aspect term if a host needs a narrow one.
    pub fn fit(aabb: (Vec3, Vec3), fov_y_radians: f32) -> Self {
        let (lo, hi) = aabb;
        let center = (lo + hi) * 0.5;
        let radius = (hi - lo).length() * 0.5;
        let distance = radius / (fov_y_radians * 0.5).sin() + 1.0;
        Self {
            center,
            radius,
            distance,
        }
    }

    /// Fit `aabb` at [`DEFAULT_FOV_Y`] (a camera left at Bevy's default
    /// projection).
    pub fn fit_default(aabb: (Vec3, Vec3)) -> Self {
        Self::fit(aabb, DEFAULT_FOV_Y)
    }

    /// A camera `Transform` looking at `center` from spherical
    /// `azimuth`/`elevation` (radians) at the fit distance — the vantage a
    /// viewer sweeps by ramping `azimuth`, and a game can use once for an
    /// establishing shot. `elevation` is measured up from the horizontal
    /// plane; the camera stays on the fit sphere, so the extent stays framed
    /// at every angle.
    ///
    /// Degenerates at `elevation` = &plusmn;&pi;/2 (looking straight down/up):
    /// the eye-to-`center` direction becomes parallel to the fixed `Vec3::Y`
    /// up vector `looking_at` uses, an undefined camera roll (in practice,
    /// `glam` returns *some* valid but arbitrarily-rolled orientation rather
    /// than panicking or NaN-ing, so this is a visual glitch at the poles,
    /// not a crash). Fine for an orbit that stays off the poles (this
    /// crate's own viewers do); clamp `elevation` short of &plusmn;&pi;/2, or
    /// pick a secondary up vector near the poles, if a host needs to reach
    /// them cleanly.
    pub fn orbit_transform(&self, azimuth: f32, elevation: f32) -> Transform {
        self.orbit_transform_zoom(azimuth, elevation, 1.0)
    }

    /// Like [`orbit_transform`](Self::orbit_transform) but at `zoom` × the fit
    /// distance: `zoom < 1` moves the eye **closer** (zoom in), `zoom > 1`
    /// **further** (zoom out); `zoom == 1` reproduces `orbit_transform`
    /// exactly. This is the reusable primitive an interactive viewer drives
    /// from keyboard/mouse-wheel input — the fit itself is unchanged, only the
    /// standoff scales, so the look-at target and framing geometry stay put.
    ///
    /// `zoom` is floored at a small positive value, so a zero or negative input
    /// can never collapse the standoff onto `center` (which would make
    /// `looking_at`'s direction undefined). Interactive callers clamp zoom well
    /// above this floor; it exists only to keep a stray direct call well-defined.
    pub fn orbit_transform_zoom(&self, azimuth: f32, elevation: f32, zoom: f32) -> Transform {
        const MIN_ZOOM: f32 = 1e-4;
        let distance = self.distance * zoom.max(MIN_ZOOM);
        let (sa, ca) = azimuth.sin_cos();
        let (se, ce) = elevation.sin_cos();
        let horizontal = distance * ce;
        let eye = self.center + Vec3::new(sa * horizontal, distance * se, ca * horizontal);
        Transform::from_translation(eye).looking_at(self.center, Vec3::Y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_centres_and_solves_the_tangent_distance() {
        // A 10 m cube centred at (1, 2, 3): half-diagonal r = 5*sqrt(3).
        let f = Framing::fit(
            (Vec3::new(-4.0, -3.0, -2.0), Vec3::new(6.0, 7.0, 8.0)),
            DEFAULT_FOV_Y,
        );
        assert!((f.center - Vec3::new(1.0, 2.0, 3.0)).length() < 1e-4);
        let r = 5.0 * 3.0_f32.sqrt();
        assert!((f.radius - r).abs() < 1e-4);
        assert!((f.distance - (r / (DEFAULT_FOV_Y * 0.5).sin() + 1.0)).abs() < 1e-4);
    }

    #[test]
    fn degenerate_aabb_still_yields_a_positive_distance() {
        let p = Vec3::new(7.0, 8.0, 9.0);
        let f = Framing::fit((p, p), DEFAULT_FOV_Y);
        assert_eq!(f.radius, 0.0);
        assert!((f.distance - 1.0).abs() < 1e-6); // the one-metre floor
    }

    #[test]
    fn orbit_transform_stays_on_the_fit_sphere_and_looks_at_centre() {
        let f = Framing::fit(
            (Vec3::new(-4.0, -3.0, -2.0), Vec3::new(6.0, 7.0, 8.0)),
            DEFAULT_FOV_Y,
        );
        for &(az, el) in &[(0.0, 0.6), (1.2, 0.3), (3.0, 1.0)] {
            let t = f.orbit_transform(az, el);
            // Eye sits exactly `distance` from the centre...
            assert!((t.translation.distance(f.center) - f.distance).abs() < 1e-3);
            // ...and the camera's forward (-Z) points at the centre.
            let to_centre = (f.center - t.translation).normalize();
            assert!((t.forward().as_vec3() - to_centre).length() < 1e-3);
        }
    }

    #[test]
    fn orbit_transform_zoom_scales_the_standoff_and_keeps_the_target() {
        let f = Framing::fit(
            (Vec3::new(-4.0, -3.0, -2.0), Vec3::new(6.0, 7.0, 8.0)),
            DEFAULT_FOV_Y,
        );
        // zoom == 1 is identical to the plain orbit transform.
        let base = f.orbit_transform(1.2, 0.3);
        let same = f.orbit_transform_zoom(1.2, 0.3, 1.0);
        assert!((base.translation - same.translation).length() < 1e-6);
        // zoom scales the eye distance linearly; the look-at target is unmoved.
        for &z in &[0.25_f32, 0.5, 2.0, 4.0] {
            let t = f.orbit_transform_zoom(1.2, 0.3, z);
            assert!((t.translation.distance(f.center) - f.distance * z).abs() < 1e-3);
            let to_centre = (f.center - t.translation).normalize();
            assert!((t.forward().as_vec3() - to_centre).length() < 1e-3);
        }
        // Non-positive zoom is floored to a tiny positive standoff, never zero.
        let degenerate = f.orbit_transform_zoom(0.0, 0.6, 0.0);
        assert!(degenerate.translation.distance(f.center) > 0.0);
    }
}
