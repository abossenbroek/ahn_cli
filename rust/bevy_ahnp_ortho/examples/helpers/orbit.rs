//! Shared orbit-camera controls for the example viewers.
//!
//! Included via `#[path = "helpers/orbit.rs"] mod orbit;` — it lives in a
//! subdirectory so Cargo does not auto-discover it as its own example binary.
//! Factors the previously-copied `Orbit` resource + `orbit_camera` system into
//! one place, and adds keyboard / mouse-wheel **zoom** on top of the automatic
//! azimuth sweep:
//!
//! - `=`/`+` (and numpad `+`), or `]`, or scroll up  → zoom **in**
//! - `-` (and numpad `-`), or `[`, or scroll down    → zoom **out**
//!
//! Zoom is a standoff multiplier on the pack's [`Framing`] fit distance,
//! clamped to a sane range; the fit and look-at target never move. The three
//! `viewer*` examples use the whole [`Orbit`] + [`orbit_camera`] pair; `demo`
//! keeps its own egui-driven `OrbitState` but reuses [`zoom_delta`] /
//! [`apply_zoom`] so the zoom mapping lives in exactly one place.
//!
//! Each example uses only a subset of this module, so unused items here are
//! expected.
#![allow(dead_code)]

use bevy::input::mouse::{MouseScrollUnit, MouseWheel};
use bevy::prelude::*;
use bevy_ahnp_ortho::Framing;

/// Elevation the camera holds while orbiting (radians up from horizontal).
pub const ELEVATION: f32 = 0.6;

/// Zoom bounds (standoff = `zoom` × fit distance) and per-input steps.
const ZOOM_MIN: f32 = 0.1;
const ZOOM_MAX: f32 = 5.0;
/// Keyboard zoom rate while a key is held, in zoom units per second — scaled by
/// the frame delta time so hold-to-zoom speed is framerate-independent.
const KEY_RATE: f32 = 1.8;
/// Mouse-wheel step per notch of `Line`-unit scroll (a physical wheel).
const WHEEL_STEP_LINE: f32 = 0.12;
/// Per pixel of `Pixel`-unit scroll (a trackpad) — ~1/16 of a line so a
/// trackpad swipe and a wheel notch move the zoom by comparable amounts.
const WHEEL_STEP_PIXEL: f32 = 0.0075;

/// The pack's [`Framing`] (computed once it is open) plus the current zoom the
/// user has dialed in. Inserted by each viewer's `frame_camera` once the pack
/// loads; absent until then, so [`orbit_camera`] no-ops early.
#[derive(Resource)]
pub struct Orbit {
    pub framing: Framing,
    pub zoom: f32,
}

impl Orbit {
    /// Start framed at the fit distance (`zoom == 1`).
    pub fn new(framing: Framing) -> Self {
        Self { framing, zoom: 1.0 }
    }
}

/// The zoom change this frame from keyboard + mouse wheel, given the frame's
/// delta time `dt` (seconds). Negative zooms in (shorter standoff), positive
/// zooms out. Keyboard is a rate scaled by `dt`, so hold-to-zoom is
/// framerate-independent (matching the auto-orbit sweep); the wheel is
/// event-driven and unit-aware (`Line` wheel vs `Pixel` trackpad). Shared by
/// the viewers and `demo` so the key/wheel bindings live in one place.
pub fn zoom_delta(
    keys: &ButtonInput<KeyCode>,
    wheel: &mut MessageReader<MouseWheel>,
    dt: f32,
) -> f32 {
    let mut delta = 0.0;
    if keys.pressed(KeyCode::Minus)
        || keys.pressed(KeyCode::NumpadSubtract)
        || keys.pressed(KeyCode::BracketLeft)
    {
        delta += KEY_RATE * dt;
    }
    if keys.pressed(KeyCode::Equal)
        || keys.pressed(KeyCode::NumpadAdd)
        || keys.pressed(KeyCode::BracketRight)
    {
        delta -= KEY_RATE * dt;
    }
    for ev in wheel.read() {
        let step = match ev.unit {
            MouseScrollUnit::Line => WHEEL_STEP_LINE,
            MouseScrollUnit::Pixel => WHEEL_STEP_PIXEL,
        };
        delta -= ev.y * step;
    }
    delta
}

/// Fold a [`zoom_delta`] into the current zoom, clamped to the allowed range.
pub fn apply_zoom(current: f32, delta: f32) -> f32 {
    (current + delta).clamp(ZOOM_MIN, ZOOM_MAX)
}

/// Sweep azimuth over time and fold keyboard / mouse-wheel input into the zoom,
/// then place the camera. A no-op until the pack's `Orbit` exists.
pub fn orbit_camera(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    mut wheel: MessageReader<MouseWheel>,
    orbit: Option<ResMut<Orbit>>,
    mut cameras: Query<&mut Transform, With<Camera3d>>,
) {
    let Some(mut orbit) = orbit else {
        wheel.clear();
        return;
    };
    let delta = zoom_delta(&keys, &mut wheel, time.delta_secs());
    if delta != 0.0 {
        orbit.zoom = apply_zoom(orbit.zoom, delta);
    }
    let azimuth = time.elapsed_secs() * 0.15;
    for mut transform in &mut cameras {
        *transform = orbit
            .framing
            .orbit_transform_zoom(azimuth, ELEVATION, orbit.zoom);
    }
}
