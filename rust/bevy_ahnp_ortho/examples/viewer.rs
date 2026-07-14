//! Orbits a real AHNP pack (heightfield or splat `content_kind`), streaming
//! tiles via [`bevy_ahnp_ortho::AhnpOrthoPlugin`]. Ortho colours are shown
//! 1:1: unlit material, `Tonemapping::None`.
//!
//! ```text
//! cargo run --example viewer -- path/to/tiles.hfp
//! ```

use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin};
use bevy::prelude::*;
use bevy_ahnp_ortho::render::AhnpPack;
use bevy_ahnp_ortho::{AhnpOrthoPlugin, Framing};

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: viewer <path/to/tiles.hfp>");
        std::process::exit(2);
    });

    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "bevy_ahnp_ortho viewer".into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(FrameTimeDiagnosticsPlugin::default())
        .add_plugins(LogDiagnosticsPlugin::default())
        .add_plugins(AhnpOrthoPlugin)
        .insert_resource(PackPath(path))
        .add_systems(Startup, open_pack)
        .add_systems(Update, (frame_camera, orbit_camera).chain())
        .run();
}

#[derive(Resource)]
struct PackPath(String);

fn open_pack(mut commands: Commands, path: Res<PackPath>) {
    match AhnpPack::open(&path.0) {
        Ok(pack) => {
            commands.spawn(pack);
        }
        Err(e) => {
            eprintln!("failed to open {}: {e}", path.0);
            std::process::exit(1);
        }
    }
}

/// The pack's [`Framing`], computed once the pack is open — the reusable
/// library helper does the actual fit; the viewer just sweeps its azimuth.
#[derive(Resource)]
struct Orbit(Framing);

/// Elevation the camera holds while orbiting (radians up from horizontal).
const ELEVATION: f32 = 0.6;

/// Spawn the camera once, framed on the pack's full world AABB via
/// [`Framing::fit_default`], and store the framing. Runs every frame but
/// no-ops after the first (guarded by the `Orbit` resource) — the pack
/// entity isn't visible until the first `Update`, since `open_pack`'s spawn
/// command applies at the Startup sync point.
fn frame_camera(mut commands: Commands, packs: Query<&AhnpPack>, orbit: Option<Res<Orbit>>) {
    if orbit.is_some() {
        return;
    }
    let Some(pack) = packs.iter().next() else {
        return;
    };
    let framing = Framing::fit_default(pack.world_aabb());
    commands.spawn((
        Camera3d::default(),
        framing.orbit_transform(0.0, ELEVATION),
        Tonemapping::None,
    ));
    commands.insert_resource(Orbit(framing));
}

fn orbit_camera(
    time: Res<Time>,
    orbit: Option<Res<Orbit>>,
    mut q: Query<&mut Transform, With<Camera3d>>,
) {
    let Some(orbit) = orbit else {
        return;
    };
    let azimuth = time.elapsed_secs() * 0.15;
    for mut t in &mut q {
        *t = orbit.0.orbit_transform(azimuth, ELEVATION);
    }
}
