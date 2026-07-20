//! Orbits a real `content_kind = 2` (splat) AHNP pack. Also the runnable
//! closer of the C-0 load-proof: if the pack's `.ply` tiles didn't parse
//! through `bevy_gaussian_splatting`'s `io_ply` reader, this window would
//! stay empty (or the app would have already panicked in `decode_tile`'s
//! error path, logged via `warn!`).
//!
//! ```text
//! cargo run --example viewer_splat --features splat -- path/to/tiles.hfp
//! ```

use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin};
use bevy::prelude::*;
use bevy_ahnp_ortho::render::AhnpPack;
use bevy_ahnp_ortho::splat::SplatSettings;
use bevy_ahnp_ortho::{AhnpOrthoPlugin, Framing};
use bevy_gaussian_splatting::{CloudSettings, GaussianSplattingPlugin};

#[path = "helpers/orbit.rs"]
mod orbit;
use orbit::{ELEVATION, Orbit, orbit_camera};

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: viewer_splat <path/to/tiles.hfp>");
        std::process::exit(2);
    });

    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "bevy_ahnp_ortho splat viewer".into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(FrameTimeDiagnosticsPlugin::default())
        .add_plugins(LogDiagnosticsPlugin::default())
        // Callers add this themselves (see `splat::spawn_cloud`'s doc comment)
        // so an app that never opens a splat pack never pays for it.
        .add_plugins(GaussianSplattingPlugin)
        .add_plugins(AhnpOrthoPlugin)
        // Demonstrates the consumer-facing splat render API: the pack encoding
        // is fixed, but the LOOK is ours to choose. Tune live without
        // recompiling, e.g. `AHNP_SPLAT_SCALE=3 AHNP_SPLAT_OPACITY=0.6
        // cargo run --example viewer_splat --features splat -- pack.hfp` —
        // a larger scale overlaps the sparse gaussians on 2.5D wall faces.
        .insert_resource(SplatSettings(CloudSettings {
            global_scale: env_f32("AHNP_SPLAT_SCALE", 1.0),
            global_opacity: env_f32("AHNP_SPLAT_OPACITY", 1.0),
            ..default()
        }))
        .insert_resource(PackPath(path))
        .add_systems(Startup, open_pack)
        .add_systems(Update, (frame_camera, orbit_camera).chain())
        .run();
}

/// Parse an `f32` from environment variable `key`, falling back to `default`
/// when it is unset or unparseable.
fn env_f32(key: &str, default: f32) -> f32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
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

/// Spawn the camera once, framed on the pack's full world AABB via
/// [`Framing::fit_default`] (see the heightfield viewer for why this is an
/// `Update` run-once rather than a `Startup` system), storing the framing in
/// the shared [`Orbit`] resource that [`orbit_camera`] then sweeps + zooms.
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
    commands.insert_resource(Orbit::new(framing));
}
