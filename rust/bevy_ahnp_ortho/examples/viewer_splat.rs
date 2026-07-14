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
use bevy_ahnp_ortho::AhnpOrthoPlugin;
use bevy_ahnp_ortho::render::AhnpPack;
use bevy_gaussian_splatting::GaussianSplattingPlugin;

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
        .insert_resource(PackPath(path))
        .add_systems(Startup, (open_pack, setup_camera))
        .add_systems(Update, orbit_camera)
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

const ORBIT_RADIUS_M: f32 = 400.0;

fn setup_camera(mut commands: Commands) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, ORBIT_RADIUS_M * 0.7, ORBIT_RADIUS_M * 0.7)
            .looking_at(Vec3::ZERO, Vec3::Y),
        Tonemapping::None,
    ));
}

fn orbit_camera(time: Res<Time>, mut q: Query<&mut Transform, With<Camera3d>>) {
    let a = time.elapsed_secs() * 0.15;
    let (x, z) = (
        a.sin() * ORBIT_RADIUS_M * 0.75,
        a.cos() * ORBIT_RADIUS_M * 0.75,
    );
    for mut t in &mut q {
        *t = Transform::from_xyz(x, ORBIT_RADIUS_M * 0.7, z).looking_at(Vec3::ZERO, Vec3::Y);
    }
}
