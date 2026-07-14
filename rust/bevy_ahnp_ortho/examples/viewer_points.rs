//! Orbits a real `.copc.laz` point cloud, loaded once at startup (not
//! per-frame LOD-streamed — see `points::load_points`'s doc comment).
//!
//! ```text
//! cargo run --example viewer_points --features points -- path/to/file.copc.laz
//! ```

use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin};
use bevy::prelude::*;
use bevy_ahnp_ortho::Framing;
use bevy_ahnp_ortho::points::{LodSelection, load_points};
use bevy_ahnp_ortho::render::mesh_points;

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: viewer_points <path/to/file.copc.laz>");
        std::process::exit(2);
    });

    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "bevy_ahnp_ortho points viewer".into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(FrameTimeDiagnosticsPlugin::default())
        .add_plugins(LogDiagnosticsPlugin::default())
        .insert_resource(PointsPath(path))
        .add_systems(Startup, spawn_points)
        .add_systems(Update, orbit_camera)
        .run();
}

#[derive(Resource)]
struct PointsPath(String);

/// The cloud's [`Framing`], computed at load.
#[derive(Resource)]
struct Orbit(Framing);

/// Elevation the camera holds while orbiting (radians up from horizontal).
const ELEVATION: f32 = 0.6;

fn spawn_points(
    mut commands: Commands,
    path: Res<PointsPath>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let cloud = match load_points(&path.0, LodSelection::All) {
        Ok(cloud) => cloud,
        Err(e) => {
            eprintln!("failed to load {}: {e}", path.0);
            std::process::exit(1);
        }
    };
    println!("loaded {} points", cloud.positions.len());
    let framing = Framing::fit_default(cloud.world_aabb());
    let mesh = meshes.add(mesh_points::build_mesh(&cloud));
    let material = materials.add(StandardMaterial {
        unlit: true,
        ..default()
    });
    commands.spawn((Mesh3d(mesh), MeshMaterial3d(material), Transform::IDENTITY));
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
