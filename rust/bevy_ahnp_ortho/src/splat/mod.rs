//! Gaussian-splat rendering (`splat` feature): spawns a decoded
//! `content_kind = 2` tile's [`PlanarGaussian3d`] cloud via
//! `bevy_gaussian_splatting`, and (via [`spawn_external_ply`]) an external
//! `.ply`/`.gcloud` cloud through the same crate's own `io_ply`/`io_bincode2`
//! `AssetLoader` (registered by `GaussianSplattingPlugin`'s `IoPlugin`, which
//! claims exactly those two extensions — **not** `.spz`, a different,
//! more-compact splat format this feature set doesn't load).
//!
//! The camera MUST carry `GaussianCamera` or the splat render pass silently
//! draws nothing — [`crate::render::tag_ortho_camera`] adds it to every
//! `Camera3d` automatically when this feature is enabled. Callers must also
//! add `bevy_gaussian_splatting::GaussianSplattingPlugin` to their `App`
//! themselves (this crate does not add it implicitly, so an app that never
//! touches splat packs never pays for the plugin's systems).

use bevy::asset::AssetServer;
use bevy::prelude::*;
use bevy_gaussian_splatting::{CloudSettings, PlanarGaussian3d, PlanarGaussian3dHandle};

/// Spawn `cloud` as a hidden gaussian-splat entity, returning its `Entity` so
/// the caller can toggle `Visibility` per-frame like every other tile kind.
///
/// Deferred via [`Commands::queue`]: the `Assets<PlanarGaussian3d>` resource
/// insert happens on a synchronously-reserved [`Entity`] id
/// ([`Commands::spawn_empty`]), so the id is usable immediately even though
/// the asset add is applied later in the same frame.
pub fn spawn_cloud(commands: &mut Commands, cloud: PlanarGaussian3d) -> Entity {
    let entity = commands.spawn_empty().id();
    commands.queue(move |world: &mut World| {
        let handle = world.resource_mut::<Assets<PlanarGaussian3d>>().add(cloud);
        world.entity_mut(entity).insert((
            PlanarGaussian3dHandle(handle),
            CloudSettings::default(),
            Transform::default(),
            Visibility::Hidden,
        ));
    });
    entity
}

/// Spawn an external `.ply`/`.gcloud` gaussian cloud (any file the standard
/// Bevy `AssetServer` can resolve — a bare path loads from the app's default
/// assets folder), visible immediately: unlike [`spawn_cloud`], there is no
/// already-decoded value to insert synchronously, so this hands the
/// `AssetServer`'s handle straight to the entity and lets Bevy's normal
/// asset-loading systems populate it whenever the load completes (no
/// `Visibility::Hidden` gating here — this is for standalone external
/// clouds, not a pack tile the LOD selection show/hides).
pub fn spawn_external_ply(
    commands: &mut Commands,
    asset_server: &AssetServer,
    path: impl Into<String>,
) -> Entity {
    let handle: Handle<PlanarGaussian3d> = asset_server.load(path.into());
    commands
        .spawn((
            PlanarGaussian3dHandle(handle),
            CloudSettings::default(),
            Transform::default(),
            Visibility::Visible,
        ))
        .id()
}
