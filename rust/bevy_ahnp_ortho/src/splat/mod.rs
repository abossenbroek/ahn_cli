//! Gaussian-splat rendering (`splat` feature): spawns a decoded
//! `content_kind = 2` tile's [`PlanarGaussian3d`] cloud via
//! `bevy_gaussian_splatting`.
//!
//! The camera MUST carry `GaussianCamera` or the splat render pass silently
//! draws nothing — [`crate::render::tag_ortho_camera`] adds it to every
//! `Camera3d` automatically when this feature is enabled. Callers must also
//! add `bevy_gaussian_splatting::GaussianSplattingPlugin` to their `App`
//! themselves (this crate does not add it implicitly, so an app that never
//! touches splat packs never pays for the plugin's systems).

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
