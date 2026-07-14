//! The `Plugin`: opens AHNP packs, drives per-frame LOD selection
//! ([`engine::tree::select`]), and spawns/hides tile entities to match.
//!
//! Content decode runs on `AsyncComputeTaskPool` (see [`poll_tasks`]):
//! selecting a `Pending` tile spawns a `Task` and flips it to `Loading` (not
//! loadable again until it settles — [`crate::engine::tree::TileContent`]),
//! so a tile is never decoded twice concurrently. Tasks are polled once per
//! frame with `poll_once`, never blocked on; a pack whose entity is despawned
//! mid-load drops its `Vec<Option<Task<_>>>` along with it, which cancels
//! whatever's still in flight (`bevy_tasks::Task`'s own `Drop`) — there is no
//! separate "stale task" bookkeeping to do because tile indices never change
//! after `AhnpPack::open` (the tree is built once and never mutated).

#[cfg(feature = "gpu_textures")]
pub mod gpu_texture;
pub mod material;
pub mod mesh_glb;
pub mod mesh_hf;
#[cfg(feature = "points")]
pub mod mesh_points;

use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;

use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::log::warn;
use bevy::math::DVec3;
use bevy::prelude::*;
use bevy::tasks::{AsyncComputeTaskPool, Task, poll_once};

use crate::ahnp::loader::{DecodedContent, decode_tile};
use crate::ahnp::source::AhnpSource;
use crate::engine::tree::{
    DEFAULT_SSE_THRESHOLD_PX, History, SelectParams, TileContent, TileNode, select,
};
use crate::errors::AhnpError;

/// A decode task's output: the tile it was decoding for (needed on
/// completion to report/log against, since the task itself doesn't know its
/// own index) plus the decode result.
type DecodeTask = Task<(TileNode, Result<DecodedContent, AhnpError>)>;

/// One opened pack's streaming state, as a component on a scene-owning
/// entity. Add via [`AhnpPack::open`] + `commands.spawn(pack)`.
///
/// `source` is `Arc`-wrapped so an in-flight decode task can hold its own
/// clone independent of the ECS borrow — `ahn_heightfield::Archive<R>` is
/// `Send + Sync` whenever `R` is (it is, for `std::fs::File`) by design, so
/// concurrent tile decodes from the same open archive are exactly what the
/// underlying crate expects.
#[derive(Component)]
pub struct AhnpPack {
    source: Arc<AhnpSource>,
    content: Vec<TileContent>,
    history: History,
    entities: Vec<Option<Entity>>,
    tasks: Vec<Option<DecodeTask>>,
}

impl AhnpPack {
    /// Open `path` and prepare (but do not yet spawn) its tile stream.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, AhnpError> {
        let source = AhnpSource::open(path)?;
        let n = source.tree.len();
        let mut history = History::default();
        history.resize(n);
        Ok(Self {
            content: vec![TileContent::Pending; n],
            history,
            entities: vec![None; n],
            tasks: (0..n).map(|_| None).collect(),
            source: Arc::new(source),
        })
    }

    /// World-space `(min, max)` axis-aligned bounding box of the pack — lets
    /// a host frame the camera on the pack's real extent instead of guessing
    /// an orbit radius (pair with [`crate::frame::Framing::fit`]). Delegates
    /// to [`crate::ahnp::source::AhnpSource::world_aabb`]; exists on the
    /// public component because `source` is private.
    pub fn world_aabb(&self) -> (Vec3, Vec3) {
        self.source.world_aabb()
    }
}

/// The plugin: registers [`stream_tiles`] and [`tag_ortho_camera`].
pub struct AhnpOrthoPlugin;

impl Plugin for AhnpOrthoPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, (tag_ortho_camera, poll_tasks, stream_tiles).chain());
        #[cfg(feature = "splat")]
        {
            app.add_systems(Update, tag_gaussian_camera);
            app.init_resource::<crate::splat::SplatSettings>();
        }
    }
}

/// Every `Camera3d` gets `Tonemapping::None` (ortho colours shown 1:1, no
/// filmic curve). Runs every frame but is a no-op once a camera already
/// carries `Tonemapping`, so a camera spawned at any time picks this up.
pub fn tag_ortho_camera(
    mut commands: Commands,
    q: Query<Entity, (With<Camera3d>, Without<Tonemapping>)>,
) {
    for e in &q {
        commands.entity(e).insert(Tonemapping::None);
    }
}

/// Every `Camera3d` gets `GaussianCamera` (`splat` feature) — the splat
/// render pass only draws to cameras carrying that marker; without it the
/// pass silently renders nothing.
///
/// Deliberately a **separate** system, guarded on `Without<GaussianCamera>`
/// rather than folded into [`tag_ortho_camera`]'s `Without<Tonemapping>`
/// guard: a host (or our own example viewers) that spawns its camera already
/// carrying `Tonemapping` would never match that guard, so it would never be
/// tagged and would render an empty splat scene. The two markers are
/// independent and each needs its own guard.
#[cfg(feature = "splat")]
pub fn tag_gaussian_camera(
    mut commands: Commands,
    q: Query<
        Entity,
        (
            With<Camera3d>,
            Without<bevy_gaussian_splatting::GaussianCamera>,
        ),
    >,
) {
    for e in &q {
        commands
            .entity(e)
            .insert(bevy_gaussian_splatting::GaussianCamera::default());
    }
}

/// Poll every in-flight decode task; on completion, build its mesh/material
/// (or gaussian cloud) and spawn its (hidden) entity, or log+mark `Failed`.
/// Runs before [`stream_tiles`] each frame so a tile that finished loading
/// this frame is already `Ready` in time for this frame's selection.
#[allow(clippy::too_many_arguments)]
fn poll_tasks(
    mut commands: Commands,
    mut packs: Query<&mut AhnpPack>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    for mut pack in &mut packs {
        let AhnpPack {
            source,
            content,
            entities,
            tasks,
            ..
        } = &mut *pack;
        for i in 0..tasks.len() {
            let Some(task) = &mut tasks[i] else { continue };
            let Some((node, result)) = bevy::tasks::block_on(poll_once(task)) else {
                continue;
            };
            tasks[i] = None;
            match result {
                Ok(DecodedContent::Heightfield {
                    heightfield,
                    texture,
                }) => {
                    let mesh = meshes.add(mesh_hf::build_mesh(source, &heightfield));
                    let material_handle =
                        textured_material(&mut materials, &mut images, texture.as_deref(), &node);
                    let entity = commands
                        .spawn((
                            Mesh3d(mesh),
                            MeshMaterial3d(material_handle),
                            Transform::IDENTITY,
                            Visibility::Hidden,
                        ))
                        .id();
                    entities[i] = Some(entity);
                    content[i] = TileContent::Ready;
                }
                Ok(DecodedContent::Game(tile)) => {
                    let mesh = meshes.add(mesh_glb::build_mesh(source, &tile));
                    let material_handle =
                        textured_material(&mut materials, &mut images, Some(&tile.texture), &node);
                    let entity = commands
                        .spawn((
                            Mesh3d(mesh),
                            MeshMaterial3d(material_handle),
                            Transform::IDENTITY,
                            Visibility::Hidden,
                        ))
                        .id();
                    entities[i] = Some(entity);
                    content[i] = TileContent::Ready;
                }
                #[cfg(feature = "splat")]
                Ok(DecodedContent::Splat(cloud)) => {
                    entities[i] = Some(crate::splat::spawn_cloud(&mut commands, cloud));
                    content[i] = TileContent::Ready;
                }
                Err(e) => {
                    warn!(
                        "tile ({}, {}, {}) failed to decode: {e}",
                        node.key.level, node.key.tx, node.key.ty
                    );
                    content[i] = TileContent::Failed;
                }
            }
        }
    }
}

/// Per-frame: select the LOD cut for every open [`AhnpPack`] against the
/// first `Camera3d` found, spawn decode tasks for newly-selected `Pending`
/// tiles, and show/hide already-decoded entities to match the selection.
fn stream_tiles(
    mut commands: Commands,
    mut packs: Query<&mut AhnpPack>,
    cameras: Query<(&GlobalTransform, &Projection, &Camera), With<Camera3d>>,
) {
    let Some((cam_gt, proj, camera)) = cameras.iter().next() else {
        return;
    };
    let Projection::Perspective(persp) = proj else {
        return;
    };
    let Some(viewport) = camera.logical_viewport_size() else {
        return;
    };
    let k_px = f64::from(viewport.y) / (2.0 * (f64::from(persp.fov) / 2.0).tan());

    for mut pack in &mut packs {
        let AhnpPack {
            source,
            content,
            history,
            entities,
            tasks,
        } = &mut *pack;
        let inv = source.world_from_ecef.inverse();
        let cam_w = cam_gt.translation();
        let cam_pos =
            inv.transform_point3(DVec3::new(cam_w.x.into(), cam_w.y.into(), cam_w.z.into()));
        let fwd_w = cam_gt.forward();
        let cam_forward = inv
            .transform_vector3(DVec3::new(fwd_w.x.into(), fwd_w.y.into(), fwd_w.z.into()))
            .normalize_or(DVec3::NEG_Z);

        let params = SelectParams {
            cam_pos,
            cam_forward,
            k_px,
            sse_threshold_px: DEFAULT_SSE_THRESHOLD_PX,
            detail_falloff_m: 0.0,
            cam_height_m: 0.0,
        };
        let sel = select(&source.tree, content, history, &|_| false, params);

        let pool = AsyncComputeTaskPool::get();
        for req in &sel.loads {
            // `select()` only ever requests a `loadable()` (i.e. `Pending`)
            // tile — `Loading`/`Ready`/`Failed` are all excluded by
            // construction (see `TileContent::loadable`) — so this check is
            // a redundant, cheap belt-and-suspenders guard against ever
            // double-spawning a task for the same tile.
            if content[req.tile] != TileContent::Pending {
                continue;
            }
            let node = source.tree.nodes[req.tile].clone();
            let archive_source = Arc::clone(source);
            let task = pool.spawn(async move {
                let result = decode_tile(&archive_source, &node);
                (node, result)
            });
            tasks[req.tile] = Some(task);
            content[req.tile] = TileContent::Loading;
        }

        let want: HashSet<usize> = sel.render.iter().copied().collect();
        for (i, ent) in entities.iter().enumerate() {
            if let Some(e) = ent {
                let visibility = if want.contains(&i) {
                    Visibility::Visible
                } else {
                    Visibility::Hidden
                };
                commands.entity(*e).insert(visibility);
            }
        }
        history.absorb(&sel, source.tree.len());
    }
}

/// Decode `texture` (if present) into an unlit ortho material, falling back
/// to flat grey on a missing or undecodable JPEG (logged as a `warn!`, never
/// a hard failure — a bad texture shouldn't sink an otherwise-good tile).
fn textured_material(
    materials: &mut Assets<StandardMaterial>,
    images: &mut Assets<Image>,
    texture: Option<&[u8]>,
    node: &TileNode,
) -> Handle<StandardMaterial> {
    #[cfg(feature = "gpu_textures")]
    let decoded = texture.map(gpu_texture::decode_jpeg_bc1);
    #[cfg(not(feature = "gpu_textures"))]
    let decoded = texture.map(material::decode_jpeg);

    match decoded {
        Some(Ok(image)) => materials.add(material::ortho_material(images.add(image))),
        Some(Err(e)) => {
            warn!(
                "tile ({}, {}, {}) texture decode failed, using flat grey: {e}",
                node.key.level, node.key.tx, node.key.ty
            );
            materials.add(material::untextured_material())
        }
        None => materials.add(material::untextured_material()),
    }
}

#[cfg(all(test, feature = "splat"))]
mod splat_camera_tests {
    use super::*;
    use bevy_gaussian_splatting::GaussianCamera;

    /// Regression: a camera spawned *already carrying* `Tonemapping` (as our
    /// example viewers, and any host that sets it, do) must still receive
    /// `GaussianCamera`. Folding the marker into `tag_ortho_camera`'s
    /// `Without<Tonemapping>` guard silently skipped such cameras, and the
    /// splat pass then drew nothing.
    #[test]
    fn camera_with_preexisting_tonemapping_still_gets_gaussian_camera() {
        let mut app = App::new();
        app.add_systems(Update, (tag_ortho_camera, tag_gaussian_camera));
        let cam = app
            .world_mut()
            .spawn((Camera3d::default(), Tonemapping::None))
            .id();
        app.update();
        assert!(
            app.world().get::<GaussianCamera>(cam).is_some(),
            "camera pre-carrying Tonemapping must still receive GaussianCamera"
        );
    }
}
