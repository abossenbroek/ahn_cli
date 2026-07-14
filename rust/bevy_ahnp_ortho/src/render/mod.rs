//! The `Plugin`: opens AHNP packs, drives per-frame LOD selection
//! ([`engine::tree::select`]), and spawns/hides tile entities to match.
//!
//! Content decode is synchronous today (inline in [`stream_tiles`]) — handing
//! it to `AsyncComputeTaskPool` so a big pack doesn't stall a frame is a
//! documented follow-up, not a design constraint of this module (see the
//! Track C report).

pub mod material;
pub mod mesh_glb;
pub mod mesh_hf;

use std::collections::HashSet;
use std::path::Path;

use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::log::warn;
use bevy::math::DVec3;
use bevy::prelude::*;

use crate::ahnp::loader::{DecodedContent, decode_tile};
use crate::ahnp::source::AhnpSource;
use crate::engine::tree::{DEFAULT_SSE_THRESHOLD_PX, History, SelectParams, TileContent, select};
use crate::errors::AhnpError;

/// One opened pack's streaming state, as a component on a scene-owning
/// entity. Add via [`AhnpPack::open`] + `commands.spawn(pack)`.
#[derive(Component)]
pub struct AhnpPack {
    source: AhnpSource,
    content: Vec<TileContent>,
    history: History,
    entities: Vec<Option<Entity>>,
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
            source,
        })
    }
}

/// The plugin: registers [`stream_tiles`] and [`tag_ortho_camera`].
pub struct AhnpOrthoPlugin;

impl Plugin for AhnpOrthoPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, (tag_ortho_camera, stream_tiles));
    }
}

/// Every `Camera3d` gets `Tonemapping::None` (ortho colours shown 1:1, no
/// filmic curve) — and, with the `splat` feature, `GaussianCamera` (the
/// splat render pass only draws to cameras carrying that marker; without it
/// the pass silently renders nothing). Runs every frame but is a no-op once a
/// camera already carries the marker(s), so a camera spawned at any time
/// picks this up.
pub fn tag_ortho_camera(
    mut commands: Commands,
    q: Query<Entity, (With<Camera3d>, Without<Tonemapping>)>,
) {
    for e in &q {
        commands.entity(e).insert(Tonemapping::None);
        #[cfg(feature = "splat")]
        commands
            .entity(e)
            .insert(bevy_gaussian_splatting::GaussianCamera::default());
    }
}

/// Per-frame: select the LOD cut for every open [`AhnpPack`] against the
/// first `Camera3d` found, decode newly-selected tiles, and show/hide their
/// entities to match the selection.
#[allow(clippy::too_many_arguments)]
fn stream_tiles(
    mut commands: Commands,
    mut packs: Query<&mut AhnpPack>,
    cameras: Query<(&GlobalTransform, &Projection, &Camera), With<Camera3d>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut images: ResMut<Assets<Image>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
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

        for req in &sel.loads {
            if content[req.tile] != TileContent::Pending {
                continue;
            }
            let node = &source.tree.nodes[req.tile];
            match decode_tile(source, node) {
                Ok(DecodedContent::Heightfield {
                    heightfield,
                    texture,
                }) => {
                    let mesh = meshes.add(mesh_hf::build_mesh(source, &heightfield));
                    let material_handle =
                        textured_material(&mut materials, &mut images, texture.as_deref(), node);
                    let entity = commands
                        .spawn((
                            Mesh3d(mesh),
                            MeshMaterial3d(material_handle),
                            Transform::IDENTITY,
                            Visibility::Hidden,
                        ))
                        .id();
                    entities[req.tile] = Some(entity);
                    content[req.tile] = TileContent::Ready;
                }
                Ok(DecodedContent::Game(tile)) => {
                    let mesh = meshes.add(mesh_glb::build_mesh(source, &tile));
                    let material_handle =
                        textured_material(&mut materials, &mut images, Some(&tile.texture), node);
                    let entity = commands
                        .spawn((
                            Mesh3d(mesh),
                            MeshMaterial3d(material_handle),
                            Transform::IDENTITY,
                            Visibility::Hidden,
                        ))
                        .id();
                    entities[req.tile] = Some(entity);
                    content[req.tile] = TileContent::Ready;
                }
                #[cfg(feature = "splat")]
                Ok(DecodedContent::Splat(cloud)) => {
                    entities[req.tile] = Some(crate::splat::spawn_cloud(&mut commands, cloud));
                    content[req.tile] = TileContent::Ready;
                }
                Err(e) => {
                    warn!(
                        "tile ({}, {}, {}) failed to decode: {e}",
                        node.key.level, node.key.tx, node.key.ty
                    );
                    content[req.tile] = TileContent::Failed;
                }
            }
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
    node: &crate::engine::tree::TileNode,
) -> Handle<StandardMaterial> {
    match texture.map(material::decode_jpeg) {
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
