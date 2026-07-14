//! `bevy_ahnp_ortho`: a Bevy renderer for AHNP (`tiles.hfp`) terrain packs.
//!
//! Streams a pack's quadtree of ortho-draped terrain tiles (heightfield grid
//! meshes or `game`-profile glTF, selected per-frame by screen-space error —
//! see [`engine::tree`]), with optional COPC point-cloud rendering (`points`
//! feature) and optional 3D Gaussian Splatting (`splat` feature: renders both
//! this pack format's own `content_kind = 2` tiles and external `.ply`/
//! `.gcloud` clouds via `bevy_gaussian_splatting`'s own `io_ply`/
//! `io_bincode2` `AssetLoader` — see [`splat::spawn_external_ply`]).
//!
//! Ported building blocks (tile-tree/LOD selection, geodesy, the meshopt
//! decoder) come from github.com/Arvikasoft/bevy_3d_tiles (dual
//! MIT/Apache-2.0) — see `NOTICE` for the full attribution and this crate's
//! own dual license.

pub mod ahnp;
pub mod engine;
pub mod errors;
pub mod render;

#[cfg(feature = "points")]
pub mod points;
#[cfg(feature = "splat")]
pub mod splat;

pub use errors::AhnpError;
pub use render::AhnpOrthoPlugin;
