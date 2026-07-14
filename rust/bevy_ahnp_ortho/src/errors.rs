//! This crate's single typed error.

use std::path::PathBuf;

/// Everything that can go wrong opening or rendering an AHNP pack.
#[derive(Debug, thiserror::Error)]
pub enum AhnpError {
    #[error("opening AHNP pack {path}: {source}")]
    Open {
        path: PathBuf,
        #[source]
        source: ahn_heightfield::HfError,
    },

    #[error("building tile tree from {path}: {reason}")]
    Tree { path: PathBuf, reason: String },

    #[error("decoding tile ({level}, {tx}, {ty}): {source}")]
    DecodeTile {
        level: u32,
        tx: u32,
        ty: u32,
        #[source]
        source: ahn_heightfield::HfError,
    },

    #[error("decoding tile ({level}, {tx}, {ty}) glb: {source}")]
    Glb {
        level: u32,
        tx: u32,
        ty: u32,
        #[source]
        source: crate::ahnp::glb::GlbError,
    },

    #[cfg(feature = "splat")]
    #[error("tile ({level}, {tx}, {ty}): splat ply is not valid: {reason}")]
    Splat {
        level: u32,
        tx: u32,
        ty: u32,
        reason: String,
    },

    #[cfg(not(feature = "splat"))]
    #[error(
        "tile ({level}, {tx}, {ty}): content_kind = 2 (`splat` profile) needs this \
         crate's `splat` feature, which is not enabled"
    )]
    SplatFeatureDisabled { level: u32, tx: u32, ty: u32 },
}
