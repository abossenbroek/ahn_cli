//! COPC point-cloud rendering (`.copc.laz` -> point-cloud entities) —
//! **reserved, not yet implemented.**
//!
//! The `points` feature flag exists so downstream code can already depend on
//! the name, but this module is intentionally empty: a real implementation
//! needs a LAS/COPC header + VLR/EVLR reader and the COPC hierarchy VLR walk
//! (to find which octree nodes a view needs) before `laz` (a pure point-record
//! compressor/decompressor with no file-format awareness of its own) can even
//! be reached — materially more work than this pass's mesh+ortho+splat core,
//! and lower priority per the Track C plan when time-constrained. See the
//! Track C report for the concrete next steps.
