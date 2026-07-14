//! Pure data + math: the tile tree, LOD selection, geodesy, and the meshopt
//! decoder. No ECS, no I/O — everything here is unit-tested without a GPU or
//! an open pack file.

pub mod geo;
pub mod geodesy;
pub mod meshopt;
pub mod tree;
