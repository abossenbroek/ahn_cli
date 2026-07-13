//! A decoder for the AHN heightfield (`.hf`) chunk format and the `AHNP` pack
//! container.
//!
//! This crate is a **decoder, not a producer**: the `ahn_cli` Python tool's
//! `tiles3d --profile heightfield` / `--profile game` writes these artifacts,
//! and this crate reads them back for a runtime. It codes against the two
//! normative specifications, not against the Python source:
//!
//! - the `.hf` chunk format —
//!   `docs/superpowers/specs/2026-07-12-heightfield-chunk-format.md`
//! - the `AHNP` pack format —
//!   `docs/superpowers/specs/2026-07-12-hfp-pack-format.md`
//!
//! The API is two layers. The **chunk layer** ([`ChunkHeader`],
//! [`Heightfield`]) decodes a single `.hf` height chunk. The **archive layer**
//! bundles many chunks (or `.glb` tiles) with a binary scene index into one
//! pack; its foundational types ([`ReadAt`], [`TileKey`], [`BlobSlot`]) are
//! defined here, with the `Archive` reader itself arriving in a later phase.
//!
//! Every multi-byte field is read with explicit little-endian byte reads (no
//! `repr(C)`/`bytemuck` casts), so decoding from an unaligned slice or an mmap
//! is bit-identical, and the crate is `#![forbid(unsafe_code)]`.
//!
//! # Examples
//!
//! Decode one height chunk and read a dequantized NAP height:
//!
//! ```
//! use ahn_heightfield::Heightfield;
//!
//! let bytes = include_bytes!("../tests/data/leaf.hf");
//! let tile = Heightfield::decode(bytes)?;
//! let height_m = tile.dequantize_at(0, 0);
//! assert!(height_m.is_finite());
//! # Ok::<(), ahn_heightfield::HfError>(())
//! ```

#![forbid(unsafe_code)]

mod archive;
mod chunk;
mod error;

pub use archive::{BlobSlot, ReadAt, TileKey};
pub use chunk::{
    ChunkHeader, Heightfield, ABSOLUTE_ERROR_CAP_M, CHUNK_HEADER_LEN, CHUNK_MAGIC, CHUNK_VERSION,
    MAX_QUANTIZED_LEVEL,
};
pub use error::{Format, HfError};
