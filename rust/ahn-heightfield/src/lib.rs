//! A decoder for the AHN heightfield (`.hf`) chunk format and the `AHNP` pack
//! container.
//!
//! This crate is a **decoder, not a producer**: the `ahn_cli` Python tool's
//! `tiles3d --profile heightfield` / `--profile game` writes these artifacts,
//! and this crate reads them back for a runtime. It codes against the two
//! normative specifications, not against the Python source:
//!
//! - the `.hf` chunk format —
//!   `docs/specs/2026-07-12-heightfield-chunk-format.md`
//! - the `AHNP` pack format —
//!   `docs/specs/2026-07-12-hfp-pack-format.md`
//!
//! The API is two layers. The **chunk layer** ([`ChunkHeader`],
//! [`Heightfield`]) decodes a single `.hf` height chunk. The **archive layer**
//! ([`Archive`], reading over any [`ReadAt`] backing store) opens a `tiles.hfp`
//! pack, validates its header and binary scene index, and hands out
//! [`Entry`] records, opaque blob bytes, and decoded [`Heightfield`] tiles.
//!
//! Every multi-byte field is read with explicit little-endian byte reads (no
//! `repr(C)`/`bytemuck` casts), so decoding from an unaligned slice or an mmap
//! is bit-identical, and the crate is `#![forbid(unsafe_code)]`.
//!
//! An optional, off-by-default `encode` feature adds a `.hf` chunk **encoder**
//! (`encode_chunk`, `quantize_levels`, `ChunkFields`) held to semantic
//! round-trip equality with the decoder — not byte parity with the Python
//! producer. It changes nothing about the default decode-only API.
//!
//! # Examples
//!
//! Open a pack, decode its root tile, and read a dequantized NAP height:
//!
//! ```
//! use ahn_heightfield::{Archive, TileKey};
//!
//! let pack = include_bytes!("../tests/data/tiles.hfp");
//! let archive = Archive::open(&pack[..])?;
//! let root = archive
//!     .find(TileKey { level: 0, tx: 0, ty: 0, tz: 0 })
//!     .expect("root tile present");
//! let tile = archive.decode_tile(root)?;
//! let height_m = tile.dequantize_at(0, 0);
//! assert!(height_m.is_finite());
//! # Ok::<(), ahn_heightfield::HfError>(())
//! ```

#![forbid(unsafe_code)]

mod archive;
mod chunk;
#[cfg(feature = "encode")]
mod encode;
mod error;

pub use archive::{
    Archive, BlobSlot, Entry, LevelRun, PackHeader, ReadAt, TileKey, PACK_BLOB_ALIGN,
    PACK_DIR_ENTRY_LEN, PACK_FORMAT_VERSION, PACK_HASH_ENTRY_LEN, PACK_HEADER_LEN,
    PACK_INDEX_ENTRY_LEN, PACK_MAGIC,
};
pub use chunk::{
    ChunkHeader, Heightfield, ABSOLUTE_ERROR_CAP_M, CHUNK_HEADER_LEN, CHUNK_MAGIC, CHUNK_VERSION,
    MAX_QUANTIZED_LEVEL, NAP_VERTICAL_DATUM,
};
#[cfg(feature = "encode")]
pub use encode::{encode_chunk, quantize_levels, ChunkFields};
pub use error::{Format, HfError};
