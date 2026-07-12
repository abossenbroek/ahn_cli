//! Throwaway hf-v2 interop spike: a Rust decoder for the v2 `.hf` chunk and
//! the `AHNP` pack container, mirroring the Python producer POC exactly.
//!
//! This is NOT the shippable `ahn-heightfield` crate — it exists only to
//! rule out the Task-1b interop doubts in CI on three OSes. Every multi-byte
//! field is read with explicit little-endian `from_le_bytes`, and the crate
//! forbids `unsafe`, so alignment is never a factor.

#![forbid(unsafe_code)]

use std::convert::TryInto;

/// Every normative reject the spike decoders can raise. Mirrors the Python
/// `PackError` / `Tiles3dError` reject set one-to-one.
#[derive(Debug, PartialEq, Eq)]
pub enum HfError {
    TooShort,
    BadMagic,
    BadVersion,
    BadContentKind,
    HeaderCrc,
    IndexCrc,
    DatasetId,
    IndexOffset,
    IndexSize,
    HashLayout,
    FileSize,
    SectionBeyondEof,
    BlobBeyondEof,
    NotAligned,
    NotSorted,
    Overlap,
    TzNonZero,
    Pad,
    Zstd,
    BadLength,
}

// ---- little-endian primitive reads (explicit; no casting) ----------------

fn le_u32(b: &[u8], off: usize) -> Result<u32, HfError> {
    let s = b.get(off..off + 4).ok_or(HfError::TooShort)?;
    Ok(u32::from_le_bytes(s.try_into().unwrap()))
}

fn le_u64(b: &[u8], off: usize) -> Result<u64, HfError> {
    let s = b.get(off..off + 8).ok_or(HfError::TooShort)?;
    Ok(u64::from_le_bytes(s.try_into().unwrap()))
}

fn le_f64(b: &[u8], off: usize) -> Result<f64, HfError> {
    let s = b.get(off..off + 8).ok_or(HfError::TooShort)?;
    Ok(f64::from_le_bytes(s.try_into().unwrap()))
}

/// u64 file offset/size -> usize, rejecting anything past `usize` on 32-bit.
fn as_usize(v: u64) -> Result<usize, HfError> {
    usize::try_from(v).map_err(|_| HfError::BadLength)
}

// ========================================================================
// v2 `.hf` chunk header (120 bytes) + zstd frame
// ========================================================================

pub const CHUNK_MAGIC: &[u8; 4] = b"AHNH";
pub const CHUNK_VERSION: u32 = 2;
pub const CHUNK_HEADER_SIZE: usize = 120;
const CHUNK_CRC_SPAN: usize = 112; // header_crc32 covers [0,112)

#[derive(Debug, Clone, PartialEq)]
pub struct ChunkHeader {
    pub version: u32,
    pub width: u32,
    pub height: u32,
    pub z_offset: f64,
    pub z_scale: f64,
    pub region: [f64; 6],
    pub payload_len: u64,
}

impl ChunkHeader {
    /// Parse and validate a v2 `.hf` header WITHOUT decompressing. The header
    /// CRC-32 is verified before any width/height is trusted, so a corrupt
    /// header can never drive a giant `width*height*2` allocation.
    pub fn parse(data: &[u8]) -> Result<ChunkHeader, HfError> {
        if data.len() < CHUNK_HEADER_SIZE {
            return Err(HfError::TooShort);
        }
        if &data[0..4] != CHUNK_MAGIC {
            return Err(HfError::BadMagic);
        }
        let version = le_u32(data, 4)?;
        if version != CHUNK_VERSION {
            return Err(HfError::BadVersion);
        }
        // CRC over [0,112) verified BEFORE trusting dims.
        let stored_crc = le_u32(data, 112)?;
        let mut h = crc32fast::Hasher::new();
        h.update(&data[0..CHUNK_CRC_SPAN]);
        if h.finalize() != stored_crc {
            return Err(HfError::HeaderCrc);
        }
        if le_u32(data, 116)? != 0 {
            return Err(HfError::Pad);
        }
        let width = le_u32(data, 8)?;
        let height = le_u32(data, 12)?;
        if width == 0 || height == 0 {
            return Err(HfError::BadLength);
        }
        let region = [
            le_f64(data, 56)?,
            le_f64(data, 64)?,
            le_f64(data, 72)?,
            le_f64(data, 80)?,
            le_f64(data, 88)?,
            le_f64(data, 96)?,
        ];
        Ok(ChunkHeader {
            version,
            width,
            height,
            z_offset: le_f64(data, 16)?,
            z_scale: le_f64(data, 24)?,
            region,
            payload_len: le_u64(data, 104)?,
        })
    }

    /// Expected decompressed byte count, computed in u64 (never u32).
    pub fn plane_bytes(&self) -> u64 {
        (self.width as u64) * (self.height as u64) * 2
    }
}

/// Fully decode a v2 `.hf` chunk to its `u16` height plane. Verifies the
/// zstd content checksum (native, via libzstd), the exact trailing-frame
/// length, and the decompressed length `== width*height*2`.
pub fn decode_chunk(data: &[u8]) -> Result<(ChunkHeader, Vec<u16>), HfError> {
    let hdr = ChunkHeader::parse(data)?;
    let frame = &data[CHUNK_HEADER_SIZE..];
    if frame.len() as u64 != hdr.payload_len {
        return Err(HfError::BadLength);
    }
    let raw = zstd::decode_all(frame).map_err(|_| HfError::Zstd)?;
    if raw.len() as u64 != hdr.plane_bytes() {
        return Err(HfError::BadLength);
    }
    let plane = raw
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    Ok((hdr, plane))
}

// ========================================================================
// `AHNP` pack container (128-B header + level dir + index + hash + blobs)
// ========================================================================

pub const PACK_MAGIC: &[u8; 4] = b"AHNP";
pub const PACK_FORMAT_VERSION: u32 = 1;
pub const PACK_HEADER_SIZE: usize = 128;
pub const DIR_ENTRY_SIZE: usize = 16;
pub const INDEX_ENTRY_SIZE: usize = 96;
pub const HASH_ENTRY_SIZE: usize = 64;
pub const BLOB_ALIGN: u64 = 16;
const PACK_HEADER_CRC_SPAN: usize = 100;

pub const KIND_HEIGHTFIELD: u32 = 0;
pub const KIND_GAME: u32 = 1;

#[derive(Debug, Clone)]
pub struct PackHeader {
    pub tile_count: u32,
    pub level_count: u32,
    pub index_offset: u64,
    pub index_size: u64,
    pub hash_offset: u64,
    pub hash_size: u64,
    pub file_size: u64,
    pub root_geometric_error: f64,
    pub dataset_id: [u8; 32],
    pub index_crc32: u32,
    pub content_kind: u32,
}

#[derive(Debug, Clone)]
pub struct Entry {
    pub level: u32,
    pub tx: u32,
    pub ty: u32,
    pub tz: u32,
    pub region: [f64; 6],
    pub geometric_error: f64,
    pub primary_offset: u64,
    pub texture_offset: u64,
    pub primary_size: u32,
    pub texture_size: u32,
}

#[derive(Debug, Clone)]
pub struct LevelRun {
    pub first_entry: u32,
    pub entry_count: u32,
    pub tx_count: u32,
    pub ty_count: u32,
}

/// A fully parsed + validated pack: header, level directory, index entries.
#[derive(Debug)]
pub struct Pack {
    pub header: PackHeader,
    pub directory: Vec<LevelRun>,
    pub entries: Vec<Entry>,
}

impl Pack {
    /// Parse and validate every normative property against `data`. Works from
    /// any slice — a plain buffer, `&padded[1..]`, or an mmap — identically,
    /// because all reads are explicit little-endian byte reads.
    pub fn open(data: &[u8]) -> Result<Pack, HfError> {
        if data.len() < PACK_HEADER_SIZE {
            return Err(HfError::TooShort);
        }
        if &data[0..4] != PACK_MAGIC {
            return Err(HfError::BadMagic);
        }
        if le_u32(data, 4)? != PACK_FORMAT_VERSION {
            return Err(HfError::BadVersion);
        }
        let content_kind = le_u32(data, 104)?;
        if content_kind != KIND_HEIGHTFIELD && content_kind != KIND_GAME {
            return Err(HfError::BadContentKind);
        }
        // header CRC over [0,100) verified before trusting the section table.
        let stored_hdr_crc = le_u32(data, 100)?;
        let mut hh = crc32fast::Hasher::new();
        hh.update(&data[0..PACK_HEADER_CRC_SPAN]);
        if hh.finalize() != stored_hdr_crc {
            return Err(HfError::HeaderCrc);
        }

        let tile_count = le_u32(data, 8)?;
        let level_count = le_u32(data, 12)?;
        let index_offset = le_u64(data, 16)?;
        let index_size = le_u64(data, 24)?;
        let hash_offset = le_u64(data, 32)?;
        let hash_size = le_u64(data, 40)?;
        let file_size = le_u64(data, 48)?;
        let root_geometric_error = le_f64(data, 56)?;
        let mut dataset_id = [0u8; 32];
        dataset_id.copy_from_slice(&data[64..96]);
        let index_crc32 = le_u32(data, 96)?;

        if index_offset != PACK_HEADER_SIZE as u64 {
            return Err(HfError::IndexOffset);
        }
        let want_index = level_count as u64 * DIR_ENTRY_SIZE as u64
            + tile_count as u64 * INDEX_ENTRY_SIZE as u64;
        if index_size != want_index {
            return Err(HfError::IndexSize);
        }
        if hash_offset != index_offset + index_size {
            return Err(HfError::HashLayout);
        }
        if hash_size != tile_count as u64 * HASH_ENTRY_SIZE as u64 {
            return Err(HfError::HashLayout);
        }
        if file_size != data.len() as u64 {
            return Err(HfError::FileSize);
        }
        let hash_end = as_usize(hash_offset + hash_size)?;
        if hash_end > data.len() {
            return Err(HfError::SectionBeyondEof);
        }

        // index CRC
        let idx_start = as_usize(index_offset)?;
        let idx_end = as_usize(hash_offset)?;
        let mut ih = crc32fast::Hasher::new();
        ih.update(&data[idx_start..idx_end]);
        if ih.finalize() != index_crc32 {
            return Err(HfError::IndexCrc);
        }

        // dataset_id == sha256(hash section)
        use sha2::{Digest, Sha256};
        let mut ds = Sha256::new();
        ds.update(&data[as_usize(hash_offset)?..hash_end]);
        if ds.finalize().as_slice() != dataset_id {
            return Err(HfError::DatasetId);
        }

        // level directory
        let mut directory = Vec::with_capacity(level_count as usize);
        for i in 0..level_count as usize {
            let base = idx_start + i * DIR_ENTRY_SIZE;
            directory.push(LevelRun {
                first_entry: le_u32(data, base)?,
                entry_count: le_u32(data, base + 4)?,
                tx_count: le_u32(data, base + 8)?,
                ty_count: le_u32(data, base + 12)?,
            });
        }

        // index entries
        let entries_base = idx_start + level_count as usize * DIR_ENTRY_SIZE;
        let mut entries = Vec::with_capacity(tile_count as usize);
        for i in 0..tile_count as usize {
            let b = entries_base + i * INDEX_ENTRY_SIZE;
            let region = [
                le_f64(data, b + 16)?,
                le_f64(data, b + 24)?,
                le_f64(data, b + 32)?,
                le_f64(data, b + 40)?,
                le_f64(data, b + 48)?,
                le_f64(data, b + 56)?,
            ];
            let e = Entry {
                level: le_u32(data, b)?,
                tx: le_u32(data, b + 4)?,
                ty: le_u32(data, b + 8)?,
                tz: le_u32(data, b + 12)?,
                region,
                geometric_error: le_f64(data, b + 64)?,
                primary_offset: le_u64(data, b + 72)?,
                texture_offset: le_u64(data, b + 80)?,
                primary_size: le_u32(data, b + 88)?,
                texture_size: le_u32(data, b + 92)?,
            };
            if e.tz != 0 {
                return Err(HfError::TzNonZero);
            }
            if !e.primary_offset.is_multiple_of(BLOB_ALIGN)
                || !e.texture_offset.is_multiple_of(BLOB_ALIGN)
            {
                return Err(HfError::NotAligned);
            }
            let pend = e.primary_offset + e.primary_size as u64;
            if pend > data.len() as u64 {
                return Err(HfError::BlobBeyondEof);
            }
            if e.texture_offset != 0 && e.texture_offset + e.texture_size as u64 > data.len() as u64
            {
                return Err(HfError::BlobBeyondEof);
            }
            entries.push(e);
        }

        // sort order (level, tz, ty, tx)
        for w in entries.windows(2) {
            let a = (w[0].level, w[0].tz, w[0].ty, w[0].tx);
            let b = (w[1].level, w[1].tz, w[1].ty, w[1].tx);
            if a >= b {
                return Err(HfError::NotSorted);
            }
        }

        // non-overlapping blobs
        let mut spans: Vec<(u64, u64)> = Vec::new();
        for e in &entries {
            spans.push((e.primary_offset, e.primary_size as u64));
            if e.texture_offset != 0 {
                spans.push((e.texture_offset, e.texture_size as u64));
            }
        }
        spans.sort();
        for w in spans.windows(2) {
            if w[0].0 + w[0].1 > w[1].0 {
                return Err(HfError::Overlap);
            }
        }

        Ok(Pack {
            header: PackHeader {
                tile_count,
                level_count,
                index_offset,
                index_size,
                hash_offset,
                hash_size,
                file_size,
                root_geometric_error,
                dataset_id,
                index_crc32,
                content_kind,
            },
            directory,
            entries,
        })
    }

    /// Opaque primary blob bytes for `entry` (`.hf` or `.glb`).
    pub fn primary<'a>(&self, data: &'a [u8], e: &Entry) -> &'a [u8] {
        let s = e.primary_offset as usize;
        &data[s..s + e.primary_size as usize]
    }
}
