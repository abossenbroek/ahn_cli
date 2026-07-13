//! The `AHNP` pack reader.
//!
//! An [`Archive`] opens a `tiles.hfp` pack — a fixed 128-byte header, a binary
//! scene index (level directory + one 96-byte entry per tile), a cold hash
//! section, and the concatenated content blobs — and hands out validated
//! [`PackHeader`] / [`Entry`] / [`LevelRun`] records, opaque blob bytes, and
//! decoded height tiles. [`Archive::open`] performs the full structural,
//! ordering, alignment and CRC validation of the header and index; the hash
//! section is read only by [`Archive::verify_blobs`], never at open. All reads
//! go through the positioned-read [`ReadAt`] trait, so tile loads are
//! cursor-independent and safe to run concurrently.
//!
//! The normative byte layout is
//! `docs/superpowers/specs/2026-07-12-hfp-pack-format.md`.

use std::cmp::Ordering;
use std::io;

use sha2::{Digest, Sha256};

use crate::chunk::Heightfield;
use crate::error::{Format, HfError};

/// The pack magic, ASCII `AHNP`.
pub const PACK_MAGIC: &[u8; 4] = b"AHNP";
/// The pack format version this crate reads.
pub const PACK_FORMAT_VERSION: u32 = 1;
/// The fixed pack-header length, in bytes.
pub const PACK_HEADER_LEN: usize = 128;
/// The level-directory record length, in bytes.
pub const PACK_DIR_ENTRY_LEN: usize = 16;
/// The index-entry record length, in bytes.
pub const PACK_INDEX_ENTRY_LEN: usize = 96;
/// The hash-section record length, in bytes.
pub const PACK_HASH_ENTRY_LEN: usize = 64;
/// The blob-region alignment, in bytes: every blob offset is a multiple of this.
pub const PACK_BLOB_ALIGN: u64 = 16;

/// `header_crc32` covers header bytes `[0, 124)`.
const PACK_CRC_SPAN: usize = 124;
/// Fixed offset of the index region (`index_offset`).
const PACK_INDEX_OFFSET: u64 = 128;

// ---- little-endian primitive reads (private; explicit, no casts) ----------
//
// Every caller has already bounds-checked the backing slice, so these
// fixed-offset reads never go out of range.

fn le_u32(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

fn le_u64(b: &[u8], off: usize) -> u64 {
    u64::from_le_bytes([
        b[off],
        b[off + 1],
        b[off + 2],
        b[off + 3],
        b[off + 4],
        b[off + 5],
        b[off + 6],
        b[off + 7],
    ])
}

fn le_f64(b: &[u8], off: usize) -> f64 {
    f64::from_bits(le_u64(b, off))
}

/// Positioned, cursor-independent reads over a pack's backing storage.
///
/// This mirrors the platform positioned-read extension traits
/// ([`std::os::unix::fs::FileExt::read_at`] /
/// `std::os::windows::fs::FileExt::seek_read`): every read names its own
/// `offset`, so there is no shared cursor and concurrent tile loads need no
/// lock. It takes `&self`, not `&mut self`, for exactly that reason.
///
/// The trait is **not sealed** — implement it for custom transports (HTTP
/// range reads, test mocks, an mmap slice). Only [`ReadAt::read_at`] is
/// required; [`ReadAt::read_exact_at`] is provided.
///
/// # Examples
///
/// A byte slice is the simplest backing store:
///
/// ```
/// use ahn_heightfield::ReadAt;
///
/// let data: &[u8] = b"AHNP....payload";
/// let mut buf = [0u8; 4];
/// let n = ReadAt::read_at(&data, &mut buf, 0)?;
/// assert_eq!(&buf[..n], b"AHNP");
/// # Ok::<(), std::io::Error>(())
/// ```
pub trait ReadAt {
    /// Reads into `buf` starting at `offset`, cursor-independent.
    ///
    /// Returns the number of bytes read (`0` at end of input); may read fewer
    /// than `buf.len()` bytes.
    ///
    /// # Errors
    ///
    /// Returns any [`std::io::Error`] the underlying storage reports.
    fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<usize>;

    /// Reads exactly `buf.len()` bytes starting at `offset`, looping over
    /// short reads.
    ///
    /// # Errors
    ///
    /// Returns [`io::ErrorKind::UnexpectedEof`] if the storage ends before
    /// `buf` is filled, or any error [`ReadAt::read_at`] reports. Provided; do
    /// not override unless a faster exact-read path exists.
    fn read_exact_at(&self, buf: &mut [u8], offset: u64) -> io::Result<()> {
        let mut filled = 0usize;
        while filled < buf.len() {
            let got = self.read_at(&mut buf[filled..], offset + filled as u64)?;
            if got == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "failed to fill whole buffer",
                ));
            }
            filled += got;
        }
        Ok(())
    }
}

#[cfg(unix)]
impl ReadAt for std::fs::File {
    fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<usize> {
        std::os::unix::fs::FileExt::read_at(self, buf, offset)
    }
}

#[cfg(windows)]
impl ReadAt for std::fs::File {
    fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<usize> {
        std::os::windows::fs::FileExt::seek_read(self, buf, offset)
    }
}

impl ReadAt for &[u8] {
    fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<usize> {
        let start = usize::try_from(offset)
            .unwrap_or(usize::MAX)
            .min(self.len());
        let src = &self[start..];
        let n = src.len().min(buf.len());
        buf[..n].copy_from_slice(&src[..n]);
        Ok(n)
    }
}

impl<T: ReadAt + ?Sized> ReadAt for &T {
    fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<usize> {
        (**self).read_at(buf, offset)
    }
}

/// Which blob slot of a pack entry an error or lookup refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlobSlot {
    /// The primary blob (`.hf` chunk or `.glb`).
    Primary,
    /// The texture blob (`.jpg`), present only for heightfield packs.
    Texture,
}

/// A pack lookup key: a tile's `(level, tx, ty, tz)` coordinates.
///
/// Field order is `(level, tx, ty, tz)` to match the on-disk entry layout,
/// but [`Ord`] uses the spec's sort key `(level, tz, ty, tx)` — a **different**
/// order. `Ord`/`PartialOrd` are therefore hand-written, not derived, so this
/// mismatch cannot be introduced silently. `tz` is `0` in this format version.
///
/// # Examples
///
/// ```
/// use ahn_heightfield::TileKey;
///
/// let a = TileKey { level: 1, tx: 1, ty: 0, tz: 0 };
/// let b = TileKey { level: 1, tx: 0, ty: 1, tz: 0 };
/// // Sorted by (level, tz, ty, tx): b's ty is larger, so a < b.
/// assert!(a < b);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileKey {
    /// Quadtree level (`0` = root).
    pub level: u32,
    /// Tile column index at this level.
    pub tx: u32,
    /// Tile row index at this level.
    pub ty: u32,
    /// Tile depth index (always `0` in this version).
    pub tz: u32,
}

impl Ord for TileKey {
    /// Orders by the spec's `(level, tz, ty, tx)` sort key — **not** the
    /// struct's `(level, tx, ty, tz)` field order.
    fn cmp(&self, other: &Self) -> Ordering {
        (self.level, self.tz, self.ty, self.tx).cmp(&(other.level, other.tz, other.ty, other.tx))
    }
}

impl PartialOrd for TileKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// A parsed, validated `AHNP` pack header.
///
/// A read-only view of the header's semantically-meaningful fields. The
/// mechanically-derived offset/size fields (`index_offset`, `index_size`,
/// `hash_offset`, `hash_size`) and the two CRCs are **not** exposed: a
/// successfully [`opened`](Archive::open) archive already guarantees they
/// equal their derivation from `tile_count` / `level_count`, so surfacing
/// them would create a second source of truth. It is `#[non_exhaustive]`.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PackHeader {
    /// Number of tiles = index entries = hash records.
    pub tile_count: u32,
    /// Number of quadtree levels = level-directory records.
    pub level_count: u32,
    /// Total byte length of the file (validated against the actual length).
    pub file_size: u64,
    /// The tileset's top-level geometric error (SSE traversal seed).
    pub root_geometric_error: f64,
    /// SHA-256 of the hash section — the content version (Merkle-style root).
    pub dataset_id: [u8; 32],
    /// `0` = heightfield (`.hf` + `.jpg`), `1` = game (`.glb`).
    pub content_kind: u32,
}

/// One 96-byte pack index entry: a tile's key, bounding region and blob spans.
///
/// A read-only view of one entry's wire fields, validated by
/// [`Archive::open`]. It is `#[non_exhaustive]`.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Entry {
    /// Quadtree level (`0` = root).
    pub level: u32,
    /// Tile column index at this level.
    pub tx: u32,
    /// Tile row index at this level.
    pub ty: u32,
    /// Tile depth index (always `0` in this version).
    pub tz: u32,
    /// The tile's **enclosing** region `(west, south, east, north, minH, maxH)`
    /// — its own mesh region unioned with every descendant's, bit-equal to the
    /// `tileset.json` bounding volume. Longitudes/latitudes in radians
    /// (EPSG:4979), heights in metres.
    pub region: [f64; 6],
    /// Tile geometric error (leaves are `0`).
    pub geometric_error: f64,
    /// Absolute file offset of the primary blob (`.hf` or `.glb`), 16-aligned.
    pub primary_offset: u64,
    /// Absolute file offset of the texture blob (`.jpg`), or `0` when there is
    /// no texture (`content_kind == 1`).
    pub texture_offset: u64,
    /// Byte length of the primary blob.
    pub primary_size: u32,
    /// Byte length of the texture blob, or `0` when there is no texture.
    pub texture_size: u32,
}

/// One 16-byte level-directory record: the contiguous entry run for a level.
///
/// A read-only view validated by [`Archive::open`]; it is `#[non_exhaustive]`.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LevelRun {
    /// Index of this level's first entry, into the entry array.
    pub first_entry: u32,
    /// Number of entries at this level.
    pub entry_count: u32,
    /// Number of distinct `tx` columns at this level.
    pub tx_count: u32,
    /// Number of distinct `ty` rows at this level.
    pub ty_count: u32,
}

/// A validated, open `AHNP` pack over some positioned-read backing store `R`.
///
/// [`Archive::open`] validates the header and index once; per-tile blob reads
/// and decodes happen lazily through the retained reader. `Archive<R>` is
/// `Send + Sync` whenever `R` is (e.g. `Archive<std::fs::File>`), so a single
/// opened archive can serve concurrent tile loads from many threads.
///
/// # Examples
///
/// ```
/// use ahn_heightfield::{Archive, TileKey};
///
/// let pack = include_bytes!("../tests/data/tiles.hfp");
/// let archive = Archive::open(&pack[..])?;
/// let root = archive
///     .find(TileKey { level: 0, tx: 0, ty: 0, tz: 0 })
///     .expect("root tile present");
/// let tile = archive.decode_tile(root)?;
/// assert!(tile.dequantize_at(0, 0).is_finite());
/// # Ok::<(), ahn_heightfield::HfError>(())
/// ```
pub struct Archive<R> {
    header: PackHeader,
    directory: Vec<LevelRun>,
    entries: Vec<Entry>,
    reader: R,
}

/// The `TileKey` a pack entry is keyed by (its `(level, tx, ty, tz)`).
fn entry_key(e: &Entry) -> TileKey {
    TileKey {
        level: e.level,
        tx: e.tx,
        ty: e.ty,
        tz: e.tz,
    }
}

/// Reads up to `buf.len()` bytes at `offset`, looping over short reads, and
/// reports how many were available (`< buf.len()` only at end of input).
fn read_available<R: ReadAt>(reader: &R, buf: &mut [u8], offset: u64) -> io::Result<usize> {
    let mut filled = 0usize;
    while filled < buf.len() {
        let got = reader.read_at(&mut buf[filled..], offset + filled as u64)?;
        if got == 0 {
            break;
        }
        filled += got;
    }
    Ok(filled)
}

/// The exact byte length of `reader`, found by positioned-read probing alone
/// (the spec forbids trusting `stat`). `O(log len)` single-byte reads.
fn probe_len<R: ReadAt>(reader: &R) -> io::Result<u64> {
    let mut one = [0u8; 1];
    // Gallop to an offset that reads 0 (at or past the end).
    let mut hi = 1u64;
    while reader.read_at(&mut one, hi)? != 0 {
        hi = hi.saturating_mul(2);
        if hi == u64::MAX {
            break;
        }
    }
    // Binary-search the boundary: smallest offset that reads 0 is the length.
    let mut low = 0u64;
    let mut high = hi;
    while low < high {
        let mid = low + (high - low) / 2;
        if reader.read_at(&mut one, mid)? == 0 {
            high = mid;
        } else {
            low = mid + 1;
        }
    }
    Ok(low)
}

/// Rejects any non-zero byte in the (inter-blob padding) range `[start, end)`,
/// reading in bounded chunks so a pathological gap cannot force a large buffer.
fn reject_nonzero_padding<R: ReadAt>(reader: &R, start: u64, end: u64) -> Result<(), HfError> {
    let mut off = start;
    let mut buf = [0u8; 512];
    while off < end {
        let want = (end - off).min(buf.len() as u64) as usize;
        let got = reader.read_at(&mut buf[..want], off)?;
        if got == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "pack truncated within inter-blob padding",
            )
            .into());
        }
        for (k, &byte) in buf[..got].iter().enumerate() {
            if byte != 0 {
                return Err(HfError::InterBlobPadding {
                    offset: off + k as u64,
                });
            }
        }
        off += got as u64;
    }
    Ok(())
}

impl<R: ReadAt> Archive<R> {
    /// Opens and validates a pack: one header read, one contiguous index read,
    /// and full structural + ordering + alignment + CRC validation. The hash
    /// section is **not** read — call [`Archive::verify_blobs`] for that.
    ///
    /// Validation follows the spec's order so no unverified count or length is
    /// ever trusted: magic, version and `header_crc32` are checked before any
    /// count sizes an allocation; derived offsets/sizes are recomputed and
    /// compared; `file_size` is checked against the actual length by positioned
    /// reads; then `index_crc32`, the level directory, and every entry.
    ///
    /// # Errors
    ///
    /// Returns [`HfError::TooShort`], [`HfError::BadMagic`],
    /// [`HfError::BadVersion`], [`HfError::HeaderCrc`],
    /// [`HfError::DegenerateCounts`], [`HfError::NonFiniteRootGeometricError`],
    /// [`HfError::ReservedNonZero`], [`HfError::BadContentKind`],
    /// [`HfError::IndexOffset`], [`HfError::IndexSize`], [`HfError::HashOffset`],
    /// [`HfError::HashSize`], [`HfError::FileSize`],
    /// [`HfError::IndexHashBeyondEof`], [`HfError::IndexCrc`],
    /// [`HfError::DirectoryDiscontinuity`], [`HfError::EntryCountMismatch`],
    /// [`HfError::NotSorted`], [`HfError::TzNonZero`],
    /// [`HfError::NonFiniteEntryField`], [`HfError::RegionNotWellOrdered`],
    /// [`HfError::LevelOutOfDirectory`], [`HfError::NotAligned`],
    /// [`HfError::BlobOrder`], [`HfError::BlobOverlap`],
    /// [`HfError::BlobBeyondEof`], [`HfError::InterBlobPadding`],
    /// [`HfError::TrailingBytes`], [`HfError::TextureConsistency`], or
    /// [`HfError::Io`].
    ///
    /// # Examples
    ///
    /// ```
    /// use ahn_heightfield::Archive;
    ///
    /// let pack = include_bytes!("../tests/data/tiles.hfp");
    /// let archive = Archive::open(&pack[..])?;
    /// assert_eq!(archive.entries().len() as u32, archive.header().tile_count);
    /// # Ok::<(), ahn_heightfield::HfError>(())
    /// ```
    pub fn open(reader: R) -> Result<Self, HfError> {
        // ---- header (verified before any count is trusted) ----------------
        let mut hdr = [0u8; PACK_HEADER_LEN];
        let filled = read_available(&reader, &mut hdr, 0)?;
        if filled < PACK_HEADER_LEN {
            return Err(HfError::TooShort {
                format: Format::Pack,
                minimum: PACK_HEADER_LEN,
                actual: filled,
            });
        }
        let magic: [u8; 4] = [hdr[0], hdr[1], hdr[2], hdr[3]];
        if &magic != PACK_MAGIC {
            return Err(HfError::BadMagic {
                format: Format::Pack,
                expected: *PACK_MAGIC,
                found: magic,
            });
        }
        let format_version = le_u32(&hdr, 4);
        if format_version != PACK_FORMAT_VERSION {
            return Err(HfError::BadVersion {
                format: Format::Pack,
                expected: PACK_FORMAT_VERSION,
                found: format_version,
            });
        }
        let stored_header_crc = le_u32(&hdr, 124);
        let computed_header_crc = crc32fast::hash(&hdr[..PACK_CRC_SPAN]);
        if computed_header_crc != stored_header_crc {
            return Err(HfError::HeaderCrc {
                format: Format::Pack,
                expected: stored_header_crc,
                computed: computed_header_crc,
            });
        }

        // ---- counts and header fields (now CRC-trusted) -------------------
        let tile_count = le_u32(&hdr, 8);
        let level_count = le_u32(&hdr, 12);
        if tile_count == 0 || level_count == 0 || level_count > tile_count {
            return Err(HfError::DegenerateCounts {
                tile_count,
                level_count,
            });
        }
        let root_geometric_error = le_f64(&hdr, 56);
        if !root_geometric_error.is_finite() {
            return Err(HfError::NonFiniteRootGeometricError {
                value: root_geometric_error,
            });
        }
        if le_u32(&hdr, 100) != 0 {
            return Err(HfError::ReservedNonZero {
                format: Format::Pack,
                byte_offset: 100,
            });
        }
        for (i, &byte) in hdr[108..124].iter().enumerate() {
            if byte != 0 {
                return Err(HfError::ReservedNonZero {
                    format: Format::Pack,
                    byte_offset: 108 + i,
                });
            }
        }
        let content_kind = le_u32(&hdr, 104);
        if content_kind > 1 {
            return Err(HfError::BadContentKind {
                found: content_kind,
            });
        }

        // ---- derived offsets/sizes (64-bit arithmetic) --------------------
        let index_offset = le_u64(&hdr, 16);
        if index_offset != PACK_INDEX_OFFSET {
            return Err(HfError::IndexOffset {
                expected: PACK_INDEX_OFFSET,
                found: index_offset,
            });
        }
        let index_size = u64::from(level_count) * 16 + u64::from(tile_count) * 96;
        let stored_index_size = le_u64(&hdr, 24);
        if stored_index_size != index_size {
            return Err(HfError::IndexSize {
                expected: index_size,
                found: stored_index_size,
            });
        }
        let hash_offset = index_offset + index_size;
        let stored_hash_offset = le_u64(&hdr, 32);
        if stored_hash_offset != hash_offset {
            return Err(HfError::HashOffset {
                expected: hash_offset,
                found: stored_hash_offset,
            });
        }
        let hash_size = u64::from(tile_count) * 64;
        let stored_hash_size = le_u64(&hdr, 40);
        if stored_hash_size != hash_size {
            return Err(HfError::HashSize {
                expected: hash_size,
                found: stored_hash_size,
            });
        }
        let file_size = le_u64(&hdr, 48);
        check_file_size(&reader, file_size)?;

        // Bound every count-sized allocation before it is made: the index and
        // hash regions must fit within the (already file-length-verified)
        // `file_size`. `hash_offset + hash_size` is the blob-region start; both
        // are the stored values, already checked equal to their count-derived
        // forms, so this bounds `index_size` — and thus `tile_count` /
        // `level_count` — to the file length. Without it, a `file_size`-honest
        // header declaring giant counts would drive a giant `index` allocation.
        let region_end = hash_offset.saturating_add(hash_size);
        if region_end > file_size {
            return Err(HfError::IndexHashBeyondEof {
                region_end,
                file_size,
            });
        }

        // ---- index region: one contiguous read + CRC ---------------------
        let index_len = usize::try_from(index_size).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "index too large for this platform",
            )
        })?;
        let mut index = vec![0u8; index_len];
        reader.read_exact_at(&mut index, index_offset)?;
        let stored_index_crc = le_u32(&hdr, 96);
        let computed_index_crc = crc32fast::hash(&index);
        if computed_index_crc != stored_index_crc {
            return Err(HfError::IndexCrc {
                expected: stored_index_crc,
                computed: computed_index_crc,
            });
        }

        // ---- level directory ---------------------------------------------
        let mut directory = Vec::with_capacity(level_count as usize);
        for l in 0..level_count as usize {
            let off = l * PACK_DIR_ENTRY_LEN;
            directory.push(LevelRun {
                first_entry: le_u32(&index, off),
                entry_count: le_u32(&index, off + 4),
                tx_count: le_u32(&index, off + 8),
                ty_count: le_u32(&index, off + 12),
            });
        }
        let mut expected_first = 0u64;
        for (l, run) in directory.iter().enumerate() {
            if u64::from(run.first_entry) != expected_first {
                return Err(HfError::DirectoryDiscontinuity {
                    level: l as u32,
                    expected_first_entry: expected_first as u32,
                    found: run.first_entry,
                });
            }
            expected_first += u64::from(run.entry_count);
        }
        if expected_first != u64::from(tile_count) {
            return Err(HfError::EntryCountMismatch {
                sum: expected_first,
                tile_count,
            });
        }

        // ---- entries ------------------------------------------------------
        let entries_base = level_count as usize * PACK_DIR_ENTRY_LEN;
        let mut entries = Vec::with_capacity(tile_count as usize);
        for i in 0..tile_count as usize {
            entries.push(parse_entry(&index, entries_base + i * PACK_INDEX_ENTRY_LEN));
        }

        // Strict ascending sort by (level, tz, ty, tx) — duplicates rejected.
        for i in 1..entries.len() {
            if entry_key(&entries[i - 1]) >= entry_key(&entries[i]) {
                return Err(HfError::NotSorted { index: i });
            }
        }

        // Per-entry field validation.
        for (i, e) in entries.iter().enumerate() {
            if e.level >= level_count {
                return Err(HfError::LevelOutOfDirectory {
                    index: i,
                    level: e.level,
                    level_count,
                });
            }
            let run = directory[e.level as usize];
            let start = run.first_entry as usize;
            let end = start + run.entry_count as usize;
            if i < start || i >= end {
                return Err(HfError::LevelOutOfDirectory {
                    index: i,
                    level: e.level,
                    level_count,
                });
            }
            if e.tz != 0 {
                return Err(HfError::TzNonZero { index: i, tz: e.tz });
            }
            for (field, value) in [
                ("region[0]", e.region[0]),
                ("region[1]", e.region[1]),
                ("region[2]", e.region[2]),
                ("region[3]", e.region[3]),
                ("region[4]", e.region[4]),
                ("region[5]", e.region[5]),
                ("geometric_error", e.geometric_error),
            ] {
                if !value.is_finite() {
                    return Err(HfError::NonFiniteEntryField {
                        index: i,
                        field,
                        value,
                    });
                }
            }
            if e.region[0] > e.region[2] || e.region[1] > e.region[3] || e.region[4] > e.region[5] {
                return Err(HfError::RegionNotWellOrdered { index: i });
            }
            if e.primary_offset % PACK_BLOB_ALIGN != 0 {
                return Err(HfError::NotAligned {
                    index: i,
                    slot: BlobSlot::Primary,
                    offset: e.primary_offset,
                });
            }
            if e.texture_size > 0 && e.texture_offset % PACK_BLOB_ALIGN != 0 {
                return Err(HfError::NotAligned {
                    index: i,
                    slot: BlobSlot::Texture,
                    offset: e.texture_offset,
                });
            }
            match content_kind {
                1 if e.texture_offset != 0 || e.texture_size != 0 => {
                    return Err(HfError::TextureConsistency {
                        index: i,
                        content_kind,
                    });
                }
                0 if e.texture_size == 0 => {
                    return Err(HfError::TextureConsistency {
                        index: i,
                        content_kind,
                    });
                }
                _ => {}
            }
        }

        // Blob layout: strict-ascending offsets, no overlap, zero padding,
        // wholly within [blob_start, file_size), in index/slot order.
        let blob_start = hash_offset + hash_size;
        let mut prev_offset: Option<u64> = None;
        let mut cursor = blob_start;
        let mut prev_index = 0usize;
        for (i, e) in entries.iter().enumerate() {
            let mut check = |offset: u64, size: u32, slot: BlobSlot| -> Result<(), HfError> {
                let end = offset.saturating_add(u64::from(size));
                if offset < blob_start || end > file_size {
                    return Err(HfError::BlobBeyondEof {
                        index: i,
                        slot,
                        end,
                        file_size,
                    });
                }
                if let Some(po) = prev_offset {
                    if offset <= po {
                        return Err(HfError::BlobOrder { index: i });
                    }
                }
                if offset < cursor {
                    return Err(HfError::BlobOverlap {
                        first_index: prev_index,
                        second_index: i,
                    });
                }
                reject_nonzero_padding(&reader, cursor, offset)?;
                prev_offset = Some(offset);
                cursor = end;
                prev_index = i;
                Ok(())
            };
            check(e.primary_offset, e.primary_size, BlobSlot::Primary)?;
            if e.texture_size > 0 {
                check(e.texture_offset, e.texture_size, BlobSlot::Texture)?;
            }
        }
        // A conforming pack ends exactly at the last blob; any bytes between the
        // final blob's end and `file_size` are trailing/gap bytes and rejected.
        if cursor != file_size {
            return Err(HfError::TrailingBytes {
                blob_region_end: cursor,
                file_size,
            });
        }

        Ok(Archive {
            header: PackHeader {
                tile_count,
                level_count,
                file_size,
                root_geometric_error,
                dataset_id: hdr[64..96].try_into().expect("32-byte dataset_id"),
                content_kind,
            },
            directory,
            entries,
            reader,
        })
    }

    /// The validated pack header.
    #[must_use]
    pub fn header(&self) -> &PackHeader {
        &self.header
    }

    /// The pack's content version (`== self.header().dataset_id`).
    #[must_use]
    pub fn dataset_id(&self) -> [u8; 32] {
        self.header.dataset_id
    }

    /// All entries, sorted ascending by `(level, tz, ty, tx)`.
    #[must_use]
    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    /// The level-directory record for `level`, if present.
    #[must_use]
    pub fn level_run(&self, level: u32) -> Option<LevelRun> {
        self.directory.get(level as usize).copied()
    }

    /// The contiguous entry slice for `level`, per the level directory.
    #[must_use]
    pub fn level(&self, level: u32) -> Option<&[Entry]> {
        let run = self.level_run(level)?;
        let start = run.first_entry as usize;
        let end = start + run.entry_count as usize;
        Some(&self.entries[start..end])
    }

    /// Binary-searches for the entry at `key`. `O(log entries.len())`.
    ///
    /// # Examples
    ///
    /// ```
    /// use ahn_heightfield::{Archive, TileKey};
    ///
    /// let pack = include_bytes!("../tests/data/tiles.hfp");
    /// let archive = Archive::open(&pack[..])?;
    /// assert!(archive.find(TileKey { level: 0, tx: 0, ty: 0, tz: 0 }).is_some());
    /// assert!(archive.find(TileKey { level: 9, tx: 9, ty: 9, tz: 0 }).is_none());
    /// # Ok::<(), ahn_heightfield::HfError>(())
    /// ```
    #[must_use]
    pub fn find(&self, key: TileKey) -> Option<&Entry> {
        self.entries
            .binary_search_by(|e| entry_key(e).cmp(&key))
            .ok()
            .map(|i| &self.entries[i])
    }

    /// The (0 to 4) implicit children of `entry`, at
    /// `(level + 1, {2·tx, 2·tx + 1}, {2·ty, 2·ty + 1}, tz = 0)`, in ascending
    /// key order. A child absent from the index simply does not exist.
    pub fn children<'a>(&'a self, entry: &Entry) -> impl Iterator<Item = &'a Entry> + 'a {
        child_keys(entry)
            .into_iter()
            .flatten()
            .filter_map(move |k| self.find(k))
    }

    /// Opaque primary blob bytes (`.hf` or `.glb`) for `entry`.
    ///
    /// # Errors
    ///
    /// Returns [`HfError::Io`] if the underlying reader fails or the blob range
    /// is no longer present (e.g. the file was truncated after `open`).
    pub fn read_primary(&self, entry: &Entry) -> Result<Vec<u8>, HfError> {
        Ok(self.read_blob(entry.primary_offset, entry.primary_size)?)
    }

    /// Opaque texture blob bytes (`.jpg`) for `entry`, or `None` when the pack
    /// is a game pack (`content_kind == 1`, no separate texture blob).
    ///
    /// # Errors
    ///
    /// Returns [`HfError::Io`] on an underlying reader failure.
    pub fn read_texture(&self, entry: &Entry) -> Result<Option<Vec<u8>>, HfError> {
        if self.header.content_kind == 1 {
            return Ok(None);
        }
        Ok(Some(
            self.read_blob(entry.texture_offset, entry.texture_size)?,
        ))
    }

    /// Decodes `entry`'s primary `.hf` blob and cross-checks it against the
    /// entry: the chunk header's four horizontal region doubles must be
    /// **bit-equal** to the entry's, its height range must be **contained**
    /// within the entry's, and it must satisfy the absolute-error cap. Only
    /// meaningful for a heightfield pack (`content_kind == 0`).
    ///
    /// See [`Heightfield::decode`] for the chunk-level decode, and the
    /// [`HfError::BlobContentMismatch`] variant for the cross-check.
    ///
    /// # Errors
    ///
    /// Returns [`HfError::BadContentKind`] when `content_kind != 0`, any error
    /// [`Heightfield::decode`] can raise, [`HfError::ErrorCapExceeded`],
    /// [`HfError::BlobContentMismatch`], or [`HfError::Io`].
    ///
    /// # Examples
    ///
    /// ```
    /// use ahn_heightfield::{Archive, TileKey};
    ///
    /// let pack = include_bytes!("../tests/data/tiles.hfp");
    /// let archive = Archive::open(&pack[..])?;
    /// for entry in archive.entries() {
    ///     let tile = archive.decode_tile(entry)?;
    ///     assert_eq!(tile.levels().len(), (tile.width() * tile.height()) as usize);
    /// }
    /// # Ok::<(), ahn_heightfield::HfError>(())
    /// ```
    pub fn decode_tile(&self, entry: &Entry) -> Result<Heightfield, HfError> {
        if self.header.content_kind != 0 {
            return Err(HfError::BadContentKind {
                found: self.header.content_kind,
            });
        }
        let bytes = self.read_primary(entry)?;
        let tile = Heightfield::decode(&bytes)?;
        tile.header().check_error_cap()?;
        let ch = tile.header().region;
        let horizontal_equal = (0..4).all(|k| ch[k].to_bits() == entry.region[k].to_bits());
        let height_contained = ch[4] >= entry.region[4] && ch[5] <= entry.region[5];
        if !horizontal_equal || !height_contained {
            return Err(HfError::BlobContentMismatch {
                index: self.entry_index(entry),
            });
        }
        Ok(tile)
    }

    /// Cold install/repair path: reads the hash section, verifies it against
    /// `dataset_id`, then checks every blob's SHA-256 against its record.
    ///
    /// The hash section is verified as a whole first (recomputing `dataset_id`
    /// — the Merkle root that anchors it), so a corrupted hash section is
    /// reported as [`HfError::DatasetIdMismatch`]; then, with the records
    /// trusted, each blob is hashed and compared. Fails fast on the first
    /// mismatch.
    ///
    /// # Errors
    ///
    /// Returns [`HfError::DatasetIdMismatch`], [`HfError::BlobHashMismatch`],
    /// [`HfError::TextureHashNotZero`], or [`HfError::Io`].
    ///
    /// # Examples
    ///
    /// ```
    /// use ahn_heightfield::Archive;
    ///
    /// let pack = include_bytes!("../tests/data/tiles.hfp");
    /// let archive = Archive::open(&pack[..])?;
    /// archive.verify_blobs()?;
    /// # Ok::<(), ahn_heightfield::HfError>(())
    /// ```
    pub fn verify_blobs(&self) -> Result<(), HfError> {
        let tile_count = u64::from(self.header.tile_count);
        let index_size = u64::from(self.header.level_count) * 16 + tile_count * 96;
        let hash_offset = PACK_INDEX_OFFSET + index_size;
        let hash_size = tile_count * 64;
        let hash_len = usize::try_from(hash_size)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "hash section too large"))?;
        let mut section = vec![0u8; hash_len];
        self.reader.read_exact_at(&mut section, hash_offset)?;

        let computed: [u8; 32] = Sha256::digest(&section).into();
        if computed != self.header.dataset_id {
            return Err(HfError::DatasetIdMismatch {
                expected: self.header.dataset_id,
                computed,
            });
        }

        for (i, entry) in self.entries.iter().enumerate() {
            let rec = &section[i * PACK_HASH_ENTRY_LEN..(i + 1) * PACK_HASH_ENTRY_LEN];
            let primary = self.read_primary(entry)?;
            if Sha256::digest(&primary).as_slice() != &rec[..32] {
                return Err(HfError::BlobHashMismatch {
                    index: i,
                    slot: BlobSlot::Primary,
                });
            }
            if self.header.content_kind == 1 {
                if rec[32..64] != [0u8; 32] {
                    return Err(HfError::TextureHashNotZero { index: i });
                }
            } else {
                let texture = self.read_blob(entry.texture_offset, entry.texture_size)?;
                if Sha256::digest(&texture).as_slice() != &rec[32..64] {
                    return Err(HfError::BlobHashMismatch {
                        index: i,
                        slot: BlobSlot::Texture,
                    });
                }
            }
        }
        Ok(())
    }

    /// Reads `size` bytes at `offset` into a fresh `Vec`.
    fn read_blob(&self, offset: u64, size: u32) -> io::Result<Vec<u8>> {
        let mut buf = vec![0u8; size as usize];
        self.reader.read_exact_at(&mut buf, offset)?;
        Ok(buf)
    }

    /// The index of an entry obtained from this archive (for diagnostics),
    /// found by its key.
    fn entry_index(&self, entry: &Entry) -> usize {
        self.entries
            .binary_search_by(|e| entry_key(e).cmp(&entry_key(entry)))
            .unwrap_or_else(|i| i)
    }
}

/// Parses one 96-byte index entry at `off` within `b`.
fn parse_entry(b: &[u8], off: usize) -> Entry {
    Entry {
        level: le_u32(b, off),
        tx: le_u32(b, off + 4),
        ty: le_u32(b, off + 8),
        tz: le_u32(b, off + 12),
        region: [
            le_f64(b, off + 16),
            le_f64(b, off + 24),
            le_f64(b, off + 32),
            le_f64(b, off + 40),
            le_f64(b, off + 48),
            le_f64(b, off + 56),
        ],
        geometric_error: le_f64(b, off + 64),
        primary_offset: le_u64(b, off + 72),
        texture_offset: le_u64(b, off + 80),
        primary_size: le_u32(b, off + 88),
        texture_size: le_u32(b, off + 92),
    }
}

/// The up-to-four implicit child keys of `entry`, in ascending key order.
/// `None` slots arise only from `u32` overflow at the extreme edge of the grid.
fn child_keys(entry: &Entry) -> [Option<TileKey>; 4] {
    let Some(level) = entry.level.checked_add(1) else {
        return [None, None, None, None];
    };
    let tx0 = entry.tx.checked_mul(2);
    let ty0 = entry.ty.checked_mul(2);
    let tx1 = tx0.and_then(|x| x.checked_add(1));
    let ty1 = ty0.and_then(|y| y.checked_add(1));
    let mk = |tx: Option<u32>, ty: Option<u32>| {
        Some(TileKey {
            level,
            tx: tx?,
            ty: ty?,
            tz: 0,
        })
    };
    [mk(tx0, ty0), mk(tx1, ty0), mk(tx0, ty1), mk(tx1, ty1)]
}

/// Rejects a `file_size` that does not equal the reader's actual length, found
/// by positioned reads alone (never trusting `stat`, per the spec).
fn check_file_size<R: ReadAt>(reader: &R, file_size: u64) -> Result<(), HfError> {
    let mut one = [0u8; 1];
    let past = reader.read_at(&mut one, file_size)?;
    let last = if file_size == 0 {
        0
    } else {
        reader.read_at(&mut one, file_size - 1)?
    };
    if past == 0 && last == 1 {
        return Ok(());
    }
    Err(HfError::FileSize {
        header: file_size,
        actual: probe_len(reader)?,
    })
}
