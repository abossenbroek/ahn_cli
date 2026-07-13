//! The crate's single error type, shared by the chunk and archive layers.
//!
//! [`HfError`] is one flat `#[non_exhaustive]` enum with one variant per
//! normative reject in the two specs. [`Format`] tags the four checks
//! (magic, version, header CRC, reserved-must-be-zero) that are structurally
//! identical between the `.hf` chunk and the `AHNP` pack, so they collapse to
//! one variant each rather than eight near-duplicates.

use std::fmt;

use crate::archive::BlobSlot;

/// Which of the two wire formats a shared reject came from.
///
/// Interpolates as `"chunk"` / `"pack"` in [`HfError`] messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Format {
    /// The `.hf` heightfield chunk format (magic `AHNH`).
    Chunk,
    /// The `AHNP` pack container format (magic `AHNP`).
    Pack,
}

impl fmt::Display for Format {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Format::Chunk => "chunk",
            Format::Pack => "pack",
        })
    }
}

/// Every normative decode/read reject the crate can raise.
///
/// One flat enum shared by both the chunk decoder ([`crate::Heightfield`])
/// and the pack reader (the archive layer, added in a later phase). It is
/// `#[non_exhaustive]`: a future format revision may add rejects, so every
/// external `match` must carry a wildcard arm. It derives `Debug` only —
/// several variants carry a source `std::io::Error` or `f64` fields, so it is
/// deliberately neither `Clone`, `PartialEq`, nor `Eq`. `Display` and
/// `std::error::Error` (with source chaining) come from `thiserror`.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum HfError {
    // ---- shared (chunk + pack) -------------------------------------------
    /// Input is shorter than the fixed header.
    #[error("{format} input shorter than the {minimum}-byte header ({actual} bytes)")]
    TooShort {
        /// Which format was being decoded.
        format: Format,
        /// Required header length in bytes.
        minimum: usize,
        /// Actual input length in bytes.
        actual: usize,
    },
    /// The 4-byte ASCII magic did not match.
    #[error("{format} magic mismatch: expected {expected:?}, found {found:?}")]
    BadMagic {
        /// Which format was being decoded.
        format: Format,
        /// The expected magic bytes (`AHNH` / `AHNP`).
        expected: [u8; 4],
        /// The magic bytes actually found.
        found: [u8; 4],
    },
    /// The format version field did not match.
    #[error("{format} version mismatch: expected {expected}, found {found}")]
    BadVersion {
        /// Which format was being decoded.
        format: Format,
        /// The version this crate decodes.
        expected: u32,
        /// The version actually found.
        found: u32,
    },
    /// The fixed header's CRC-32/ISO-HDLC did not match its stored value.
    #[error(
        "{format} header CRC-32 mismatch: expected {expected:#010x}, computed {computed:#010x}"
    )]
    HeaderCrc {
        /// Which format was being decoded.
        format: Format,
        /// The CRC stored in the header.
        expected: u32,
        /// The CRC computed over the header bytes.
        computed: u32,
    },
    /// A reserved/pad byte that must be zero was non-zero.
    #[error("{format} reserved byte at offset {byte_offset} is not zero")]
    ReservedNonZero {
        /// Which format was being decoded.
        format: Format,
        /// Byte offset of the offending reserved byte.
        byte_offset: usize,
    },
    /// A positioned read failed.
    #[error("i/o error")]
    Io(#[from] std::io::Error),

    // ---- chunk-only ------------------------------------------------------
    /// `width == 0` or `height == 0`.
    #[error("chunk grid has a zero dimension: width {width}, height {height}")]
    ZeroDimension {
        /// The `width` field.
        width: u32,
        /// The `height` field.
        height: u32,
    },
    /// A `float64` header field was `NaN` or infinite.
    #[error("chunk header field {field} is not finite: {value}")]
    NonFiniteHeaderField {
        /// Name of the offending field.
        field: &'static str,
        /// The non-finite value.
        value: f64,
    },
    /// `z_scale <= 0` (would divide-by-zero the dequantizer).
    #[error("chunk z_scale must be positive, found {z_scale}")]
    NonPositiveZScale {
        /// The offending `z_scale`.
        z_scale: f64,
    },
    /// The bytes after the header are not exactly `payload_len` (truncated or
    /// trailing data).
    #[error("chunk payload length mismatch: expected {expected}, actual {actual}")]
    PayloadLengthMismatch {
        /// `payload_len` from the header.
        expected: u64,
        /// The actual number of trailing bytes.
        actual: u64,
    },
    /// The zstd frame failed to decompress (including a content-checksum
    /// mismatch surfaced at the frame epilogue).
    #[error("chunk zstd frame failed to decode")]
    Zstd(#[source] std::io::Error),
    /// The decompressed plane length is not `width * height * 2`.
    #[error("chunk plane length mismatch: expected {expected}, actual {actual}")]
    PlaneLengthMismatch {
        /// `width * height * 2`.
        expected: u64,
        /// The actual decompressed length.
        actual: u64,
    },
    /// A stored quantization level exceeds the 12-bit maximum (`4095`).
    #[error("chunk level at (row {row}, col {col}) out of range: {level} > 4095")]
    LevelOutOfRange {
        /// Row of the offending level.
        row: u32,
        /// Column of the offending level.
        col: u32,
        /// The out-of-range level value.
        level: u16,
    },
    /// The exported round-trip bound `z_scale / 2` exceeds the 25 mm absolute
    /// cap. Only raised by the opt-in [`crate::ChunkHeader::check_error_cap`]
    /// and the archive layer's tile decode, never by `Heightfield::decode`.
    #[error("chunk exceeds the 0.025 m absolute-error cap: z_scale {z_scale}, bound {bound_m} m")]
    ErrorCapExceeded {
        /// The offending `z_scale`.
        z_scale: f64,
        /// The exported bound `z_scale / 2`, in metres.
        bound_m: f64,
    },

    // ---- pack-only -------------------------------------------------------
    /// `content_kind` is not `0` (heightfield) or `1` (game).
    #[error("pack content_kind is invalid: {found}")]
    BadContentKind {
        /// The out-of-range value.
        found: u32,
    },
    /// `tile_count`/`level_count` are degenerate (`0`, or `level_count >
    /// tile_count`).
    #[error("pack has degenerate counts: tile_count {tile_count}, level_count {level_count}")]
    DegenerateCounts {
        /// The `tile_count` field.
        tile_count: u32,
        /// The `level_count` field.
        level_count: u32,
    },
    /// `root_geometric_error` is `NaN` or infinite.
    #[error("pack root_geometric_error is not finite: {value}")]
    NonFiniteRootGeometricError {
        /// The non-finite value.
        value: f64,
    },
    /// `index_offset != 128`.
    #[error("pack index_offset mismatch: expected {expected}, found {found}")]
    IndexOffset {
        /// The required value (`128`).
        expected: u64,
        /// The value found.
        found: u64,
    },
    /// `index_size != level_count * 16 + tile_count * 96`.
    #[error("pack index_size mismatch: expected {expected}, found {found}")]
    IndexSize {
        /// The derived value.
        expected: u64,
        /// The value found.
        found: u64,
    },
    /// `hash_offset != index_offset + index_size`.
    #[error("pack hash_offset mismatch: expected {expected}, found {found}")]
    HashOffset {
        /// The derived value.
        expected: u64,
        /// The value found.
        found: u64,
    },
    /// `hash_size != tile_count * 64`.
    #[error("pack hash_size mismatch: expected {expected}, found {found}")]
    HashSize {
        /// The derived value.
        expected: u64,
        /// The value found.
        found: u64,
    },
    /// `file_size` does not equal the actual file length.
    #[error("pack file_size mismatch: header says {header}, actual {actual}")]
    FileSize {
        /// `file_size` from the header.
        header: u64,
        /// The actual file length.
        actual: u64,
    },
    /// The index region's CRC-32/ISO-HDLC did not match.
    #[error("pack index CRC-32 mismatch: expected {expected:#010x}, computed {computed:#010x}")]
    IndexCrc {
        /// The CRC stored in the header.
        expected: u32,
        /// The CRC computed over the index region.
        computed: u32,
    },
    /// The level directory is discontinuous (`directory[0].first_entry != 0`
    /// or a run gap).
    #[error("pack level {level} directory discontinuity: expected first_entry {expected_first_entry}, found {found}")]
    DirectoryDiscontinuity {
        /// The affected level.
        level: u32,
        /// The required `first_entry`.
        expected_first_entry: u32,
        /// The `first_entry` found.
        found: u32,
    },
    /// `sum(entry_count) != tile_count`.
    #[error("pack level entry_count sum {sum} does not equal tile_count {tile_count}")]
    EntryCountMismatch {
        /// The summed entry counts.
        sum: u64,
        /// The `tile_count` field.
        tile_count: u32,
    },
    /// Entries are not strictly ascending by `(level, tz, ty, tx)` (unsorted or
    /// a duplicate key).
    #[error("pack entries not strictly sorted at index {index}")]
    NotSorted {
        /// Index of the first out-of-order entry.
        index: usize,
    },
    /// An entry's `tz` is non-zero (reserved in this version).
    #[error("pack entry {index} has non-zero tz: {tz}")]
    TzNonZero {
        /// Index of the offending entry.
        index: usize,
        /// The non-zero `tz`.
        tz: u32,
    },
    /// A `float64` entry field (`region[i]` or `geometric_error`) is not
    /// finite.
    #[error("pack entry {index} field {field} is not finite: {value}")]
    NonFiniteEntryField {
        /// Index of the offending entry.
        index: usize,
        /// Name of the offending field.
        field: &'static str,
        /// The non-finite value.
        value: f64,
    },
    /// An entry's `region` is not well-ordered (`west > east`, `south > north`,
    /// or `minHeight > maxHeight`).
    #[error("pack entry {index} region is not well-ordered")]
    RegionNotWellOrdered {
        /// Index of the offending entry.
        index: usize,
    },
    /// An entry's `level` is outside `[0, level_count)` or outside its
    /// directory run.
    #[error("pack entry {index} level {level} outside directory (level_count {level_count})")]
    LevelOutOfDirectory {
        /// Index of the offending entry.
        index: usize,
        /// The out-of-range level.
        level: u32,
        /// The pack's `level_count`.
        level_count: u32,
    },
    /// A blob offset is not 16-byte aligned.
    #[error("pack entry {index} {slot:?} blob offset {offset} is not 16-aligned")]
    NotAligned {
        /// Index of the offending entry.
        index: usize,
        /// Which blob slot.
        slot: BlobSlot,
        /// The misaligned offset.
        offset: u64,
    },
    /// Blob offsets are not strictly ascending in index order.
    #[error("pack entry {index} blob offsets out of order")]
    BlobOrder {
        /// Index of the offending entry.
        index: usize,
    },
    /// Two blob ranges overlap.
    #[error("pack entry {second_index}: blob range overlaps with entry {first_index}")]
    BlobOverlap {
        /// Index of the earlier blob.
        first_index: usize,
        /// Index of the overlapping blob.
        second_index: usize,
    },
    /// A blob range extends past `file_size` or begins before the blob region.
    #[error("pack entry {index} {slot:?} blob ends at {end}, past file_size {file_size}")]
    BlobBeyondEof {
        /// Index of the offending entry.
        index: usize,
        /// Which blob slot.
        slot: BlobSlot,
        /// The blob's end offset.
        end: u64,
        /// The pack's `file_size`.
        file_size: u64,
    },
    /// A non-zero inter-blob padding byte.
    #[error("pack inter-blob padding at offset {offset} is not zero")]
    InterBlobPadding {
        /// Offset of the non-zero padding byte.
        offset: u64,
    },
    /// A texture slot inconsistent with `content_kind` (kind 1 with a texture,
    /// or kind 0 with an empty texture).
    #[error("pack entry {index} texture slot inconsistent with content_kind {content_kind}")]
    TextureConsistency {
        /// Index of the offending entry.
        index: usize,
        /// The pack's `content_kind`.
        content_kind: u32,
    },
    /// A `.hf` chunk header's region does not match its entry (horizontally
    /// bit-equal / height-contained). Only from the archive layer's tile
    /// decode, never `open`.
    #[error("pack entry {index}: chunk-header region does not match the entry region")]
    BlobContentMismatch {
        /// Index of the offending entry.
        index: usize,
    },
    /// A blob's SHA-256 did not match its hash-section record.
    #[error("pack entry {index} {slot:?} blob SHA-256 mismatch")]
    BlobHashMismatch {
        /// Index of the offending entry.
        index: usize,
        /// Which blob slot.
        slot: BlobSlot,
    },
    /// A `content_kind == 1` tile's `texture_sha256` was not 32 zero bytes.
    #[error("pack entry {index} texture SHA-256 is not the zero sentinel")]
    TextureHashNotZero {
        /// Index of the offending entry.
        index: usize,
    },
    /// `dataset_id` did not equal SHA-256 of the hash section.
    #[error("pack dataset_id mismatch")]
    DatasetIdMismatch {
        /// The `dataset_id` stored in the header.
        expected: [u8; 32],
        /// The digest computed over the hash section.
        computed: [u8; 32],
    },
}
