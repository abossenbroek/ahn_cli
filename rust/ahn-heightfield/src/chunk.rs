//! The `.hf` heightfield chunk decoder (format version 3).
//!
//! A chunk is a fixed 120-byte little-endian header immediately followed by
//! exactly one zstandard frame that decompresses to the tile's row-major
//! `uint16` quantized height plane. [`ChunkHeader::parse`] validates the
//! header without decompressing; [`Heightfield::decode`] additionally
//! decompresses and validates the plane.
//!
//! Every height in a chunk — the stored plane, `z_offset`, and
//! `region[4]/[5]` — is a **NAP** height (EPSG:5709), never ellipsoidal; see
//! [`NAP_VERTICAL_DATUM`] and the spec's *Coordinate contract*.
//!
//! The normative byte layout is
//! `docs/specs/2026-07-12-heightfield-chunk-format.md`.

use crate::error::{Format, HfError};

/// The chunk magic, ASCII `AHNH`.
pub const CHUNK_MAGIC: &[u8; 4] = b"AHNH";
/// The chunk format version this crate decodes.
pub const CHUNK_VERSION: u32 = 3;
/// The fixed chunk-header length, in bytes.
pub const CHUNK_HEADER_LEN: usize = 120;
/// The maximum stored quantization level (12-bit range).
pub const MAX_QUANTIZED_LEVEL: u16 = 4095;
/// The absolute round-trip-error cap on the height axis, in metres.
pub const ABSOLUTE_ERROR_CAP_M: f64 = 0.025;
/// The only `vertical_datum` value a v3 chunk carries: EPSG:5709, NAP height
/// (*Normaal Amsterdams Peil*, the Dutch national vertical datum). Every
/// height in the chunk — the stored plane, `z_offset`, `region[4]/[5]` — is
/// in this datum; see the spec's *Coordinate contract*.
pub const NAP_VERTICAL_DATUM: u32 = 5709;

/// `header_crc32` covers header bytes `[0, 116)` (v3: grown from v2's
/// `[0, 112)` to also cover `vertical_datum`).
const CHUNK_CRC_SPAN: usize = 116;

// ---- little-endian primitive reads (private; explicit, no casts) ----------
//
// Every caller has already checked the slice is at least `CHUNK_HEADER_LEN`
// long and only reads fixed offsets within `[0, 120)`, so the fixed-size
// slice conversions never fail.

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

/// A parsed, validated `.hf` chunk header.
///
/// A read-only view of the header's wire fields, with no invariant beyond what
/// [`ChunkHeader::parse`] already validated, so its fields are public. It is
/// `#[non_exhaustive]`: a future format revision can add a field without that
/// being a breaking change for readers.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChunkHeader {
    /// Format version (always [`CHUNK_VERSION`] for a valid header).
    pub version: u32,
    /// Vertex-grid column count.
    pub width: u32,
    /// Vertex-grid row count.
    pub height: u32,
    /// Height-axis quantizer translation, in metres.
    pub z_offset: f64,
    /// Height-axis quantizer scale, in metres per level.
    pub z_scale: f64,
    /// ECEF y-up RTC centre of the A-profile tile (convenience anchor only).
    pub rtc_centre: [f64; 3],
    /// The tile's own mesh region, `(west, south, east, north, minH, maxH)`;
    /// longitudes/latitudes in radians (EPSG:4979), heights in **NAP** metres
    /// (EPSG:5709 — see [`NAP_VERTICAL_DATUM`]), the same datum as the
    /// stored plane.
    pub region: [f64; 6],
    /// Byte length of the zstandard frame that follows the header.
    pub payload_len: u64,
    /// EPSG CRS code of the height datum; always [`NAP_VERTICAL_DATUM`] for a
    /// valid header (any other value is a decode error).
    pub vertical_datum: u32,
}

impl ChunkHeader {
    /// Parses and validates a v3 `.hf` header without decompressing the
    /// payload.
    ///
    /// `header_crc32` is verified **before** `width`/`height`/`payload_len`
    /// (or `vertical_datum`) are trusted, so a corrupt header can never drive
    /// a giant allocation or silently mislabel the plane's datum.
    ///
    /// # Errors
    ///
    /// Returns [`HfError::TooShort`], [`HfError::BadMagic`],
    /// [`HfError::BadVersion`], [`HfError::HeaderCrc`],
    /// [`HfError::BadVerticalDatum`], [`HfError::ZeroDimension`],
    /// [`HfError::NonFiniteHeaderField`], or [`HfError::NonPositiveZScale`].
    ///
    /// # Examples
    ///
    /// ```
    /// use ahn_heightfield::{ChunkHeader, CHUNK_VERSION};
    ///
    /// let bytes = include_bytes!("../tests/data/leaf.hf");
    /// let header = ChunkHeader::parse(bytes)?;
    /// assert_eq!(header.version, CHUNK_VERSION);
    /// assert_eq!((header.width, header.height), (6, 6));
    /// # Ok::<(), ahn_heightfield::HfError>(())
    /// ```
    pub fn parse(bytes: &[u8]) -> Result<Self, HfError> {
        if bytes.len() < CHUNK_HEADER_LEN {
            return Err(HfError::TooShort {
                format: Format::Chunk,
                minimum: CHUNK_HEADER_LEN,
                actual: bytes.len(),
            });
        }
        let magic: [u8; 4] = [bytes[0], bytes[1], bytes[2], bytes[3]];
        if &magic != CHUNK_MAGIC {
            return Err(HfError::BadMagic {
                format: Format::Chunk,
                expected: *CHUNK_MAGIC,
                found: magic,
            });
        }
        let version = le_u32(bytes, 4);
        if version != CHUNK_VERSION {
            return Err(HfError::BadVersion {
                format: Format::Chunk,
                expected: CHUNK_VERSION,
                found: version,
            });
        }
        // CRC over [0, 116) is verified before width/height/payload_len (or
        // vertical_datum) are trusted or used to size any allocation.
        let stored_crc = le_u32(bytes, 116);
        let computed_crc = crc32fast::hash(&bytes[..CHUNK_CRC_SPAN]);
        if computed_crc != stored_crc {
            return Err(HfError::HeaderCrc {
                format: Format::Chunk,
                expected: stored_crc,
                computed: computed_crc,
            });
        }
        let vertical_datum = le_u32(bytes, 112);
        if vertical_datum != NAP_VERTICAL_DATUM {
            return Err(HfError::BadVerticalDatum {
                expected: NAP_VERTICAL_DATUM,
                found: vertical_datum,
            });
        }
        let width = le_u32(bytes, 8);
        let height = le_u32(bytes, 12);
        if width == 0 || height == 0 {
            return Err(HfError::ZeroDimension { width, height });
        }
        let z_offset = le_f64(bytes, 16);
        let z_scale = le_f64(bytes, 24);
        let rtc_centre = [le_f64(bytes, 32), le_f64(bytes, 40), le_f64(bytes, 48)];
        let region = [
            le_f64(bytes, 56),
            le_f64(bytes, 64),
            le_f64(bytes, 72),
            le_f64(bytes, 80),
            le_f64(bytes, 88),
            le_f64(bytes, 96),
        ];
        // Any non-finite float in the header poisons downstream geodesy /
        // dequant math and is rejected.
        for (field, value) in [
            ("z_offset", z_offset),
            ("z_scale", z_scale),
            ("rtc_centre[0]", rtc_centre[0]),
            ("rtc_centre[1]", rtc_centre[1]),
            ("rtc_centre[2]", rtc_centre[2]),
            ("region[0]", region[0]),
            ("region[1]", region[1]),
            ("region[2]", region[2]),
            ("region[3]", region[3]),
            ("region[4]", region[4]),
            ("region[5]", region[5]),
        ] {
            if !value.is_finite() {
                return Err(HfError::NonFiniteHeaderField { field, value });
            }
        }
        if z_scale <= 0.0 {
            return Err(HfError::NonPositiveZScale { z_scale });
        }
        Ok(ChunkHeader {
            version,
            width,
            height,
            z_offset,
            z_scale,
            rtc_centre,
            region,
            payload_len: le_u64(bytes, 104),
            vertical_datum,
        })
    }

    /// Expected decompressed payload length, `width * height * 2`, always
    /// evaluated in 64-bit arithmetic.
    ///
    /// A CRC-consistent but corrupt header can name dims whose product exceeds
    /// `u64`; the multiplication saturates to [`u64::MAX`] rather than
    /// overflowing, which still rejects cleanly on the length check (no real
    /// plane is that long) instead of panicking in a debug build.
    #[must_use]
    pub fn plane_len(&self) -> u64 {
        u64::from(self.width)
            .saturating_mul(u64::from(self.height))
            .saturating_mul(2)
    }

    /// `true` if this header's exported round-trip bound (`z_scale / 2`)
    /// exceeds the [`ABSOLUTE_ERROR_CAP_M`] cap.
    ///
    /// [`Heightfield::decode`] never enforces this (a lightweight runtime may
    /// skip it); call this or [`ChunkHeader::check_error_cap`] to enforce it.
    #[must_use]
    pub fn exceeds_error_cap(&self) -> bool {
        self.z_scale / 2.0 > ABSOLUTE_ERROR_CAP_M
    }

    /// [`ChunkHeader::exceeds_error_cap`] as a `Result`, for callers that want
    /// the cap enforced with the same error type as `decode`.
    ///
    /// # Errors
    ///
    /// Returns [`HfError::ErrorCapExceeded`] if `z_scale / 2` exceeds
    /// [`ABSOLUTE_ERROR_CAP_M`].
    pub fn check_error_cap(&self) -> Result<(), HfError> {
        if self.exceeds_error_cap() {
            return Err(HfError::ErrorCapExceeded {
                z_scale: self.z_scale,
                bound_m: self.z_scale / 2.0,
            });
        }
        Ok(())
    }
}

impl TryFrom<&[u8]> for ChunkHeader {
    type Error = HfError;

    fn try_from(bytes: &[u8]) -> Result<Self, HfError> {
        Self::parse(bytes)
    }
}

/// A fully decoded `.hf` chunk: its header plus the `width * height` `uint16`
/// quantized height plane.
///
/// A behavioural type: it holds the cross-field invariant `levels.len() ==
/// width * height` (enforced at decode) and exposes computed accessors, so its
/// fields are private.
#[derive(Debug, Clone, PartialEq)]
pub struct Heightfield {
    header: ChunkHeader,
    levels: Vec<u16>,
}

impl Heightfield {
    /// Fully decodes a v3 `.hf` chunk: header, then the zstd frame (with native
    /// content-checksum verification), then the exact `width * height` `uint16`
    /// plane.
    ///
    /// Does **not** enforce the absolute-error cap; see
    /// [`ChunkHeader::check_error_cap`].
    ///
    /// # Errors
    ///
    /// Returns any error [`ChunkHeader::parse`] can, plus
    /// [`HfError::PayloadLengthMismatch`], [`HfError::Zstd`],
    /// [`HfError::PlaneLengthMismatch`], or [`HfError::LevelOutOfRange`].
    ///
    /// # Examples
    ///
    /// ```
    /// use ahn_heightfield::Heightfield;
    ///
    /// let bytes = include_bytes!("../tests/data/leaf.hf");
    /// let tile = Heightfield::decode(bytes)?;
    /// assert_eq!(tile.levels().len(), (tile.width() * tile.height()) as usize);
    /// let h = tile.dequantize_at(0, 0);
    /// assert!(h.is_finite());
    /// # Ok::<(), ahn_heightfield::HfError>(())
    /// ```
    pub fn decode(bytes: &[u8]) -> Result<Self, HfError> {
        let header = ChunkHeader::parse(bytes)?;
        let frame = &bytes[CHUNK_HEADER_LEN..];
        if frame.len() as u64 != header.payload_len {
            return Err(HfError::PayloadLengthMismatch {
                expected: header.payload_len,
                actual: frame.len() as u64,
            });
        }
        let raw = zstd::decode_all(frame).map_err(HfError::Zstd)?;
        let expected = header.plane_len();
        if raw.len() as u64 != expected {
            return Err(HfError::PlaneLengthMismatch {
                expected,
                actual: raw.len() as u64,
            });
        }
        let levels: Vec<u16> = raw
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        // Free integrity check: every stored level is within the 12-bit range.
        for (i, &level) in levels.iter().enumerate() {
            if level > MAX_QUANTIZED_LEVEL {
                let i = i as u32;
                return Err(HfError::LevelOutOfRange {
                    row: i / header.width,
                    col: i % header.width,
                    level,
                });
            }
        }
        Ok(Heightfield { header, levels })
    }

    /// The parsed header.
    #[must_use]
    pub fn header(&self) -> &ChunkHeader {
        &self.header
    }

    /// Vertex-grid column count.
    #[must_use]
    pub fn width(&self) -> u32 {
        self.header.width
    }

    /// Vertex-grid row count.
    #[must_use]
    pub fn height(&self) -> u32 {
        self.header.height
    }

    /// Row-major raw quantized levels, top row first — `levels()[r * width + c]`.
    #[must_use]
    pub fn levels(&self) -> &[u16] {
        &self.levels
    }

    /// Raw quantized level at `(row, col)`.
    ///
    /// # Panics
    ///
    /// Panics if `row >= self.height()` or `col >= self.width()`.
    #[must_use]
    pub fn level_at(&self, row: u32, col: u32) -> u16 {
        self.levels[self.flat_index(row, col)]
    }

    /// Dequantized NAP height at `(row, col)`: `level * z_scale + z_offset`.
    ///
    /// # Panics
    ///
    /// Panics if `row >= self.height()` or `col >= self.width()`.
    #[must_use]
    pub fn dequantize_at(&self, row: u32, col: u32) -> f64 {
        f64::from(self.level_at(row, col)) * self.header.z_scale + self.header.z_offset
    }

    fn flat_index(&self, row: u32, col: u32) -> usize {
        assert!(
            row < self.header.height && col < self.header.width,
            "index ({row}, {col}) out of range for {}x{} grid",
            self.header.width,
            self.header.height,
        );
        (row * self.header.width + col) as usize
    }
}
