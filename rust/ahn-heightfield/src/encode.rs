//! The optional `encode` feature: a v2 `.hf` chunk **encoder**.
//!
//! Available only with the `encode` feature enabled. This is the inverse of
//! [`crate::Heightfield::decode`] for the chunk layer: it quantizes a plane of
//! source NAP heights and writes a complete, self-consistent `.hf` chunk that
//! this crate (and the Python reference decoder) reads back.
//!
//! # Determinism scope
//!
//! This encoder is held to **semantic round-trip equality**, not byte parity
//! with the Python producer: `decode(encode(x))` reproduces the identical
//! quantized levels and header fields, but the exact compressed frame bytes may
//! differ from the Python frame (libzstd builds and encoder versions legitimately
//! differ). Both normative specs make byte-for-byte determinism a property of
//! the Python producer alone; the Rust side is contracted to semantic
//! equivalence. See the chunk spec's *Producer byte-determinism scope*.
//!
//! Every other producer rule **is** followed exactly: 12-bit round-half-even
//! quantization (via [`f64::round_ties_even`], the spec's normative rounding),
//! the `1e-9` epsilon scale for a flat tile, the 25 mm absolute-error-cap
//! refusal, a single one-shot zstandard frame at level 3 with the RFC 8878
//! content checksum and embedded content size, and a `header_crc32` signed over
//! bytes `[0, 112)`.

use crate::chunk::{
    ABSOLUTE_ERROR_CAP_M, CHUNK_HEADER_LEN, CHUNK_MAGIC, CHUNK_VERSION, MAX_QUANTIZED_LEVEL,
};
use crate::error::HfError;

/// The pinned zstandard compression level (chunk spec: level 3, one-shot).
const ZSTD_LEVEL: i32 = 3;
/// `header_crc32` is signed over header bytes `[0, 112)` (all fields through
/// `payload_len`, excluding `header_crc32` and `pad`).
const CRC_SPAN: usize = CHUNK_HEADER_LEN - 8;

/// The header fields and source heights for one `.hf` chunk to encode.
///
/// Available only with the `encode` feature enabled.
///
/// `heights` is the tile's `width * height` source NAP heights (EPSG:7415
/// vertical / metres), row-major with the **top row first**, matching the
/// decoded [`crate::Heightfield::levels`] layout. The quantizer (`z_offset`,
/// `z_scale`) is derived from `heights` by [`quantize_levels`]; the caller
/// supplies only the grid dimensions and the geometry metadata.
///
/// All `heights` values must be finite: the encoder has no reject variant for a
/// non-finite source sample (the decoder never needs one, as heights become
/// `uint16` levels), so a non-finite height is a caller contract violation, not
/// a returned error. The header floats (`rtc_centre`, `region`) *are* validated.
#[derive(Debug, Clone, Copy)]
pub struct ChunkFields<'a> {
    /// Vertex-grid column count (must be non-zero).
    pub width: u32,
    /// Vertex-grid row count (must be non-zero).
    pub height: u32,
    /// ECEF y-up RTC centre of the A-profile tile (convenience anchor).
    pub rtc_centre: [f64; 3],
    /// The tile's own mesh region `(west, south, east, north, minH, maxH)` —
    /// longitudes/latitudes in radians (EPSG:4979), heights in metres.
    pub region: [f64; 6],
    /// The `width * height` source NAP heights, row-major, top row first.
    pub heights: &'a [f64],
}

/// Quantizes source NAP heights to 12-bit levels with the spec's affine scheme.
///
/// Available only with the `encode` feature enabled.
///
/// Returns `(z_offset, z_scale, levels)` where `z_offset = min(heights)`,
/// `z_scale = (max - min) / 4095` (or the `1e-9` epsilon scale for a flat tile),
/// and each level is `clip(round_ties_even((h - z_offset) / z_scale), 0, 4095)`.
/// Rounding is **round-half-even** ([`f64::round_ties_even`]), applied before
/// the clip — not truncation, not round-half-away-from-zero — so a value exactly
/// halfway between two levels goes to the even neighbour, matching the Python
/// producer's `numpy.rint` bit for bit.
///
/// Dequantization is `h' = level * z_scale + z_offset`, reproduced by
/// [`crate::Heightfield::dequantize_at`].
///
/// # Panics
///
/// Panics if `heights` is empty (a valid tile has at least one sample). All
/// values are assumed finite (see [`ChunkFields`]); a non-finite input yields
/// an unspecified result rather than a panic.
///
/// # Examples
///
/// ```
/// use ahn_heightfield::quantize_levels;
///
/// // A flat tile stores all zeros and keeps a non-zero (epsilon) scale.
/// let (z_offset, z_scale, levels) = quantize_levels(&[3.0, 3.0, 3.0, 3.0]);
/// assert_eq!(z_offset, 3.0);
/// assert!(z_scale > 0.0);
/// assert_eq!(levels, vec![0, 0, 0, 0]);
/// ```
#[must_use]
pub fn quantize_levels(heights: &[f64]) -> (f64, f64, Vec<u16>) {
    assert!(
        !heights.is_empty(),
        "quantize_levels requires at least one height",
    );
    let mut min = heights[0];
    let mut max = heights[0];
    for &h in &heights[1..] {
        if h < min {
            min = h;
        }
        if h > max {
            max = h;
        }
    }
    let extent = max - min;
    let max_level = f64::from(MAX_QUANTIZED_LEVEL);
    let z_scale = if extent > 0.0 {
        extent / max_level
    } else {
        1e-9
    };
    let levels = heights
        .iter()
        .map(|&h| {
            let scaled = ((h - min) / z_scale)
                .round_ties_even()
                .clamp(0.0, max_level);
            // `scaled` is an exact non-negative integer in `[0, 4095]`, so the
            // cast is lossless and never saturates.
            scaled as u16
        })
        .collect();
    (min, z_scale, levels)
}

/// Encodes a complete v2 `.hf` chunk: quantizes `fields.heights`, compresses the
/// `uint16` plane into one zstandard frame (level 3, content checksum + embedded
/// size), and writes the 120-byte header with a signed `header_crc32`.
///
/// Available only with the `encode` feature enabled.
///
/// The result is a self-consistent chunk that [`crate::Heightfield::decode`]
/// accepts and round-trips: the decoded header fields equal `fields` (dimensions,
/// `rtc_centre`, `region`, the derived `z_offset`/`z_scale`) and the decoded
/// levels equal [`quantize_levels`]`(fields.heights).2`.
///
/// # Errors
///
/// Returns [`HfError::ZeroDimension`] if `width == 0` or `height == 0`;
/// [`HfError::PlaneLengthMismatch`] if `heights.len() != width * height`;
/// [`HfError::NonFiniteHeaderField`] if any `rtc_centre` or `region` value is
/// non-finite; [`HfError::ErrorCapExceeded`] if the derived `z_scale / 2`
/// exceeds the [`ABSOLUTE_ERROR_CAP_M`] cap; or [`HfError::Io`] if the
/// zstandard encoder fails.
///
/// # Examples
///
/// ```
/// use ahn_heightfield::{encode_chunk, ChunkFields, Heightfield};
///
/// let heights = [0.0, 0.5, 1.0, 1.5];
/// let bytes = encode_chunk(ChunkFields {
///     width: 2,
///     height: 2,
///     rtc_centre: [0.0, 0.0, 0.0],
///     region: [0.1, 0.2, 0.3, 0.4, 0.0, 1.5],
///     heights: &heights,
/// })?;
/// let tile = Heightfield::decode(&bytes)?;
/// assert_eq!(tile.width(), 2);
/// assert_eq!(tile.dequantize_at(0, 0), 0.0);
/// # Ok::<(), ahn_heightfield::HfError>(())
/// ```
pub fn encode_chunk(fields: ChunkFields<'_>) -> Result<Vec<u8>, HfError> {
    let ChunkFields {
        width,
        height,
        rtc_centre,
        region,
        heights,
    } = fields;

    if width == 0 || height == 0 {
        return Err(HfError::ZeroDimension { width, height });
    }
    let sample_count = u64::from(width) * u64::from(height);
    if heights.len() as u64 != sample_count {
        return Err(HfError::PlaneLengthMismatch {
            expected: sample_count * 2,
            actual: heights.len() as u64 * 2,
        });
    }
    // Reject non-finite header floats so the encoder can never emit a chunk its
    // own decoder would reject on these fields.
    for (field, value) in [
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

    let (z_offset, z_scale, levels) = quantize_levels(heights);
    let bound_m = z_scale / 2.0;
    if bound_m > ABSOLUTE_ERROR_CAP_M {
        return Err(HfError::ErrorCapExceeded { z_scale, bound_m });
    }

    let mut raw = Vec::with_capacity(levels.len() * 2);
    for &level in &levels {
        raw.extend_from_slice(&level.to_le_bytes());
    }
    let frame = compress_frame(&raw)?;

    let mut out = vec![0u8; CHUNK_HEADER_LEN];
    out[0..4].copy_from_slice(CHUNK_MAGIC);
    out[4..8].copy_from_slice(&CHUNK_VERSION.to_le_bytes());
    out[8..12].copy_from_slice(&width.to_le_bytes());
    out[12..16].copy_from_slice(&height.to_le_bytes());
    out[16..24].copy_from_slice(&z_offset.to_le_bytes());
    out[24..32].copy_from_slice(&z_scale.to_le_bytes());
    for (i, v) in rtc_centre.iter().enumerate() {
        let off = 32 + i * 8;
        out[off..off + 8].copy_from_slice(&v.to_le_bytes());
    }
    for (i, v) in region.iter().enumerate() {
        let off = 56 + i * 8;
        out[off..off + 8].copy_from_slice(&v.to_le_bytes());
    }
    out[104..112].copy_from_slice(&(frame.len() as u64).to_le_bytes());
    // `pad` at [116, 120) stays zero. Sign header_crc32 over [0, 112) last.
    let crc = crc32fast::hash(&out[..CRC_SPAN]);
    out[112..116].copy_from_slice(&crc.to_le_bytes());

    out.extend_from_slice(&frame);
    Ok(out)
}

/// One-shot zstandard compression at level 3 with the RFC 8878 content checksum
/// and the embedded content size both written, matching the chunk spec's decode
/// contract (`decode_all` reaches and verifies the checksum epilogue).
fn compress_frame(raw: &[u8]) -> Result<Vec<u8>, HfError> {
    use zstd::zstd_safe::CParameter;

    let mut compressor = zstd::bulk::Compressor::new(ZSTD_LEVEL)?;
    compressor.set_parameter(CParameter::ChecksumFlag(true))?;
    compressor.set_parameter(CParameter::ContentSizeFlag(true))?;
    // Map any zstd encode failure through the shared `HfError::Io` catch-all —
    // there is no encode-side reject variant, and this path is unreachable for
    // valid input.
    Ok(compressor.compress(raw)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_half_even_matches_the_spec_examples() {
        // The spec's worked examples: 2.5 -> 2 (even), 3.5 -> 4 (even). Build a
        // two-level ramp so the scaled value lands exactly on a .5 boundary.
        assert_eq!(2.5_f64.round_ties_even(), 2.0);
        assert_eq!(3.5_f64.round_ties_even(), 4.0);
    }

    #[test]
    fn full_range_maps_endpoints_to_zero_and_max() {
        let (_, _, levels) = quantize_levels(&[0.0, 4095.0]);
        assert_eq!(levels, vec![0, MAX_QUANTIZED_LEVEL]);
    }
}
