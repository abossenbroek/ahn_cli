//! One negative per normative chunk-decode reject. Each input is a valid chunk
//! with exactly one field corrupted (CRC re-signed where the target lies inside
//! the CRC span), so only the intended reject fires.

mod common;

use ahn_heightfield::{ChunkHeader, Format, Heightfield, HfError};

/// The committed 6x6 leaf chunk — a known-good base to corrupt.
const LEAF: &[u8] = include_bytes!("data/leaf.hf");

fn leaf() -> Vec<u8> {
    LEAF.to_vec()
}

#[test]
fn too_short() {
    let err = Heightfield::decode(&LEAF[..119]).unwrap_err();
    assert!(matches!(
        err,
        HfError::TooShort {
            format: Format::Chunk,
            minimum: 120,
            actual: 119,
        }
    ));
}

#[test]
fn bad_magic() {
    let mut b = leaf();
    b[0] = b'X';
    // Magic is checked before the CRC, so no re-sign is needed.
    let err = Heightfield::decode(&b).unwrap_err();
    assert!(matches!(
        err,
        HfError::BadMagic {
            format: Format::Chunk,
            ..
        }
    ));
}

#[test]
fn bad_version() {
    let mut b = leaf();
    b[4..8].copy_from_slice(&3u32.to_le_bytes());
    let err = Heightfield::decode(&b).unwrap_err();
    assert!(matches!(
        err,
        HfError::BadVersion {
            format: Format::Chunk,
            expected: 2,
            found: 3,
        }
    ));
}

#[test]
fn header_crc_mismatch() {
    let mut b = leaf();
    // Flip a byte inside the CRC span [0, 112) without re-signing.
    b[56] ^= 0xFF;
    let err = ChunkHeader::parse(&b).unwrap_err();
    assert!(matches!(
        err,
        HfError::HeaderCrc {
            format: Format::Chunk,
            ..
        }
    ));
}

#[test]
fn reserved_pad_non_zero() {
    let mut b = leaf();
    // `pad` at offset 116 is outside the CRC span, so the CRC stays valid.
    b[116] = 1;
    let err = ChunkHeader::parse(&b).unwrap_err();
    assert!(matches!(
        err,
        HfError::ReservedNonZero {
            format: Format::Chunk,
            byte_offset: 116,
        }
    ));
}

#[test]
fn zero_dimension() {
    let mut b = leaf();
    b[8..12].copy_from_slice(&0u32.to_le_bytes());
    common::resign_chunk_crc(&mut b);
    let err = ChunkHeader::parse(&b).unwrap_err();
    assert!(matches!(
        err,
        HfError::ZeroDimension {
            width: 0,
            height: 6,
        }
    ));
}

#[test]
fn non_finite_header_field() {
    let mut b = leaf();
    b[16..24].copy_from_slice(&f64::NAN.to_le_bytes());
    common::resign_chunk_crc(&mut b);
    let err = ChunkHeader::parse(&b).unwrap_err();
    assert!(matches!(
        err,
        HfError::NonFiniteHeaderField {
            field: "z_offset",
            ..
        }
    ));
}

#[test]
fn non_positive_z_scale() {
    let mut b = leaf();
    b[24..32].copy_from_slice(&0.0f64.to_le_bytes());
    common::resign_chunk_crc(&mut b);
    let err = ChunkHeader::parse(&b).unwrap_err();
    assert!(matches!(err, HfError::NonPositiveZScale { z_scale } if z_scale == 0.0));
}

#[test]
fn payload_length_mismatch() {
    let mut b = leaf();
    let actual_frame = (b.len() - 120) as u64;
    b[104..112].copy_from_slice(&(actual_frame + 1).to_le_bytes());
    common::resign_chunk_crc(&mut b);
    let err = Heightfield::decode(&b).unwrap_err();
    assert!(matches!(
        err,
        HfError::PayloadLengthMismatch { expected, actual }
            if expected == actual_frame + 1 && actual == actual_frame
    ));
}

#[test]
fn zstd_frame_corrupt() {
    let mut b = leaf();
    // Flip a byte inside the frame (offset >= 120); the header CRC is
    // unaffected, so the zstd content checksum is what catches it.
    let last = b.len() - 1;
    b[last] ^= 0xFF;
    let err = Heightfield::decode(&b).unwrap_err();
    assert!(matches!(err, HfError::Zstd(_)));
}

#[test]
fn plane_length_mismatch() {
    let mut b = leaf();
    // Widen width 6 -> 7: the frame still decodes to 6*6*2 bytes but the
    // header now claims 7*6*2.
    b[8..12].copy_from_slice(&7u32.to_le_bytes());
    common::resign_chunk_crc(&mut b);
    let err = Heightfield::decode(&b).unwrap_err();
    assert!(matches!(
        err,
        HfError::PlaneLengthMismatch {
            expected: 84,
            actual: 72,
        }
    ));
}

#[test]
fn level_out_of_range() {
    // A synthetic 2x1 chunk whose first level exceeds the 12-bit maximum.
    let hf = common::synth_chunk(2, 1, 0.0, 1.0, &[4096, 0]);
    let err = Heightfield::decode(&hf).unwrap_err();
    assert!(matches!(
        err,
        HfError::LevelOutOfRange {
            row: 0,
            col: 0,
            level: 4096,
        }
    ));
}

#[test]
fn decode_from_unaligned_slice_matches_aligned() {
    let aligned = Heightfield::decode(LEAF).expect("aligned decode");
    let mut padded = vec![0u8];
    padded.extend_from_slice(LEAF);
    let unaligned = Heightfield::decode(&padded[1..]).expect("unaligned decode");
    assert_eq!(aligned.header(), unaligned.header());
    assert_eq!(aligned.levels(), unaligned.levels());
}

#[test]
fn giant_dims_do_not_allocate() {
    let mut b = leaf();
    // A CRC-consistent header declaring width = height = u32::MAX must be
    // rejected on the length check, never driving a width*height*2 allocation.
    b[8..12].copy_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
    b[12..16].copy_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
    common::resign_chunk_crc(&mut b);
    // The header itself parses (CRC valid, dims non-zero, fields finite), and
    // plane_len saturates instead of overflowing.
    let header = ChunkHeader::parse(&b).expect("header parses");
    assert_eq!(header.plane_len(), u64::MAX);
    // Decode rejects on the plane-length check without a huge allocation, so
    // this test completing near-instantly is the practical proof.
    let err = Heightfield::decode(&b).unwrap_err();
    assert!(matches!(err, HfError::PlaneLengthMismatch { .. }));
}

#[test]
fn error_cap_is_opt_in() {
    // A synthetic chunk with z_scale = 1.0 has bound 0.5 m, far past the cap.
    let hf = common::synth_chunk(1, 1, 0.0, 1.0, &[0]);
    // decode never enforces the cap.
    let tile = Heightfield::decode(&hf).expect("decode ignores the cap");
    assert!(tile.header().exceeds_error_cap());
    // The explicit check does enforce it.
    let err = tile.header().check_error_cap().unwrap_err();
    assert!(matches!(
        err,
        HfError::ErrorCapExceeded { z_scale, bound_m }
            if z_scale == 1.0 && bound_m == 0.5
    ));
}
