//! One negative per normative pack-reader reject, plus the truncation matrix.
//!
//! Each case starts from a pristine committed pack, corrupts exactly one field
//! with exact-byte edits, and re-signs the CRC(s) whose span the edit lies
//! inside (via `common::resign_pack`) so that only the intended reject fires.
//! Edits outside every CRC span (blob-region bytes) need no re-sign.

mod common;

use ahn_heightfield::{Archive, BlobSlot, Format, HfError};

fn hf() -> Vec<u8> {
    common::read_pack("heightfield")
}

fn game() -> Vec<u8> {
    common::read_pack("game")
}

fn open_err(bytes: &[u8]) -> HfError {
    match Archive::open(bytes) {
        Ok(_) => panic!("expected open to reject this pack"),
        Err(e) => e,
    }
}

// ---- header rejects -------------------------------------------------------

#[test]
fn too_short() {
    let pack = hf();
    let err = open_err(&pack[..127]);
    assert!(matches!(
        err,
        HfError::TooShort {
            format: Format::Pack,
            minimum: 128,
            actual: 127,
        }
    ));
}

#[test]
fn bad_magic() {
    let mut p = hf();
    p[0] = b'X'; // magic is checked before the CRC — no re-sign.
    assert!(matches!(
        open_err(&p),
        HfError::BadMagic {
            format: Format::Pack,
            ..
        }
    ));
}

#[test]
fn bad_version() {
    let mut p = hf();
    p[4..8].copy_from_slice(&2u32.to_le_bytes()); // version before CRC — no re-sign.
    assert!(matches!(
        open_err(&p),
        HfError::BadVersion {
            format: Format::Pack,
            expected: 1,
            found: 2,
        }
    ));
}

#[test]
fn header_crc_mismatch() {
    let mut p = hf();
    p[64] ^= 0xFF; // flip a dataset_id byte inside [0,124) without re-signing.
    assert!(matches!(
        open_err(&p),
        HfError::HeaderCrc {
            format: Format::Pack,
            ..
        }
    ));
}

#[test]
fn degenerate_counts() {
    let mut p = hf();
    p[8..12].copy_from_slice(&0u32.to_le_bytes()); // tile_count = 0
    common::resign_pack(&mut p);
    assert!(matches!(
        open_err(&p),
        HfError::DegenerateCounts {
            tile_count: 0,
            level_count: 2,
        }
    ));
}

#[test]
fn non_finite_root_geometric_error() {
    let mut p = hf();
    p[56..64].copy_from_slice(&f64::NAN.to_le_bytes());
    common::resign_pack(&mut p);
    assert!(matches!(
        open_err(&p),
        HfError::NonFiniteRootGeometricError { .. }
    ));
}

#[test]
fn reserved_non_zero() {
    let mut p = hf();
    p[100] = 1; // reserved @ 100
    common::resign_pack(&mut p);
    assert!(matches!(
        open_err(&p),
        HfError::ReservedNonZero {
            format: Format::Pack,
            byte_offset: 100,
        }
    ));
}

#[test]
fn pad_non_zero() {
    let mut p = hf();
    p[115] = 1; // a byte of pad [108,124)
    common::resign_pack(&mut p);
    assert!(matches!(
        open_err(&p),
        HfError::ReservedNonZero {
            format: Format::Pack,
            byte_offset: 115,
        }
    ));
}

#[test]
fn bad_content_kind() {
    let mut p = hf();
    p[104..108].copy_from_slice(&2u32.to_le_bytes());
    common::resign_pack(&mut p);
    assert!(matches!(open_err(&p), HfError::BadContentKind { found: 2 }));
}

#[test]
fn index_offset_wrong() {
    let mut p = hf();
    p[16..24].copy_from_slice(&129u64.to_le_bytes());
    common::resign_pack(&mut p);
    assert!(matches!(
        open_err(&p),
        HfError::IndexOffset {
            expected: 128,
            found: 129
        }
    ));
}

#[test]
fn index_size_wrong() {
    let mut p = hf();
    p[24..32].copy_from_slice(&999u64.to_le_bytes());
    common::resign_pack(&mut p);
    assert!(matches!(
        open_err(&p),
        HfError::IndexSize {
            expected: 512,
            found: 999
        }
    ));
}

#[test]
fn hash_offset_wrong() {
    let mut p = hf();
    p[32..40].copy_from_slice(&999u64.to_le_bytes());
    common::resign_pack(&mut p);
    assert!(matches!(
        open_err(&p),
        HfError::HashOffset {
            expected: 640,
            found: 999
        }
    ));
}

#[test]
fn hash_size_wrong() {
    let mut p = hf();
    p[40..48].copy_from_slice(&999u64.to_le_bytes());
    common::resign_pack(&mut p);
    assert!(matches!(
        open_err(&p),
        HfError::HashSize {
            expected: 320,
            found: 999
        }
    ));
}

#[test]
fn file_size_wrong() {
    let mut p = hf();
    p[48..56].copy_from_slice(&9999u64.to_le_bytes());
    common::resign_pack(&mut p);
    assert!(matches!(
        open_err(&p),
        HfError::FileSize {
            header: 9999,
            actual: 5635
        }
    ));
}

#[test]
fn index_crc_mismatch() {
    let mut p = hf();
    // Flip a byte inside the index region [128,640) without re-signing the
    // index CRC; header CRC is unaffected (byte is outside [0,124)).
    p[300] ^= 0xFF;
    assert!(matches!(open_err(&p), HfError::IndexCrc { .. }));
}

// ---- level-directory rejects ----------------------------------------------

#[test]
fn directory_discontinuity() {
    let mut p = hf();
    p[128..132].copy_from_slice(&1u32.to_le_bytes()); // directory[0].first_entry = 1
    common::resign_pack(&mut p);
    assert!(matches!(
        open_err(&p),
        HfError::DirectoryDiscontinuity {
            level: 0,
            expected_first_entry: 0,
            found: 1,
        }
    ));
}

#[test]
fn entry_count_mismatch() {
    let mut p = hf();
    // Shrink level 1's entry_count 4 -> 3: continuity still holds (level 1 is
    // last), but sum(entry_count) = 4 != tile_count 5.
    p[148..152].copy_from_slice(&3u32.to_le_bytes());
    common::resign_pack(&mut p);
    assert!(matches!(
        open_err(&p),
        HfError::EntryCountMismatch {
            sum: 4,
            tile_count: 5
        }
    ));
}

// ---- index-entry rejects --------------------------------------------------

#[test]
fn not_sorted() {
    let mut p = hf();
    let e1 = common::entry_offset(&p, 1);
    p[e1 + 4..e1 + 8].copy_from_slice(&99u32.to_le_bytes()); // entry 1 tx -> 99
    common::resign_pack(&mut p);
    assert!(matches!(open_err(&p), HfError::NotSorted { index: 2 }));
}

#[test]
fn tz_non_zero() {
    let mut p = hf();
    let e4 = common::entry_offset(&p, 4);
    p[e4 + 12..e4 + 16].copy_from_slice(&1u32.to_le_bytes()); // last entry tz -> 1
    common::resign_pack(&mut p);
    assert!(matches!(
        open_err(&p),
        HfError::TzNonZero { index: 4, tz: 1 }
    ));
}

#[test]
fn non_finite_entry_field() {
    let mut p = hf();
    let e0 = common::entry_offset(&p, 0);
    p[e0 + 16..e0 + 24].copy_from_slice(&f64::INFINITY.to_le_bytes()); // region[0]
    common::resign_pack(&mut p);
    assert!(matches!(
        open_err(&p),
        HfError::NonFiniteEntryField {
            index: 0,
            field: "region[0]",
            ..
        }
    ));
}

#[test]
fn region_not_well_ordered() {
    let mut p = hf();
    let e0 = common::entry_offset(&p, 0);
    p[e0 + 16..e0 + 24].copy_from_slice(&2.0f64.to_le_bytes()); // west = 2
    p[e0 + 32..e0 + 40].copy_from_slice(&1.0f64.to_le_bytes()); // east = 1 (< west)
    common::resign_pack(&mut p);
    assert!(matches!(
        open_err(&p),
        HfError::RegionNotWellOrdered { index: 0 }
    ));
}

#[test]
fn level_out_of_directory() {
    let mut p = hf();
    let e4 = common::entry_offset(&p, 4);
    p[e4..e4 + 4].copy_from_slice(&5u32.to_le_bytes()); // last entry level -> 5
    common::resign_pack(&mut p);
    assert!(matches!(
        open_err(&p),
        HfError::LevelOutOfDirectory {
            index: 4,
            level: 5,
            level_count: 2
        }
    ));
}

#[test]
fn not_aligned_primary() {
    let mut p = hf();
    let e0 = common::entry_offset(&p, 0);
    p[e0 + 72..e0 + 80].copy_from_slice(&961u64.to_le_bytes());
    common::resign_pack(&mut p);
    assert!(matches!(
        open_err(&p),
        HfError::NotAligned {
            index: 0,
            slot: BlobSlot::Primary,
            offset: 961,
        }
    ));
}

#[test]
fn not_aligned_texture() {
    let mut p = hf();
    let e0 = common::entry_offset(&p, 0);
    p[e0 + 80..e0 + 88].copy_from_slice(&1201u64.to_le_bytes());
    common::resign_pack(&mut p);
    assert!(matches!(
        open_err(&p),
        HfError::NotAligned {
            index: 0,
            slot: BlobSlot::Texture,
            offset: 1201,
        }
    ));
}

#[test]
fn blob_order() {
    let mut p = hf();
    let e1 = common::entry_offset(&p, 1);
    p[e1 + 72..e1 + 80].copy_from_slice(&960u64.to_le_bytes()); // entry1 primary -> 960
    common::resign_pack(&mut p);
    assert!(matches!(open_err(&p), HfError::BlobOrder { index: 1 }));
}

#[test]
fn blob_overlap() {
    let mut p = hf();
    let e0 = common::entry_offset(&p, 0);
    // entry0 texture at 1184 lands inside entry0's primary range [960,1191).
    p[e0 + 80..e0 + 88].copy_from_slice(&1184u64.to_le_bytes());
    common::resign_pack(&mut p);
    assert!(matches!(
        open_err(&p),
        HfError::BlobOverlap {
            first_index: 0,
            second_index: 0
        }
    ));
}

#[test]
fn blob_beyond_eof() {
    let mut p = hf();
    let e4 = common::entry_offset(&p, 4);
    p[e4 + 80..e4 + 88].copy_from_slice(&5616u64.to_le_bytes()); // 5616 + 707 > 5635
    common::resign_pack(&mut p);
    assert!(matches!(
        open_err(&p),
        HfError::BlobBeyondEof {
            index: 4,
            slot: BlobSlot::Texture,
            end: 6323,
            file_size: 5635,
        }
    ));
}

#[test]
fn inter_blob_padding_non_zero() {
    let mut p = hf();
    // A byte of the zero padding between entry 0's primary (ends 1191) and its
    // 16-aligned texture (1200). Not inside any CRC span, so no re-sign.
    p[1195] = 1;
    assert!(matches!(
        open_err(&p),
        HfError::InterBlobPadding { offset: 1195 }
    ));
}

#[test]
fn texture_consistency_heightfield_missing_texture() {
    let mut p = hf();
    let e0 = common::entry_offset(&p, 0);
    p[e0 + 92..e0 + 96].copy_from_slice(&0u32.to_le_bytes()); // texture_size -> 0
    common::resign_pack(&mut p);
    assert!(matches!(
        open_err(&p),
        HfError::TextureConsistency {
            index: 0,
            content_kind: 0
        }
    ));
}

#[test]
fn texture_consistency_game_unexpected_texture() {
    let mut p = game();
    let e0 = common::entry_offset(&p, 0);
    p[e0 + 92..e0 + 96].copy_from_slice(&1u32.to_le_bytes()); // texture_size -> 1
    common::resign_pack(&mut p);
    assert!(matches!(
        open_err(&p),
        HfError::TextureConsistency {
            index: 0,
            content_kind: 1
        }
    ));
}

// ---- truncation matrix ----------------------------------------------------

#[test]
fn truncation_before_full_header_is_too_short() {
    let pack = hf();
    for n in [0usize, 64, 123, 127] {
        assert!(
            matches!(
                open_err(&pack[..n]),
                HfError::TooShort {
                    format: Format::Pack,
                    minimum: 128,
                    ..
                }
            ),
            "truncation to {n} bytes should be TooShort",
        );
    }
}

#[test]
fn truncation_after_header_is_file_size_mismatch() {
    let pack = hf();
    // Header present (>=128) but body cut mid-directory / mid-index / mid-hash /
    // mid-blob / last byte: the file_size invariant catches every case.
    for n in [128usize, 200, 640, 700, 1000, 5634] {
        assert!(
            matches!(
                open_err(&pack[..n]),
                HfError::FileSize { header: 5635, actual } if actual as usize == n
            ),
            "truncation to {n} bytes should be a FileSize mismatch",
        );
    }
}
