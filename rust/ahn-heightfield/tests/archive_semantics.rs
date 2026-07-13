//! `ReadAt` semantics (slice vs `File`, concurrency, misalignment), the
//! cold-hash-section promise of `open`, the `verify_blobs` failure paths, and
//! `decode_tile`'s cross-check / error-cap enforcement.

mod common;

use std::cell::RefCell;
use std::fs::File;
use std::io;
use std::sync::Arc;

use ahn_heightfield::{Archive, BlobSlot, Entry, HfError, PackHeader, ReadAt, TileKey};

use common::TileSpec;

/// The heightfield fixture's cold hash section, `[hash_offset, hash_offset +
/// hash_size)` — `[640, 960)` for the committed 5-tile / 2-level pack.
const HASH_START: u64 = 640;
const HASH_END: u64 = 960;

// ---- ReadAt: slice vs File give identical results -------------------------

#[test]
fn slice_and_file_readers_agree() {
    let bytes = common::read_pack("heightfield");
    let slice_archive = Archive::open(&bytes[..]).expect("slice opens");

    let file = File::open(common::pack_fixture_path("heightfield")).expect("open file");
    let file_archive = Archive::open(file).expect("file opens");

    assert_eq!(slice_archive.header(), file_archive.header());
    assert_eq!(slice_archive.entries(), file_archive.entries());
    for (a, b) in slice_archive.entries().iter().zip(file_archive.entries()) {
        assert_eq!(
            slice_archive.decode_tile(a).expect("slice decode"),
            file_archive.decode_tile(b).expect("file decode"),
        );
        assert_eq!(
            slice_archive.read_primary(a).expect("slice primary"),
            file_archive.read_primary(b).expect("file primary"),
        );
    }
}

#[test]
fn decode_from_a_misaligned_slice_matches() {
    let bytes = common::read_pack("heightfield");
    let aligned = Archive::open(&bytes[..]).expect("aligned");

    let mut padded = vec![0u8];
    padded.extend_from_slice(&bytes);
    let misaligned = Archive::open(&padded[1..]).expect("misaligned");

    assert_eq!(aligned.header(), misaligned.header());
    assert_eq!(aligned.entries(), misaligned.entries());
    for (a, b) in aligned.entries().iter().zip(misaligned.entries()) {
        assert_eq!(
            aligned.decode_tile(a).expect("a"),
            misaligned.decode_tile(b).expect("b"),
        );
    }
}

#[test]
fn concurrent_positioned_reads_on_one_file() {
    let file = File::open(common::pack_fixture_path("heightfield")).expect("open file");
    let archive = Arc::new(Archive::open(file).expect("file opens"));

    // Single-threaded baseline.
    let baseline: Vec<_> = archive
        .entries()
        .iter()
        .map(|e| archive.decode_tile(e).expect("decode"))
        .collect();

    let handles: Vec<_> = (0..4)
        .map(|_| {
            let archive = Arc::clone(&archive);
            let baseline = baseline.clone();
            std::thread::spawn(move || {
                for _ in 0..50 {
                    for (i, e) in archive.entries().iter().enumerate() {
                        let tile = archive.decode_tile(e).expect("concurrent decode");
                        assert_eq!(tile, baseline[i], "concurrent decode diverged");
                    }
                }
            })
        })
        .collect();
    for h in handles {
        h.join().expect("worker thread panicked");
    }
}

/// `Archive<File>` and the record types are `Send + Sync`, so one open archive
/// serves concurrent loads. Verified at compile time.
#[test]
fn public_types_are_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Archive<File>>();
    assert_send_sync::<PackHeader>();
    assert_send_sync::<Entry>();
    assert_send_sync::<TileKey>();
    assert_send_sync::<HfError>();
}

// ---- open does not read the cold hash section -----------------------------

/// A `ReadAt` that records every positioned read's byte range.
struct SpyReader<'a> {
    data: &'a [u8],
    reads: RefCell<Vec<(u64, u64)>>,
}

impl ReadAt for SpyReader<'_> {
    fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<usize> {
        let n = self.data.read_at(buf, offset)?;
        self.reads.borrow_mut().push((offset, offset + n as u64));
        Ok(n)
    }
}

fn touched_hash_section(reads: &[(u64, u64)]) -> bool {
    reads
        .iter()
        .any(|&(start, end)| start < HASH_END && end > HASH_START)
}

#[test]
fn open_never_reads_the_hash_section_but_verify_blobs_does() {
    let bytes = common::read_pack("heightfield");
    let spy = SpyReader {
        data: &bytes,
        reads: RefCell::new(Vec::new()),
    };

    let archive = Archive::open(&spy).expect("opens through spy");
    assert!(
        !touched_hash_section(&spy.reads.borrow()),
        "open must not read the cold hash section [{HASH_START}, {HASH_END})",
    );

    // The cold section is only touched by the install/repair path.
    spy.reads.borrow_mut().clear();
    archive.verify_blobs().expect("verifies");
    assert!(
        touched_hash_section(&spy.reads.borrow()),
        "verify_blobs must read the hash section",
    );
}

// ---- verify_blobs failure paths -------------------------------------------

#[test]
fn hash_section_bit_flip_passes_open_but_fails_verify_via_dataset_id() {
    let mut bytes = common::read_pack("heightfield");
    // Flip a byte inside the hash section. It is not covered by either CRC and
    // is never read at open, so open still succeeds (the cold-section promise).
    bytes[700] ^= 0xFF;
    let archive = Archive::open(&bytes[..]).expect("open ignores the cold section");
    let err = archive.verify_blobs().unwrap_err();
    assert!(matches!(err, HfError::DatasetIdMismatch { .. }));
}

#[test]
fn corrupt_primary_blob_fails_verify_with_blob_hash_mismatch() {
    let mut bytes = common::read_pack("heightfield");
    // A byte in entry 0's primary blob interior [960, 1191). The hash section is
    // untouched, so dataset_id still matches; the blob's own hash does not.
    bytes[1010] ^= 0xFF;
    let archive = Archive::open(&bytes[..]).expect("open ignores blob contents");
    let err = archive.verify_blobs().unwrap_err();
    assert!(matches!(
        err,
        HfError::BlobHashMismatch {
            index: 0,
            slot: BlobSlot::Primary
        }
    ));
}

#[test]
fn corrupt_texture_blob_fails_verify_with_blob_hash_mismatch() {
    let mut bytes = common::read_pack("heightfield");
    // A byte in entry 0's texture blob [1200, 1903).
    bytes[1500] ^= 0xFF;
    let archive = Archive::open(&bytes[..]).expect("open");
    let err = archive.verify_blobs().unwrap_err();
    assert!(matches!(
        err,
        HfError::BlobHashMismatch {
            index: 0,
            slot: BlobSlot::Texture
        }
    ));
}

#[test]
fn game_texture_hash_not_zero_is_rejected_by_verify() {
    let mut bytes = common::read_pack("game");
    // Make entry 0's texture_sha256 record non-zero, then re-sign dataset_id so
    // the section stays internally consistent (open + dataset_id both pass) and
    // the per-record check is what fires.
    bytes[672] = 1; // hash_offset(640) + 32 = first texture_sha256 byte
    common::resign_pack_dataset_id(&mut bytes);
    let archive = Archive::open(&bytes[..]).expect("open");
    let err = archive.verify_blobs().unwrap_err();
    assert!(matches!(err, HfError::TextureHashNotZero { index: 0 }));
}

// ---- decode_tile cross-check and error cap --------------------------------

#[test]
fn decode_tile_rejects_a_region_mismatch() {
    let mut bytes = common::read_pack("heightfield");
    let e1 = common::entry_offset(&bytes, 1);
    // A finite, still-well-ordered west that no longer matches the chunk header's
    // own west, so decode_tile's horizontal bit-equal cross-check fails.
    bytes[e1 + 16..e1 + 24].copy_from_slice(&(-10.0f64).to_le_bytes());
    common::resign_pack(&mut bytes);
    let archive = Archive::open(&bytes[..]).expect("open accepts a well-ordered region");
    let entry = &archive.entries()[1];
    let err = archive.decode_tile(entry).unwrap_err();
    assert!(matches!(err, HfError::BlobContentMismatch { index: 1 }));
}

#[test]
fn decode_tile_enforces_the_absolute_error_cap() {
    // A synthetic single-tile heightfield pack whose chunk has z_scale = 1.0
    // (bound 0.5 m, far past the 0.025 m cap). The entry region matches the
    // chunk's own region, so the cap — checked before the cross-check — fires.
    let chunk = common::synth_chunk(2, 2, 0.0, 1.0, &[0, 1, 2, 3]);
    let pack = common::build_pack(
        0,
        &[TileSpec {
            level: 0,
            tx: 0,
            ty: 0,
            region: [0.1, 0.2, 0.3, 0.4, -1.0, 1.0], // == synth_chunk's region
            primary: chunk,
            texture: Some(vec![0xFF, 0xD8, 0xFF, 0xD9]), // JPEG-stand-in bytes
        }],
    );
    let archive = Archive::open(&pack[..]).expect("synthetic pack opens");
    let err = archive.decode_tile(&archive.entries()[0]).unwrap_err();
    assert!(matches!(err, HfError::ErrorCapExceeded { .. }));
}

#[test]
fn synthetic_heightfield_tile_round_trips_within_the_cap() {
    // Same shape but a within-cap z_scale, proving the builder produces packs
    // that open and decode_tile cleanly (cross-check + cap both pass).
    let chunk = common::synth_chunk(2, 2, 0.0, 0.001, &[0, 1, 2, 3]);
    let pack = common::build_pack(
        0,
        &[TileSpec {
            level: 0,
            tx: 0,
            ty: 0,
            region: [0.1, 0.2, 0.3, 0.4, -1.0, 1.0],
            primary: chunk,
            texture: Some(vec![0xFF, 0xD8, 0xFF, 0xD9]),
        }],
    );
    let archive = Archive::open(&pack[..]).expect("opens");
    let tile = archive.decode_tile(&archive.entries()[0]).expect("decodes");
    assert_eq!(tile.levels(), &[0, 1, 2, 3]);
    archive.verify_blobs().expect("synthetic pack verifies");
    assert_eq!(
        archive
            .read_texture(&archive.entries()[0])
            .expect("texture"),
        Some(vec![0xFF, 0xD8, 0xFF, 0xD9]),
    );
}

// ---- implicit children over a sparse pack ---------------------------------

#[test]
fn children_over_a_sparse_pack() {
    // Root plus only two of its four possible children (a sparse level 1).
    let tiles = [(0u32, 0u32, 0u32), (1, 0, 0), (1, 1, 1)];
    let primaries = vec![vec![1u8; 8], vec![2u8; 8], vec![3u8; 8]];
    let pack = common::build_game_pack(&tiles, &primaries);
    let archive = Archive::open(&pack[..]).expect("sparse pack opens");

    let root = archive
        .find(TileKey {
            level: 0,
            tx: 0,
            ty: 0,
            tz: 0,
        })
        .expect("root");
    let children: Vec<(u32, u32, u32)> = archive
        .children(root)
        .map(|e| (e.level, e.tx, e.ty))
        .collect();
    // Only the two present children come back, in ascending key order; the two
    // absent child keys (1,1,0) and (1,0,1) are silently skipped.
    assert_eq!(children, vec![(1, 0, 0), (1, 1, 1)]);

    // A present leaf still has no children of its own.
    let leaf = archive
        .find(TileKey {
            level: 1,
            tx: 1,
            ty: 1,
            tz: 0,
        })
        .expect("leaf");
    assert_eq!(archive.children(leaf).count(), 0);
}
