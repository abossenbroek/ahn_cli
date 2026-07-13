//! Golden opens of the committed `game` and `heightfield` packs: header fields,
//! entry count/sort order, level-directory consistency, `read_primary`
//! byte-equality against the recomputed per-blob SHA-256, `decode_tile` of every
//! heightfield tile, and `verify_blobs` green on the pristine packs.

mod common;

use ahn_heightfield::{Archive, TileKey};
use sha2::{Digest, Sha256};

/// Extracts the `dataset_id` hex string from a fixture `manifest.json`.
fn manifest_dataset_id(profile: &str) -> String {
    let path = common::repo_fixture(&format!(
        "tests/tiles3d/fixtures/rust-consumer/{profile}/manifest.json"
    ));
    let text = std::fs::read_to_string(path).expect("read manifest.json");
    let needle = "\"dataset_id\": \"";
    let start = text.find(needle).expect("dataset_id key") + needle.len();
    text[start..start + 64].to_string()
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

const SORTED_KEYS: [(u32, u32, u32); 5] = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 0, 1), (1, 1, 1)];

#[test]
fn opens_both_packs_with_expected_header_and_index() {
    for (profile, expected_kind) in [("heightfield", 0u32), ("game", 1u32)] {
        let pack = common::read_pack(profile);
        let archive = Archive::open(&pack[..]).expect("pack opens");
        let header = archive.header();

        assert_eq!(header.tile_count, 5, "{profile} tile_count");
        assert_eq!(header.level_count, 2, "{profile} level_count");
        assert_eq!(header.content_kind, expected_kind, "{profile} content_kind");
        assert_eq!(header.file_size, pack.len() as u64, "{profile} file_size");
        assert_eq!(header.root_geometric_error, 8.0, "{profile} root_ge");

        // dataset_id agrees with the header field, the accessor, and manifest.json.
        assert_eq!(archive.dataset_id(), header.dataset_id);
        assert_eq!(
            hex(&archive.dataset_id()),
            manifest_dataset_id(profile),
            "{profile} dataset_id matches manifest.json",
        );

        // Entry count + strict (level, tz, ty, tx) sort order.
        let keys: Vec<(u32, u32, u32)> = archive
            .entries()
            .iter()
            .map(|e| (e.level, e.tx, e.ty))
            .collect();
        assert_eq!(keys, SORTED_KEYS, "{profile} entry sort order");
        for e in archive.entries() {
            assert_eq!(e.tz, 0);
        }

        // Level directory: one root, four leaves; contiguous runs.
        let root_run = archive.level_run(0).expect("level 0 run");
        assert_eq!((root_run.first_entry, root_run.entry_count), (0, 1));
        let leaf_run = archive.level_run(1).expect("level 1 run");
        assert_eq!((leaf_run.first_entry, leaf_run.entry_count), (1, 4));
        assert!(archive.level_run(2).is_none());
        assert_eq!(archive.level(0).unwrap().len(), 1);
        assert_eq!(archive.level(1).unwrap().len(), 4);
        assert!(archive.level(2).is_none());

        // find round-trips every key; a missing key is None.
        for &(level, tx, ty) in &SORTED_KEYS {
            let key = TileKey {
                level,
                tx,
                ty,
                tz: 0,
            };
            let found = archive.find(key).expect("entry found");
            assert_eq!((found.level, found.tx, found.ty), (level, tx, ty));
        }
        assert!(archive
            .find(TileKey {
                level: 5,
                tx: 5,
                ty: 5,
                tz: 0
            })
            .is_none());

        // read_primary returns exactly the blob the hash section records.
        let hash_base =
            128 + (header.level_count as usize) * 16 + (header.tile_count as usize) * 96;
        for (i, entry) in archive.entries().iter().enumerate() {
            let primary = archive.read_primary(entry).expect("read primary");
            let recorded = &pack[hash_base + i * 64..hash_base + i * 64 + 32];
            assert_eq!(
                Sha256::digest(&primary).as_slice(),
                recorded,
                "{profile} entry {i} primary blob matches its recorded SHA-256",
            );
        }

        // verify_blobs is green on the pristine pack.
        archive.verify_blobs().expect("pristine pack verifies");
    }
}

#[test]
fn root_children_are_the_four_leaves() {
    let pack = common::read_pack("heightfield");
    let archive = Archive::open(&pack[..]).expect("pack opens");
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
    // Ascending key order: (ty, tx) row-major within level 1.
    assert_eq!(children, vec![(1, 0, 0), (1, 1, 0), (1, 0, 1), (1, 1, 1)],);
    // A leaf has no children.
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

#[test]
fn decode_tile_succeeds_for_every_heightfield_tile() {
    let pack = common::read_pack("heightfield");
    let archive = Archive::open(&pack[..]).expect("pack opens");
    for entry in archive.entries() {
        let tile = archive
            .decode_tile(entry)
            .expect("tile decodes + cross-checks");
        assert_eq!(tile.levels().len(), (tile.width() * tile.height()) as usize,);
        // The chunk's own horizontal region is bit-equal to the entry's (the
        // cross-check decode_tile enforces), and every height is finite.
        assert_eq!(tile.header().region[0].to_bits(), entry.region[0].to_bits());
        assert_eq!(tile.header().region[2].to_bits(), entry.region[2].to_bits());
        for row in 0..tile.height() {
            for col in 0..tile.width() {
                assert!(tile.dequantize_at(row, col).is_finite());
            }
        }
    }
}

#[test]
fn decode_tile_rejects_a_game_pack() {
    let pack = common::read_pack("game");
    let archive = Archive::open(&pack[..]).expect("pack opens");
    let entry = &archive.entries()[0];
    let err = archive.decode_tile(entry).unwrap_err();
    assert!(matches!(
        err,
        ahn_heightfield::HfError::BadContentKind { found: 1 }
    ));
    // A game pack has no separate texture blob.
    assert!(archive.read_texture(entry).unwrap().is_none());
}

#[test]
fn crate_local_doc_fixture_matches_the_repo_heightfield_pack() {
    // The doctests include `tests/data/tiles.hfp`; guard it against drift from
    // the canonical committed fixture (regenerated by the Python suite).
    let local: &[u8] = include_bytes!("data/tiles.hfp");
    let repo = common::read_pack("heightfield");
    assert_eq!(local, &repo[..], "crate-local doc pack is a stale copy");
}
