//! Golden-decode tests: decode every `.hf` chunk in the committed heightfield
//! pack fixture and assert the header/quantization invariants the spec and the
//! fixtures' provenance record.

mod common;

use ahn_heightfield::{ChunkHeader, Heightfield, TileKey, CHUNK_VERSION, MAX_QUANTIZED_LEVEL};

#[test]
fn decodes_every_fixture_chunk_with_valid_invariants() {
    let chunks = common::heightfield_chunks();
    // 12x12 synthetic scene at tile_pixels=8: one root + four leaves.
    assert_eq!(chunks.len(), 5, "expected five packed tiles");

    // The pack stores entries sorted by (level, tz, ty, tx); the extracted keys
    // must be exactly that set in that order.
    let keys: Vec<(u32, u32, u32)> = chunks.iter().map(|c| (c.level, c.tx, c.ty)).collect();
    assert_eq!(
        keys,
        vec![(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 0, 1), (1, 1, 1)],
        "tile keys in pack sort order",
    );

    for chunk in &chunks {
        let header = ChunkHeader::parse(&chunk.hf).expect("header parses");
        assert_eq!(header.version, CHUNK_VERSION);
        assert!(header.width > 0 && header.height > 0);
        assert!(header.z_scale > 0.0);
        // No production tile approaches the 0.025 m cap; these are ~0.005 m.
        assert!(!header.exceeds_error_cap());
        header.check_error_cap().expect("within absolute-error cap");

        let tile = Heightfield::decode(&chunk.hf).expect("chunk decodes");
        assert_eq!(tile.header(), &header, "decode header matches parse");
        assert_eq!(
            tile.levels().len() as u64,
            header.plane_len() / 2,
            "plane has width*height levels",
        );

        let bound = header.z_scale / 2.0;
        let (min_h, max_h) = (header.region[4], header.region[5]);
        for &level in tile.levels() {
            assert!(level <= MAX_QUANTIZED_LEVEL, "level within 12-bit range");
        }
        // Every dequantized height lies within the header's [minH, maxH] range
        // widened by the quantization bound (z_scale / 2).
        for row in 0..tile.height() {
            for col in 0..tile.width() {
                let h = tile.dequantize_at(row, col);
                assert!(
                    h >= min_h - bound && h <= max_h + bound,
                    "dequantized height {h} within [{min_h}, {max_h}] +/- {bound}",
                );
            }
        }

        // level_at agrees with the row-major levels() slice.
        let w = tile.width();
        assert_eq!(tile.level_at(0, 0), tile.levels()[0]);
        let (row, col) = (1u32, 2u32);
        assert_eq!(
            tile.level_at(row, col),
            tile.levels()[(row * w + col) as usize],
            "level_at is row-major",
        );
    }
}

#[test]
fn tilekey_orders_by_spec_sort_key_matching_pack_order() {
    let chunks = common::heightfield_chunks();
    let mut keys: Vec<TileKey> = chunks
        .iter()
        .map(|c| TileKey {
            level: c.level,
            tx: c.tx,
            ty: c.ty,
            tz: 0,
        })
        .collect();
    let as_extracted = keys.clone();
    keys.sort();
    assert_eq!(
        keys, as_extracted,
        "TileKey Ord reproduces the pack's on-disk entry order",
    );
}
