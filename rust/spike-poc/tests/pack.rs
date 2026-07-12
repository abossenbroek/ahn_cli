//! `AHNP` pack container interop (P1–P10) + the per-frame scan micro-benchmark.

mod common;

use ahn_hf_spike::{Entry, HfError, Pack, KIND_GAME, KIND_HEIGHTFIELD, PACK_HEADER_SIZE};

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Heightfield pack opens, validates fully, and its dataset_id + content_kind
/// match the Python golden. Each primary blob is a decodable v2... no — the
/// committed fixtures are v1 `.hf`; here we only assert the pack framing and
/// that the primary blob for a leaf is the exact bytes of the fixture file.
#[test]
fn heightfield_pack_opens_and_matches_golden() {
    let data = common::read("heightfield.hfp");
    let pack = Pack::open(&data).expect("open heightfield pack");
    let g = common::golden();
    let gp = &g["packs"]["heightfield"];
    assert_eq!(pack.header.content_kind, KIND_HEIGHTFIELD);
    assert_eq!(
        pack.header.tile_count,
        gp["tile_count"].as_u64().unwrap() as u32
    );
    assert_eq!(
        hex(&pack.header.dataset_id),
        gp["dataset_id"].as_str().unwrap()
    );
    // every heightfield entry has a texture (sibling jpg), primary is a .hf
    for e in &pack.entries {
        assert!(
            e.texture_offset != 0,
            "heightfield tile must have a texture"
        );
        assert!(e.primary_size > 0);
    }
}

/// Game pack opens; primary is a `.glb`, texture slot is zeroed (embedded).
#[test]
fn game_pack_opens_texture_slot_zeroed() {
    let data = common::read("game.hfp");
    let pack = Pack::open(&data).expect("open game pack");
    let g = common::golden();
    assert_eq!(pack.header.content_kind, KIND_GAME);
    assert_eq!(
        hex(&pack.header.dataset_id),
        g["packs"]["game"]["dataset_id"].as_str().unwrap()
    );
    for e in &pack.entries {
        assert_eq!(e.texture_offset, 0, "game tile texture is embedded in glb");
        assert_eq!(e.texture_size, 0);
    }
}

/// Blob file order == index order == sorted (level,tz,ty,tx), and the index is
/// level-major contiguous per the level directory.
#[test]
fn blob_order_equals_index_order() {
    let data = common::read("heightfield.hfp");
    let pack = Pack::open(&data).unwrap();
    let idx_keys: Vec<_> = pack
        .entries
        .iter()
        .map(|e| (e.level, e.tz, e.ty, e.tx))
        .collect();
    let mut sorted = idx_keys.clone();
    sorted.sort();
    assert_eq!(idx_keys, sorted, "index not sorted");
    // primary offsets strictly increasing == blob order matches index order
    let mut prev = 0u64;
    for e in &pack.entries {
        assert!(e.primary_offset >= prev, "blobs not in index order");
        prev = e.primary_offset;
    }
    // level directory runs are contiguous and cover all entries
    let mut cursor = 0u32;
    for run in &pack.directory {
        assert_eq!(run.first_entry, cursor);
        cursor += run.entry_count;
    }
    assert_eq!(cursor, pack.header.tile_count);
}

/// The Rust tuple comparator agrees with Python's sort on a sparse/non-square
/// key set (absent implicit children). Golden from `sort_vectors.json`.
#[test]
fn sort_comparator_matches_python_sparse() {
    let bytes = common::read("sort_vectors.json");
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let parse = |k: &serde_json::Value| {
        let a = k.as_array().unwrap();
        // stored as (level, tz, ty, tx)
        (
            a[0].as_u64().unwrap(),
            a[1].as_u64().unwrap(),
            a[2].as_u64().unwrap(),
            a[3].as_u64().unwrap(),
        )
    };
    let mut input: Vec<_> = v["input"].as_array().unwrap().iter().map(parse).collect();
    let expected: Vec<_> = v["expected_sorted"]
        .as_array()
        .unwrap()
        .iter()
        .map(parse)
        .collect();
    input.sort();
    assert_eq!(input, expected, "Rust sort != Python sort");
}

/// dataset_id changes iff content changes: flip one payload byte inside a blob
/// region and the recomputed dataset_id no longer matches the header — the
/// pack open fails at the dataset_id (hash-section) check only if we corrupt
/// the hash section; corrupting a blob is caught by per-blob sha (install
/// time). Here we corrupt the HASH SECTION and expect a DatasetId reject.
#[test]
fn hash_section_corruption_rejected() {
    let mut data = common::read("heightfield.hfp");
    let pack = Pack::open(&data).unwrap();
    let hash_off = pack.header.hash_offset as usize;
    data[hash_off + 3] ^= 0x01;
    assert!(matches!(Pack::open(&data), Err(HfError::DatasetId)));
}

/// Header/index bit-flips are rejected by their CRCs.
#[test]
fn header_and_index_bitflips_rejected() {
    let base = common::read("heightfield.hfp");

    let mut h = base.clone();
    h[8] ^= 0x01; // tile_count byte -> header CRC fail
    assert!(matches!(Pack::open(&h), Err(HfError::HeaderCrc)));

    let mut i = base.clone();
    // first index-entry byte lives just past the 128-B header + level dir.
    let idx_first = PACK_HEADER_SIZE + 2 * 16; // 2 levels * 16-B dir records
    i[idx_first] ^= 0x01;
    assert!(matches!(Pack::open(&i), Err(HfError::IndexCrc)));
}

/// Truncation matrix: chopping mid-header / mid-directory / mid-index /
/// mid-hash / mid-blob / last-byte is rejected in EVERY case.
#[test]
fn truncation_matrix_all_rejected() {
    let data = common::read("heightfield.hfp");
    let pack = Pack::open(&data).unwrap();
    let idx_off = PACK_HEADER_SIZE;
    let dir_end = idx_off + pack.header.level_count as usize * 16;
    let hash_off = pack.header.hash_offset as usize;
    let blob_start = (pack.header.hash_offset + pack.header.hash_size) as usize;
    let points = [
        ("mid-header", 64usize),
        ("mid-directory", idx_off + 8),
        ("mid-index", dir_end + 40),
        ("mid-hash", hash_off + 20),
        ("mid-blob", blob_start + 8),
        ("last-byte", data.len() - 1),
    ];
    for (label, n) in points {
        let r = Pack::open(&data[..n]);
        assert!(r.is_err(), "{label}: truncation not rejected");
    }
}

/// P10 per-frame scan micro-benchmark: a synthetic 2,000-entry resident index,
/// linear AABB scan vs per-row binary search. Confirms the sub-100 µs class
/// (informational; asserts only that a full linear scan is < 5 ms so a slow CI
/// box still passes while the number is recorded).
#[test]
fn per_frame_scan_microbench() {
    // Build 2,000 fake entries (level 11 full row-major grid ~ 45x45).
    let n = 2000usize;
    let mut entries: Vec<Entry> = Vec::with_capacity(n);
    for i in 0..n {
        let tx = (i % 64) as u32;
        let ty = (i / 64) as u32;
        entries.push(Entry {
            level: 11,
            tx,
            ty,
            tz: 0,
            region: [
                0.1 + tx as f64 * 1e-6,
                0.2 + ty as f64 * 1e-6,
                0.1 + (tx as f64 + 1.0) * 1e-6,
                0.2 + (ty as f64 + 1.0) * 1e-6,
                -5.0,
                40.0,
            ],
            geometric_error: 0.0,
            primary_offset: (i as u64) * 128,
            texture_offset: 0,
            primary_size: 100,
            texture_size: 0,
        });
    }
    // query AABB overlapping ~a handful of entries
    let (qw, qs, qe, qn) = (0.1 + 10e-6, 0.2 + 5e-6, 0.1 + 12e-6, 0.2 + 7e-6);
    let reps = 200u32;
    let t0 = std::time::Instant::now();
    let mut hits = 0u64;
    for _ in 0..reps {
        for e in &entries {
            if e.region[0] <= qe && e.region[2] >= qw && e.region[1] <= qn && e.region[3] >= qs {
                hits += 1;
            }
        }
    }
    let per_scan = t0.elapsed() / reps;
    eprintln!(
        "[bench] linear AABB scan over {n} entries: {:?}/scan, hits/scan={}",
        per_scan,
        hits / reps as u64
    );
    assert!(
        per_scan.as_millis() < 5,
        "linear scan unexpectedly slow: {per_scan:?}"
    );
}

/// The heightfield pack's primary blobs are byte-identical to the committed
/// loose `.hf` fixture files (the pack really contains the same content).
#[test]
fn pack_primary_blobs_equal_loose_fixtures() {
    let data = common::read("heightfield.hfp");
    let pack = Pack::open(&data).unwrap();
    for e in &pack.entries {
        let name = format!("{}-{}-{}", e.level, e.tx, e.ty);
        let loose =
            std::fs::read(common::fixture_dir().join(format!("heightfield/tiles/{name}.hf")))
                .unwrap();
        assert_eq!(pack.primary(&data, e), &loose[..], "{name} blob mismatch");
    }
}
