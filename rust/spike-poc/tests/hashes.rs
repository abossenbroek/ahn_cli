//! Hash cross-language golden vectors: `crc32fast` == Python `zlib.crc32`,
//! `sha2::Sha256` == Python `hashlib.sha256`.

mod common;

#[test]
fn crc32fast_matches_python_zlib_crc32() {
    let g = common::golden();
    let input = g["crc32"]["input_utf8"]
        .as_str()
        .unwrap()
        .as_bytes()
        .to_vec();
    let want = g["crc32"]["crc32"].as_u64().unwrap() as u32;
    let mut h = crc32fast::Hasher::new();
    h.update(&input);
    assert_eq!(h.finalize(), want, "crc32fast != zlib.crc32");
}

#[test]
fn sha256_matches_python_hashlib() {
    use sha2::{Digest, Sha256};
    let g = common::golden();
    let input = g["sha256"]["input_utf8"]
        .as_str()
        .unwrap()
        .as_bytes()
        .to_vec();
    let want = g["sha256"]["sha256"].as_str().unwrap().to_string();
    let mut h = Sha256::new();
    h.update(&input);
    let got: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
    assert_eq!(got, want, "sha2 != hashlib.sha256");
}

/// The committed zstd frame's sha256 matches the Python golden (pins the exact
/// checksum-on frame bytes across the two bundled libzstd builds).
#[test]
fn zstd_frame_sha256_matches_golden() {
    use sha2::{Digest, Sha256};
    let g = common::golden();
    let frame = common::read("zstd_frame.bin");
    let mut h = Sha256::new();
    h.update(&frame);
    let got: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
    let want = g["zstd"]["frame_sha256"].as_str().unwrap();
    assert_eq!(got, want, "zstd frame bytes differ across libzstd builds");
}
