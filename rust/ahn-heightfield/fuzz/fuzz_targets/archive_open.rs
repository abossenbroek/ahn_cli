#![no_main]
//! Fuzz the `AHNP` pack reader against arbitrary bytes.
//!
//! `Archive::open` over a `&[u8]` must never panic on malformed input. When a
//! pack happens to open, the target also exercises the cold `verify_blobs` path
//! and every per-tile blob read/decode — all cheap at fuzz-corpus sizes — so the
//! deeper index/blob/chunk code paths are covered too, not just `open`.

use ahn_heightfield::Archive;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(archive) = Archive::open(data) {
        let _ = archive.verify_blobs();
        for entry in archive.entries() {
            let _ = archive.read_primary(entry);
            let _ = archive.read_texture(entry);
            let _ = archive.decode_tile(entry);
        }
    }
});
