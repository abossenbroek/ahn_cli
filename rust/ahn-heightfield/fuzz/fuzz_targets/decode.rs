#![no_main]
//! Fuzz the `.hf` chunk layer against arbitrary bytes.
//!
//! The invariant under test: neither the header parse nor the full decode may
//! ever panic, over-allocate, or hang — every malformed input must surface as a
//! clean `HfError`. The header CRC guards the giant-allocation window, so a
//! corrupt `width`/`height` is rejected before any plane is sized.

use ahn_heightfield::{ChunkHeader, Heightfield};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = ChunkHeader::parse(data);
    let _ = Heightfield::decode(data);
});
