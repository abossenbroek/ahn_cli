// Each integration-test binary pulls in this shared module but uses only a
// subset of its helpers, so per-binary dead-code warnings are expected here.
#![allow(dead_code)]

//! Test-local helpers shared across the integration tests.
//!
//! The committed golden fixtures are packed (`tiles.hfp`), and this crate's
//! phase does not yet implement the `Archive` reader, so a minimal slicing
//! helper reads the pack index directly to extract each tile's primary `.hf`
//! blob. It is deliberately throwaway: the archive layer replaces it.

use std::path::PathBuf;

/// Recomputes and writes `header_crc32` over bytes `[0, 112)`, so that a
/// header patched anywhere in the CRC span still has a matching CRC (only the
/// deliberately-corrupted-outside-or-target field then fires).
pub fn resign_chunk_crc(bytes: &mut [u8]) {
    let crc = crc32fast::hash(&bytes[..112]);
    bytes[112..116].copy_from_slice(&crc.to_le_bytes());
}

/// Builds a complete, valid v2 `.hf` chunk from an explicit level plane, using
/// `zstd::encode_all` for the frame. For test inputs only — the normative
/// producer is the Python one-shot compressor.
pub fn synth_chunk(
    width: u32,
    height: u32,
    z_offset: f64,
    z_scale: f64,
    levels: &[u16],
) -> Vec<u8> {
    assert_eq!(levels.len() as u64, u64::from(width) * u64::from(height));
    let mut raw = Vec::with_capacity(levels.len() * 2);
    for &l in levels {
        raw.extend_from_slice(&l.to_le_bytes());
    }
    let frame = zstd::encode_all(&raw[..], 3).expect("encode zstd frame");

    let mut hf = vec![0u8; 120];
    hf[0..4].copy_from_slice(b"AHNH");
    hf[4..8].copy_from_slice(&2u32.to_le_bytes());
    hf[8..12].copy_from_slice(&width.to_le_bytes());
    hf[12..16].copy_from_slice(&height.to_le_bytes());
    hf[16..24].copy_from_slice(&z_offset.to_le_bytes());
    hf[24..32].copy_from_slice(&z_scale.to_le_bytes());
    // rtc_centre[3] left as three finite zeros.
    // region: a finite, well-ordered placeholder.
    for (i, v) in [0.1_f64, 0.2, 0.3, 0.4, -1.0, 1.0].into_iter().enumerate() {
        let off = 56 + i * 8;
        hf[off..off + 8].copy_from_slice(&v.to_le_bytes());
    }
    hf[104..112].copy_from_slice(&(frame.len() as u64).to_le_bytes());
    resign_chunk_crc(&mut hf);
    hf.extend_from_slice(&frame);
    hf
}

/// One extracted tile: its `(level, tx, ty)` key and its primary `.hf` bytes.
pub struct PackedChunk {
    pub level: u32,
    pub tx: u32,
    pub ty: u32,
    pub hf: Vec<u8>,
}

fn le_u32(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

fn le_u64(b: &[u8], off: usize) -> u64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(&b[off..off + 8]);
    u64::from_le_bytes(a)
}

/// Absolute path to a committed repo fixture, resolved from the crate root.
pub fn repo_fixture(rel: &str) -> PathBuf {
    // CARGO_MANIFEST_DIR = <repo>/rust/ahn-heightfield
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop(); // rust
    p.pop(); // repo root
    p.push(rel);
    p
}

/// Reads the heightfield `tiles.hfp` fixture and slices out every tile's
/// primary `.hf` blob using only the pack index (no `Archive` dependency).
pub fn heightfield_chunks() -> Vec<PackedChunk> {
    let pack = std::fs::read(repo_fixture(
        "tests/tiles3d/fixtures/rust-consumer/heightfield/tiles.hfp",
    ))
    .expect("read heightfield tiles.hfp fixture");

    assert_eq!(&pack[0..4], b"AHNP", "pack magic");
    let tile_count = le_u32(&pack, 8) as usize;
    let level_count = le_u32(&pack, 12) as usize;
    let index_offset = le_u64(&pack, 16) as usize;
    assert_eq!(index_offset, 128, "index_offset");

    let entries_base = index_offset + level_count * 16;
    let mut out = Vec::with_capacity(tile_count);
    for i in 0..tile_count {
        let b = entries_base + i * 96;
        let level = le_u32(&pack, b);
        let tx = le_u32(&pack, b + 4);
        let ty = le_u32(&pack, b + 8);
        let primary_offset = le_u64(&pack, b + 72) as usize;
        let primary_size = le_u32(&pack, b + 88) as usize;
        let hf = pack[primary_offset..primary_offset + primary_size].to_vec();
        out.push(PackedChunk { level, tx, ty, hf });
    }
    out
}
