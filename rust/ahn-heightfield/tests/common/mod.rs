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

/// Absolute path to the committed `tiles.hfp` fixture for a profile
/// (`"heightfield"` or `"game"`).
pub fn pack_fixture_path(profile: &str) -> PathBuf {
    repo_fixture(&format!(
        "tests/tiles3d/fixtures/rust-consumer/{profile}/tiles.hfp"
    ))
}

/// Reads a committed `tiles.hfp` fixture into memory.
pub fn read_pack(profile: &str) -> Vec<u8> {
    std::fs::read(pack_fixture_path(profile)).expect("read tiles.hfp fixture")
}

/// Reads a `uint32` field at `off` in a pack image (little-endian).
pub fn pack_u32(b: &[u8], off: usize) -> u32 {
    le_u32(b, off)
}

/// Reads a `uint64` field at `off` in a pack image (little-endian).
pub fn pack_u64(b: &[u8], off: usize) -> u64 {
    le_u64(b, off)
}

/// Absolute file offset of index entry `i` in a pack image.
pub fn entry_offset(pack: &[u8], i: usize) -> usize {
    let level_count = le_u32(pack, 12) as usize;
    128 + level_count * 16 + i * 96
}

/// Recomputes `index_crc32` (offset 96) over the index region and
/// `header_crc32` (offset 124) over `[0, 124)`, so a pack patched inside either
/// CRC span still validates and only the deliberately-corrupted field fires.
pub fn resign_pack(bytes: &mut [u8]) {
    let level_count = le_u32(bytes, 12) as usize;
    let tile_count = le_u32(bytes, 8) as usize;
    let index_size = level_count * 16 + tile_count * 96;
    let index_crc = crc32fast::hash(&bytes[128..128 + index_size]);
    bytes[96..100].copy_from_slice(&index_crc.to_le_bytes());
    let header_crc = crc32fast::hash(&bytes[..124]);
    bytes[124..128].copy_from_slice(&header_crc.to_le_bytes());
}

/// Recomputes `dataset_id` (SHA-256 of the hash section) into the header, then
/// re-signs both CRCs — for constructing a pack whose hash section was
/// deliberately altered but remains internally consistent.
pub fn resign_pack_dataset_id(bytes: &mut [u8]) {
    use sha2::{Digest, Sha256};
    let level_count = le_u32(bytes, 12) as usize;
    let tile_count = le_u32(bytes, 8) as usize;
    let index_size = level_count * 16 + tile_count * 96;
    let hash_offset = 128 + index_size;
    let hash_size = tile_count * 64;
    let digest: [u8; 32] = Sha256::digest(&bytes[hash_offset..hash_offset + hash_size]).into();
    bytes[64..96].copy_from_slice(&digest);
    resign_pack(bytes);
}

fn align16(x: u64) -> u64 {
    (x + 15) & !15
}

/// One tile to assemble into a synthetic pack.
pub struct TileSpec {
    pub level: u32,
    pub tx: u32,
    pub ty: u32,
    /// Enclosing region `(west, south, east, north, minH, maxH)`.
    pub region: [f64; 6],
    /// Primary blob (`.hf` or `.glb`).
    pub primary: Vec<u8>,
    /// Texture blob (`.jpg`), or `None` for a game pack.
    pub texture: Option<Vec<u8>>,
}

/// Assembles a minimal, fully-valid pack from tiles given in `(level, tz=0, ty,
/// tx)` sort order. Levels `0..=max(level)` must each hold at least one tile.
/// Every blob is 16-aligned with zero inter-blob padding, the hash section and
/// `dataset_id` are computed from the blobs, and both CRCs are signed — so the
/// result opens cleanly and a single deliberate mutation isolates one reject.
pub fn build_pack(content_kind: u32, tiles: &[TileSpec]) -> Vec<u8> {
    use sha2::{Digest, Sha256};
    let tile_count = tiles.len() as u32;
    let level_count = tiles
        .iter()
        .map(|t| t.level)
        .max()
        .expect("at least one tile")
        + 1;
    let index_size = u64::from(level_count) * 16 + u64::from(tile_count) * 96;
    let hash_size = u64::from(tile_count) * 64;
    let blob_start = 128 + index_size + hash_size;

    // Blob offsets, in index/slot order (primary then texture), 16-aligned.
    let mut primary_off = Vec::with_capacity(tiles.len());
    let mut texture_off = Vec::with_capacity(tiles.len());
    let mut cur = blob_start;
    for t in tiles {
        cur = align16(cur);
        primary_off.push(cur);
        cur += t.primary.len() as u64;
        match &t.texture {
            Some(tex) => {
                cur = align16(cur);
                texture_off.push(cur);
                cur += tex.len() as u64;
            }
            None => texture_off.push(0),
        }
    }
    let file_size = cur;

    let mut buf = vec![0u8; file_size as usize];
    buf[0..4].copy_from_slice(b"AHNP");
    buf[4..8].copy_from_slice(&1u32.to_le_bytes());
    buf[8..12].copy_from_slice(&tile_count.to_le_bytes());
    buf[12..16].copy_from_slice(&level_count.to_le_bytes());
    buf[16..24].copy_from_slice(&128u64.to_le_bytes());
    buf[24..32].copy_from_slice(&index_size.to_le_bytes());
    buf[32..40].copy_from_slice(&(128 + index_size).to_le_bytes());
    buf[40..48].copy_from_slice(&hash_size.to_le_bytes());
    buf[48..56].copy_from_slice(&file_size.to_le_bytes());
    buf[56..64].copy_from_slice(&8.0f64.to_le_bytes()); // root_geometric_error
    buf[104..108].copy_from_slice(&content_kind.to_le_bytes());

    // Level directory.
    for l in 0..level_count {
        let idxs: Vec<usize> = (0..tiles.len()).filter(|&i| tiles[i].level == l).collect();
        let first_entry = *idxs.first().expect("each level holds a tile") as u32;
        let mut txs: Vec<u32> = idxs.iter().map(|&i| tiles[i].tx).collect();
        let mut tys: Vec<u32> = idxs.iter().map(|&i| tiles[i].ty).collect();
        txs.sort_unstable();
        txs.dedup();
        tys.sort_unstable();
        tys.dedup();
        let off = 128 + l as usize * 16;
        buf[off..off + 4].copy_from_slice(&first_entry.to_le_bytes());
        buf[off + 4..off + 8].copy_from_slice(&(idxs.len() as u32).to_le_bytes());
        buf[off + 8..off + 12].copy_from_slice(&(txs.len() as u32).to_le_bytes());
        buf[off + 12..off + 16].copy_from_slice(&(tys.len() as u32).to_le_bytes());
    }

    // Index entries + hash section + blobs.
    let entries_base = 128 + level_count as usize * 16;
    let hash_base = (128 + index_size) as usize;
    for (i, t) in tiles.iter().enumerate() {
        let off = entries_base + i * 96;
        buf[off..off + 4].copy_from_slice(&t.level.to_le_bytes());
        buf[off + 4..off + 8].copy_from_slice(&t.tx.to_le_bytes());
        buf[off + 8..off + 12].copy_from_slice(&t.ty.to_le_bytes());
        // tz = 0 (already zero)
        for (k, v) in t.region.into_iter().enumerate() {
            let ro = off + 16 + k * 8;
            buf[ro..ro + 8].copy_from_slice(&v.to_le_bytes());
        }
        let ge: f64 = if t.level + 1 == level_count { 0.0 } else { 4.0 };
        buf[off + 64..off + 72].copy_from_slice(&ge.to_le_bytes());
        buf[off + 72..off + 80].copy_from_slice(&primary_off[i].to_le_bytes());
        buf[off + 80..off + 88].copy_from_slice(&texture_off[i].to_le_bytes());
        buf[off + 88..off + 92].copy_from_slice(&(t.primary.len() as u32).to_le_bytes());
        let tex_size = t.texture.as_ref().map_or(0, Vec::len) as u32;
        buf[off + 92..off + 96].copy_from_slice(&tex_size.to_le_bytes());

        let ps = primary_off[i] as usize;
        buf[ps..ps + t.primary.len()].copy_from_slice(&t.primary);
        let ho = hash_base + i * 64;
        buf[ho..ho + 32].copy_from_slice(&<[u8; 32]>::from(Sha256::digest(&t.primary)));
        if let Some(tex) = &t.texture {
            let ts = texture_off[i] as usize;
            buf[ts..ts + tex.len()].copy_from_slice(tex);
            buf[ho + 32..ho + 64].copy_from_slice(&<[u8; 32]>::from(Sha256::digest(tex)));
        }
        // No-texture tiles keep texture_sha256 = 32 zero bytes (already zero).
    }

    resign_pack_dataset_id(&mut buf);
    buf
}

/// A `content_kind = 1` (game) pack over `(level, tx, ty)` tiles with arbitrary
/// primary `.glb`-stand-in blobs and no textures.
pub fn build_game_pack(tiles: &[(u32, u32, u32)], primaries: &[Vec<u8>]) -> Vec<u8> {
    assert_eq!(tiles.len(), primaries.len());
    let specs: Vec<TileSpec> = tiles
        .iter()
        .zip(primaries)
        .map(|(&(level, tx, ty), primary)| TileSpec {
            level,
            tx,
            ty,
            region: [0.0, 0.0, 1.0, 1.0, -1.0, 1.0],
            primary: primary.clone(),
            texture: None,
        })
        .collect();
    build_pack(1, &specs)
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
