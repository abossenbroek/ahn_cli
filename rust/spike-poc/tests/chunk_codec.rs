//! v2 `.hf` chunk + zstd-frame interop doubts (checksum on, negatives).

mod common;

use ahn_hf_spike::{decode_chunk, ChunkHeader, HfError, CHUNK_HEADER_SIZE};

/// zstd checksum-on frame from python-zstandard 0.23.0 decodes in the Rust
/// `zstd` crate, and the decompressed bytes match the committed plane. Pins
/// the decode API to `zstd::decode_all`.
#[test]
fn zstd_frame_decodes_and_matches_plane() {
    let frame = common::read("zstd_frame.bin");
    let plane = common::read("zstd_plane.bin");
    let out = zstd::decode_all(&frame[..]).expect("decode_all");
    assert_eq!(out, plane, "cross-language zstd decode mismatch");
}

/// PROVE THE NEGATIVE: a single flipped payload bit makes the content-checksum
/// verification fail — libzstd rejects it, not a silent wrong-bytes decode.
#[test]
fn zstd_bitflip_payload_fails_checksum() {
    let mut frame = common::read("zstd_frame.bin");
    // Flip a bit inside the compressed body (not the 4-byte magic).
    let i = frame.len() / 2;
    frame[i] ^= 0x01;
    let r = zstd::decode_all(&frame[..]);
    assert!(r.is_err(), "bit-flipped frame must NOT decode silently");
}

/// PROVE THE NEGATIVE: a truncated frame is a hard error via `decode_all`
/// (the epilogue/checksum can't be reached).
#[test]
fn zstd_truncated_frame_hard_errors() {
    let frame = common::read("zstd_frame.bin");
    let chopped = &frame[..frame.len() - 4];
    assert!(
        zstd::decode_all(chopped).is_err(),
        "truncated frame must error"
    );
    // one-byte "frame" too
    assert!(zstd::decode_all(&frame[..1]).is_err());
}

/// The committed v2 chunk decodes end-to-end and the plane sha256 matches the
/// Python golden (full cross-language agreement on the whole codec).
#[test]
fn v2_chunk_decodes_to_golden_plane() {
    let chunk = common::read("chunk_v2.hf");
    let (hdr, plane) = decode_chunk(&chunk).expect("decode v2 chunk");
    assert_eq!(hdr.version, 2);
    assert_eq!(hdr.width, 33);
    assert_eq!(hdr.height, 33);
    assert_eq!(plane.len(), 33 * 33);

    // sha256 of the raw little-endian plane == golden.plane_sha256
    use sha2::{Digest, Sha256};
    let mut raw = Vec::with_capacity(plane.len() * 2);
    for v in &plane {
        raw.extend_from_slice(&v.to_le_bytes());
    }
    let mut h = Sha256::new();
    h.update(&raw);
    let got = hex(&h.finalize());
    let want = common::golden()["chunk_v2"]["chunk_v2.hf"]["plane_sha256"]
        .as_str()
        .unwrap()
        .to_string();
    assert_eq!(got, want, "decoded plane != python golden");
}

/// v2 header CRC is verified BEFORE dims are trusted: a flipped width byte
/// (CRC now wrong) is rejected as a CRC error, never a giant allocation.
#[test]
fn v2_header_crc_guards_dims() {
    let mut chunk = common::read("chunk_v2.hf");
    chunk[8] ^= 0x01; // corrupt width
    assert_eq!(ChunkHeader::parse(&chunk), Err(HfError::HeaderCrc));
}

/// A corrupt-header giant-dims chunk (width=height=65535, payload_len=1, CRC
/// VALID) must be rejected on the length/frame path — never allocating the
/// 8.59 GB `width*height*2` plane. Proves `plane_bytes()` is computed in u64.
#[test]
fn giant_dims_rejected_without_allocation() {
    let chunk = common::read("chunk_v2_giant.hf");
    // header parses (CRC is valid) but reports an 8.59 GB plane in u64.
    let hdr = ChunkHeader::parse(&chunk).expect("giant header parses");
    assert_eq!(hdr.plane_bytes(), 65535u64 * 65535 * 2);
    assert!(hdr.plane_bytes() > u32::MAX as u64, "must exceed u32");
    // full decode rejects at the frame stage (1-byte non-frame), no alloc.
    assert_eq!(decode_chunk(&chunk), Err(HfError::Zstd));
}

/// Alignment-independence: decoding from a slice offset by 1 byte
/// (`&padded[1..]`) yields identical results. The crate does explicit LE
/// reads and forbids unsafe, so misalignment cannot matter.
#[test]
fn decode_from_misaligned_slice() {
    let chunk = common::read("chunk_v2.hf");
    let mut padded = vec![0xAAu8];
    padded.extend_from_slice(&chunk);
    let (_h0, p0) = decode_chunk(&chunk).unwrap();
    let (_h1, p1) = decode_chunk(&padded[1..]).unwrap();
    assert_eq!(p0, p1);
}

/// Decoding an mmap'd fixture gives the same result as an in-memory buffer.
#[test]
fn decode_from_mmap() {
    let path = common::data_dir().join("chunk_v2.hf");
    let file = std::fs::File::open(&path).unwrap();
    // SAFETY note: memmap2 is safe API; our crate reads the &[u8] explicitly.
    let mmap = unsafe { memmap2::Mmap::map(&file).unwrap() };
    let (_h, p_mmap) = decode_chunk(&mmap).unwrap();
    let (_h2, p_buf) = decode_chunk(&common::read("chunk_v2.hf")).unwrap();
    assert_eq!(p_mmap, p_buf);
}

/// Header shorter than 120 bytes is rejected, not indexed out of bounds.
#[test]
fn short_header_rejected() {
    let chunk = common::read("chunk_v2.hf");
    assert_eq!(
        ChunkHeader::parse(&chunk[..CHUNK_HEADER_SIZE - 1]),
        Err(HfError::TooShort)
    );
}

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}
