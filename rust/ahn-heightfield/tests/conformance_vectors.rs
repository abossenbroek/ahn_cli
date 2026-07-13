//! The specs' permanent, normative algorithm-conformance vectors. These pin
//! the exact CRC-32/ISO-HDLC and SHA-256 implementations the crate relies on,
//! independent of any codec or quantization choice.

use sha2::{Digest, Sha256};

const GOLDEN_INPUT: &[u8] = b"AHN heightfield spike golden vector 0123456789";

#[test]
fn crc32_iso_hdlc_golden_vector() {
    assert_eq!(crc32fast::hash(GOLDEN_INPUT), 0xb4b4_1f5a);
}

#[test]
fn sha256_golden_vector() {
    let digest = Sha256::digest(GOLDEN_INPUT);
    let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
    assert_eq!(
        hex,
        "6a8998f8fab0139aaff77ccb9ab123907d58bc55d8b7244e2394f41418731b54",
    );
}
