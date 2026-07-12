//! Shared helpers for the spike integration tests: locate committed fixtures
//! and golden vectors, load bytes.
//!
//! Compiled into each test binary; not every binary uses every helper.
#![allow(dead_code)]

use std::path::PathBuf;

pub fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data")
}

/// The repo's committed loose-file fixtures (two dirs up from the crate).
pub fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/tiles3d/fixtures/rust-consumer")
}

pub fn read(path: &str) -> Vec<u8> {
    let p = data_dir().join(path);
    std::fs::read(&p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()))
}

pub fn golden() -> serde_json::Value {
    let bytes = read("golden.json");
    serde_json::from_slice(&bytes).expect("parse golden.json")
}
