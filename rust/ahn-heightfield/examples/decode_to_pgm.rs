//! Decode a `.hf` chunk and write its quantized height plane as a binary PGM
//! (`P5`) greyscale image — a quick way to eyeball a tile's terrain.
//!
//! Paths are relative to the crate root; with no argument it reads the committed
//! fixture and writes the PGM into the system temp directory:
//!
//! ```text
//! cargo run --example decode_to_pgm
//! cargo run --example decode_to_pgm -- path/to/tile.hf out.pgm
//! ```
//!
//! The PGM uses a `maxval` of 4095 (the 12-bit quantization range), so each
//! sample is written as two big-endian bytes per the PGM spec.

use std::error::Error;
use std::path::PathBuf;

use ahn_heightfield::{Heightfield, MAX_QUANTIZED_LEVEL};

const DEFAULT_CHUNK: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/leaf.hf");

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args_os().skip(1);
    let input = args
        .next()
        .map_or_else(|| PathBuf::from(DEFAULT_CHUNK), PathBuf::from);
    let output = args.next().map_or_else(
        || std::env::temp_dir().join("ahn_heightfield.pgm"),
        PathBuf::from,
    );

    let bytes = std::fs::read(&input)?;
    let tile = Heightfield::decode(&bytes)?;
    let (width, height) = (tile.width(), tile.height());

    // `P5\n<w> <h>\n<maxval>\n` header, then raw big-endian 16-bit samples.
    let mut pgm = format!("P5\n{width} {height}\n{MAX_QUANTIZED_LEVEL}\n").into_bytes();
    for &level in tile.levels() {
        pgm.extend_from_slice(&level.to_be_bytes());
    }
    std::fs::write(&output, &pgm)?;

    println!("decoded {width} x {height} tile from {}", input.display());
    println!("wrote {}-byte PGM to {}", pgm.len(), output.display());
    Ok(())
}
