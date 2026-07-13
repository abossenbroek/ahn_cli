//! Parse a `.hf` chunk header and print its fields — a header-only read that
//! never decompresses the payload.
//!
//! Paths are relative to the crate root; with no argument it reads the
//! committed fixture:
//!
//! ```text
//! cargo run --example dump_header
//! cargo run --example dump_header -- path/to/tile.hf
//! ```

use std::error::Error;
use std::path::PathBuf;

use ahn_heightfield::ChunkHeader;

/// The committed single-tile fixture, resolved at compile time so the example
/// runs from any working directory.
const DEFAULT_CHUNK: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/leaf.hf");

fn main() -> Result<(), Box<dyn Error>> {
    let path = std::env::args_os()
        .nth(1)
        .map_or_else(|| PathBuf::from(DEFAULT_CHUNK), PathBuf::from);
    let bytes = std::fs::read(&path)?;
    let header = ChunkHeader::parse(&bytes)?;

    println!("file         {}", path.display());
    println!("version      {}", header.version);
    println!("dimensions   {} x {}", header.width, header.height);
    println!("z_offset     {}", header.z_offset);
    println!("z_scale      {}", header.z_scale);
    println!("rtc_centre   {:?}", header.rtc_centre);
    println!("region       {:?}", header.region);
    println!("payload_len  {}", header.payload_len);
    println!("plane_len    {}", header.plane_len());
    println!(
        "error_cap    {}",
        if header.exceeds_error_cap() {
            "EXCEEDED (z_scale/2 > 0.025 m)"
        } else {
            "within 0.025 m"
        }
    );
    Ok(())
}
