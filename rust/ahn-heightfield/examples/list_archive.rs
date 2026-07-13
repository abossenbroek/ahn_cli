//! Open an `AHNP` pack and list its header and index entries — the cheap
//! `open` path only (no blob reads, no hash-section verification).
//!
//! Paths are relative to the crate root; with no argument it reads the committed
//! fixture:
//!
//! ```text
//! cargo run --example list_archive
//! cargo run --example list_archive -- path/to/tiles.hfp
//! ```

use std::error::Error;
use std::fmt::Write as _;
use std::path::PathBuf;

use ahn_heightfield::Archive;

const DEFAULT_PACK: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/tiles.hfp");

fn main() -> Result<(), Box<dyn Error>> {
    let path = std::env::args_os()
        .nth(1)
        .map_or_else(|| PathBuf::from(DEFAULT_PACK), PathBuf::from);
    let bytes = std::fs::read(&path)?;
    let archive = Archive::open(&bytes[..])?;
    let header = archive.header();

    println!("pack          {}", path.display());
    println!(
        "content_kind  {} ({})",
        header.content_kind,
        if header.content_kind == 0 {
            "heightfield: .hf + .jpg"
        } else {
            "game: .glb"
        }
    );
    println!("tiles         {}", header.tile_count);
    println!("levels        {}", header.level_count);
    println!("root_gerror   {}", header.root_geometric_error);
    println!("dataset_id    {}", hex32(&archive.dataset_id()));
    println!("entries:");
    for (i, e) in archive.entries().iter().enumerate() {
        println!(
            "  [{i}] level {} tile ({},{})  primary {} B  texture {} B  gerror {}",
            e.level, e.tx, e.ty, e.primary_size, e.texture_size, e.geometric_error
        );
    }
    Ok(())
}

/// Renders a 32-byte digest as 64 lowercase hex characters.
fn hex32(bytes: &[u8; 32]) -> String {
    let mut s = String::with_capacity(64);
    for b in bytes {
        let _ = write!(s, "{b:02x}");
    }
    s
}
