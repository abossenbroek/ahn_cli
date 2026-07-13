# ahn-heightfield

A Rust **decoder** for the AHN heightfield (`.hf`) chunk format and the `AHNP`
pack container. It reads the artifacts produced by the `ahn_cli` Python tool's
`tiles3d --profile heightfield` / `--profile game` commands, coding against the
two normative specifications, not against the Python source:

- the `.hf` chunk format —
  [`docs/superpowers/specs/2026-07-12-heightfield-chunk-format.md`](../../docs/superpowers/specs/2026-07-12-heightfield-chunk-format.md)
- the `AHNP` pack format —
  [`docs/superpowers/specs/2026-07-12-hfp-pack-format.md`](../../docs/superpowers/specs/2026-07-12-hfp-pack-format.md)

The crate is `#![forbid(unsafe_code)]`; every multi-byte field is read with
explicit little-endian byte reads (no `repr(C)` / `bytemuck` casts), so
decoding from an unaligned slice or an mmap is bit-identical.

## Quickstart

```rust
use ahn_heightfield::Heightfield;

// `bytes` is one `.hf` chunk (e.g. a pack's primary blob, or a loose file).
let tile = Heightfield::decode(bytes)?;
let height_m = tile.dequantize_at(row, col); // NAP height in metres
# Ok::<(), ahn_heightfield::HfError>(())
```

`ChunkHeader::parse` validates a header without decompressing (it verifies the
header CRC-32 before trusting any dimension); `Heightfield::decode` additionally
decompresses and validates the `width * height` `uint16` quantized height plane.

To read a whole `tiles.hfp` pack, open it over any positioned-read backing
store (a byte slice, an mmap slice, or a `std::fs::File`):

```rust
use ahn_heightfield::{Archive, TileKey};

// `reader` is anything implementing `ReadAt` — `&[u8]`, a File, an mmap slice.
let archive = Archive::open(reader)?;
let root = archive.find(TileKey { level: 0, tx: 0, ty: 0, tz: 0 }).unwrap();
let tile = archive.decode_tile(root)?;         // decode + entry cross-check
let height_m = tile.dequantize_at(0, 0);       // NAP height in metres
for child in archive.children(root) { /* implicit quadtree children */ }
archive.verify_blobs()?;                        // cold install/repair check
# Ok::<(), ahn_heightfield::HfError>(())
```

`Archive::open` performs the full structural / ordering / alignment / CRC
validation of the header and index but never reads the cold hash section;
`verify_blobs` is the separate install/repair path that checks `dataset_id`
and every per-blob SHA-256. `Archive<R>` is `Send + Sync` whenever `R` is, so
one opened archive serves concurrent tile loads.

## Sibling texture convention

A `.hf` height chunk carries only geometry. The draped texture is a **baseline
sequential JPEG** stored alongside it: for a loose tile `<level>-<tx>-<ty>.hf`
the texture is `<level>-<tx>-<ty>.jpg` (same base name); inside an `AHNP` pack
the `.hf` blob is the entry's *primary* blob and the `.jpg` is its *texture*
blob. This crate decodes the `.hf` geometry; the JPEG is an opaque blob a
runtime hands to its own image decoder.

## Scope

This is a decode-first library. The chunk layer (`ChunkHeader`, `Heightfield`)
and the archive layer (`Archive` over the `ReadAt` positioned-read trait, plus
`PackHeader`, `Entry`, `LevelRun`, `TileKey`, `BlobSlot`) are implemented; an
optional `encode` feature arrives in a later phase.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
