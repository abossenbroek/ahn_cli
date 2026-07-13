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

## Sibling texture convention

A `.hf` height chunk carries only geometry. The draped texture is a **baseline
sequential JPEG** stored alongside it: for a loose tile `<level>-<tx>-<ty>.hf`
the texture is `<level>-<tx>-<ty>.jpg` (same base name); inside an `AHNP` pack
the `.hf` blob is the entry's *primary* blob and the `.jpg` is its *texture*
blob. This crate decodes the `.hf` geometry; the JPEG is an opaque blob a
runtime hands to its own image decoder.

## Scope

This is a decode-first library. The chunk layer (`ChunkHeader`, `Heightfield`)
is implemented; the archive layer's foundational types (`ReadAt`, `TileKey`,
`BlobSlot`) are in place, with the `Archive` reader and an optional `encode`
feature arriving in later phases.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
