# ahn-heightfield

A Rust **decoder** — and, behind the optional `encode` feature, an encoder for
the chunk layer — for the AHN heightfield (`.hf`) chunk format and the `AHNP`
(`tiles.hfp`) pack container. It reads the artifacts produced by the `ahn_cli`
Python tool's `tiles3d --profile heightfield` / `game` / `splat` commands,
coding against the two normative specifications, not against the Python source:

- the `.hf` chunk format —
  [`docs/specs/2026-07-12-heightfield-chunk-format.md`](../../docs/specs/2026-07-12-heightfield-chunk-format.md)
- the `AHNP` pack format —
  [`docs/specs/2026-07-12-hfp-pack-format.md`](../../docs/specs/2026-07-12-hfp-pack-format.md)

The crate is `#![forbid(unsafe_code)]`; every multi-byte field is read with
explicit little-endian byte reads (no `repr(C)` / `bytemuck` casts), so
decoding from an unaligned slice or an mmap is bit-identical.

- **License:** `MIT`
- **MSRV:** Rust `1.77` (the encoder's `f64::round_ties_even`, the spec's
  normative round-half-even quantization rounding, stabilized in 1.77).

### MSRV policy

The **MSRV** (Minimum Supported Rust Version) is the oldest Rust toolchain
this crate promises to compile on — a compatibility contract for consumers
who pin an older `rustc` (common in CI, enterprises, and Linux distributions).
Cargo reads the `rust-version` field and refuses to build on anything older
with a clear error rather than a confusing compiler failure. Raising an MSRV
can break such consumers, so it is a deliberate, documented change, not an
incidental one.

`rust-version = "1.77"` in `Cargo.toml` is enforced in CI on all three
shipped OSes (`.github/workflows/rust.yml`'s `rust-test` matrix builds and
tests at both `stable` and pinned `1.77`, on Ubuntu, macOS and Windows) — a
transitive dependency bump that raises the effective MSRV fails that job.
Transitive build-dependency drift is additionally capped at the manifest
level: `jobserver` (pulled in via `cc` → `zstd-sys`) declares a higher
`rust-version` from `0.1.35` onward, so `Cargo.toml` carries an explicit
`jobserver = ">=0.1.30, <0.1.35"` constraint (the "phantom pin") alongside
the committed `Cargo.lock`. This means a consumer building on Rust 1.77
resolves a working dependency set by construction, not only because our own
lockfile happens to pin one.

## Features

| Feature  | Default | What it adds                                                                 |
|----------|:-------:|------------------------------------------------------------------------------|
| *(none)* | ✔       | The decoder: the chunk layer (`ChunkHeader`, `Heightfield`) and the archive layer (`Archive`, `ReadAt`, `PackHeader`, `Entry`, …). |
| `encode` |         | The `.hf` chunk **encoder** (`encode_chunk`, `quantize_levels`, `ChunkFields`). Adds no dependencies. |

The `encode` feature is off by default and does not change the default API or
behaviour. It is held to **semantic round-trip** equality (`decode(encode(x))`
reproduces the identical levels and header fields), **not** byte parity with the
Python producer — both specs scope byte-for-byte determinism to the Python
producer alone.

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

## Encoding (`encode` feature)

With `--features encode`, the crate can also **write** a `.hf` chunk: it
quantizes a plane of source NAP heights (12-bit, round-half-even, epsilon scale
for a flat tile), refuses a tile past the 25 mm absolute-error cap, and emits one
zstandard frame (level 3, content checksum) under a CRC-signed header.

```rust
use ahn_heightfield::{encode_chunk, ChunkFields, Heightfield};

let heights = [0.0, 0.5, 1.0, 1.5]; // width*height source NAP heights, row-major
let bytes = encode_chunk(ChunkFields {
    width: 2,
    height: 2,
    rtc_centre: [0.0, 0.0, 0.0],
    region: [0.1, 0.2, 0.3, 0.4, 0.0, 1.5],
    heights: &heights,
})?;
let tile = Heightfield::decode(&bytes)?; // round-trips its own output
# Ok::<(), ahn_heightfield::HfError>(())
```

## Content kinds and the sibling texture convention

A pack's `content_kind` (in the `AHNP` header, `archive.header().content_kind`)
fixes the primary/texture blob layout for every tile in it:

- **`0` (heightfield)** — primary blob is a `.hf` chunk; texture blob is a
  baseline-sequential `.jpg`.
- **`1` (game)** — primary blob is a `.glb` with its texture embedded; **no**
  separate texture blob.
- **`2` (splat)** — primary blob is a zstd-wrapped 3DGS `.ply`; **no** texture
  blob (colour lives in the gaussians). `decode_tile` returns a typed error for
  this kind — it is not a heightfield — but the archive still opens and its
  blobs are readable.

A `.hf` height chunk carries only geometry; the draped texture is stored
alongside it (for a loose tile `<level>-<tx>-<ty>.hf` the texture is
`<level>-<tx>-<ty>.jpg`, same base name). This crate decodes the `.hf`
geometry; the JPEG, the `.glb`, and the `.ply` are opaque blobs a runtime
hands to its own image / glTF / splat decoder.

## Examples

Runnable against the committed fixtures (paths resolve relative to the crate, so
no argument is needed):

```bash
cargo run --example dump_header      # parse a .hf header and print its fields
cargo run --example decode_to_pgm    # decode a .hf and write a greyscale PGM
cargo run --example list_archive     # open a tiles.hfp and list its entries
```

Each accepts an optional path argument to point at your own artifact.

## Fuzzing

Two [`cargo-fuzz`](https://github.com/rust-fuzz/cargo-fuzz) targets live in
`fuzz/` (their own isolated workspace): `decode` (the `.hf` chunk layer) and
`archive_open` (the `AHNP` reader, including `verify_blobs` and per-tile decode).
Both assert the decoder never panics on arbitrary bytes. Fuzzing is **not** part
of `cargo test`; running it needs a nightly toolchain:

```bash
cargo install cargo-fuzz
cargo +nightly fuzz run decode
cargo +nightly fuzz run archive_open
```

On a stable toolchain the fuzz crate still `cargo check`s (`cd fuzz && cargo
check`). The corpora are seeded from the committed fixtures plus truncated and
bad-magic reject variants.

## Scope

This is a decode-first library. The chunk layer (`ChunkHeader`, `Heightfield`)
and the archive layer (`Archive` over the `ReadAt` positioned-read trait, plus
`PackHeader`, `Entry`, `LevelRun`, `TileKey`, `BlobSlot`) are implemented, plus
the optional `encode` feature's `.hf` chunk encoder. The pack (`AHNP`) side is
read-only: the normative pack producer is the Python tool.

## License

Licensed under the [MIT license](LICENSE-MIT), matching the parent `ahn_cli`
project.
