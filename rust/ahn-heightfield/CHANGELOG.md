# Changelog

All notable changes to this crate are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the crate adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Chunk layer: `ChunkHeader` (with `parse`, `plane_len`, `exceeds_error_cap`,
  `check_error_cap`, and `TryFrom<&[u8]>`) and `Heightfield` (with `decode` and
  the `header`/`width`/`height`/`levels`/`level_at`/`dequantize_at` accessors)
  for the v2 `.hf` heightfield chunk format.
- The shared `HfError` error enum (`#[non_exhaustive]`, one variant per
  normative reject in both specs) and the `Format` tag.
- Archive-layer foundations: the `ReadAt` positioned-read trait (with `File`,
  `&[u8]`, and `&T` implementations), the `TileKey` lookup key (with the spec's
  hand-written `(level, tz, ty, tx)` ordering), and the `BlobSlot` tag.
- Archive layer: `Archive<R>` — the `AHNP` pack reader — with `open` (full
  header/index structural, ordering, alignment and CRC validation; the hash
  section is never read at open), the `header`/`dataset_id`/`entries`/
  `level_run`/`level`/`find`/`children` accessors, `read_primary`/`read_texture`
  opaque-blob reads, `decode_tile` (chunk decode + the entry cross-check and the
  absolute-error cap), and the cold `verify_blobs` install/repair path
  (`dataset_id` + per-blob SHA-256). Plus the `PackHeader`, `Entry`, and
  `LevelRun` parsed-record types.
- Public format constants (`CHUNK_MAGIC`, `CHUNK_VERSION`, `CHUNK_HEADER_LEN`,
  `MAX_QUANTIZED_LEVEL`, `ABSOLUTE_ERROR_CAP_M`; `PACK_MAGIC`,
  `PACK_FORMAT_VERSION`, `PACK_HEADER_LEN`, `PACK_DIR_ENTRY_LEN`,
  `PACK_INDEX_ENTRY_LEN`, `PACK_HASH_ENTRY_LEN`, `PACK_BLOB_ALIGN`).

### Changed

- `sha2` is now a normal dependency (the archive layer's `verify_blobs` needs
  it), rather than a dev-dependency.
