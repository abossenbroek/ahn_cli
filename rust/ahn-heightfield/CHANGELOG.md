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
- Public format constants (`CHUNK_MAGIC`, `CHUNK_VERSION`, `CHUNK_HEADER_LEN`,
  `MAX_QUANTIZED_LEVEL`, `ABSOLUTE_ERROR_CAP_M`).
