# The `AHNP` pack format (`tiles.hfp`, version 1) — normative

This document is **normative**: the Rust runtime (`ahn-heightfield`
crate's `Archive` layer) and the stakeholder's game code against *this
specification*, not against the Python source. `ahn_cli`'s `tiles3d`
`--profile game` and `--profile heightfield` write a single
`tiles.hfp` pack that bundles every content blob plus a self-describing
binary scene index. **The pack is the scene**: the runtime opens the pack
and, on its play-time path, never parses JSON. The Python reference
producer/reader (`ahn_cli/tiles3d/pack.py`) and verifier implement exactly
what follows.

The individual `.hf` height chunks a heightfield pack carries are specified
in [`2026-07-12-heightfield-chunk-format.md`](2026-07-12-heightfield-chunk-format.md);
this document specifies only the container.

All multi-byte integers and floats are **little-endian**. Floats are IEEE
754 (`float64` = binary64). Hashes are SHA-256; CRCs are CRC-32/ISO-HDLC
(bit-identical between Python `zlib.crc32` and the Rust `crc32fast`
crate). The pack is the runtime's **only** input besides the blobs it
already contains; `tileset.json`, `provenance.json` and `manifest.json`
are demoted to deterministic debug / interop / integrity sidecars.

## File layout (regions, in file order)

A `tiles.hfp` file is five contiguous regions in this fixed order:

| Region | Offset | Size | Contents |
|---|---|---|---|
| Header | `0` | `128` | Fixed pack header (below). |
| Index | `index_offset` (`= 128`) | `index_size` | Level directory (`level_count × 16 B`) immediately followed by the index entries (`tile_count × 96 B`). |
| Hash section | `hash_offset` | `hash_size` (`= tile_count × 64 B`) | Per-tile `primary_sha256` + `texture_sha256`, in index order. Cold: read only for install/repair, never at open. |
| Blob region | first 16-aligned offset `>= hash_offset + hash_size` | to EOF | Every content blob, concatenated in **index order**, each 16-byte aligned with zero inter-blob padding. |

with the invariants:

```
index_offset = 128
index_size   = level_count * 16 + tile_count * 96
hash_offset  = index_offset + index_size          (16-aligned: both terms are multiples of 16)
hash_size    = tile_count * 64
file_size    = total byte length of the file
```

Because `128`, `index_size` and `hash_size` are all multiples of 16, the
blob region begins at a 16-aligned offset (`hash_offset + hash_size`) with
no leading pad. A reader recomputes each of `index_offset`, `index_size`,
`hash_offset`, `hash_size` from the header counts and **rejects** any that
disagree with the stored header values (see *Rejects*).

All derived sizes and offsets (`index_size`, `hash_size`, every blob
offset/size comparison, and every bounds check against `file_size`) **must
be computed in 64-bit arithmetic** (`u64`), converting to `usize` only
through a checked conversion that itself rejects on a 32-bit host. The
counts are `uint32`, so `level_count * 16 + tile_count * 96` cannot wrap a
`u64`, but a native-`usize` intermediate on a 32-bit target could — do the
arithmetic in `u64`, mirroring the chunk spec's `width * height * 2` rule.

## Pack header (128 bytes, offsets in bytes)

| Offset | Size | Type       | Field                  | Meaning / reject condition                                              |
|-------:|-----:|------------|------------------------|------------------------------------------------------------------------|
| 0      | 4    | `char[4]`  | `magic`                | ASCII `AHNP` (`0x41 0x48 0x4E 0x50`). Any other value is rejected.      |
| 4      | 4    | `uint32`   | `format_version`       | Pack format version. This document is version `1`. Any other value is rejected. |
| 8      | 4    | `uint32`   | `tile_count`           | Number of tiles = number of index entries = number of hash records.    |
| 12     | 4    | `uint32`   | `level_count`          | Number of quadtree levels = number of level-directory records.         |
| 16     | 8    | `uint64`   | `index_offset`         | Must equal `128`.                                                       |
| 24     | 8    | `uint64`   | `index_size`           | Must equal `level_count * 16 + tile_count * 96`.                        |
| 32     | 8    | `uint64`   | `hash_offset`          | Must equal `index_offset + index_size`.                                |
| 40     | 8    | `uint64`   | `hash_size`            | Must equal `tile_count * 64`.                                           |
| 48     | 8    | `uint64`   | `file_size`            | Must equal the actual file length (truncation/trailing-byte check without trusting `stat`). |
| 56     | 8    | `float64`  | `root_geometric_error` | Bit-equal to the `tileset.json` document's top-level `geometricError`. Lets SSE traversal start with no JSON. |
| 64     | 32   | `u8[32]`   | `dataset_id`           | SHA-256 over the **hash section** bytes `[hash_offset, hash_offset + hash_size)`. The content version (Merkle-style root). |
| 96     | 4    | `uint32`   | `index_crc32`          | CRC-32/ISO-HDLC over the **index region** bytes `[index_offset, index_offset + index_size)`. Mismatch is rejected. |
| 100    | 4    | `uint32`   | `reserved`             | Reserved; must be `0`. Any other value is rejected. (Vacated by moving `header_crc32` to offset 124; see *Header CRC*.) |
| 104    | 4    | `uint32`   | `content_kind`         | `0` = heightfield, `1` = game, `2` = splat. Any other value is rejected. See *Content kind*. |
| 108    | 16   | `u8[16]`   | `pad`                  | Reserved; every byte must be `0`. Any non-zero byte is rejected.        |
| 124    | 4    | `uint32`   | `header_crc32`         | CRC-32/ISO-HDLC over header bytes `[0, 124)`. Mismatch is rejected.     |

Total: `128` bytes.

### Header CRC (integrity of the fixed header)

`header_crc32` covers header bytes `[0, 124)` — **every** header field
from `magic` through `pad` inclusive, i.e. including `index_crc32`,
`reserved`, `content_kind` and `pad`, and excluding only `header_crc32`
itself. A conforming reader **verifies `header_crc32` first**, before
trusting any count and before sizing any allocation from `tile_count` /
`level_count`. `index_crc32` (at offset 96) is thus doubly protected: by
`header_crc32` as a header field, and it in turn protects the index region.

> Note on the layout: an earlier spike draft placed `header_crc32` at
> offset 100 covering only `[0, 100)`, which left `content_kind` (then at
> 104) outside the CRC. This version moves `header_crc32` to offset 124 and
> extends its span to `[0, 124)` so `content_kind` and `pad` are covered,
> and defines the vacated `[100, 104)` slot as a zero `reserved` field.

### Content kind

`content_kind` selects the on-disk representation of both blob slots for
**every** tile in the pack (a pack is homogeneous — all tiles share one
kind):

| `content_kind` | Profile | Primary blob | Texture blob |
|---:|---|---|---|
| `0` | heightfield | `.hf` chunk (see the chunk spec) | baseline JPEG (`.jpg`) — always present, `texture_size > 0` |
| `1` | game | binary glTF (`.glb`), texture embedded in the glb | **none**: `texture_offset = 0` and `texture_size = 0` |
| `2` | splat | `.ply` (3DGS, zstd-wrapped) | **none**: `texture_offset = 0` and `texture_size = 0` (colour lives in the gaussians' SH coefficients) |

Any other `content_kind` value is rejected. The reader uses
`content_kind` to resolve the URI↔key file extension (see *URI ↔ key
mapping*) and to know whether a texture blob exists.

> **Revision note (splat, non-breaking).** `content_kind = 2` (splat) was
> added after this document's initial version, alongside every other
> `content_kind ∈ {1, 2} ⇒ no texture` rule above. This is a **non-breaking
> addition**, not a format-version bump: `format_version` stays `1`, the
> container layout is unchanged, and `content_kind` was always inside the
> `header_crc32` span (see *Header CRC*) — a pack written before this
> revision simply never had a tile with `content_kind = 2`. A conforming
> reader built against the pre-splat `{0, 1}` set must be updated to accept
> `2` and to treat it as a no-texture kind (like `1`); it otherwise needs no
> other change, since a splat pack's primary blob is opaque bytes to the
> container exactly like a `.hf` chunk or a `.glb` is.

### `dataset_id` (content version)

`dataset_id` is the SHA-256 digest of the **hash-section bytes**
`[hash_offset, hash_offset + hash_size)` — that is, SHA-256 over the
concatenation of every tile's `primary_sha256 || texture_sha256` in index
order. It is a Merkle-style content root:

- deterministic and content-derived (no timestamps, no counters): an
  identical rebuild yields an identical `dataset_id`;
- any change to any blob, or any added/removed tile, changes at least one
  record in the hash section and therefore changes `dataset_id`;
- a client staleness check is a 32-byte compare read from one 128-byte
  header read — no blob or index scan.

A reader **may** recompute `dataset_id` from the hash section and reject a
mismatch (an install/repair-time check; anchors the hash section against
header corruption).

## Level directory (`level_count × 16 B`)

Immediately after the header, at `index_offset = 128`, comes the level
directory: one 16-byte record per quadtree level, ordered by ascending
`level` (`0` = root). Each LOD's index entries form one contiguous run.

| Offset (within record) | Size | Type     | Field         | Meaning                                                    |
|-----------------------:|-----:|----------|---------------|------------------------------------------------------------|
| 0                      | 4    | `uint32` | `first_entry` | Index of this level's first entry (0-based, into the entry array). |
| 4                      | 4    | `uint32` | `entry_count` | Number of entries at this level.                           |
| 8                      | 4    | `uint32` | `tx_count`    | Number of distinct `tx` columns at this level.             |
| 12                     | 4    | `uint32` | `ty_count`    | Number of distinct `ty` rows at this level.                |

Record `l` describes level `l`. The runs are contiguous and cover the
whole entry array exactly once: `directory[0].first_entry == 0`;
`directory[l].first_entry == directory[l-1].first_entry +
directory[l-1].entry_count`; and `sum(entry_count) == tile_count`. A
reader rejects any directory that violates these (see *Rejects*).

`tx_count` / `ty_count` bound a level's grid for row-slice addressing; a
level need not be a full dense `tx_count × ty_count` grid (sparse /
non-square source rasters produce absent `(tx, ty)` cells), so a reader
must not assume `entry_count == tx_count * ty_count`.

## Index entry (`96 B` each) — one per tile

The index entries follow the level directory, at `128 + level_count * 16`.
There is exactly one entry per tile, **sorted ascending by
`(level, tz, ty, tx)`** — level-major then row-major — so altitude-driven
look-ahead queries over a level are contiguous scans.

| Offset (within entry) | Size | Type      | Field            | Meaning / reject condition                                          |
|----------------------:|-----:|-----------|------------------|--------------------------------------------------------------------|
| 0                     | 4    | `uint32`  | `level`          | Quadtree level (`0` = root).                                       |
| 4                     | 4    | `uint32`  | `tx`             | Tile column index at this level.                                   |
| 8                     | 4    | `uint32`  | `ty`             | Tile row index at this level.                                     |
| 12                    | 4    | `uint32`  | `tz`             | Tile depth index. **Must be `0`** in this version; any other value is rejected. |
| 16                    | 8    | `float64` | `region[0]`      | West longitude, radians (EPSG:4979). The **union/enclosing** region — see *Region semantics*. |
| 24                    | 8    | `float64` | `region[1]`      | South latitude, radians.                                          |
| 32                    | 8    | `float64` | `region[2]`      | East longitude, radians.                                          |
| 40                    | 8    | `float64` | `region[3]`      | North latitude, radians.                                          |
| 48                    | 8    | `float64` | `region[4]`      | Minimum ellipsoidal height, metres.                              |
| 56                    | 8    | `float64` | `region[5]`      | Maximum ellipsoidal height, metres.                              |
| 64                    | 8    | `float64` | `geometric_error`| Tile geometric error (leaves `0`). Bit-equal to the tile's `tileset.json` `geometricError`. |
| 72                    | 8    | `uint64`  | `primary_offset` | Absolute file offset of the primary blob. **16-byte aligned.**   |
| 80                    | 8    | `uint64`  | `texture_offset` | Absolute file offset of the texture blob (16-aligned when `texture_size > 0`; `0` when there is no texture). |
| 88                    | 4    | `uint32`  | `primary_size`   | Byte length of the primary blob.                                 |
| 92                    | 4    | `uint32`  | `texture_size`   | Byte length of the texture blob (`0` when there is no texture). |

Total: `96` bytes.

**Children are implicit.** A tile's children are exactly those entries
present at `(level + 1, {2·tx, 2·tx + 1}, {2·ty, 2·ty + 1}, tz = 0)`; a
child that is absent from the index simply does not exist (sparse rasters).
There are no child pointers.

### Region semantics (enclosing region — differs from the chunk header)

Each entry's `region` is the tile's **union / enclosing** region: the
tile's own mesh region unioned with all its descendants' regions, so a
parent bounding volume encloses its children (a 3D Tiles validity rule).
This is **bit-equal to the `tileset.json` bounding volume** for the same
tile — leaf and parent alike — and it is what culling / LOD selection
needs.

This is **not** the same six doubles as the tile's `.hf` chunk-header
`region`, which is the tile's *own* mesh region. For leaf tiles the two
regions are fully bit-equal; for parent tiles the four horizontal doubles
are bit-equal but the chunk header's height range is a **contained subset**
of this entry's height range (see the chunk spec's *Region semantics*).
Consequently the wrong-tile-under-right-key guard is **not** a blanket
six-double bit-compare of the chunk header against this entry; it is
horizontal-bit-equal + height-containment + the per-blob SHA-256 below.

### Blob offsets, alignment and order

- `primary_offset` and (when present) `texture_offset` are **absolute file
  offsets**, each a multiple of **16**.
- Blobs are stored in **index order** (`primary` before its own `texture`;
  entry `i`'s blobs before entry `i+1`'s), so blob offsets are strictly
  ascending in index order.
- Inter-blob padding (the bytes between the end of one blob and the
  16-aligned start of the next) is **zero**. A reader rejects non-zero
  padding.
- Blobs do not overlap and lie wholly within `[blob_region_start,
  file_size)`.
- For `content_kind = 1` (game) or `content_kind = 2` (splat) every entry
  has `texture_offset = 0` and `texture_size = 0` (no texture blob is
  stored). For `content_kind = 0` (heightfield) every entry has a texture
  blob (`texture_size > 0`, offset 16-aligned).

## Hash section (`tile_count × 64 B`)

At `hash_offset`, one 64-byte record per tile in index order:

| Offset (within record) | Size | Type      | Field            | Meaning                                                        |
|-----------------------:|-----:|-----------|------------------|----------------------------------------------------------------|
| 0                      | 32   | `u8[32]`  | `primary_sha256` | SHA-256 of the tile's primary blob bytes.                     |
| 32                     | 32   | `u8[32]`  | `texture_sha256` | SHA-256 of the tile's texture blob bytes, or a sentinel when there is no texture (below). |

**No-texture sentinel (decided).** When `content_kind = 1` (game) or
`content_kind = 2` (splat) there is no texture blob, so `texture_sha256` is
**32 zero bytes** (all-zeros), matching the zeroed `texture_offset` /
`texture_size` slots. The absence of a texture is signalled uniformly by
zeros; a verifier checks `content_kind ∈ {1, 2} ⇒ texture_sha256 == 0…0`
directly, without hashing an empty input. (The alternative — SHA-256 of the
empty string — was rejected as implying a present-but-empty texture and
requiring a hash to express "no texture".)

The hash section is **cold**: it is read only by install/repair and by the
verifier's per-blob integrity pass, never on the play-time open path. It is
anchored by `dataset_id` (SHA-256 of this whole section).

## URI ↔ key mapping (`tileset.json` sidecar)

`tileset.json` is retained as a deterministic sidecar. Its `content.uri`
values map one-to-one onto pack index keys by a single **strict** parse:

```
tiles/<level>-<tx>-<ty>.<ext>
```

where `<level>`, `<tx>`, `<ty>` are the entry's `level`, `tx`, `ty`
rendered as base-10 ASCII integers with **no leading zeros** (a bare `0`
where the value is zero), and `<ext>` is fixed by `content_kind`:

| `content_kind` | `<ext>` |
|---:|---|
| `0` (heightfield) | `hf` |
| `1` (game) | `glb` |
| `2` (splat) | `ply` |

`tz` is `0` in this version and does **not** appear in the URI. A parse is
strict: the leading `tiles/` segment, the two `-` separators, a `.`
before a matching `<ext>`, base-10 integers with no leading zeros, and
nothing else — any deviation is rejected. Each pack entry maps to exactly
one `tileset.json` URI and vice versa (a one-to-one, onto mapping the
verifier checks in both directions).

## Rejects (conforming-reader checklist)

A conforming reader rejects (does not silently repair) any of the
following. Verify in an order that never trusts an unverified count or
length:

**Header**

- `magic != "AHNP"`;
- `format_version != 1`;
- input shorter than 128 bytes;
- `header_crc32` does not match CRC-32/ISO-HDLC of bytes `[0, 124)`
  (checked **before** any count is trusted);
- `tile_count == 0` or `level_count == 0` (an empty pack is never
  produced; every valid pack has at least one tile and one level), or
  `level_count > tile_count` (each level holds at least one tile);
- `root_geometric_error` is non-finite (`NaN` / `±Inf`);
- `reserved` (offset 100) `!= 0`; any byte of `pad` (offset `[108, 124)`)
  `!= 0`;
- `content_kind` not in `{0, 1, 2}`;
- `index_offset != 128`;
- `index_size != level_count * 16 + tile_count * 96`;
- `hash_offset != index_offset + index_size`;
- `hash_size != tile_count * 64`;
- `file_size` does not equal the actual file length (truncation or
  trailing bytes);
- `index_crc32` does not match CRC-32/ISO-HDLC of the index region
  `[index_offset, index_offset + index_size)`.

**Level directory**

- `directory[0].first_entry != 0`; a run discontinuity
  (`directory[l].first_entry != directory[l-1].first_entry +
  directory[l-1].entry_count`); or `sum(entry_count) != tile_count`.

**Index entries**

- entries not strictly ascending by `(level, tz, ty, tx)` (unsorted or a
  duplicate key);
- any `tz != 0`;
- any **non-finite** value (`NaN` / `±Inf`) in an entry's six `region`
  doubles or its `geometric_error`;
- an entry `region` that is not well-ordered (`region[0] > region[2]`,
  `region[1] > region[3]`, or `region[4] > region[5]` — west/south/min
  must not exceed east/north/max);
- a `level` outside `[0, level_count)`, or an entry whose `level` places
  it outside its level-directory run;
- `primary_offset` not 16-aligned, or (when `texture_size > 0`)
  `texture_offset` not 16-aligned;
- blob offsets not strictly ascending in index order, blob ranges
  overlapping, or any range extending past `file_size` or starting before
  the blob region;
- bytes present after the last blob — the read cursor at the end of the
  blob scan does not equal `file_size` (a conforming pack ends exactly at
  its final blob, so any trailing bytes or trailing gap is rejected);
- non-zero inter-blob padding;
- for `content_kind = 1` or `content_kind = 2`: any entry with
  `texture_offset != 0` or `texture_size != 0`; for `content_kind = 0`:
  any entry with `texture_size == 0`;
- a blob whose decoded content does not match its entry (e.g. a `.hf` blob
  whose chunk-header region is not horizontally bit-equal to and
  height-contained within the entry region — a semantic cross-check the
  verifier performs).

**Hash section / dataset_id** (install/repair, not at open)

- any `primary_sha256` / `texture_sha256` that does not match the SHA-256
  of the corresponding blob (a `texture_sha256` for a `content_kind = 1`
  or `content_kind = 2` tile must be 32 zero bytes);
- `dataset_id` not equal to SHA-256 of the hash section.

**Truncation** at any point (mid-header, mid-directory, mid-index,
mid-hash, mid-blob, or a missing final byte) is caught by the combination
of the length invariants (`index_size`, `hash_size`, `file_size`), the
CRCs, and — for a corrupted-but-right-length blob — the per-frame content
checksums (zstd for `.hf`) and the hash section.

## Producer write protocol (informative)

The producer is single-pass and bounded-memory (all sizes are known up
front from the quadtree plan — `tile_count`, `level_count`, and each tile's
blob sizes):

1. Compute the tiling plan and, for each tile in index order, the blob
   bytes and their sizes.
2. Write the 128-byte header with the counts and offsets filled in but
   `dataset_id`, `index_crc32` and `header_crc32` left zero; write a
   zeroed level directory, a zeroed index region and a zeroed hash section.
3. Stream the blobs into the blob region in index order, 16-aligning each
   (writing zero padding), recording each blob's offset, size and SHA-256.
4. Seek back and patch the level directory, the index entries (offsets /
   sizes / regions / errors) and the hash section.
5. Compute `dataset_id` = SHA-256 of the now-final hash section, then
   `index_crc32` over the index region, then `header_crc32` over `[0, 124)`;
   patch the header.

The spike verified this single-pass seek-back build is **byte-identical**
to a two-pass in-memory build, and that re-encoding the same inputs
reproduces the identical pack (encode-twice determinism), on all three
OSes.

**Producer determinism scope.** Byte-for-byte pack determinism is a
property of the **Python producer** with its pinned libraries (recorded in
`provenance.json`, including the producer platform, since JPEG bytes are
pinned per Pillow / libjpeg-turbo build). A Rust encoder, if one is ever
added, is held to semantic equivalence, not byte parity.

## Runtime access pattern (informative)

- **Open**: one 128-byte header read → verify `header_crc32` → compare
  `dataset_id` (staleness) → one contiguous read of the index region
  (~205 KB/city) → verify `index_crc32` → parse the level directory and
  entries once into native structures (explicit little-endian field reads;
  no unaligned casts). The hash section is **not** read at open.
- **Per frame**: pick the level-directory run for the active LOD, then a
  row-slice or a linear AABB scan over the resident entries (the spike
  measured ~8–12 µs for a 2000-entry linear scan on all three OSes — well
  inside the sub-100 µs budget), then one positioned read per blob entering
  the resident set. Children are found by probing the index for the four
  implicit child keys.
- Chunk / glb headers are touched only at decode time. Positioned reads
  (`pread` on unix, `seek_read` on Windows, or an mmap slice) are
  cursor-independent and safe for concurrent tile loads; the format uses
  **explicit little-endian reads**, never `repr(C)` / `bytemuck` casts, so
  decoding from an mmap or an unaligned slice is bit-identical (the 16-byte
  blob alignment is a spec property reserved for possible future zero-copy,
  not a current parsing requirement).

## Deliverable root layout

All three lossy profiles (game, heightfield and splat) write the same set
of files; only `content_kind` and the blob contents differ:

```
out/
  tiles.hfp        # all content blobs + the binary scene index (the runtime's ONLY input besides its own blobs)
  tileset.json     # demoted sidecar: deterministic debug/interop witness; the game never opens it
  provenance.json  # gains a "pack" block (magic, version, alignment, sha256 algorithm, dataset_id hex, producer platform) + existing fields
  manifest.json    # integrity manifest over every loose file + tiles.hfp
```

The `strict` profile is untouched by this format: it keeps its loose
`tiles/` directory of float32 glTF + PNG and writes no pack. The
build's crash-safe two-phase accept-marker swap is unchanged; its
tool-owned artifact set for the lossy profiles becomes `{tiles.hfp,
tileset.json, provenance.json, manifest.json}`.

### `manifest.json` shape

`manifest.json` is a byte-oriented integrity sidecar written with **LF
(`\n`) newlines on every platform**, UTF-8, `sort_keys`, 2-space indent,
and a trailing newline (byte-deterministic across OSes):

```json
{
  "algorithm": "sha256",
  "dataset_id": "<64 hex chars>",
  "files": {
    "<relative-path>": { "sha256": "<64 hex chars>", "size": <integer bytes> }
  }
}
```

- `algorithm` is the fixed string `"sha256"`.
- `files` maps each **loose file plus `tiles.hfp`** (relative path from
  `out/`, forward-slash separated) to its SHA-256 hex digest and byte
  size. Keys are sorted.
- `dataset_id` repeats the pack's `dataset_id` as 64 lowercase hex
  characters, tying the manifest to the exact pack content.

`tileset.json`, `provenance.json` and `manifest.json` (and any other text
sidecar) are all written with LF newlines on every platform — the Windows
CRLF text-mode bug the v1 producer had is fixed, and LF is stated here
normatively for any file the manifest hashes, so a manifest recomputed on
any OS matches.

## Two-encodings witness (informative — build-time verifier)

The build produces two encodings of the same scene — the binary pack and
the `tileset.json` sidecar — and the verifier rejects the build unless they
agree bit-for-bit where they overlap: every tile's six-double `region`
(f64 bit patterns after JSON round-trip) and `geometricError`, the
`root_geometric_error` vs the tileset's top-level `geometricError`, and the
URI↔key mapping (one-to-one, onto). It also recomputes `dataset_id`, the
hash section and `manifest.json`, and byte-compares the whole artifact set
against an independent rebuild. This is a producer-side guarantee; a
runtime consumer needs only the pack.

## HTTP delivery (informative)

The pack is designed for local-disk streaming, but its single-file +
absolute-offset layout also suits HTTP range delivery: a client fetches
the 128-byte header (a `Range: bytes=0-127` request), then the index region
(`[index_offset, index_offset + index_size)`), then per-tile blob ranges on
demand. The hash section and `manifest.json` support integrity /
resumable-repair over an unreliable transport. No part of the format
requires the whole file to be resident.

## Rejected alternatives (recorded)

These were considered and rejected during design (spike-validated where
noted); recorded so they are not re-litigated:

- **CRC32C / XXH64 for the header/index hashes** — no Python stdlib
  implementation; CRC-32/ISO-HDLC (`zlib.crc32`) is stdlib on the producer
  and `crc32fast` on the consumer, bit-identical (golden-vector verified).
- **`zip` / 3TZ container** — general-purpose, not byte-deterministic
  across toolchains, and carries per-entry overhead and a central directory
  we do not need.
- **SQLite** (e.g. MBTiles-style) — not byte-deterministic; a heavyweight
  dependency for a read-mostly, append-once index.
- **RPF-style name hashing / a string table** — unnecessary: `(level, tx,
  ty)` keys map to URIs by one strict parse; no string interning needed.
- **zstd dictionaries** — the per-tile height payload is near-incompressible
  (§codec bake-off); a shared dictionary buys nothing and adds a
  cross-version coupling.
- **4096-byte blob alignment** — 16-byte alignment is enough for any future
  zero-copy field read and wastes far less padding on ~100 KB blobs.
- **Per-entry child pointers** — children are implicit in the `(level+1,
  2tx{+1}, 2ty{+1})` key presence; explicit pointers spend bytes for
  nothing.
- **Octree subdivision of the terrain** — the data is 2.5D; a real z-axis
  would be fabricated. `tz` is kept in the key (always `0`) so a future
  COPC-style lidar pack can reuse this exact header / directory / index /
  hash machinery with `tz` meaningful, at zero cost now — but nothing in
  this version subdivides in depth.
- **Cryptographic signing** — out of scope for v1; `dataset_id` provides
  content addressing and tamper-evidence within the trust boundary. Signing
  is a possible future layer over `dataset_id`, noted here only so the field
  choice is understood.

## Unified `(level, x, y, z)` key rationale (informative)

Every index entry and the sort order carry a `tz` (depth) component that is
always `0` in this version. It is retained deliberately: the same header /
level-directory / 96-byte-entry / hash-section machinery is meant to be
reused, unchanged, by a future COPC-style lidar pack where `tz` is a
meaningful octree depth. Keeping the key shape `(level, x, y, z)` now costs
four zero bytes per entry and spares a format-version break later; this
version builds nothing that uses `tz != 0` and rejects it.
