# The `.hf` heightfield chunk format (version 2) — normative

This document is **normative**: the Rust runtime decoder (the
`ahn-heightfield` crate) codes against *this specification*, not against
the Python source. `ahn_cli`'s `tiles3d --profile heightfield` writes one
`.hf` chunk plus one baseline JPEG per quadtree tile; the Python reference
codec (`ahn_cli/tiles3d/heightfield.py`) and its verifier
(`ahn_cli/tiles3d/verify_heightfield.py`) implement exactly what follows.
The companion pack container that bundles these chunks for the runtime is
specified separately in
[`2026-07-12-hfp-pack-format.md`](2026-07-12-hfp-pack-format.md).

A `.hf` chunk is a fixed little-endian header immediately followed by a
single zstandard frame. The frame decompresses to the tile's quantized
elevation plane; everything else a runtime needs to rebuild the tile mesh
(vertex X/Y, texture coordinates, triangle connectivity) is **implicit**
and reconstructed from the header — see *Grid-reconstruction contract*.

All multi-byte integers and floats are **little-endian**. Floats are IEEE
754 (`float64` = binary64). There is no padding *inside* the field block:
the header is exactly `120` bytes, laid out with the equivalent of Python
`struct.pack("<4sIII" + "d" * 11 + "Q" + "II", ...)`.

## v1 → v2 delta (v1 never shipped)

Version 1 of this format was specified but **never shipped** (no `.hf`
chunk produced by a released tool is version 1). Version 2 is the first
normative, consumed-by-Rust version. The differences from the v1 draft:

1. **Header grows 112 → 120 bytes.** All v1 fields keep their v1 offsets.
   Two `uint32` fields are appended: `header_crc32` at offset `112`
   (CRC-32/ISO-HDLC over header bytes `[0, 112)`) and `pad` at offset
   `116` (must be `0`). The header stays an 8-byte multiple.
2. **`version` field value is `2`.** A `version` of `1` (or anything
   else) is a decode error.
3. **Payload codec: zstd level 19 → level 3, with the RFC 8878 content
   checksum now written.** The frame is still exactly one zstandard frame,
   but produced with `write_checksum=True` at level `3`. The checksum is a
   new integrity layer verified natively by libzstd on decode.
4. **Height quantization: 16-bit → 12-bit levels.** The affine scheme is
   unchanged, but the maximum quantization level is `4095` (not `65535`),
   so `z_scale = extent / 4095`. Levels are still stored as `uint16` LE in
   the same 2-bytes-per-sample container.
5. **New absolute-error cap (normative reject).** A tile whose exported
   height bound `z_scale / 2` exceeds `0.025 m` is refused by the producer
   and the verifier (see *Quantization*).
6. **New decode rejects:** `width == 0` or `height == 0`; `header_crc32`
   mismatch; `pad != 0`; zstd content-checksum mismatch; any stored level
   `> 4095`. See *Decode errors*.
7. **Region-semantics correction.** The v1 text claimed the header
   `region` is "bit-for-bit the same six doubles as this tile's
   `tileset.json` bounding volume." This is **false for parent tiles** and
   is corrected here: the header `region` is the tile's *own* mesh region,
   which is contained within — not always equal to — the enclosing
   `tileset.json` / pack-index region. See *Region semantics*.

## Coordinate contract (load-bearing — read this first)

The stored plane is the **quantized NAP height plane** — the genuine
sampled source heights (`payload.z`, EPSG:7415 vertical / NAP metres),
quantized to `uint16` along that single axis. It is **not** an
ECEF-swizzled mesh axis. This is the defining difference from the
`strict`/`game` (Approach A) profiles, whose glTF stores all three
quantized ECEF-RTC axes:

- A heightfield tile is a **geodetic-grid product**. The tile's geodetic
  footprint (longitude/latitude extent and the min/max ellipsoidal
  height) travels in the header `region`. The runtime reconstructs each
  vertex's geodetic position `(lon, lat, h)` from `region` + the implicit
  uniform vertex grid + the dequantized height `h`, then transforms to
  whatever frame it renders in.
- `rtc_centre` is carried in the header **only as a convenience anchor**:
  it is the ECEF y-up RTC centre of the *same tile in the A profile*, so
  a consumer can align a heightfield tile with an A-profile tile without
  recomputing geodesy. It is metadata, not the vertical datum of the
  stored plane. A decoder that only wants heights can ignore it.

This keeps the format authentic (every stored value is a real source
sample requantized, never averaged or infilled) and hits the geometry
size target of ~1–1.5 bytes/pixel.

## Header layout (120 bytes, offsets in bytes)

| Offset | Size | Type       | Field           | Meaning                                                                 |
|-------:|-----:|------------|-----------------|-------------------------------------------------------------------------|
| 0      | 4    | `char[4]`  | `magic`         | ASCII `AHNH` (`0x41 0x48 0x4E 0x48`). Any other value is a decode error. |
| 4      | 4    | `uint32`   | `version`       | Format version. This document is version `2`. Any other value is a decode error. |
| 8      | 4    | `uint32`   | `width`         | Vertex-grid column count (number of sampled columns). `0` is a decode error. |
| 12     | 4    | `uint32`   | `height`        | Vertex-grid row count (number of sampled rows). `0` is a decode error.  |
| 16     | 8    | `float64`  | `z_offset`      | Height-axis quantizer translation (metres). See *Quantization*.         |
| 24     | 8    | `float64`  | `z_scale`       | Height-axis quantizer scale (metres per level). See *Quantization*.     |
| 32     | 8    | `float64`  | `rtc_centre[0]` | ECEF y-up RTC centre X (A-profile anchor; see contract).                |
| 40     | 8    | `float64`  | `rtc_centre[1]` | ECEF y-up RTC centre Y.                                                  |
| 48     | 8    | `float64`  | `rtc_centre[2]` | ECEF y-up RTC centre Z.                                                  |
| 56     | 8    | `float64`  | `region[0]`     | West longitude, radians (EPSG:4979).                                    |
| 64     | 8    | `float64`  | `region[1]`     | South latitude, radians.                                                |
| 72     | 8    | `float64`  | `region[2]`     | East longitude, radians.                                                |
| 80     | 8    | `float64`  | `region[3]`     | North latitude, radians.                                                |
| 88     | 8    | `float64`  | `region[4]`     | Minimum ellipsoidal height, metres.                                     |
| 96     | 8    | `float64`  | `region[5]`     | Maximum ellipsoidal height, metres.                                     |
| 104    | 8    | `uint64`   | `payload_len`   | Byte length of the zstandard frame that follows the header.             |
| 112    | 4    | `uint32`   | `header_crc32`  | CRC-32/ISO-HDLC over header bytes `[0, 112)`. Mismatch is a decode error. |
| 116    | 4    | `uint32`   | `pad`           | Reserved; must be `0`. Any other value is a decode error.               |

`region` uses the OGC 3D Tiles region ordering
`(west, south, east, north, minHeight, maxHeight)`.

### `header_crc32` (integrity of the fixed header)

`header_crc32` is the CRC-32/ISO-HDLC checksum (the polynomial and
reflection of `zlib.crc32` in Python and the `crc32fast` crate in Rust —
the two are bit-identical) computed over the **first 112 header bytes**,
i.e. every field from `magic` through `payload_len` inclusive, *excluding*
`header_crc32` and `pad` themselves.

A conforming decoder **must verify `header_crc32` before trusting
`width`, `height` or `payload_len`** — in particular before allocating any
buffer sized from them. This closes the corrupt-header giant-allocation
window: a flipped `width` byte fails the CRC check and is rejected before
`width * height * 2` is ever computed or allocated.

The CRC-32/ISO-HDLC algorithm itself is pinned by an algorithm-conformance
golden vector (see *Test vectors*), so a clean-room decoder can validate
its CRC implementation independently of any `.hf` chunk.

## Payload

Immediately after the header (at byte offset `120`) comes exactly one
zstandard frame of `payload_len` bytes, and **nothing after it** — any
trailing byte, or a frame shorter than `payload_len`, is a decode error.

The frame decompresses to exactly `width * height * 2` bytes: a
row-major, **top row first** array of `width * height` `uint16`
little-endian quantized height levels. Element `(r, c)` (row `r` from the
top, column `c` from the left) is at flat index `r * width + c`. A
decompressed length that is not `width * height * 2` is a decode error.

The size expression `width * height * 2` **must be evaluated in 64-bit
arithmetic** on both the producer and the decoder. At the maximum grid a
plan can emit this stays well within `uint32`, but a corrupt (yet
CRC-consistent) header could in principle name dims whose product exceeds
`uint32`; a decoder computing the expected length in 32-bit could wrap and
mis-validate. Compute it as `u64`/`usize` (Rust: `u64`, converting to
`usize` with a checked `try_from` that itself rejects on a 32-bit host).

### zstandard framing (deterministic)

- Compression level is fixed at **3** (a module constant). This is the
  stakeholder-pinned middle ground from the codec bake-off: the height
  payload is near-incompressible (per-tile quantization makes the low byte
  noise-like), so level 3 gives essentially the same ratio as level 19 at
  ~5× faster Python encode and materially faster decode, at these ≤132 KB
  raw granules.
- The RFC 8878 **content checksum is written** (`write_checksum=True`):
  libzstd appends an XXH64-low32 checksum of the uncompressed content to
  the frame epilogue and verifies it on a full-frame decode. A bit-flipped
  payload therefore fails to decode (it is *not* a silent wrong-bytes
  decode) — a new integrity layer on top of `header_crc32`.
- Single-threaded only (`threads = 0`): worker threads can change frame
  boundaries and thus the exact bytes.
- The frame **embeds the decompressed content size** (zstd frame content
  size field is written, `write_content_size=True`), so a decoder can
  allocate exactly and reject a wrong-length payload without an
  out-of-band size.
- **Producer call is one-shot.** The normative producer is the one-shot
  `ZstdCompressor(level=3, write_checksum=True, write_content_size=True,
  threads=0).compress(raw)` call. The streamed `stream_writer` path emits
  a *different* frame for the same input (an off-by-one-byte difference was
  measured in the spike) and is **forbidden** for the producer.

**Producer byte-determinism scope.** Given the same input plane and the
same pinned `python-zstandard` / `libzstd` build (recorded in
`provenance.json`; the spike pinned python-zstandard `0.23.0` bundling
libzstd `1.5.6`), the produced frame bytes are identical (encode-twice
equality is a test). Byte-for-byte producer determinism is a property of
**the Python producer only**. A Rust encoder (the crate's optional,
off-by-default `encode` feature) is held to **semantic round-trip
equality** — it must produce a frame that decodes to the identical plane —
**not** byte parity with the Python frame; libzstd builds and encoder
versions legitimately differ in the exact compressed bytes.

**Decode API.** The pinned Rust decode call is `zstd::decode_all(&[u8])`,
which runs the streaming decoder to frame end. This forces libzstd to
reach and verify the RFC 8878 content-checksum epilogue and to reject a
truncated frame (the epilogue is never reached), so both a bit-flip and a
truncation surface as hard errors rather than partial output.

**Band-framing revisit trigger (informative).** One zstd frame per tile is
the streaming granule. At the current ≤257×257 tile (≤132 KB raw,
≤~124 KB compressed) seekable-zstd / intra-tile band framing solves a
non-existent problem and is deliberately not used. **If a future plan ever
emits a tile dimension `≥ 2048` px** (raw plane `≥ 8 MB`), revisit whether
the single-frame granule should become band-framed for partial-decode /
streaming-upload; nothing in this version supports it.

## Quantization (height axis only)

The height plane is quantized with the **same per-axis affine scheme** as
the A-profile position quantizer (`ahn_cli/tiles3d/quantize.py`), applied
to the single height axis — but at a **12-bit maximum level of `4095`**
rather than the A profile's 16-bit `65535`:

```
z_offset = min(z)
extent   = max(z) - min(z)
z_scale  = extent / 4095           (if extent > 0)
z_scale  = 1e-9                    (epsilon scale, if extent == 0)
level    = clip(round_half_even((z - z_offset) / z_scale), 0, 4095)   # uint16
```

Rounding is **round-half-even** (banker's rounding / round-ties-to-even —
Python `numpy.rint`, Rust `f64::round_ties_even`), applied to the scaled
value *before* the clip. This is **not** truncation (`as` cast) nor
round-half-away-from-zero (`f64::round`): a value exactly halfway between
two levels goes to the even neighbour, so e.g. `round_half_even(2.5) == 2`
and `round_half_even(3.5) == 4`. Choosing the wrong rounding function
yields off-by-one levels on halfway samples — a silent divergence, so this
is normative for the producer and for any semantic-round-trip encoder.
Dequantization is
`z' = level * z_scale + z_offset`. A zero-extent (flat) tile stores all
zeros and dequantizes to exactly `z_offset` (error 0), keeping `z_scale`
non-zero (the epsilon scale). Levels are stored as `uint16` LE (the
container is unchanged from the A profile; only the value range narrows to
`[0, 4095]`), **not** bit-packed.

**Rationale for 12-bit (state-of-record).** At 16-bit the quantizer spends
the whole `uint16` range resolving sub-mm detail (~0.2 mm median) roughly
250× finer than AHN's ~5 cm vertical accuracy — i.e. encoding sensor
noise. 12-bit keeps the worst-case round-trip error ~10× inside AHN's
accuracy on both production grids (≤18 mm worst-case on 50 cm tiles,
≤7 mm on the 8 cm ortho-grid tiles) while cutting the *compressed*
footprint ~30 %. It applies to the `.hf` height axis **only**; the game
profile's glTF POSITION quantization stays 16-bit.

**Exported bound.** The worst-case round-trip error on the height axis is
half a quantization step:

```
|z' - z| <= z_scale / 2 == extent / 4095 / 2
```

Every stored level satisfies this bound against its source height by
construction, and the verifier asserts it against `axis_error_bound`
(`= z_scale / 2`), never a literal.

**Absolute-error cap (normative reject).** A heightfield tile is only
valid if its exported bound is within an absolute ceiling:

```
z_scale / 2 <= 0.025 m        (equivalently: height extent <= 204.75 m)
```

The **producer refuses** to encode a tile beyond this cap (a typed
`Tiles3dError`), and the **verifier enforces** it. A decoder **may** check
it as a validity condition (see *Decode errors*); it is derivable from the
header alone (`z_scale`). No production AHN tile approaches 204.75 m of
height extent, so this cap never rejects genuine data; it exists to make
an over-tall (therefore under-resolved) tile a hard error rather than a
silently lossy one.

## Region semantics (corrected in v2)

The header `region` is the tile's **own** mesh region — the geodetic
bounding volume of *this tile's* sampled vertices
(`ahn_cli/tiles3d/mesh.py`'s `Region`). It is a **distinct quantity** from
the enclosing region that the `tileset.json` sidecar and the pack index
carry for the same tile:

- **`tileset.json` / pack-index region** = the *union* region: a tile's
  own region unioned with all its descendants' regions, because a parent
  bounding volume must enclose its children (a hard 3D Tiles validity
  rule). This is what culling / LOD selection needs.
- **`.hf` header region** = the tile's own region only.

Their exact relationship (verified across all spike fixtures):

- The **4 horizontal doubles** (`west, south, east, north`) are
  **bit-equal** between the header region and the enclosing region for
  **every** tile — leaf and parent alike. (A parent's horizontal footprint
  is exactly the union of its children's footprints, which tile the parent
  with shared boundaries, so no horizontal widening occurs.)
- The **2 height doubles** (`minHeight, maxHeight`) are **bit-equal only
  for leaf tiles**. For a parent tile the header height range is a
  (possibly strict) **subset** of the enclosing height range: finer-strided
  descendants reach elevations the parent's own coarse samples do not, so
  `enclosing.minHeight <= header.minHeight` and `header.maxHeight <=
  enclosing.maxHeight`.
- Therefore for **leaf tiles all six doubles are bit-equal**; for parent
  tiles only the four horizontal ones are guaranteed bit-equal and the
  header height range is contained.

**Consumer wrong-tile guard (normative SHOULD).** A runtime that loads a
`.hf` chunk by pack-index key and wants to catch a swapped or mis-indexed
tile **must not** use a blanket six-double bit-compare of the header
region against the index region — that would falsely reject every valid
parent tile. Instead the guard is:

1. the **four horizontal doubles** of the header region are **bit-equal**
   to the index entry's region (catches a swapped horizontal footprint);
2. the header height range is **contained** in the index entry's height
   range (`index.minHeight <= header.minHeight` and `header.maxHeight <=
   index.maxHeight`);
3. the **per-blob SHA-256** in the pack hash section is the definitive
   wrong-blob detector (see the pack spec).

Reconstructed (dequantized) heights may exceed the header's stored
`[minHeight, maxHeight]` range by up to `z_scale / 2` (the quantization
bound), so consumers must not assume exact containment of dequantized
values within the header region either.

## Grid-reconstruction contract (implicit geometry)

Only the height plane is stored. A runtime rebuilds the full tile mesh
exactly as the A-profile mesh (`ahn_cli/tiles3d/mesh.py`) so the two
profiles render the same surface:

- **Vertices**: a uniform `width × height` grid. Vertex `(r, c)` maps to a
  geodetic position whose longitude/latitude come from the tile's
  `region` by uniform spacing across the sampled span, and whose height is
  the dequantized `level[r * width + c]`. (The exact per-vertex geodetic
  X/Y are the sampled EPSG:28992 pixel centres of the A profile; a runtime
  that only needs a rendered surface may interpolate them uniformly across
  `region`.)
- **Texture coordinates**: implicit **texel-centre** UVs. Vertex `(r, c)`
  has `u = (c + 0.5) / width`, `v = (r + 0.5) / height`. The draped
  texture is the sibling `.jpg` file.
- **Connectivity**: implicit two-triangles-per-cell, matching
  `mesh.py`'s diagonal orientation exactly. For cell `(r, c)` with
  row-major vertex indices

  ```
  a = r * width + c
  b = a + 1                 # (r,   c+1)
  c_ = a + width            # (r+1, c)
  d = c_ + 1                # (r+1, c+1)
  ```

  emit the two triangles `(a, c_, d)` and `(a, d, b)` — the same winding
  the A-profile index buffer uses, so normals and back-face culling agree
  across profiles.

## Sibling texture

Alongside `<level>-<tx>-<ty>.hf` the profile writes
`<level>-<tx>-<ty>.jpg`: a **baseline sequential** JPEG of the exact same
sampled ortho pixels the A-profile game texture uses, produced by the
shared `ahn_cli/tiles3d/jpeg.py` codec at its pinned settings (quality 85,
4:2:0 subsampling, non-progressive, Huffman-optimization off — unchanged
by this format version). When these chunks are bundled into a pack
(`tiles.hfp`), the `.hf` blob is the entry's *primary* blob and the `.jpg`
blob is the entry's *texture* blob (see the pack spec); when written as
loose files, the `.jpg` is referenced only by this reconstruction contract
(same base name, by convention).

## Manifest (`tileset.json`)

The heightfield profile writes the same `tileset.json`-shaped index as the
A profiles: a quadtree of tiles with `REPLACE` refinement, exact EPSG:4979
`region` bounding volumes (the *union*/enclosing regions per *Region
semantics*), and per-tile `geometricError` (leaves `0`). The only
difference is each tile's `content.uri` ends in `.hf`. The file remains
valid 3D Tiles 1.1 JSON and is written with **LF (`\n`) newlines on every
platform** (no CR/CRLF), UTF-8, sorted keys, 2-space indent, trailing
newline — a hard byte-determinism requirement for any file the pack
`manifest.json` hashes. **Note:** `.hf` is a vendor content type that the
generic OGC `3d-tiles-validator` does not recognize, so a heightfield
tileset is not run through that validator (only the `strict`/`game`
profiles are); the `.hf` decoder is validated against this document and by
`ahn_cli`'s own post-write verifier.

## Decode errors (normative rejects)

A conforming decoder rejects (does not silently repair). Verify these in
an order that never trusts an unverified length:

- input shorter than the 120-byte header;
- `magic != "AHNH"`;
- `version != 2`;
- **`header_crc32` does not match** the CRC-32/ISO-HDLC of bytes `[0, 112)`
  — checked **before** `width`/`height`/`payload_len` are trusted or used
  to size any allocation;
- `pad != 0`;
- `width == 0` or `height == 0`;
- any **non-finite** value (`NaN`, `+Inf`, `-Inf`) in a `float64` header
  field (`z_offset`, `z_scale`, the three `rtc_centre` components, or any
  of the six `region` doubles) — the producer only ever writes finite
  values, and a non-finite would poison downstream geodesy / dequant math;
- `z_scale <= 0` (a valid chunk has `z_scale = extent / 4095 > 0` or the
  positive epsilon scale; a zero or negative scale would divide-by-zero the
  dequant and the cap check);
- the trailing bytes after the header not being exactly `payload_len`
  bytes (truncated or trailing data);
- a zstandard frame that fails to decompress, **including a content-checksum
  mismatch** (a corrupted payload) surfaced by `decode_all` reaching the
  frame epilogue;
- a decompressed length `!= width * height * 2` (computed in 64-bit);
- **any stored level `> 4095`** (a value outside the 12-bit quantization
  range — a free integrity check on the decompressed plane).

A conforming decoder **may** additionally reject, as a validity condition
derivable from the header alone, a tile whose `z_scale / 2 > 0.025 m`
(the absolute-error cap). The producer and the reference verifier always
enforce this; a lightweight runtime decoder may skip it.

## Test vectors

Two categories. The **algorithm-conformance vectors** are pure functions
of fixed inputs and are **permanent and normative** — a clean-room decoder
validates its CRC-32 and SHA-256 implementations against them regardless
of any codec or quantization choice:

| Vector | Input | Expected |
|---|---|---|
| CRC-32/ISO-HDLC | ASCII `"AHN heightfield spike golden vector 0123456789"` | `0xb4b41f5a` (`3031703386`) |
| SHA-256 | same string | `6a8998f8fab0139aaff77ccb9ab123907d58bc55d8b7244e2394f41418731b54` |

The **format-instance vectors** below were produced during the interop
spike under the spike's settings (zstd **level 19**, 16-bit height
quantization) and are **illustrative, not v2-normative byte values** — the
v2 codec pins zstd level 3 and 12-bit quantization, so the exact frame
size, the `.hf` `header_crc32`, and any pack `dataset_id` change under the
final settings. The **definitive v2 instance vectors are regenerated by
the Phase 3 Python producer under the pinned settings and committed as
binary fixtures**; the Phase 4 Rust tests recompute and byte-compare
against those committed fixtures (not against the spike-era numbers below):

| Vector (spike-era, illustrative) | Input | Spike value |
|---|---|---|
| zstd frame, checksum on, **level 19** | 33×33 `uint16` ramp `arange%500` | 727 B, sha256 `e3c8788c…490cbc70` |
| `.hf` `header_crc32`, **level 19 / 16-bit** | `chunk_v2.hf` bytes `[0, 112)` | `0x6ffe3f40`, `pad = 0` |

The structural shape these illustrate is normative (the header CRC covers
`[0, 112)`; `pad = 0`; the frame carries a content checksum and embedded
size); only their concrete bytes are settings-dependent and superseded by
the committed v2 fixtures.
