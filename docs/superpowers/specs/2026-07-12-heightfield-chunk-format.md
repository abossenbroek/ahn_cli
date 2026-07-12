# The `.hf` heightfield chunk format (version 1) — normative

This document is **normative**: the Rust runtime decoder codes against
*this specification*, not against the Python source. `ahn_cli`'s
`tiles3d --profile heightfield` writes one `.hf` chunk plus one baseline
JPEG per quadtree tile; the Python reference codec
(`ahn_cli/tiles3d/heightfield.py`) and its verifier
(`ahn_cli/tiles3d/verify_heightfield.py`) implement exactly what follows.

A `.hf` chunk is a fixed little-endian header immediately followed by a
single zstandard frame. The frame decompresses to the tile's quantized
elevation plane; everything else a runtime needs to rebuild the tile mesh
(vertex X/Y, texture coordinates, triangle connectivity) is **implicit**
and reconstructed from the header — see *Grid-reconstruction contract*.

All multi-byte integers and floats are **little-endian**. Floats are IEEE
754 (`float64` = binary64). There is no padding: the header is exactly
`112` bytes, laid out with the equivalent of Python
`struct.pack("<4sIII" + "d" * 11 + "Q", ...)`.

## Coordinate contract (load-bearing — read this first)

The stored plane is the **quantized NAP height plane** — the genuine
sampled source heights (`payload.z`, EPSG:7415 vertical / NAP metres),
quantized to `uint16` along that single axis. It is **not** an
ECEF-swizzled mesh axis. This is the defining difference from the
`strict`/`game` (Approach A) profiles, whose glTF stores all three
quantized ECEF-RTC axes:

- A heightfield tile is a **geodetic-grid product**. The tile's geodetic
  footprint (longitude/latitude extent and the min/max ellipsoidal
  height) travels in the header `region` (identical to the tile's
  `tileset.json` bounding-volume region). The runtime reconstructs each
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

## Header layout (112 bytes, offsets in bytes)

| Offset | Size | Type       | Field         | Meaning                                                                 |
|-------:|-----:|------------|---------------|-------------------------------------------------------------------------|
| 0      | 4    | `char[4]`  | `magic`       | ASCII `AHNH` (`0x41 0x48 0x4E 0x48`). Any other value is a decode error. |
| 4      | 4    | `uint32`   | `version`     | Format version. This document is version `1`. Any other value is a decode error. |
| 8      | 4    | `uint32`   | `width`       | Vertex-grid column count (number of sampled columns).                   |
| 12     | 4    | `uint32`   | `height`      | Vertex-grid row count (number of sampled rows).                         |
| 16     | 8    | `float64`  | `z_offset`    | Height-axis quantizer translation (metres). See *Quantization*.         |
| 24     | 8    | `float64`  | `z_scale`     | Height-axis quantizer scale (metres per level). See *Quantization*.     |
| 32     | 8    | `float64`  | `rtc_centre[0]` | ECEF y-up RTC centre X (A-profile anchor; see contract).              |
| 40     | 8    | `float64`  | `rtc_centre[1]` | ECEF y-up RTC centre Y.                                                |
| 48     | 8    | `float64`  | `rtc_centre[2]` | ECEF y-up RTC centre Z.                                                |
| 56     | 8    | `float64`  | `region[0]`   | West longitude, radians (EPSG:4979).                                     |
| 64     | 8    | `float64`  | `region[1]`   | South latitude, radians.                                                 |
| 72     | 8    | `float64`  | `region[2]`   | East longitude, radians.                                                 |
| 80     | 8    | `float64`  | `region[3]`   | North latitude, radians.                                                 |
| 88     | 8    | `float64`  | `region[4]`   | Minimum ellipsoidal height, metres.                                      |
| 96     | 8    | `float64`  | `region[5]`   | Maximum ellipsoidal height, metres.                                      |
| 104    | 8    | `uint64`   | `payload_len` | Byte length of the zstandard frame that follows the header.             |

`region` uses the OGC 3D Tiles region ordering
`(west, south, east, north, minHeight, maxHeight)` and is bit-for-bit the
same six doubles as this tile's `tileset.json` bounding volume.
Reconstructed (dequantized) heights may exceed the stored `[minHeight,
maxHeight]` range by up to `z_scale / 2` per the quantization bound below,
so consumers must not assume exact containment of dequantized values.

## Payload

Immediately after the header (at byte offset `112`) comes exactly one
zstandard frame of `payload_len` bytes, and **nothing after it** — any
trailing byte, or a frame shorter than `payload_len`, is a decode error.

The frame decompresses to exactly `width * height * 2` bytes: a
row-major, **top row first** array of `width * height` `uint16`
little-endian quantized height levels. Element `(r, c)` (row `r` from the
top, column `c` from the left) is at flat index `r * width + c`. A
decompressed length that is not `width * height * 2` is a decode error.

### zstandard framing (deterministic)

- Compression level is fixed at **19** (a module constant). It is the
  maximum stable ratio level for these write-once assets; higher `--ultra`
  levels trade determinism/compat headroom for marginal gains we do not
  need.
- Single-threaded only (`threads = 0`): worker threads can change frame
  boundaries and thus the exact bytes.
- The frame **embeds the decompressed content size** (zstd frame content
  size field is written), so a decoder can allocate exactly and reject a
  wrong-length payload without an out-of-band size.
- Given the same input plane and the same pinned `zstandard`/`libzstd`
  build, the produced frame bytes are identical (encode-twice equality is
  a test). The `zstandard` version is recorded in `provenance.json`.

## Quantization (height axis only)

The height plane is quantized with the **same per-axis affine scheme** as
the A-profile position quantizer (`ahn_cli/tiles3d/quantize.py`), applied
to the single height axis via `quantize_axis`:

```
z_offset = min(z)
extent   = max(z) - min(z)
z_scale  = extent / 65535          (if extent > 0)
z_scale  = 1e-9                    (epsilon scale, if extent == 0)
level    = clip(round_half_even((z - z_offset) / z_scale), 0, 65535)   # uint16
```

Rounding is round-half-even (banker's rounding). Dequantization is
`z' = level * z_scale + z_offset`. A zero-extent (flat) tile stores all
zeros and dequantizes to exactly `z_offset` (error 0), keeping `z_scale`
non-zero.

**Exported bound.** The worst-case round-trip error on the height axis is
half a quantization step:

```
|z' - z| <= z_scale / 2 == extent / 65535 / 2
```

Every stored level satisfies this bound against its source height by
construction, and the verifier asserts it against `axis_error_bound`
(`= z_scale / 2`), never a literal.

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
4:2:0 subsampling, non-progressive, Huffman-optimization off). The
tileset entry's `content.uri` points at the `.hf`; the `.jpg` is
referenced only by the reconstruction contract above (same base name, by
convention), never by `tileset.json`.

## Manifest (`tileset.json`)

The heightfield profile writes the same `tileset.json`-shaped index as the
A profiles: a quadtree of tiles with `REPLACE` refinement, exact EPSG:4979
`region` bounding volumes, and per-tile `geometricError` (leaves `0`). The
only difference is each tile's `content.uri` ends in `.hf`. The file
remains valid 3D Tiles 1.1 JSON. **Note:** `.hf` is a vendor content type
that the generic OGC `3d-tiles-validator` does not recognize, so a
heightfield tileset is not run through that validator (only the
`strict`/`game` profiles are); the `.hf` decoder is validated against this
document and by `ahn_cli`'s own post-write verifier.

## Decode errors (normative rejects)

A conforming decoder rejects (does not silently repair):

- input shorter than the 112-byte header;
- `magic != "AHNH"`;
- `version != 1`;
- the trailing bytes after the header not being exactly `payload_len`
  bytes (truncated or trailing data);
- a zstandard frame that fails to decompress;
- a decompressed length `!= width * height * 2`.
