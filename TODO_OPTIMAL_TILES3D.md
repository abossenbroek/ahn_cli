# TODO — tiles3d compression profiles (Approach A) + heightfield export (Approach C)

Target reader: senior engineer. No hand-holding. Terse by design, but every item
carries enough context to act on without re-reading the brainstorm.

Source: 2026-07-11 brainstorm (stakeholder Q&A). Consumer is the stakeholder's
own Rust game — streaming load of terrain tiles + lidar. Codec choices are
Rust-decoder-driven: `KHR_mesh_quantization`, `EXT_meshopt_compression`, baseline
JPEG. Draco explicitly rejected (C++-only decode path). Lidar side is already
served by the `copc` context (COPC = streaming octree; `laz-rs`/`copc-rs` decode
it) — nothing to build there.

Scale target that defines success: full Moerkapelle GeoJSON AOI at native 8 cm
(2.35 Gpx) as a streamable tileset. Budget math, load-bearing for every codec
decision below: strict leaf cost is ~44 B/px (float32 attrs + uint32 indices);
quantization alone lands at ~34 B/px ≈ 107 GB — indices dominate — and FAILS the
target. Quantization + meshopt ≈ 2–3 B/px geometry, + JPEG ≈ 0.5 B/px, ×4/3 LOD
overhead → **~9–11 GB** (A). Heightfield chunks ≈ 1–1.5 B/px → **~4–6 GB** (C).
The 56 GB uncompressed reconcile EXR intermediate is unchanged by this work and
fits current disk alongside either output.

Stakeholder decisions, recorded 2026-07-11 — do not relitigate silently:

- Quantized positions: **accepted** (bounded, documented error).
- Lossy textures (JPEG): **accepted**.
- Error-bounded vertex dropping / mesh simplification: **rejected** — every
  leaf-tile vertex remains one genuine sample per source pixel, full grid.
- Strict profile stays available and byte-identical to today's output.

---

## Guardrails (apply to every item, no exceptions)

- **DDD.** `tiles3d` stays one bounded context. New seam is the
  payload/encoder split: `TilePayload` (pure sampled source data) →
  `TileEncoder` (profile-specific bytes). No module outside the encoder layer
  may know about glTF, JPEG, meshopt, or zstd. `Profile` is a value object —
  no stringly-typed profile switches beyond the CLI boundary.
- **TDD.** Red-green-refactor, failing test first, every item.
- **100% test coverage.** Line + branch, `fail_under = 100`, no new
  `# pragma: no cover` without a documented reason in the PR.
- **Documentation.** Every public function/class: contract docstring (inputs,
  outputs, invariants, failure modes). CLAUDE.md + README updated in the same
  PR that lands each user-visible change.
- **No speculative code.** Build exactly what's below. No `--jpeg-quality`
  flag, no KTX2, no Draco, no implicit tiling, no simplification. Flag gaps,
  don't fill them.
- **Determinism, widened boundary — pinned, recorded, verified.** Same input →
  same output, byte-identical *per machine and lockfile* (the PROJ/geodesy
  precedent). Third-party encoders (`meshoptimizer`, Pillow JPEG) enter the
  boundary: exact versions pinned in `uv.lock`, recorded in `provenance.json`,
  and every encoder gets an encode-twice-byte-equal test. The verifier's
  whole-file byte-identity backstop (independent in-process rebuild → compare)
  survives unchanged in every profile.
- **Strict profile is frozen.** `--profile strict` output stays byte-identical
  to today's. Existing tiles3d tests must pass untouched — any diff to them in
  a PR is a red flag, bounce it.
- **Authenticity, amended for lossy artifacts.** Heights/geometry: every
  stored value must be exactly recomputable from genuine source samples via a
  documented deterministic transform (quantization qualifies; averaging and
  infill still forbidden; the vertex set is never thinned below one vertex per
  source pixel at leaves). Textures: lossy JPEG allowed in the game profile
  only, guarded by a decoded-fidelity floor against the source ortho tile —
  the `uniform_image`/`flat_surface` gates still run on the *source*, never on
  decoded JPEG.
- **Single typed error.** Everything raises/wraps into `Tiles3dError`; the CLI
  translates once. Encoder-library exceptions never escape raw.
- **Extensibility only where specified.** The `TileEncoder` seam is the one
  open extension point (it is what makes item "Heightfield Export (C)" a pure
  addition). Nowhere else.

---

## Payload/Encoder Split in `emit.py`

Today `emit.py` fuses sampling and encoding: it strides the grids and
immediately packs glb + PNG. Split it.

**Requirement:** `TilePayload` value object — sampled height grid, sampled
ortho pixels, RTC centre, EPSG:4979 region, stride, geometric error, tile
coordinates. `TileEncoder` protocol: `encode(payload) -> EncodedTile`
(content bytes + content filename + texture bytes if separate). Extract the
existing float32-glb + PNG path into `StrictEncoder` with **zero byte drift**.
`emit.py` keeps children-first ordering and region containment by
construction; `build.py`'s two-phase swap/hold-aside/accept-marker machinery
is untouched.

**Definition of done:**
- `TilePayload` + `TileEncoder` protocol defined, documented, typed strictly
- `StrictEncoder` = old behavior; existing tests pass untouched
- Byte-freeze regression test: tiny synthetic scene built pre/post refactor →
  identical file hashes (golden hashes checked in)

---

## Quantizer (`quantize.py`)

**Requirement:** pure module, no I/O. Positions → per-axis uint16 in
tile-local RTC space with `KHR_mesh_quantization` semantics: ints dequantize
via node `scale`/`translation` only. Per-tile Z offset/scale from the tile's
actual height range; XY from the tile's pixel span. Rounding: round-half-even,
stated in the docstring. Document the error bound as a formula
(`extent / 65535 / 2` per axis; at 8 cm / 256 px: XY ≤ ~0.16 mm, Z range-
dependent) and export it — the verifier asserts against the *documented*
bound, not a magic number. UVs → core-glTF normalized uint16 (no extension
needed). Degenerate case: zero height range → scale 0 is invalid glTF; use
the documented epsilon-scale rule, test it.

**Definition of done:**
- Quantize + dequantize implemented, pure, deterministic
- Property-style tests over synthetic grids: round-trip error ≤ documented
  bound, always; quantize(quantize⁻¹(q)) == q (idempotence in int domain)
- Zero-range and single-vertex-span edge cases covered

---

## Meshopt Stream Compression

**Requirement:** `EXT_meshopt_compression` on every vertex attribute and
index bufferView in the game profile. Dependency: the `meshoptimizer` PyPI
binding, pinned. First implementation step is a spike test that the binding
exposes deterministic `encodeVertexBuffer`/`encodeIndexBuffer`(/filters) —
see Blocking Questions; if it doesn't, stop and re-plan, don't work around.
glTF wiring: compressed bufferView + `extensionsRequired` fallback rules per
the extension spec (we declare it required — no fallback buffer; the Rust
`meshopt` crate decodes it).

**Definition of done:**
- Encode wired for POSITION, TEXCOORD_0, indices; modes/filters chosen and
  documented per stream
- Encode-twice-byte-equal determinism test
- Decode-round-trip test: decoded bytes == pre-encode bytes, exactly
- Library version recorded into provenance

---

## JPEG Texture Writer (`jpeg.py`)

**Requirement:** baseline sequential JPEG via Pillow, pinned settings, stated
once as module constants: quality 85, 4:2:0 subsampling, no progressive, no
optimize-Huffman drift (fix `optimize` explicitly). Encoder input is the
exact sampled ortho pixels from the payload — same array the strict PNG path
would have written. Also provide decode (for the verifier's fidelity check).
No quality flag — YAGNI, revisit only on stakeholder ask.

**Definition of done:**
- Writer + reader implemented; settings are module constants with a contract
  docstring
- Encode-twice-byte-equal determinism test
- Fidelity-floor test: decode(encode(tile)) vs source tile mean-abs-error
  under the fixed threshold; a deliberately garbage encode fails it
- Pillow version recorded into provenance

---

## Game glTF Writer

**Requirement:** extend `gltf.py` (or sibling `gltf_quant.py`, implementer's
call — keep one obvious home) to emit the quantized + meshopt + JPEG tile:
uint16 POSITION accessors with node scale/translation dequantization,
normalized-uint16 TEXCOORD_0, meshopt bufferViews, JPEG `image/jpeg` texture,
`extensionsUsed`/`extensionsRequired` = `KHR_mesh_quantization`,
`EXT_meshopt_compression`. Accessor min/max are the exact int extremes of the
written data (verifier bit-compares, as today). glb container framing rules
unchanged (4-byte alignment, LE).

**Definition of done:**
- Valid glb per profile; `3d-tiles-validator` + glTF validator green on a
  sample game-profile tileset (record invocation in docs, as done for strict)
- Accessor extremes bit-exact tests
- tileset.json carries the extension declarations only under the game profile

---

## CLI: `--profile strict|game`

**Requirement:** one new Click option on `tiles3d`, default `strict`.
Validation and error translation in `cli/app.py` only; profile parsing to the
`Profile` value object at the boundary. Provenance gains: profile name,
quantization bit depths + per-tile scheme note, JPEG settings, encoder
library versions. CLAUDE.md command examples updated.

**Definition of done:**
- Flag wired end-to-end; default runs are byte-identical to pre-change output
- Provenance fields present under game profile, absent/unchanged under strict
- CLI tests cover both profiles + rejection of unknown values

---

## Verifier Extensions (`verify.py`)

**Requirement:** profile-aware verification, strictest-in-class as today. The
whole-file byte-identity check against an independent in-process rebuild stays
the backstop for both profiles. Game profile adds, per tile:
1. meshopt bufferViews decode to exactly the pre-encode bytes;
2. quantized POSITION ints == an independent requantization of the genuine
   source samples, bit-exact; dequantized values within the documented bound
   of the source samples;
3. JPEG parses as baseline JPEG, byte-equals the independent re-encode, and
   decodes within the fidelity floor of the sampled source ortho tile;
4. structural glTF checks extended: extension declarations, component types,
   normalized flags, meshopt bufferView framing.

**Definition of done:**
- All four check families implemented and run unconditionally post-write
- Every new check has a negative test corrupting the exact bytes it guards
  (bit-flip in a meshopt stream, off-by-one quantized int, recompressed
  JPEG at different quality, dropped extension declaration, …)
- Verification failure still triggers full cleanup + hold-aside restore —
  covered by a fault-injection test under the game profile

---

## Heightfield Export (Approach C) — second phase, after A is green

The C seam exists precisely so this lands without touching A. Build only
after the game profile ships and the Rust game consumes it; revisit need
then (the stakeholder may find A sufficient).

**Requirement:** `HeightfieldEncoder` implementing `TileEncoder`. Per tile:
uint16 Z plane (same quantizer, same bound) + fixed little-endian header
(magic, version, dims, Z offset/scale, RTC centre, region) + zstd-compressed
payload (`zstandard` pinned); JPEG texture alongside (reuse `jpeg.py`);
X/Y/UV/connectivity implicit — the game reconstructs the grid. Manifest: the
same tileset.json-shaped index (regions, geometricError, REPLACE refinement)
with tile content pointing at `.hf` chunks — the chunk format spec lives in
`docs/` and is normative for the Rust decoder. Target ~1–1.5 B/px geometry.

**Definition of done:**
- Chunk format documented byte-for-byte in `docs/` (the Rust side codes
  against that doc, not against our source)
- Encoder + Python reference decoder implemented; round-trip test:
  decode(encode(payload)) reproduces quantized ints exactly
- Verifier: byte-identity backstop + requantization check + zstd
  decode-round-trip, same negative-test discipline
- Size benchmark (informational, non-blocking): B/px vs the A profile on the
  same synthetic scene

---

## Integration Tests

- Synthetic small site (reuse tiles3d conftest fixtures) built under
  `--profile game`: verifier green end-to-end, then every negative test.
- Byte-freeze goldens for strict (see "Payload/Encoder Split").
- Determinism: build the same inputs twice in one process → identical file
  hashes, both profiles.
- Size budget (informational, non-blocking CI): geometry B/px on the
  synthetic scene, printed; regression alarm if leaf geometry exceeds
  ~4 B/px post-meshopt.
- Rust-side decode is out of CI scope here (separate repo), but the chunk/glb
  fixtures produced by the tests are committed small so the game repo can
  consume them as its own test fixtures.

---

## Blocking Questions — Resolve Before Writing Code

1. **`meshoptimizer` PyPI binding surface.** Confirm it exposes the codec
   (`encodeVertexBuffer`/`encodeIndexBuffer`/`encodeFilter*`) and not just the
   optimizer passes, and that output is deterministic for a pinned version.
   If not: candidate fallbacks (ctypes over libmeshopt, or vendoring the
   ~1-file C codec behind a thin cffi shim) — decide with stakeholder.
2. **Pillow JPEG cross-version drift policy.** Accepted as per-machine
   determinism (lockfile-pinned)? Or does the stakeholder want the hand-packed
   baseline JPEG encoder (~500 lines, deterministic forever) now rather than
   later? Default assumption: Pillow now, hand-packed later if drift bites.
3. **Node-scale vs KHR_texture_transform interplay** — none expected (we
   don't use texture transforms), but confirm the Rust `gltf` crate applies
   node scale to quantized POSITION the way the spec requires before
   committing to no-fallback `extensionsRequired`.

---

## Explicitly Out of Scope — Do Not Touch

- Draco, KTX2/Basis, WebP — rejected codecs for this consumer.
- Mesh simplification / error-bounded vertex dropping — stakeholder-rejected;
  leaves keep one vertex per source pixel, full stop.
- Implicit tiling, external tileset subtrees, any server/hosting component —
  static files are the streaming story.
- `reconcile`'s EXR format (the 56 GB intermediate is a separate concern).
- Any byte change to the strict profile, the COPC context, or `copc`'s
  streaming behavior (COPC already serves the game's lidar needs as-is).
