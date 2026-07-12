# hf-v2 spike results (Task 1b — interop/platform)

Status of every design doubt in the approved v2 plan
(`resilient-riding-sprout.md`, "Format v2 design" + "Phase 1 — 1b"), each
marked **GREEN** (ruled out, with evidence) or **RED** (design must change,
with the concrete failure). A RED is a *success* of the spike.

**Summary: 2 RED, the rest GREEN, none deferred-blocking.** RED-1 — the `.hf`
chunk-header region is not bit-equal to the tileset/pack region for *parent*
tiles (narrow one SHOULD + one spec sentence; container design unaffected).
RED-2 — the existing v1 tiles3d producer cannot run on Windows (CRLF in
text-mode writes); pre-existing, Phase-3 fix is a one-line newline change.
Neither blocks the v2 container/chunk design; both are concrete, evidence-backed
adjustments for Phase 2/3. All Rust interop, hash, zstd-checksum, pack, and
platform doubts are GREEN on all three OSes.

- Throwaway POC: `rust/spike-poc/` (Rust decoder + `cargo test`s) and
  `rust/spike-poc/tools/{pack_poc,gen_vectors}.py` (Python producer POC).
- Golden vectors + pack fixtures: `rust/spike-poc/tests/data/` (regenerable,
  byte-deterministic via `uv run python rust/spike-poc/tools/gen_vectors.py`).
- Loose-file fixtures reused: `tests/tiles3d/fixtures/rust-consumer/`.
- CI: `.github/workflows/rust.yml` — `spike-lint` (ubuntu) + `spike-test`
  (3 OS × stable) + `cross-language` (3 OS: uv sync → regen → `git diff
  --exit-code` → `cargo test`). Run link:
  https://github.com/abossenbroek/ahn_cli/actions/runs/29203051096

Task 1a (lz4-vs-zstd codec bake-off) owns the codec-winner rows; this spike
validated zstd interop only. If 1a pins lz4, every zstd row below must be
mirrored for the lz4 frame format before Phase 2 (owned by task 1a).

## Pinned toolchain / library versions (this spike)

| Component | Version |
|---|---|
| python-zstandard | 0.23.0 (bundled **libzstd 1.5.6**) |
| Pillow | 12.3.0 |
| Rust `zstd` crate | 0.13 (zstd-sys, vendored libzstd, MSVC-clean) |
| Rust `crc32fast` | 1.4 |
| Rust `sha2` | 0.10 |
| Rust `memmap2` | 0.9 |
| cargo/rustc (local) | 1.96.1 |

## Pinned Rust API choices

- **zstd decode: `zstd::decode_all(&[u8]) -> io::Result<Vec<u8>>`.** It runs
  the streaming decoder to frame end, so libzstd verifies the RFC-8878
  content checksum and rejects a truncated frame (the epilogue is never
  reached). A bit-flipped payload fails here — it is *not* a silent
  wrong-bytes decode. This is the API the shippable crate's chunk codec uses.
- **CRC-32: `crc32fast::Hasher`** (ISO-HDLC) — bit-identical to Python
  `zlib.crc32`.
- **SHA-256: `sha2::Sha256`** — bit-identical to Python `hashlib.sha256`.
- **All field reads: explicit `u32/u64/f64::from_le_bytes`** over `&[u8]`,
  `#![forbid(unsafe_code)]`, no `bytemuck`/`repr(C)` casting — so decoding
  from `&padded[1..]` or an `mmap` is bit-identical (alignment cannot matter).
- **Producer zstd: one-shot `ZstdCompressor(...).compress()`** (pinned). The
  streamed `stream_writer` path produces a *different* frame (726 B vs 727 B
  for the golden plane) — one-shot must be the normative producer call.

## Golden vectors (permanent — become Phase 4 tests)

| Vector | Input | Expected |
|---|---|---|
| CRC-32 | `"AHN heightfield spike golden vector 0123456789"` (UTF-8) | `0xb4b41f5a` (`3031703386`) |
| SHA-256 | same string | `6a8998f8fab0139aaff77ccb9ab123907d58bc55d8b7244e2394f41418731b54` |
| zstd frame (checksum on) | 33×33 `uint16` ramp `arange%500`, `zstd_plane.bin` | 727 B, sha256 `e3c8788c…490cbc70` (`zstd_frame.bin`) |
| v2 `.hf` header CRC-32 | `chunk_v2.hf` bytes [0,112) | `0x6ffe3f40`, pad `0` |
| heightfield `tiles.hfp` `dataset_id` | 5 tiles (`.hf`+`.jpg`) | `6d9909a7…20b8aac8` |
| game `tiles.hfp` `dataset_id` | 5 tiles (`.glb`, texture slot 0) | `eca10a74…60347151` |

Binary vectors are committed under `rust/spike-poc/tests/data/`; the Rust
tests recompute and byte-compare against these values, and `gen_vectors.py`
reproduces the exact bytes on any host (verified byte-identical on re-run).

## Pinned pack layout (spike decision → ratify in Phase 2 spec)

`AHNP` header **128 B, LE**: magic(4) · format_version u32=1 · tile_count u32 ·
level_count u32 · index_offset u64=128 · index_size u64 · hash_offset u64 ·
hash_size u64 · file_size u64 · root_geometric_error f64 · dataset_id[32] ·
index_crc32 u32 @96 · **header_crc32 u32 @100 (CRC over [0,100))** ·
**content_kind u32 @104** · 20 B zero pad. Level directory 16 B/level
(first_entry, entry_count, tx_count, ty_count u32). Index entry **96 B**:
level·tx·ty·**tz(=0)** u32 · region f64[6] · geometric_error f64 ·
primary_offset u64 · texture_offset u64 (**16-aligned**) · primary_size u32 ·
texture_size u32. Hash section 64 B/tile (primary_sha256·texture_sha256).
Blob region = index order `(level,tz,ty,tx)`, each blob 16-aligned, pad zero.

v2 `.hf` chunk header **120 B**: v1 112-B header + `header_crc32 u32` (CRC over
[0,112)) + `pad u32=0`; payload one checksum-on zstd frame.

## Doubt table

### Chunk / codec (local)

| Doubt | Method | Evidence | Verdict |
|---|---|---|---|
| checksum-on zstd frame (zstandard 0.23.0) decodes in Rust `zstd` | `decode_all(zstd_frame.bin)` == `zstd_plane.bin` | equal | GREEN |
| bit-flip → checksum error, not silent | flip mid-payload bit → `decode_all` | `is_err()` | GREEN |
| truncated frame → hard error (API pinned) | chop 4 B and to 1 B → `decode_all` | `is_err()` both | GREEN |
| encode-twice byte-identity (checksum on) | compress plane twice | identical 727 B | GREEN |
| one-shot vs streamed frame | compare `compress()` vs `stream_writer` | 727 vs 726 B → **pin one-shot** | GREEN |
| checksum-byte golden across libzstd builds | frame sha256 vs golden, Rust libzstd | match | GREEN |
| level-19 window at 257×257 under default decoder | encode 257×257 (132 KB), default `decode_all` | round-trips, window=132098 | GREEN |
| frame embeds content size | `get_frame_parameters` | content_size=132098 | GREEN |
| `zlib.crc32` == `crc32fast` | golden vector | `0xb4b41f5a` both | GREEN |
| `hashlib.sha256` == `sha2` | golden vector | equal | GREEN |
| decode from `&padded[1..]` | offset slice by 1 | identical plane | GREEN |
| decode from `mmap`'d fixture (memmap2) | map + decode | identical plane | GREEN |
| `width*height*2` in u64 | `plane_bytes()` returns u64 | `65535²·2 = 8 589 672 450 > u32::MAX` | GREEN |
| corrupt-header giant dims: no pre-validation alloc | `chunk_v2_giant.hf` (65535², payload_len=1, CRC valid) | rejects at frame stage (`Zstd`), never allocates 8.59 GB | GREEN |
| v2 header CRC guards dims | flip width byte | rejected `HeaderCrc` before dims trusted | GREEN |
| `width==0`/`height==0` reject | decoder guard | `BadLength` | GREEN |
| u64→usize checked conversion | `as_usize` via `try_from` | rejects on 32-bit overflow (`BadLength`) | GREEN |

### Pack (P1–P11)

| Doubt | Method | Evidence | Verdict |
|---|---|---|---|
| P1 single-pass seek-back == two-pass build | build both, compare | byte-identical (hf 5571 B, game 17380 B) | GREEN |
| P1 encode-twice pack determinism | rebuild | identical | GREEN |
| P2 blob order == index order after re-sorting children-first traversal | feed reversed tiles | identical pack; offsets ascending in index order | GREEN |
| P3 layout const offsets 128/16/96 | struct sizes | header 108 B → padded 128; dir 16; entry 96 | GREEN |
| P4 Python sort == Rust comparator, sparse/non-square | `sort_vectors.json` golden | Rust `.sort()` == Python `sorted()` | GREEN |
| **P5 region f64 bit-equality across all three encodings** | f64↔bits compare | **see RED-1 below** | **RED** |
| P5 pack entry region == tileset == json round-trip | f64 bits, incl. parents + 0.0 | all 6 doubles bit-equal (pack entry sourced from tileset union) | GREEN |
| P6 dataset_id recompute matches; rebuild stable; any change → new | mutate blob/add/remove tile | identical on rebuild; changes on 1-blob/add/remove | GREEN |
| P7 root_geometric_error bit-equals tileset.json | f64 bits | `8.0` bit-equal (incl. json round-trip) | GREEN |
| P8 per-entry validation (16-align, non-overlap, sorted, zero pad, tz==0) | reader enforces each | all pass; negatives rejected | GREEN |
| P9 truncation matrix (header/dir/index/hash/blob/last-byte) | chop at 6 points, both readers | all 6 rejected in Python **and** Rust | GREEN |
| P9 header/index bit-flip | flip 1 byte | `HeaderCrc` / `IndexCrc` (both languages) | GREEN |
| P9 hash-section corruption | flip 1 byte in hash section | `DatasetId` reject | GREEN |
| P10 per-frame scan sub-100 µs class | 2000-entry linear AABB scan, 3 OS | **~8–12 µs/scan** (mac 8.2, ubuntu 11.8, windows 11.8) — sub-100 µs on every OS | GREEN |
| P11 mmap vs positioned reads (Windows/Defender) | timed index reads, 3 OS | pread/rep: mac 572 ns, linux 956 ns, **windows 1.49 µs**; mmap-copy/rep ~95–108 ns everywhere. No Defender pathology at fixture scale — real-hardware / large-index validation deferred (non-blocking). | GREEN |
| content_kind both fixtured | heightfield (`.hf`+`.jpg`) & game (`.glb`, texture slot 0) | both packs open + validate | GREEN |
| MSVC build of `zstd-sys`, no extra system deps | windows `cargo build`/`cargo test` in CI | `spike-test (windows-latest)` **success** — no extra deps | GREEN |
| Spike producer byte-determinism per OS | `gen_vectors.py` + `git diff --exit-code rust/spike-poc/tests/data` on 3 OS | ubuntu/macos/windows byte-identical | GREEN |
| Production tiles3d producer determinism on Windows | run existing `regen_rust_fixtures` on 3 OS | ubuntu/macos GREEN; **windows FAILS** | **RED-2 below** |

## RED-1 — `.hf` chunk-header region ≠ tileset/pack region for PARENT tiles

**The plan's blanket claim** "region f64 bit-equality across pack entry vs
chunk header vs tileset.json" and its new SHOULD "the Rust loader
bit-compares chunk `region` vs the tileset entry (catches swapped/misindexed
tiles)" are **false for interior/root tiles** and would falsely reject every
parent tile.

**Root cause (confirmed in `ahn_cli/tiles3d/emit.py:179`).** The `tileset.json`
entry stores `union_region(mesh.region, <all descendant regions>)` — a parent's
bounding volume must *enclose* its children (a hard 3D Tiles validity rule).
The `.hf` chunk header stores the tile's *own* `mesh.region`. For a tile with
children the two diverge in the **height axis** (min/maxHeight), because
finer-strided children reach elevations the parent's own coarse samples do not.

**Evidence** (fixture root tile `0-0-0`, from `decode_heightfield` vs
`tileset.json`):

| Component | `.hf` header | tileset.json | bit-equal |
|---|---|---|---|
| west/south/east/north | 0.10000025 / −2.75e-6 / 0.10000575 / 2.75e-6 | identical | **yes (all 4)** |
| minHeight | −4.11798095703125 | −4.93294620513916 | **no** |
| maxHeight | 37.62133026123047 | 38.80571365356445 | **no** |

All 4 **leaf** tiles are fully bit-equal (6/6 doubles). For the parent, the
header region is a strict **subset** (contained) of the tileset region.

**Design change required before Phase 2 (evidence-backed):**

1. The **pack index-entry region = the tileset.json (union/enclosing) region**
   — this is what culling/LOD needs (the enclosing volume) and is **bit-equal
   to tileset.json for every tile** (leaf and parent). The verifier's
   two-encodings witness (pack index ↔ tileset.json, full 6-double bit-compare)
   is GREEN and stands.
2. The **`.hf` chunk-header region stays the tile's own `mesh.region`** and is
   documented as a *distinct* quantity, guaranteed only to be **contained**
   within the enclosing region (verified: containment holds for all fixtures).
   The v1 spec text calling it "bit-for-bit the same … as this tile's
   tileset.json bounding volume" is inaccurate and must be corrected in the v2
   spec.
3. The **wrong-tile-under-right-key guard** must NOT be a full 6-double
   bit-compare of chunk-header vs index for parents. Use: (a) the pack index ↔
   tileset.json full bit-compare (all tiles); (b) the Rust loader asserts the
   chunk header's 4 horizontal doubles are **bit-equal** to the index entry and
   the header height range is **contained** in the index height range
   (catches a swapped footprint without rejecting valid parents); (c) per-blob
   SHA-256 (hash section) as the definitive wrong-blob detector.

This does not block the pack/container design (all other P-rows GREEN); it
narrows one normative SHOULD and one spec sentence.

## RED-2 — the existing v1 tiles3d producer cannot run on Windows (CRLF)

Surfaced by running the **existing** `regen_rust_fixtures` on the CI
windows-latest runner: `build_tiles3d` → `verify_tiles3d` →
`_verify_byte_identity` raises `Tiles3dError: tileset.json does not byte-equal
its independent rebuild` **on Windows only** (ubuntu + macos pass).

**Root cause.** `ahn_cli/tiles3d/tileset.py:91`
`path.write_text(render_tileset(document), encoding="utf-8")` writes in **text
mode**, so on Windows Python translates every `\n` → `\r\n`. The verifier
`ahn_cli/tiles3d/verify.py:610` then compares the on-disk bytes (CRLF) against
`render_tileset(...).encode("utf-8")` (LF) and they differ. The same text-mode
write is used for `provenance.json` (`build.py:145`). The build's own
byte-identity backstop therefore fails on the first text artifact, so the
`heightfield`/`game`/`strict` builds all abort on Windows.

**Scope.** Pre-existing v1 bug (not introduced by this spike; not in my
authorized edit surface, so left unfixed). The current Python CI runs
ubuntu-only, which is why it was never seen. It is squarely in Phase 3's
rewrite scope (`emit.py`/`build.py`/`verify.py`).

**Fix (Phase 3).** Write text artifacts with an explicit LF: either
`path.write_text(text, encoding="utf-8", newline="\n")` or
`path.write_bytes(text.encode("utf-8"))`, for `tileset.json`, `provenance.json`
and every other text artifact. This makes on-disk bytes LF on all platforms —
exactly the cross-OS byte-determinism the pack/manifest goal already requires.
The spike's own `gen_vectors.py` already uses `write_bytes` for its JSON for
this reason, and its 3-OS determinism gate is GREEN.

**CI handling.** The `cross-language` job's production-regen determinism step is
`continue-on-error` **on Windows only** so this documented finding is surfaced
as a visible failed-but-allowed step without masking a genuine *new* Unix
determinism regression (which still hard-fails). Remove that guard once Phase 3
lands the LF fix.

## Newly discovered doubts (for Phase 2/3)

- **ND-1 (from RED-1).** Whether *any* interior-tile field beyond region
  height diverges between the two encodings (geometric_error, horizontal
  region) — spike shows only height diverges; Phase 3 verifier must encode the
  "horizontal bit-equal + height contained" rule, not a blanket bit-compare.
- **ND-2.** `.gitattributes` marks `tests/**/fixtures/*.json` as `text`, which
  invites autocrlf mangling on Windows. The CI sets `core.autocrlf false`
  before the determinism `git diff`; Phase 3 should confirm no committed spike
  data file carries CRLF and consider a `-text` attribute on `tests/data`.
- **ND-3.** `is_multiple_of` (used for the 16-align check) raised the crate's
  effective MSRV; the shippable crate must pin `rust-version` and either keep
  the newer MSRV or fall back to `% == 0` for a lower MSRV (Phase 4 decision).
- **ND-4.** The pinned pack header packs `content_kind` at offset 104 and
  `header_crc32` at 100 covering `[0,100)` (so it protects `index_crc32` at 96
  but *not* `content_kind`); Phase 2 must decide whether the header CRC span
  should extend to cover `content_kind` (recommended) — a spike-pinned choice,
  not yet ratified.

## JPEG / producer note (item 3, no new work)

Pillow JPEG bytes are pinned **per Pillow/libjpeg-turbo version** (quality 85,
4:2:0, non-progressive, optimize off — see `ahn_cli/tiles3d/jpeg.py` and the
existing `tests/tiles3d` determinism tests). Cross-OS JPEG byte-identity is
**empirically** checked by the `cross-language` job's `git diff --exit-code`
over the committed `.jpg` fixtures on ubuntu/macos/windows. Recording the
**producer platform in `provenance.json`** is a Phase 3 item (not a blocker).
A Windows JPEG or zstd byte mismatch in CI is a genuine finding to report, not
to paper over.
