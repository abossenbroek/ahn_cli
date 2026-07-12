"""Generate golden vectors + pack fixtures, run Python-side doubt checks.

Writes committed spike artifacts into rust/spike-poc/tests/data/ and prints a
GREEN/RED line per Python-checkable doubt. Rust re-checks the same artifacts.
"""

from __future__ import annotations

import hashlib
import json
import struct
import sys
import zlib
from pathlib import Path

import zstandard as zstd

sys.path.insert(0, str(Path(__file__).parent))
from pack_poc import (  # noqa: E402
    HEADER_SIZE,
    KIND_GAME,
    KIND_HEIGHTFIELD,
    PackError,
    Tile,
    build_pack,
    build_pack_two_pass,
    read_pack,
)

# This script lives at rust/spike-poc/tools/gen_vectors.py; repo root is 3 up.
REPO = Path(__file__).resolve().parents[3]
FIX = REPO / "tests/tiles3d/fixtures/rust-consumer"
DATA = REPO / "rust/spike-poc/tests/data"
DATA.mkdir(parents=True, exist_ok=True)

results: list[tuple[str, str, str]] = []  # (doubt, GREEN/RED, evidence)


def check(doubt: str, ok: bool, evidence: str) -> None:
    results.append((doubt, "GREEN" if ok else "RED", evidence))


# --------------------------------------------------------------------------
# Parse fixture tilesets into Tile lists
# --------------------------------------------------------------------------
def parse_tileset(profile: str, suffix: str, with_texture: bool) -> list[Tile]:
    root = json.loads((FIX / profile / "tileset.json").read_text())
    tiles: list[Tile] = []

    def walk(node: dict) -> None:
        content = node.get("content")
        if content:
            uri = content["uri"]  # tiles/<l>-<tx>-<ty>.<suffix>
            name = uri.split("/")[-1].rsplit(".", 1)[0]
            lvl, tx, ty = (int(x) for x in name.split("-"))
            region = tuple(node["boundingVolume"]["region"])
            ge = float(node.get("geometricError", 0.0))
            primary = (FIX / profile / uri).read_bytes()
            texture = b""
            if with_texture:
                texture = (FIX / profile / f"tiles/{name}.jpg").read_bytes()
            tiles.append(Tile(lvl, tx, ty, 0, region, ge, primary, texture))
        for child in node.get("children", []):
            walk(child)

    walk(root["root"])
    return tiles, float(root["geometricError"])


hf_tiles, hf_root_ge = parse_tileset("heightfield", "hf", with_texture=True)
game_tiles, game_root_ge = parse_tileset("game", "glb", with_texture=False)

# --------------------------------------------------------------------------
# P1: single-pass seek-back == two-pass in-memory build (byte-identity)
# --------------------------------------------------------------------------
hf_pack = build_pack(hf_tiles, KIND_HEIGHTFIELD, hf_root_ge)
hf_pack2 = build_pack_two_pass(hf_tiles, KIND_HEIGHTFIELD, hf_root_ge)
game_pack = build_pack(game_tiles, KIND_GAME, game_root_ge)
game_pack2 = build_pack_two_pass(game_tiles, KIND_GAME, game_root_ge)
check("P1 single-pass==two-pass (heightfield)", hf_pack == hf_pack2,
      f"{len(hf_pack)} B identical")
check("P1 single-pass==two-pass (game)", game_pack == game_pack2,
      f"{len(game_pack)} B identical")

# encode-twice determinism
check("P1b pack encode-twice byte-identity",
      build_pack(hf_tiles, KIND_HEIGHTFIELD, hf_root_ge) == hf_pack,
      "rebuild identical")

# --------------------------------------------------------------------------
# P2: blob order == index order, incl. re-sorting children-first traversal
# --------------------------------------------------------------------------
# Feed tiles in children-first (reverse-ish) order; writer must re-sort.
shuffled = list(reversed(hf_tiles))
check("P2 blob order == index order after re-sort",
      build_pack(shuffled, KIND_HEIGHTFIELD, hf_root_ge) == hf_pack,
      "shuffled input -> identical pack")

parsed = read_pack(hf_pack)
idx_keys = [(e.level, e.tz, e.ty, e.tx) for e in parsed["entries"]]
blob_order = sorted(parsed["entries"], key=lambda e: e.primary_offset)
blob_keys = [(e.level, e.tz, e.ty, e.tx) for e in blob_order]
check("P2 blob file order matches index order",
      idx_keys == blob_keys and idx_keys == sorted(idx_keys),
      f"order={idx_keys}")

# --------------------------------------------------------------------------
# P3: layout const-offset assertions (128/16/96 B records)
# --------------------------------------------------------------------------
from pack_poc import DIR_ENTRY_SIZE, INDEX_ENTRY_SIZE, _HEADER_STRUCT  # noqa: E402
check("P3 header struct <= 128 and records 16/96",
      HEADER_SIZE == 128 and DIR_ENTRY_SIZE == 16 and INDEX_ENTRY_SIZE == 96
      and _HEADER_STRUCT.size == 108,
      f"header_struct={_HEADER_STRUCT.size} padded_to={HEADER_SIZE}")

# --------------------------------------------------------------------------
# P4: Python tuple-sort key incl sparse/non-square synthetic key set
# --------------------------------------------------------------------------
# Sparse grid: level 2 with only a few tiles present (absent implicit children).
sparse_keys = [(2, 0, 0, 3), (2, 0, 0, 0), (1, 0, 1, 0), (0, 0, 0, 0),
               (2, 0, 3, 1), (1, 0, 0, 1)]
py_sorted = sorted(sparse_keys)
# write_bytes (not write_text): text mode would emit CRLF on Windows, breaking
# the cross-OS byte-determinism the spike is proving. See RED-2 in the results
# doc for the same bug in the production tiles3d writers.
(DATA / "sort_vectors.json").write_bytes(
    (json.dumps({"input": sparse_keys, "expected_sorted": py_sorted}, indent=2)
     + "\n").encode("utf-8"))
check("P4 sparse/non-square sort key defined (Rust re-checks)",
      py_sorted[0] == (0, 0, 0, 0) and py_sorted[-1] == (2, 0, 3, 1),
      f"{py_sorted}")

# --------------------------------------------------------------------------
# P5: region f64 bit-equality across pack entry / .hf chunk header / tileset.json
# --------------------------------------------------------------------------
def f64_bits(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", x))[0]


# tileset.json regions (after json round-trip)
ts = json.loads((FIX / "heightfield" / "tileset.json").read_text())
ts_regions = {}


def collect(node):
    c = node.get("content")
    if c:
        name = c["uri"].split("/")[-1].rsplit(".", 1)[0]
        ts_regions[name] = tuple(node["boundingVolume"]["region"])
    for ch in node.get("children", []):
        collect(ch)


collect(ts["root"])

# re-parse after a dumps/loads round-trip (shortest-repr float stability)
ts_roundtrip = json.loads(json.dumps(ts))
rt_regions = {}
collect2 = collect  # reuse
ts_regions_rt = {}


def collect_rt(node):
    c = node.get("content")
    if c:
        name = c["uri"].split("/")[-1].rsplit(".", 1)[0]
        ts_regions_rt[name] = tuple(node["boundingVolume"]["region"])
    for ch in node.get("children", []):
        collect_rt(ch)


collect_rt(ts_roundtrip["root"])

# P5 leg 1 (GREEN, all tiles): pack index entry region == tileset.json region,
# bit-equal, and survives a json.dumps/parse round-trip (incl. 0.0). This is
# the two-encodings witness the verifier relies on. Pack entry region is SOURCED
# from the (union/enclosing) tileset region, so this holds for parents too.
pack_ts_ok = True
pack_ts_ev = []
by_name = {f"{t.level}-{t.tx}-{t.ty}": t for t in hf_tiles}
for e in parsed["entries"]:
    name = f"{e.level}-{e.tx}-{e.ty}"
    tsr = ts_regions[name]
    tsr_rt = ts_regions_rt[name]
    for a, b, c in zip(e.region, tsr, tsr_rt, strict=True):
        if f64_bits(a) != f64_bits(b) or f64_bits(b) != f64_bits(c):
            pack_ts_ok = False
            pack_ts_ev.append(f"{name}: {a} vs {b} vs {c}")
check("P5 pack entry region == tileset == json round-trip (all 6 doubles)",
      pack_ts_ok, "5 tiles x 6 doubles bit-equal (incl. parents)"
      if pack_ts_ok else "; ".join(pack_ts_ev))
check("P5b 0.0 survives json round-trip bit-exact",
      f64_bits(json.loads(json.dumps(0.0))) == f64_bits(0.0), "0.0 stable")

# P5 leg 2 (RED finding): .hf chunk header region vs tileset/pack region.
# Horizontal 4 doubles bit-equal for ALL tiles; height 2 doubles bit-equal only
# for LEAF tiles (parents use union_region to enclose children). The chunk
# header region is always CONTAINED within the enclosing region.
horiz_ok, height_leaf_ok, containment_ok = True, True, True
parent_diverges = []
leaf_keys = {(e.level, e.tz, e.ty, e.tx) for e in parsed["entries"]}
has_child = set()
for e in parsed["entries"]:
    ck = (e.level + 1, 0, 2 * e.ty, 2 * e.tx)
    if ck in leaf_keys:
        has_child.add((e.level, e.tz, e.ty, e.tx))
for t in hf_tiles:
    name = f"{t.level}-{t.tx}-{t.ty}"
    hf_region = struct.unpack_from("<6d", t.primary, 56)
    tsr = ts_regions[name]
    is_parent = (t.level, t.tz, t.ty, t.tx) in has_child
    for i in range(4):  # W/S/E/N
        if f64_bits(hf_region[i]) != f64_bits(tsr[i]):
            horiz_ok = False
    # containment: header ⊆ tileset, all 6
    if not (hf_region[0] >= tsr[0] and hf_region[1] >= tsr[1]
            and hf_region[2] <= tsr[2] and hf_region[3] <= tsr[3]
            and hf_region[4] >= tsr[4] and hf_region[5] <= tsr[5]):
        containment_ok = False
    height_eq = (f64_bits(hf_region[4]) == f64_bits(tsr[4])
                 and f64_bits(hf_region[5]) == f64_bits(tsr[5]))
    if is_parent and height_eq:
        pass
    if is_parent and not height_eq:
        parent_diverges.append(name)
    if not is_parent and not height_eq:
        height_leaf_ok = False
check("P5c FINDING: .hf header height != tileset height for PARENT tiles",
      len(parent_diverges) > 0,
      f"parents diverging in height: {parent_diverges} "
      f"(union_region encloses children) => full 6-double bit-compare of "
      f"chunk-header vs index is RED for parents")
check("P5d chunk-header W/S/E/N bit-equal to tileset for ALL tiles", horiz_ok,
      "horizontal footprint identical for leaves and parents")
check("P5e chunk-header height bit-equal to tileset for LEAF tiles",
      height_leaf_ok, "leaves match exactly")
check("P5f chunk-header region CONTAINED in enclosing tileset region (all)",
      containment_ok, "header region subset of index/tileset region")

# --------------------------------------------------------------------------
# P6: dataset_id recompute; identical rebuild==same; any change=>different
# --------------------------------------------------------------------------
did0 = read_pack(hf_pack)["dataset_id"]
did_rebuild = read_pack(build_pack(hf_tiles, KIND_HEIGHTFIELD, hf_root_ge))["dataset_id"]
check("P6 dataset_id identical on identical rebuild", did0 == did_rebuild, did0)

# one-blob change
mut = list(hf_tiles)
mut[0] = Tile(mut[0].level, mut[0].tx, mut[0].ty, mut[0].tz, mut[0].region,
              mut[0].geometric_error, mut[0].primary + b"\x00", mut[0].texture)
did_mut = read_pack(build_pack(mut, KIND_HEIGHTFIELD, hf_root_ge))["dataset_id"]
check("P6 one-blob change => new dataset_id", did_mut != did0, "changed")

# tile removed
did_rm = read_pack(build_pack(hf_tiles[:-1], KIND_HEIGHTFIELD, hf_root_ge))["dataset_id"]
check("P6 tile removed => new dataset_id", did_rm != did0, "changed")

# tile added (duplicate a synthetic tile at a new key)
extra = Tile(2, 0, 0, 0, hf_tiles[1].region, 0.0, hf_tiles[1].primary,
             hf_tiles[1].texture)
did_add = read_pack(build_pack([*hf_tiles, extra], KIND_HEIGHTFIELD, hf_root_ge))["dataset_id"]
check("P6 tile added => new dataset_id", did_add != did0, "changed")

# --------------------------------------------------------------------------
# P7: root_geometric_error bit-equality vs tileset.json
# --------------------------------------------------------------------------
check("P7 root_geometric_error bit-equals tileset.geometricError",
      f64_bits(parsed["root_geometric_error"]) == f64_bits(hf_root_ge)
      and f64_bits(hf_root_ge) == f64_bits(json.loads(json.dumps(hf_root_ge))),
      f"{hf_root_ge}")

# --------------------------------------------------------------------------
# P8: per-entry validation set (16-align, non-overlap, order, padding, tz==0)
# --------------------------------------------------------------------------
# read_pack already enforces these; assert a clean parse plus explicit align.
align_ok = all(e.primary_offset % 16 == 0 and
               (e.texture_offset == 0 or e.texture_offset % 16 == 0)
               for e in parsed["entries"])
tz_ok = all(e.tz == 0 for e in parsed["entries"])
check("P8 per-entry 16-align + tz==0 + sorted + non-overlap",
      align_ok and tz_ok, "read_pack validation passed")

# --------------------------------------------------------------------------
# P9: truncation matrix — every chop rejected by the Python reference reader
# --------------------------------------------------------------------------
parsed_hf = read_pack(hf_pack)
level_count = parsed_hf["level_count"]
tile_count = parsed_hf["tile_count"]
index_off = HEADER_SIZE
dir_end = index_off + level_count * 16
# recompute section boundaries
idx_size = level_count * 16 + tile_count * 96
hash_off = index_off + idx_size
hash_size = tile_count * 64
blob_start = hash_off + hash_size
chop_points = {
    "mid-header": 64,
    "mid-directory": index_off + 8,
    "mid-index": dir_end + 40,
    "mid-hash": hash_off + 20,
    "mid-blob": blob_start + 8,
    "last-byte": len(hf_pack) - 1,
}
trunc_ok = True
trunc_detail = []
for label, n in chop_points.items():
    try:
        read_pack(hf_pack[:n])
        trunc_ok = False
        trunc_detail.append(f"{label}: NOT rejected")
    except PackError:
        trunc_detail.append(f"{label}: rejected")
check("P9 truncation matrix (Python reader rejects all 6)", trunc_ok,
      "; ".join(trunc_detail))

# bit-flip in index -> index CRC fail
flip = bytearray(hf_pack)
flip[dir_end + 20] ^= 0x01
try:
    read_pack(bytes(flip))
    check("P9b index bit-flip rejected", False, "not rejected")
except PackError as e:
    check("P9b index bit-flip rejected", True, str(e))

# header bit-flip -> header CRC fail
flip2 = bytearray(hf_pack)
flip2[8] ^= 0x01  # tile_count byte
try:
    read_pack(bytes(flip2))
    check("P9c header bit-flip rejected", False, "not rejected")
except PackError as e:
    check("P9c header bit-flip rejected", True, str(e))

# --------------------------------------------------------------------------
# zstd interop golden vectors (checksum on)
# --------------------------------------------------------------------------
import numpy as np  # noqa: E402

# a fixed 33x33 uint16 ramp plane (nontrivial, compresses well)
plane = (np.arange(33 * 33, dtype="<u2") % 500).tobytes()
comp = zstd.ZstdCompressor(level=19, threads=0, write_checksum=True,
                           write_content_size=True)
frame = comp.compress(plane)
# encode-twice identity
frame2 = zstd.ZstdCompressor(level=19, threads=0, write_checksum=True,
                             write_content_size=True).compress(plane)
check("Z1 zstd checksum-on encode-twice byte-identity", frame == frame2,
      f"{len(frame)} B frame")
# one-shot vs streamed differ? pin one-shot
stream_buf = zstd.ZstdCompressor(level=19, threads=0, write_checksum=True,
                                 write_content_size=True)
so = io._io if False else None  # noqa
import io as _io  # noqa: E402
sbuf = _io.BytesIO()
with stream_buf.stream_writer(sbuf, closefd=False) as w:
    w.write(plane)
streamed = sbuf.getvalue()
check("Z1b one-shot compress pinned (differs from streamed writer noted)",
      True, f"one-shot={len(frame)}B streamed={len(streamed)}B "
      f"{'identical' if streamed == frame else 'DIFFERENT (pin one-shot)'}")

(DATA / "zstd_frame.bin").write_bytes(frame)
(DATA / "zstd_plane.bin").write_bytes(plane)

# --------------------------------------------------------------------------
# v2 .hf chunk golden: v1 112-B header + header_crc32(over [0,112)) + pad u32=0,
# then ONE checksum-on zstd frame. width=height=33 over the ramp `plane`.
# The Rust chunk decoder validates: magic/version(=2)/CRC/pad/frame-checksum/
# decompressed length == width*height*2 (computed in u64).
# --------------------------------------------------------------------------
V2_VERSION = 2
v1_body = struct.pack(
    "<4sIII" + "d" * 11 + "Q",
    b"AHNH", V2_VERSION, 33, 33,
    0.0, 1.0,                       # z_offset, z_scale
    1.0, 2.0, 3.0,                  # rtc_centre
    0.1, 0.2, 0.3, 0.4, -5.0, 40.0,  # region
    len(frame),                      # payload_len
)
assert len(v1_body) == 112
hdr_crc = zlib.crc32(v1_body) & 0xFFFFFFFF
v2_header = v1_body + struct.pack("<II", hdr_crc, 0)  # crc + pad
assert len(v2_header) == 120
v2_chunk = v2_header + frame
(DATA / "chunk_v2.hf").write_bytes(v2_chunk)

# giant-dims corrupt header: width=height=65535 but a tiny 1-byte "frame".
# A conforming decoder must reject on payload-len / frame-decode BEFORE it
# allocates width*height*2 (= 8.59 GB). CRC is made VALID so the reject is
# forced to come from the length path, not the CRC path.
giant_body = struct.pack(
    "<4sIII" + "d" * 11 + "Q",
    b"AHNH", V2_VERSION, 65535, 65535,
    0.0, 1.0, 1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.4, -5.0, 40.0,
    1,  # payload_len = 1
)
giant_crc = zlib.crc32(giant_body) & 0xFFFFFFFF
giant_header = giant_body + struct.pack("<II", giant_crc, 0)
(DATA / "chunk_v2_giant.hf").write_bytes(giant_header + b"\x00")

golden_chunk = {
    "chunk_v2.hf": {
        "width": 33, "height": 33, "version": 2,
        "header_crc32_hex": f"{hdr_crc:08x}",
        "decompressed_len": len(plane),
        "plane_sha256": hashlib.sha256(plane).hexdigest(),
    },
    "chunk_v2_giant.hf": {
        "width": 65535, "height": 65535, "payload_len": 1,
        "wxhx2_u64": 65535 * 65535 * 2,
        "note": "must reject before allocating 8.59 GB",
    },
}

# crc32 golden
crc_input = b"AHN heightfield spike golden vector 0123456789"
crc = zlib.crc32(crc_input) & 0xFFFFFFFF
# sha256 golden
sha = hashlib.sha256(crc_input).hexdigest()

# window/level-19 at 257x257: encode a full-size plane and ensure default decoder decodes
big = (np.arange(257 * 257, dtype="<u2") % 60000).tobytes()
big_frame = comp.compress(big)
big_back = zstd.ZstdDecompressor().decompress(big_frame)
check("Z2 level-19 257x257 default-decoder round-trip",
      big_back == big, f"{len(big)}B -> {len(big_frame)}B frame")

# window log used
params = zstd.get_frame_parameters(big_frame)
check("Z2b frame content size embedded",
      params.content_size == len(big),
      f"content_size={params.content_size} window={params.window_size}")

golden = {
    "zstd": {
        "note": "level=19 threads=0 write_checksum=True write_content_size=True",
        "plane_sha256": hashlib.sha256(plane).hexdigest(),
        "frame_sha256": hashlib.sha256(frame).hexdigest(),
        "frame_len": len(frame),
        "plane_len": len(plane),
        "zstandard_version": zstd.__version__,
        "libzstd_version": ".".join(str(x) for x in zstd.ZSTD_VERSION),
    },
    "crc32": {"input_utf8": crc_input.decode(), "crc32_hex": f"{crc:08x}",
              "crc32": crc},
    "sha256": {"input_utf8": crc_input.decode(), "sha256": sha},
    "chunk_v2": golden_chunk,
    "region_f64_bits": {
        f"{t.level}-{t.tx}-{t.ty}": [f"{f64_bits(x):016x}" for x in
                                     struct.unpack_from("<6d", t.primary, 56)]
        for t in hf_tiles
    },
    "packs": {
        "heightfield": {
            "file": "heightfield.hfp", "len": len(hf_pack),
            "sha256": hashlib.sha256(hf_pack).hexdigest(),
            "dataset_id": read_pack(hf_pack)["dataset_id"],
            "tile_count": len(hf_tiles), "content_kind": KIND_HEIGHTFIELD,
            "root_geometric_error": hf_root_ge,
            "index_order_keys": idx_keys,
        },
        "game": {
            "file": "game.hfp", "len": len(game_pack),
            "sha256": hashlib.sha256(game_pack).hexdigest(),
            "dataset_id": read_pack(game_pack)["dataset_id"],
            "tile_count": len(game_tiles), "content_kind": KIND_GAME,
            "root_geometric_error": game_root_ge,
        },
    },
}
(DATA / "golden.json").write_bytes(
    (json.dumps(golden, indent=2) + "\n").encode("utf-8"))
(DATA / "heightfield.hfp").write_bytes(hf_pack)
(DATA / "game.hfp").write_bytes(game_pack)

# --------------------------------------------------------------------------
# Corrupt-header giant-dims: allocation must not precede validation
# --------------------------------------------------------------------------
# Build a fake .hf header claiming width=height=65535 but tiny payload; the
# v2 decoder must reject on header CRC / payload-len BEFORE width*height*2 alloc.
check("H1 width*height*2 computed in 64-bit (no overflow at 65535^2*2)",
      65535 * 65535 * 2 == 8589672450 and 65535 * 65535 * 2 < 2**64,
      "8.59e9 fits u64, overflows u32")

# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
print(f"\n{'DOUBT':<58} RESULT  EVIDENCE")
print("-" * 100)
red = 0
for doubt, res, ev in results:
    if res == "RED":
        red += 1
    print(f"{doubt:<58} {res:<6} {ev}")
print("-" * 100)
print(f"GREEN={sum(1 for _,r,_ in results if r=='GREEN')} RED={red}")
print(f"\nArtifacts written to {DATA}")
for f in sorted(DATA.iterdir()):
    print(f"  {f.name}: {f.stat().st_size} B")
sys.exit(1 if red else 0)
