//! Binary glTF (glb) parsing + dequantization for the `game` profile
//! (`content_kind = 1`): the counterpart to `ahnp::source`'s heightfield and
//! splat decode paths.
//!
//! Mirrors `ahn_cli.tiles3d.gltf_quant`'s exact wire contract (see that
//! module's docstring) rather than using a generic glTF crate: this reader
//! only ever sees glbs from our own producer, so a small, fully-validated
//! reader for the one fixed shape is simpler and more auditable than fitting
//! a general-purpose glTF importer to a custom pair of required extensions
//! (`EXT_meshopt_compression` + `KHR_mesh_quantization`) it doesn't natively
//! model. Pure module: parses bytes to plain data, no Bevy types, no I/O.
//!
//! Layout consumed (all mandatory, `extensionsRequired` = both extensions):
//! - `accessors[0]` POSITION: `uint16 x3`, `accessors[1]` TEXCOORD_0:
//!   normalized `uint16 x2`, `accessors[2]` indices: `uint32` SCALAR.
//! - `bufferViews[0..2]` each carry the *actual* compressed bytes in their
//!   `extensions.EXT_meshopt_compression` object (`buffer: 0` = the glb BIN
//!   chunk, `byteOffset`/`byteLength` there) — NOT their own top-level
//!   `buffer`/`byteOffset`, which point at a fallback buffer that is never
//!   allocated (`ahn_cli.tiles3d.gltf_quant`'s doc comment).
//! - `bufferViews[3]` (the JPEG): an ordinary bufferView into the BIN chunk,
//!   referenced by `images[0].bufferView`.
//! - `nodes[0].translation`/`.scale`: the `KHR_mesh_quantization` dequant
//!   transform, with the tile's RTC centre already folded into
//!   `translation` (see `gltf_quant.py`) — dequantizing via
//!   `translation + scale * q` yields the vertex directly in glTF-y-up
//!   ECEF-swizzled space, no separate centre add needed.

use crate::engine::meshopt::decode_buffer_view;

const GLB_MAGIC: u32 = 0x4654_6C67;
const GLB_VERSION: u32 = 2;
const CHUNK_JSON: u32 = 0x4E4F_534A;
const CHUNK_BIN: u32 = 0x004E_4942;

/// One decoded `content_kind = 1` tile: plain data, no Bevy types.
///
/// `ecef_positions` are already un-swizzled back to raw ECEF metres
/// (inverting `gltf_quant`'s `(x, z, -y)` glTF-y-up axis permutation) — a
/// renderer composes them with its own ECEF -> world anchor, exactly like
/// the heightfield and splat paths.
pub struct GlbTile {
    pub ecef_positions: Vec<[f64; 3]>,
    pub uvs: Vec<[f32; 2]>,
    pub indices: Vec<u32>,
    pub texture: Vec<u8>,
}

/// Everything that can go wrong decoding a `game` profile glb.
#[derive(Debug, thiserror::Error)]
pub enum GlbError {
    #[error("glb: {0}")]
    Malformed(String),
    #[error("glb meshopt decode: {0}")]
    Meshopt(String),
}

/// Decode a `game` profile glb blob into plain vertex/index/texture data.
///
/// # Errors
/// [`GlbError::Malformed`] if the container or JSON document doesn't match
/// the fixed shape this reader expects; [`GlbError::Meshopt`] if a
/// compressed stream fails to decode.
pub fn decode_glb(bytes: &[u8]) -> Result<GlbTile, GlbError> {
    let (json_bytes, bin) = split_chunks(bytes)?;
    let doc: serde_json::Value = serde_json::from_slice(json_bytes)
        .map_err(|e| GlbError::Malformed(format!("glb JSON: {e}")))?;

    let accessors = array(&doc, "accessors")?;
    let vertex_count = accessor_count(accessors, 0)?;
    let index_count = accessor_count(accessors, 2)?;

    let buffer_views = array(&doc, "bufferViews")?;
    let position_stream = meshopt_stream(buffer_views, 0, bin)?;
    let uv_stream = meshopt_stream(buffer_views, 1, bin)?;
    let index_stream = meshopt_stream(buffer_views, 2, bin)?;

    let position_bytes = decode_buffer_view(
        &position_stream.mode,
        "NONE",
        vertex_count,
        position_stream.byte_stride,
        position_stream.data,
    )
    .map_err(GlbError::Meshopt)?;
    let uv_bytes = decode_buffer_view(
        &uv_stream.mode,
        "NONE",
        vertex_count,
        uv_stream.byte_stride,
        uv_stream.data,
    )
    .map_err(GlbError::Meshopt)?;
    let index_bytes = decode_buffer_view(
        &index_stream.mode,
        "NONE",
        index_count,
        index_stream.byte_stride,
        index_stream.data,
    )
    .map_err(GlbError::Meshopt)?;

    let node = doc
        .get("nodes")
        .and_then(|n| n.get(0))
        .ok_or_else(|| GlbError::Malformed("missing nodes[0]".into()))?;
    let scale = f64_triple(node, "scale")?;
    let translation = f64_triple(node, "translation")?;

    let mut ecef_positions = Vec::with_capacity(vertex_count);
    for i in 0..vertex_count {
        let base = i * position_stream.byte_stride;
        let qx = f64::from(u16::from_le_bytes([
            position_bytes[base],
            position_bytes[base + 1],
        ]));
        let qy = f64::from(u16::from_le_bytes([
            position_bytes[base + 2],
            position_bytes[base + 3],
        ]));
        let qz = f64::from(u16::from_le_bytes([
            position_bytes[base + 4],
            position_bytes[base + 5],
        ]));
        // Dequantized (yx, yy, yz) is glTF-y-up ECEF-swizzled per
        // `gltf_quant.py`: `(x, y, z)_ecef -> (x, z, -y)_gltf`. Invert it.
        let yx = translation[0] + scale[0] * qx;
        let yy = translation[1] + scale[1] * qy;
        let yz = translation[2] + scale[2] * qz;
        ecef_positions.push([yx, -yz, yy]);
    }

    let mut uvs = Vec::with_capacity(vertex_count);
    for i in 0..vertex_count {
        let base = i * uv_stream.byte_stride;
        let qu = f32::from(u16::from_le_bytes([uv_bytes[base], uv_bytes[base + 1]]));
        let qv = f32::from(u16::from_le_bytes([uv_bytes[base + 2], uv_bytes[base + 3]]));
        uvs.push([qu / 65535.0, qv / 65535.0]);
    }

    let mut indices = Vec::with_capacity(index_count);
    for i in 0..index_count {
        let base = i * index_stream.byte_stride;
        indices.push(u32::from_le_bytes(
            index_bytes[base..base + 4].try_into().unwrap(),
        ));
    }

    let texture = image_bytes(&doc, buffer_views, bin)?;

    Ok(GlbTile {
        ecef_positions,
        uvs,
        indices,
        texture,
    })
}

/// Split a `.glb` container into its JSON and BIN chunk byte slices.
fn split_chunks(bytes: &[u8]) -> Result<(&[u8], &[u8]), GlbError> {
    if bytes.len() < 12 {
        return Err(GlbError::Malformed(
            "shorter than the 12-byte glb header".into(),
        ));
    }
    let magic = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
    let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    let total_len = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    if magic != GLB_MAGIC {
        return Err(GlbError::Malformed(format!("bad magic 0x{magic:08x}")));
    }
    if version != GLB_VERSION {
        return Err(GlbError::Malformed(format!(
            "bad version {version}, expected 2"
        )));
    }
    if bytes.len() < total_len {
        return Err(GlbError::Malformed(format!(
            "truncated: {} bytes, header declares {total_len}",
            bytes.len()
        )));
    }

    let mut offset = 12usize;
    let (json_len, json_type) = chunk_header(bytes, offset)?;
    offset += 8;
    if json_type != CHUNK_JSON {
        return Err(GlbError::Malformed(format!(
            "first chunk type 0x{json_type:08x}, expected JSON"
        )));
    }
    if bytes.len() < offset + json_len {
        return Err(GlbError::Malformed("JSON chunk overruns the buffer".into()));
    }
    let json_bytes = &bytes[offset..offset + json_len];
    offset += json_len;

    let (bin_len, bin_type) = chunk_header(bytes, offset)?;
    offset += 8;
    if bin_type != CHUNK_BIN {
        return Err(GlbError::Malformed(format!(
            "second chunk type 0x{bin_type:08x}, expected BIN"
        )));
    }
    if bytes.len() < offset + bin_len {
        return Err(GlbError::Malformed("BIN chunk overruns the buffer".into()));
    }
    let bin = &bytes[offset..offset + bin_len];
    Ok((json_bytes, bin))
}

fn chunk_header(bytes: &[u8], offset: usize) -> Result<(usize, u32), GlbError> {
    if bytes.len() < offset + 8 {
        return Err(GlbError::Malformed("truncated chunk header".into()));
    }
    let len = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
    let kind = u32::from_le_bytes(bytes[offset + 4..offset + 8].try_into().unwrap());
    Ok((len, kind))
}

fn array<'a>(
    doc: &'a serde_json::Value,
    key: &str,
) -> Result<&'a Vec<serde_json::Value>, GlbError> {
    doc.get(key)
        .and_then(|v| v.as_array())
        .ok_or_else(|| GlbError::Malformed(format!("missing `{key}` array")))
}

fn accessor_count(accessors: &[serde_json::Value], index: usize) -> Result<usize, GlbError> {
    accessors
        .get(index)
        .and_then(|a| a.get("count"))
        .and_then(serde_json::Value::as_u64)
        .map(|c| c as usize)
        .ok_or_else(|| GlbError::Malformed(format!("accessors[{index}].count missing/non-integer")))
}

struct RawMeshoptStream<'a> {
    mode: String,
    byte_stride: usize,
    data: &'a [u8],
}

/// Read `bufferViews[index]`'s `EXT_meshopt_compression` extension: the
/// actual compressed bytes live in the BIN chunk at its own `byteOffset`
/// (NOT the bufferView's top-level `byteOffset`, which addresses a
/// never-allocated fallback buffer).
fn meshopt_stream<'a>(
    buffer_views: &[serde_json::Value],
    index: usize,
    bin: &'a [u8],
) -> Result<RawMeshoptStream<'a>, GlbError> {
    let ext = buffer_views
        .get(index)
        .and_then(|bv| bv.get("extensions"))
        .and_then(|e| e.get("EXT_meshopt_compression"))
        .ok_or_else(|| {
            GlbError::Malformed(format!(
                "bufferViews[{index}].extensions.EXT_meshopt_compression missing"
            ))
        })?;
    let mode = ext
        .get("mode")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| GlbError::Malformed(format!("bufferViews[{index}] meshopt mode missing")))?
        .to_string();
    let byte_stride = ext
        .get("byteStride")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| {
            GlbError::Malformed(format!("bufferViews[{index}] meshopt byteStride missing"))
        })? as usize;
    let byte_offset = ext
        .get("byteOffset")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0) as usize;
    let byte_length = ext
        .get("byteLength")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| {
            GlbError::Malformed(format!("bufferViews[{index}] meshopt byteLength missing"))
        })? as usize;
    if bin.len() < byte_offset + byte_length {
        return Err(GlbError::Malformed(format!(
            "bufferViews[{index}] meshopt stream overruns the BIN chunk"
        )));
    }
    Ok(RawMeshoptStream {
        mode,
        byte_stride,
        data: &bin[byte_offset..byte_offset + byte_length],
    })
}

fn f64_triple(node: &serde_json::Value, key: &str) -> Result<[f64; 3], GlbError> {
    let arr = node
        .get(key)
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| GlbError::Malformed(format!("nodes[0].{key} missing/not an array")))?;
    if arr.len() != 3 {
        return Err(GlbError::Malformed(format!(
            "nodes[0].{key} has {} elements, expected 3",
            arr.len()
        )));
    }
    let mut out = [0.0f64; 3];
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = arr[i]
            .as_f64()
            .ok_or_else(|| GlbError::Malformed(format!("nodes[0].{key}[{i}] is not a number")))?;
    }
    Ok(out)
}

/// Read the embedded JPEG via `images[0].bufferView` (an ordinary
/// bufferView, no meshopt extension — straight into the BIN chunk).
fn image_bytes(
    doc: &serde_json::Value,
    buffer_views: &[serde_json::Value],
    bin: &[u8],
) -> Result<Vec<u8>, GlbError> {
    let bv_index = doc
        .get("images")
        .and_then(|images| images.get(0))
        .and_then(|image| image.get("bufferView"))
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| GlbError::Malformed("images[0].bufferView missing".into()))?
        as usize;
    let bv = buffer_views
        .get(bv_index)
        .ok_or_else(|| GlbError::Malformed("images[0].bufferView out of range".into()))?;
    let byte_offset = bv
        .get("byteOffset")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0) as usize;
    let byte_length = bv
        .get("byteLength")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| GlbError::Malformed("image bufferView byteLength missing".into()))?
        as usize;
    if bin.len() < byte_offset + byte_length {
        return Err(GlbError::Malformed(
            "image bufferView overruns the BIN chunk".into(),
        ));
    }
    Ok(bin[byte_offset..byte_offset + byte_length].to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A real `--profile game` tile, generated by the Python producer
    /// (20x20 synthetic ortho, `tile_pixels=8`; tile (level 2, tx 0, ty 0),
    /// the smallest leaf, 3132 bytes).
    const GAME_TILE: &[u8] = include_bytes!("../../tests/data/game_tile.glb");

    #[test]
    fn decodes_a_real_game_tile() {
        let tile = decode_glb(GAME_TILE).expect("decode");
        // This leaf's sampled grid is 5x5 (25 vertices), 4x4 cells x 2
        // triangles x 3 indices = 96 indices.
        assert_eq!(tile.ecef_positions.len(), 25);
        assert_eq!(tile.uvs.len(), 25);
        assert_eq!(tile.indices.len(), 96);
        assert!(tile.indices.iter().all(|&i| (i as usize) < 25));
        assert!(!tile.texture.is_empty());
        // Every position must be finite and at a genuine ECEF magnitude
        // (a few thousand km from the origin, not near-zero or NaN).
        for p in &tile.ecef_positions {
            assert!(p.iter().all(|c| c.is_finite()));
            let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            assert!(
                r > 1_000_000.0,
                "position {p:?} not planetary-magnitude ECEF"
            );
        }
        // UVs stay in the unit square.
        for uv in &tile.uvs {
            assert!((0.0..=1.0).contains(&uv[0]) && (0.0..=1.0).contains(&uv[1]));
        }
        // The embedded texture is a real JPEG (SOI marker).
        assert_eq!(&tile.texture[0..2], &[0xFF, 0xD8]);
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bytes = GAME_TILE.to_vec();
        bytes[0] = 0;
        assert!(decode_glb(&bytes).is_err());
    }

    #[test]
    fn rejects_truncated_input() {
        assert!(decode_glb(&GAME_TILE[..8]).is_err());
    }
}
