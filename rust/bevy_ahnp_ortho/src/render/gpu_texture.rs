//! `gpu_textures` feature: transcode a decoded tile JPEG into a GPU-native
//! block-compressed texture (BC1/DXT1 via `intel_tex_2`) at load, instead of
//! uploading the plain `Rgba8UnormSrgb` `material::decode_jpeg` produces.
//! BC1 is the smallest/fastest BCn format and has no alpha channel to lose —
//! a fit for our opaque ortho colour textures (4:1 vs. `Rgba8`, decoded
//! once at CPU load time rather than resampled every frame on the GPU).
//!
//! `intel_tex_2`'s BC1 encoder operates on whole 4x4 blocks with no partial-
//! block handling of its own, so a JPEG whose dimensions aren't a multiple
//! of 4 is padded up first (edge pixels replicated into the pad, so the
//! extra fractional block blends with its real neighbour rather than reading
//! black) before compression; the resulting `Image` is sized to the padded
//! (not the source) dimensions.
//!
//! **UV compensation.** Every tile mesh's UVs are computed against the
//! tile's own *native* pixel dimensions (`mesh_hf.rs`/`mesh_glb.rs`,
//! `(c+0.5)/tw`), texel-centre, `0..1`-normalized against `tw`/`th` — but
//! this module uploads the image at the *padded* `tw'`/`th'` instead. Sampled
//! unchanged, those UVs would run past the tile's real texels into the
//! (edge-replicated but still wrong) pad — worse, every tile pads
//! independently, so the quadtree's pixel-perfect shared boundaries would
//! misregister tile-to-tile, not just at the outer edge. [`decode_jpeg_bc1`]
//! therefore also returns the `(tw/tw', th/th')` scale a caller must apply
//! via `StandardMaterial::uv_transform` (`render::stream_tiles`'s
//! `textured_material` does this): `u' = (c+0.5)/tw * (tw/tw') =
//! (c+0.5)/tw'`, which samples texel `c` exactly, never the pad.

use bevy::asset::RenderAssetUsages;
use bevy::image::Image;
use bevy::math::Vec2;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use intel_tex_2::RgbaSurface;

use crate::render::material::DecodedTexture;

/// Decode `jpeg` and transcode it to a BC1 (`Bc1RgbaUnormSrgb`) `Image`, with
/// the UV scale (see this module's doc comment) needed to sample only its
/// valid (unpadded) region.
///
/// # Errors
/// Returns `image::ImageError` if `jpeg` doesn't decode.
pub fn decode_jpeg_bc1(jpeg: &[u8]) -> Result<DecodedTexture, image::ImageError> {
    let dyn_img = image::load_from_memory_with_format(jpeg, image::ImageFormat::Jpeg)?;
    let rgba = dyn_img.to_rgba8();
    let (width, height) = rgba.dimensions();
    let (padded_width, padded_height) = (round_up_4(width), round_up_4(height));
    let padded = pad_edge_replicate(&rgba, width, height, padded_width, padded_height);

    let surface = RgbaSurface {
        data: &padded,
        width: padded_width,
        height: padded_height,
        stride: padded_width * 4,
    };
    let blocks = intel_tex_2::bc1::compress_blocks(&surface);

    let image = Image::new(
        Extent3d {
            width: padded_width,
            height: padded_height,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        blocks,
        TextureFormat::Bc1RgbaUnormSrgb,
        RenderAssetUsages::RENDER_WORLD,
    );
    let uv_scale = Vec2::new(
        width as f32 / padded_width as f32,
        height as f32 / padded_height as f32,
    );
    Ok(DecodedTexture { image, uv_scale })
}

fn round_up_4(v: u32) -> u32 {
    v.div_ceil(4) * 4
}

/// Pad an `(width, height)` RGBA8 buffer up to `(padded_width,
/// padded_height)`, replicating the last valid row/column into the pad.
fn pad_edge_replicate(
    rgba: &image::RgbaImage,
    width: u32,
    height: u32,
    padded_width: u32,
    padded_height: u32,
) -> Vec<u8> {
    if width == padded_width && height == padded_height {
        return rgba.as_raw().clone();
    }
    let mut out = vec![0u8; (padded_width * padded_height * 4) as usize];
    for y in 0..padded_height {
        let sy = y.min(height - 1);
        for x in 0..padded_width {
            let sx = x.min(width - 1);
            let px = rgba.get_pixel(sx, sy).0;
            let dst = ((y * padded_width + x) * 4) as usize;
            out[dst..dst + 4].copy_from_slice(&px);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_jpeg(width: u32, height: u32) -> Vec<u8> {
        let mut img = image::RgbaImage::new(width, height);
        for (x, y, px) in img.enumerate_pixels_mut() {
            *px = image::Rgba([(x * 40) as u8, (y * 40) as u8, 128, 255]);
        }
        let mut jpeg_bytes = Vec::new();
        image::DynamicImage::ImageRgba8(img)
            .to_rgb8()
            .write_to(
                &mut std::io::Cursor::new(&mut jpeg_bytes),
                image::ImageFormat::Jpeg,
            )
            .expect("encode synthetic jpeg");
        jpeg_bytes
    }

    /// A tiny synthetic JPEG (not a multiple of 4 on either axis) exercises
    /// the padding path; asserts the result is a well-formed BC1 `Image` at
    /// the padded size with the expected block-compressed byte length.
    #[test]
    fn transcodes_to_a_valid_bc1_image() {
        let out = decode_jpeg_bc1(&synthetic_jpeg(6, 5)).expect("transcode");
        assert_eq!(
            out.image.texture_descriptor.format,
            TextureFormat::Bc1RgbaUnormSrgb
        );
        assert_eq!(
            out.image.texture_descriptor.size,
            Extent3d {
                width: 8,
                height: 8,
                depth_or_array_layers: 1
            }
        );
        // BC1: 8 bytes/4x4 block; 8x8 padded -> 2x2 blocks -> 32 bytes.
        assert_eq!(out.image.data.as_ref().expect("data").len(), 32);
    }

    /// The blocking 3D-graphics review finding: without the `uv_scale`
    /// compensation, a tile mesh's native-dimension texel-centre UV for its
    /// LAST column/row samples into the padding, not the tile's own last
    /// real texel -- and since every tile pads independently, that breaks
    /// the quadtree's pixel-perfect shared tile boundaries, not just the
    /// outer edge. This reproduces the bug's own numbers: a 5x5 tile (the
    /// `game` fixture's leaf grid) padded to 8x8.
    #[test]
    fn uv_scale_keeps_native_texel_centre_uvs_inside_the_real_image() {
        let out = decode_jpeg_bc1(&synthetic_jpeg(5, 5)).expect("transcode");
        assert_eq!(out.uv_scale, Vec2::new(5.0 / 8.0, 5.0 / 8.0));

        // mesh_hf.rs/mesh_glb.rs's texel-centre UV for column/row `tw - 1`
        // (the tile's own last real texel) against its NATIVE dimension.
        let tw = 5.0f32;
        let native_u = (tw - 1.0 + 0.5) / tw; // 0.9

        // Uncorrected (the bug): sampling `native_u` directly against the
        // padded 8-wide texture lands on physical column `native_u * 8 =
        // 7.2` -- inside the pad (real columns are 0..5, pad is 5..8), not
        // the tile's own last texel.
        let uncorrected_padded_column = native_u * 8.0;
        assert!(
            uncorrected_padded_column >= 5.0,
            "sanity check: the bug's own numbers should land in the pad, got column {uncorrected_padded_column}"
        );

        // Corrected: `native_u * uv_scale` lands on physical column
        // `((tw-1)+0.5)/tw' ~= 4.5`, i.e. back inside the tile's own real
        // texels (columns 0..5), not the pad.
        let corrected_u = native_u * out.uv_scale.x;
        let corrected_padded_column = corrected_u * 8.0;
        assert!(
            (4.0..5.0).contains(&corrected_padded_column),
            "corrected UV should land in the tile's last real texel, got column {corrected_padded_column}"
        );
    }
}
