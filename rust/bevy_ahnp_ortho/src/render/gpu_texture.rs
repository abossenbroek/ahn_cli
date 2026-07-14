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

use bevy::asset::RenderAssetUsages;
use bevy::image::Image;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
use intel_tex_2::RgbaSurface;

/// Decode `jpeg` and transcode it to a BC1 (`Bc1RgbaUnormSrgb`) `Image`.
///
/// # Errors
/// Returns `image::ImageError` if `jpeg` doesn't decode.
pub fn decode_jpeg_bc1(jpeg: &[u8]) -> Result<Image, image::ImageError> {
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

    Ok(Image::new(
        Extent3d {
            width: padded_width,
            height: padded_height,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        blocks,
        TextureFormat::Bc1RgbaUnormSrgb,
        RenderAssetUsages::RENDER_WORLD,
    ))
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

    /// A tiny synthetic JPEG (not a multiple of 4 on either axis) exercises
    /// the padding path; asserts the result is a well-formed BC1 `Image` at
    /// the padded size with the expected block-compressed byte length.
    #[test]
    fn transcodes_to_a_valid_bc1_image() {
        let mut img = image::RgbaImage::new(6, 5);
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

        let out = decode_jpeg_bc1(&jpeg_bytes).expect("transcode");
        assert_eq!(
            out.texture_descriptor.format,
            TextureFormat::Bc1RgbaUnormSrgb
        );
        assert_eq!(
            out.texture_descriptor.size,
            Extent3d {
                width: 8,
                height: 8,
                depth_or_array_layers: 1
            }
        );
        // BC1: 8 bytes/4x4 block; 8x8 padded -> 2x2 blocks -> 32 bytes.
        assert_eq!(out.data.as_ref().expect("data").len(), 32);
    }
}
