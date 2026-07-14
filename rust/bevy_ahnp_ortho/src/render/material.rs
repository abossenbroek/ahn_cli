//! The ortho-draped material: unlit, sRGB, colours shown 1:1 (no filmic
//! tonemapping curve) — matching what the ortho producer's JPEG bytes mean.
//! Pairs with `Camera3d` + `Tonemapping::None` (see `render::mod`'s
//! `tag_ortho_camera`).

use bevy::asset::RenderAssetUsages;
use bevy::image::Image;
use bevy::pbr::StandardMaterial;
use bevy::prelude::Color;

/// Decode `jpeg` bytes into a Bevy `Image` (sRGB, `Rgba8UnormSrgb`).
///
/// # Errors
/// Returns `image::ImageError` if `jpeg` doesn't decode.
pub fn decode_jpeg(jpeg: &[u8]) -> Result<Image, image::ImageError> {
    let dyn_img = image::load_from_memory_with_format(jpeg, image::ImageFormat::Jpeg)?;
    Ok(Image::from_dynamic(
        dyn_img,
        true,
        RenderAssetUsages::RENDER_WORLD,
    ))
}

// The `gpu_textures` feature (GPU-native BC1 transcode via `intel_tex_2` at
// load) lives in `render::gpu_texture`; `render::stream_tiles`'s
// `textured_material` dispatches to it instead of this module's plain
// `Rgba8UnormSrgb` `decode_jpeg` when the feature is on.

/// An unlit material draping `texture` 1:1 (no lighting term) — matching the
/// ortho pixels' meaning as measured colour, not a lit surface.
pub fn ortho_material(texture: bevy::asset::Handle<Image>) -> StandardMaterial {
    StandardMaterial {
        base_color_texture: Some(texture),
        unlit: true,
        base_color: Color::WHITE,
        ..Default::default()
    }
}

/// The untextured fallback: flat grey, still unlit (used when a tile has no
/// texture blob, e.g. mid-decode or a texture-forbidden profile).
pub fn untextured_material() -> StandardMaterial {
    StandardMaterial {
        unlit: true,
        base_color: Color::srgb(0.6, 0.6, 0.6),
        ..Default::default()
    }
}
