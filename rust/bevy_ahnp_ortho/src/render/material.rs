//! The ortho-draped material: unlit, sRGB, colours shown 1:1 (no filmic
//! tonemapping curve) — matching what the ortho producer's JPEG bytes mean.
//! Pairs with `Camera3d` + `Tonemapping::None` (see `render::mod`'s
//! `tag_ortho_camera`).

use bevy::asset::RenderAssetUsages;
use bevy::image::Image;
use bevy::math::{Affine2, Vec2};
use bevy::pbr::StandardMaterial;
use bevy::prelude::Color;

/// A decoded tile texture, plus the UV scale needed to sample only its
/// *valid* region.
///
/// Every tile mesh's UVs (`mesh_hf.rs`/`mesh_glb.rs`) are computed against
/// the tile's own native pixel dimensions, texel-centre, `0..1`-normalized.
/// The plain [`decode_jpeg`] path uploads the image at exactly those
/// dimensions, so `uv_scale` is always [`Vec2::ONE`] (an identity
/// transform) there. `render::gpu_texture::decode_jpeg_bc1`'s block-
/// compressed path is the one exception: it pads the texture up to a
/// multiple of 4 first, so its native-dimension UVs would run past the
/// tile's real texels into the pad — `uv_scale` there is `<1` on the
/// padded axis/axes, meant to be applied via
/// `StandardMaterial::uv_transform`.
pub struct DecodedTexture {
    pub image: Image,
    pub uv_scale: Vec2,
}

/// Decode `jpeg` bytes into a Bevy `Image` (sRGB, `Rgba8UnormSrgb`), with an
/// identity `uv_scale` (this path never pads).
///
/// # Errors
/// Returns `image::ImageError` if `jpeg` doesn't decode.
pub fn decode_jpeg(jpeg: &[u8]) -> Result<DecodedTexture, image::ImageError> {
    let dyn_img = image::load_from_memory_with_format(jpeg, image::ImageFormat::Jpeg)?;
    let image = Image::from_dynamic(dyn_img, true, RenderAssetUsages::RENDER_WORLD);
    Ok(DecodedTexture {
        image,
        uv_scale: Vec2::ONE,
    })
}

// The `gpu_textures` feature (GPU-native BC1 transcode via `intel_tex_2` at
// load) lives in `render::gpu_texture`; `render::stream_tiles`'s
// `textured_material` dispatches to it instead of this module's plain
// `Rgba8UnormSrgb` `decode_jpeg` when the feature is on.

/// An unlit material draping `texture` 1:1 (no lighting term) — matching the
/// ortho pixels' meaning as measured colour, not a lit surface. `uv_scale`
/// (from the [`DecodedTexture`] the caller decoded `texture` from) is applied
/// as the material's `uv_transform`, so a padded `gpu_textures` upload still
/// samples only its tile's real texels (see this module's/`gpu_texture`'s
/// doc comments) — `Vec2::ONE` (the plain-decode default) makes this an
/// identity transform, a no-op.
pub fn ortho_material(texture: bevy::asset::Handle<Image>, uv_scale: Vec2) -> StandardMaterial {
    StandardMaterial {
        base_color_texture: Some(texture),
        uv_transform: Affine2::from_scale(uv_scale),
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
