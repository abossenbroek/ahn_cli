//! The `gpu_textures` feature links `intel_tex_2`'s precompiled ISPC object
//! code, which references the C++ exception-handling ABI
//! (`__gxx_personality_v0`) but doesn't pull `libstdc++` in itself -- `cc`
//! only auto-links it for crates it compiles as C++, not for a dependency's
//! bundled precompiled objects. Without this, linking any binary that
//! enables `gpu_textures` on Linux fails with an undefined-symbol error.

fn main() {
    if std::env::var_os("CARGO_FEATURE_GPU_TEXTURES").is_some()
        && std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux")
    {
        println!("cargo:rustc-link-lib=stdc++");
    }
}
