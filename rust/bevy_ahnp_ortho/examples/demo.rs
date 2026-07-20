//! Interactive demo **and integration tutorial** for `bevy_ahnp_ortho`.
//!
//! This file is written to be read top-to-bottom as a walkthrough, and to be
//! the thing you copy from when wiring this crate into your own app — it is
//! the crate's recommended starting point (see the README's "Usage" section).
//! `examples/viewer.rs`, `viewer_splat.rs` and `viewer_points.rs` stay as
//! narrower, single-purpose references.
//!
//! ## The minimal integration (four steps)
//!
//! 1. **Add the plugin.** `app.add_plugins(bevy_ahnp_ortho::AhnpOrthoPlugin)`.
//!    It registers the systems that stream a pack's LOD-selected tiles every
//!    frame and tags every `Camera3d` with `Tonemapping::None` (so the ortho
//!    JPEG colours you see are the source pixels, not a filmic grade) — see
//!    [`main`] below, and `render::AhnpOrthoPlugin`'s own doc comment for what
//!    it wires up under the hood.
//! 2. **Open and spawn a pack.** `AhnpPack::open(path)` returns a
//!    `Result<AhnpPack, AhnpError>`; on success, `commands.spawn(pack)` — see
//!    [`load_pack`] below. Opening is cheap (it parses the pack's header and
//!    binary index, not its tile content); tile bytes stream in afterwards.
//! 3. **Frame a camera.** `Framing::fit_default(pack.world_aabb())` turns the
//!    pack's real world-space extent into a `center`/`radius`/`distance` a
//!    camera can use — see [`reframe_on_load`] and [`orbit_camera`] below.
//!    `world_aabb()` is available the instant `open()` returns (it comes from
//!    the pack's own header/index, not from decoded tiles), so you can frame
//!    the camera before a single tile has finished streaming in.
//! 4. **Add `bevy_gaussian_splatting::GaussianSplattingPlugin` yourself if you
//!    built with `--features splat`** (this crate never adds it implicitly,
//!    so an app that only ever opens heightfield/game packs never pays for
//!    it) — see `main` below.
//!
//! That's the whole integration surface. Everything else in this file —
//! egui sliders, the FPS readout, the runtime file switcher — is demo
//! scaffolding built *on top of* those four steps, not part of the API.
//!
//! ## What this demo shows
//!
//! - **Live FPS** (an egui overlay, reading `bevy::diagnostic`'s
//!   `FrameTimeDiagnosticsPlugin`).
//! - **Lighting sliders**: sun azimuth/elevation/illuminance, ambient
//!   brightness, and a toggle between the ortho's normal unlit (1:1 colour)
//!   material and a lit `StandardMaterial` shading mode, so you can see the
//!   difference `unlit` makes.
//! - **A screen-space-error slider** (`render::SseThreshold`): lower loads
//!   finer tiles sooner (more detail, more streaming load); higher keeps
//!   coarser tiles selected longer.
//! - **Splat sliders** (`--features splat`): `global_scale`/`global_opacity`,
//!   applied to *both* future tiles (via `splat::SplatSettings`) and every
//!   already-spawned tile's own `CloudSettings` component — see
//!   [`apply_splat_settings`] and `splat::SplatSettings`'s own doc comment on
//!   why both are necessary.
//! - **Runtime file loading**: a text field plus any paths passed on the
//!   command line as one-click buttons. Switching packs cleanly despawns the
//!   old one's tiles (via `render::AhnpPack`'s `on_remove` hook) before
//!   opening the new one, and a bad path reports an error in the UI instead
//!   of panicking.
//!
//! ```text
//! cargo run --example demo
//! cargo run --example demo -- path/to/tiles.hfp
//! cargo run --example demo --features splat -- path/to/splat_tiles.hfp
//! cargo run --example demo --features "splat gpu_textures" -- a.hfp b.hfp c.hfp
//! ```
//!
//! With no path at all, the demo still opens (an empty scene) so you can type
//! one into the UI's text field.

use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::light::{DirectionalLight, GlobalAmbientLight};
use bevy::prelude::*;
use bevy_ahnp_ortho::render::{AhnpPack, SseThreshold};
use bevy_ahnp_ortho::{AhnpOrthoPlugin, Framing};
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};

#[path = "helpers/orbit.rs"]
mod orbit;
use bevy::input::mouse::MouseWheel;
use orbit::{apply_zoom, zoom_delta};

#[cfg(feature = "splat")]
use bevy_ahnp_ortho::splat::SplatSettings;
#[cfg(feature = "splat")]
use bevy_gaussian_splatting::{CloudSettings, GaussianSplattingPlugin};

fn main() {
    // Every non-flag argument is a candidate pack path: the first one loads
    // automatically at startup, and all of them show up as one-click "load"
    // buttons in the UI (see `ui_system`).
    let candidates: Vec<String> = std::env::args().skip(1).collect();

    let mut app = App::new();
    app.add_plugins(DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window {
            title: "bevy_ahnp_ortho demo".into(),
            ..default()
        }),
        ..default()
    }))
    .add_plugins(FrameTimeDiagnosticsPlugin::default())
    .add_plugins(EguiPlugin::default())
    // ---- STEP 1: add the plugin (see the module doc comment above) --------
    .add_plugins(AhnpOrthoPlugin);

    // ---- STEP 4: the splat feature needs its render plugin added too ------
    #[cfg(feature = "splat")]
    app.add_plugins(GaussianSplattingPlugin);

    app.insert_resource(Candidates(candidates.clone()))
        .insert_resource(DemoUi::new(candidates.first().cloned()))
        .insert_resource(CurrentPack::default())
        .insert_resource(NeedsFraming::default())
        .insert_resource(OrbitState::default())
        .insert_resource(LightingSettings::default())
        .init_resource::<SseThreshold>();
    #[cfg(feature = "splat")]
    app.insert_resource(SplatUi::default());

    app.add_systems(Startup, (spawn_camera_and_light, load_initial_pack))
        .add_systems(Update, (reframe_on_load, orbit_camera, apply_lighting))
        .add_systems(EguiPrimaryContextPass, ui_system);
    #[cfg(feature = "splat")]
    app.add_systems(Update, apply_splat_settings);

    app.run();
}

/// Elevation the camera holds while auto-orbiting (radians up from
/// horizontal) — matches the other example viewers.
const ORBIT_ELEVATION: f32 = 0.5;

/// CLI-provided pack paths, shown as quick-load buttons.
#[derive(Resource)]
struct Candidates(Vec<String>);

/// The currently open pack's entity, if any — tracked so [`load_pack`] can
/// despawn it (which cascades to its tiles via `AhnpPack`'s `on_remove`
/// hook) before opening a replacement.
#[derive(Resource, Default)]
struct CurrentPack(Option<Entity>);

/// Set whenever a load just happened; [`reframe_on_load`] clears it once the
/// new pack's entity is queryable and the camera has been refit to it. A
/// plain "run once per load" flag rather than an event, since at most one
/// load is ever in flight in this demo.
#[derive(Resource, Default)]
struct NeedsFraming(bool);

/// The active [`Framing`], recomputed by [`reframe_on_load`] every time a
/// pack (re)loads; [`orbit_camera`] sweeps the camera around it and applies the
/// user's `zoom`. `framing` is `None` before the first pack has ever loaded.
#[derive(Resource)]
struct OrbitState {
    framing: Option<Framing>,
    /// Standoff multiplier on the fit distance; 1.0 == the framed fit.
    zoom: f32,
}

impl Default for OrbitState {
    fn default() -> Self {
        Self {
            framing: None,
            zoom: 1.0,
        }
    }
}

/// Text field + status line state for the file-loader panel.
#[derive(Resource)]
struct DemoUi {
    path_input: String,
    status: String,
}

impl DemoUi {
    fn new(initial: Option<String>) -> Self {
        Self {
            path_input: initial.unwrap_or_default(),
            status: "no pack loaded yet — type a path above and click Load".to_string(),
        }
    }
}

/// Sun + ambient parameters the sliders drive; [`apply_lighting`] pushes
/// these onto the `DirectionalLight`/`GlobalAmbientLight` and every ortho
/// tile's material every frame.
#[derive(Resource)]
struct LightingSettings {
    /// Compass heading of the sun, degrees.
    azimuth_deg: f32,
    /// Angle above the horizon, degrees.
    elevation_deg: f32,
    /// `DirectionalLight::illuminance`, lux.
    illuminance: f32,
    /// `GlobalAmbientLight::brightness`.
    ambient_brightness: f32,
    /// `false` switches every ortho tile's material to lit shading instead of
    /// the crate's default unlit (1:1 colour) look.
    unlit: bool,
}

impl Default for LightingSettings {
    fn default() -> Self {
        Self {
            azimuth_deg: 45.0,
            elevation_deg: 45.0,
            // A plausible overcast-day illuminance (see `DirectionalLight`'s
            // own doc comment for the full lux reference table) — not a
            // physically-derived default, just a reasonable starting point
            // for the "lit" toggle to look sensible immediately.
            illuminance: 10_000.0,
            ambient_brightness: 80.0,
            unlit: true,
        }
    }
}

/// Consumer-facing splat render knobs the UI drives (mirrors
/// `bevy_gaussian_splatting::CloudSettings`'s fields we expose sliders for).
#[cfg(feature = "splat")]
#[derive(Resource)]
struct SplatUi {
    global_scale: f32,
    global_opacity: f32,
}

#[cfg(feature = "splat")]
impl Default for SplatUi {
    fn default() -> Self {
        Self {
            global_scale: 1.0,
            global_opacity: 1.0,
        }
    }
}

/// Spawn the one persistent camera and the one directional light up front.
/// [`apply_lighting`] drives both from [`LightingSettings`] every frame, and
/// [`orbit_camera`] drives the camera's `Transform` once a pack has loaded.
fn spawn_camera_and_light(mut commands: Commands) {
    commands.spawn((Camera3d::default(), Transform::default(), Tonemapping::None));
    commands.spawn(DirectionalLight::default());
    // `GlobalAmbientLight` is inserted by Bevy's own `LightPlugin` (part of
    // `DefaultPlugins`); `apply_lighting` mutates that existing resource
    // rather than inserting a second one.
}

/// ---- STEP 2 + 3 (see the module doc comment): open, spawn, and mark for
/// framing ----------------------------------------------------------------
///
/// Despawns whatever pack is currently loaded first — `AhnpPack`'s
/// `on_remove` hook despawns its tile entities too, so this never leaks the
/// previous load's tiles — then opens `path` and spawns it if it parses.
/// Never panics on a bad path: reports the error as the returned status
/// string instead.
fn load_pack(
    commands: &mut Commands,
    path: &str,
    current_pack: &mut CurrentPack,
    needs_framing: &mut NeedsFraming,
) -> String {
    if let Some(old) = current_pack.0.take() {
        commands.entity(old).despawn();
    }
    match AhnpPack::open(path) {
        Ok(pack) => {
            let entity = commands.spawn(pack).id();
            current_pack.0 = Some(entity);
            needs_framing.0 = true;
            format!("loaded {path}")
        }
        Err(e) => format!("failed to open {path}: {e}"),
    }
}

/// Loads the first CLI-provided path, if any, at startup.
fn load_initial_pack(
    mut commands: Commands,
    candidates: Res<Candidates>,
    mut current_pack: ResMut<CurrentPack>,
    mut needs_framing: ResMut<NeedsFraming>,
    mut ui: ResMut<DemoUi>,
) {
    let Some(path) = candidates.0.first() else {
        return;
    };
    ui.status = load_pack(&mut commands, path, &mut current_pack, &mut needs_framing);
}

/// Runs every frame but only does anything right after [`load_pack`] set
/// [`NeedsFraming`] — it waits for the newly-spawned `AhnpPack` entity to
/// become queryable (spawn commands apply at the next sync point, not
/// immediately), then fits the camera to its world AABB exactly once.
fn reframe_on_load(
    packs: Query<&AhnpPack>,
    mut needs_framing: ResMut<NeedsFraming>,
    mut orbit: ResMut<OrbitState>,
) {
    if !needs_framing.0 {
        return;
    }
    let Some(pack) = packs.iter().next() else {
        return;
    };
    orbit.framing = Some(Framing::fit_default(pack.world_aabb()));
    needs_framing.0 = false;
}

/// Sweeps the camera around the active [`OrbitState`]'s framing and folds in
/// keyboard / mouse-wheel zoom (shared [`zoom_delta`]/[`apply_zoom`] mapping) —
/// a no-op until the first pack has loaded.
fn orbit_camera(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    mut wheel: MessageReader<MouseWheel>,
    mut orbit: ResMut<OrbitState>,
    mut q: Query<&mut Transform, With<Camera3d>>,
) {
    let delta = zoom_delta(&keys, &mut wheel, time.delta_secs());
    if delta != 0.0 {
        orbit.zoom = apply_zoom(orbit.zoom, delta);
    }
    let Some(framing) = orbit.framing else {
        return;
    };
    let azimuth = time.elapsed_secs() * 0.1;
    for mut t in &mut q {
        *t = framing.orbit_transform_zoom(azimuth, ORBIT_ELEVATION, orbit.zoom);
    }
}

/// Applies [`LightingSettings`] to the sun's transform/illuminance, the
/// global ambient brightness, and every ortho tile's `unlit` flag. Runs every
/// frame (egui's immediate-mode sliders touch the resource whether or not the
/// user actually dragged anything, so gating on Bevy change-detection
/// wouldn't skip much) but only *writes* a tile's material when its `unlit`
/// value actually differs, so a steady UI doesn't churn `Assets`' change
/// tracking or force needless GPU re-uploads.
fn apply_lighting(
    lighting: Res<LightingSettings>,
    mut sun: Query<(&mut DirectionalLight, &mut Transform)>,
    mut ambient: ResMut<GlobalAmbientLight>,
    tiles: Query<&MeshMaterial3d<StandardMaterial>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    for (mut light, mut transform) in &mut sun {
        light.illuminance = lighting.illuminance;
        // The light shines along its transform's local -Z; azimuth rotates
        // that around Y (compass heading), elevation pitches it around X
        // (angle above the horizon). Not geodetically meaningful — just a
        // slider-friendly parameterization for a demo sun.
        *transform = Transform::from_rotation(Quat::from_euler(
            EulerRot::YXZ,
            lighting.azimuth_deg.to_radians(),
            -lighting.elevation_deg.to_radians(),
            0.0,
        ));
    }
    ambient.brightness = lighting.ambient_brightness;

    for handle in &tiles {
        let changed = materials
            .get(&handle.0)
            .is_some_and(|m| m.unlit != lighting.unlit);
        if changed && let Some(m) = materials.get_mut(&handle.0) {
            m.unlit = lighting.unlit;
        }
    }
}

/// Pushes [`SplatUi`] onto `splat::SplatSettings` (styles *future* tiles) and
/// onto every already-spawned tile's own `CloudSettings` component (styles
/// tiles that already loaded) — see `SplatSettings`'s own doc comment on why
/// both are needed for a slider to visibly restyle a scene that's already
/// streamed in.
#[cfg(feature = "splat")]
fn apply_splat_settings(
    splat_ui: Res<SplatUi>,
    mut settings: ResMut<SplatSettings>,
    mut clouds: Query<&mut CloudSettings>,
) {
    settings.0.global_scale = splat_ui.global_scale;
    settings.0.global_opacity = splat_ui.global_opacity;
    for mut cloud in &mut clouds {
        if cloud.global_scale != splat_ui.global_scale {
            cloud.global_scale = splat_ui.global_scale;
        }
        if cloud.global_opacity != splat_ui.global_opacity {
            cloud.global_opacity = splat_ui.global_opacity;
        }
    }
}

/// The whole UI: an FPS readout, the file loader, and every parameter slider.
#[allow(clippy::too_many_arguments)]
fn ui_system(
    mut commands: Commands,
    mut contexts: EguiContexts,
    mut ui_state: ResMut<DemoUi>,
    candidates: Res<Candidates>,
    mut current_pack: ResMut<CurrentPack>,
    mut needs_framing: ResMut<NeedsFraming>,
    diagnostics: Res<DiagnosticsStore>,
    mut lighting: ResMut<LightingSettings>,
    mut sse: ResMut<SseThreshold>,
    #[cfg(feature = "splat")] mut splat_ui: ResMut<SplatUi>,
) -> Result {
    let fps = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FPS)
        .and_then(bevy::diagnostic::Diagnostic::smoothed)
        .unwrap_or(0.0);

    let mut load_requested: Option<String> = None;
    let ctx = contexts.ctx_mut()?;
    egui::Window::new("bevy_ahnp_ortho demo").show(ctx, |ui| {
        ui.heading(format!("{fps:.1} FPS"));
        ui.separator();

        ui.label("Load a pack (heightfield, game, or splat — same file loader for all three):");
        ui.horizontal(|ui| {
            ui.text_edit_singleline(&mut ui_state.path_input);
            if ui.button("Load").clicked() {
                load_requested = Some(ui_state.path_input.clone());
            }
        });
        if !candidates.0.is_empty() {
            ui.label("From the command line:");
            for path in &candidates.0 {
                if ui.button(path).clicked() {
                    load_requested = Some(path.clone());
                }
            }
        }
        ui.label(ui_state.status.as_str());

        ui.separator();
        ui.label("Lighting");
        ui.add(egui::Slider::new(&mut lighting.azimuth_deg, 0.0..=360.0).text("Sun azimuth (deg)"));
        ui.add(
            egui::Slider::new(&mut lighting.elevation_deg, 5.0..=90.0).text("Sun elevation (deg)"),
        );
        ui.add(
            egui::Slider::new(&mut lighting.illuminance, 0.0..=20_000.0)
                .text("Sun illuminance (lux)"),
        );
        ui.add(
            egui::Slider::new(&mut lighting.ambient_brightness, 0.0..=1000.0)
                .text("Ambient brightness"),
        );
        ui.checkbox(
            &mut lighting.unlit,
            "Unlit ortho colours (uncheck to see lit shading)",
        );

        ui.separator();
        ui.label("Level of detail");
        ui.add(egui::Slider::new(&mut sse.0, 1.0..=64.0).text("Screen-space-error threshold (px)"));

        #[cfg(feature = "splat")]
        {
            ui.separator();
            ui.label("Splat rendering (content_kind = 2)");
            ui.add(egui::Slider::new(&mut splat_ui.global_scale, 0.1..=5.0).text("Global scale"));
            ui.add(
                egui::Slider::new(&mut splat_ui.global_opacity, 0.0..=1.0).text("Global opacity"),
            );
        }

        ui.separator();
        ui.label(format!(
            "gpu_textures (BC1 transcode at load): {}",
            if cfg!(feature = "gpu_textures") {
                "compiled in"
            } else {
                "off — plain RGBA8 textures (recompile with --features gpu_textures)"
            }
        ));
    });

    if let Some(path) = load_requested {
        ui_state.status = load_pack(&mut commands, &path, &mut current_pack, &mut needs_framing);
    }
    Ok(())
}
