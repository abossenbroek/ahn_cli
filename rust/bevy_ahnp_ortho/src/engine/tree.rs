//! Tile-tree model + the per-frame selection algorithm.
//!
//! Ported from github.com/Arvikasoft/bevy_3d_tiles (dual MIT/Apache-2.0),
//! `src/traversal.rs`: the pure selection algorithm (`select`, `visit`,
//! `screen_space_error`, `Priority`, `Selection`, `History`) is kept
//! near-verbatim (only the now-impossible "contentless interior" branch is
//! dropped — see below). `TileTree::build`/`graft`, which parsed a
//! `tileset.json` schema and composed per-tile transforms, are REPLACED by
//! [`TileTree::build_from_entries`], which builds directly from an AHNP
//! pack's flat [`ahn_heightfield::Entry`] list: children are found by
//! quadtree doubling — `(level+1, 2tx+dx, 2ty+dy)` for `dx, dy in {0, 1}` —
//! not by walking a parsed JSON tree. This severs the `schema` dependency
//! entirely: `select()`/SSE never needed it.
//!
//! Two simplifications an AHNP pack makes safe, vs. the general 3D Tiles
//! model this was ported from:
//! - **every tile has content** (each level is its own genuine downsampled
//!   render, never a contentless grouping interior) — so the "contentless
//!   tile always refines" branch in `visit()` is dropped; a tile without
//!   children is a genuine leaf.
//! - **every tile's region is EPSG:4979** (absolute, not composed under a
//!   per-tile transform) and **refine is always REPLACE** (the producer never
//!   emits ADD) — so `TileNode` carries no transform matrices, only a
//!   region-derived [`WorldVolume`]. The `Refine` enum is kept (or a splat/
//!   points overlay embedder might one day graft an ADD-refined tree) but
//!   [`TileTree::build_from_entries`] always assigns `Refine::Replace`.
//!
//! `TileNode` is keyed by [`ahn_heightfield::TileKey`] rather than embedding
//! the (`#[non_exhaustive]`, not constructible outside its crate)
//! [`ahn_heightfield::Entry`] directly — a loader maps a selected node back to
//! its `Entry` via `Archive::find(node.key)` when it actually needs the blob
//! offsets. This also keeps the tree model testable with synthetic fixtures,
//! with no archive/pack file needed.

use std::collections::HashMap;

use ahn_heightfield::{Entry, TileKey};
use bevy::math::DVec3;

use super::geo::region_to_ecef_volume;

/// Refine when a tile's screen-space error exceeds this many pixels.
/// CesiumJS's default `maximumScreenSpaceError`.
pub const DEFAULT_SSE_THRESHOLD_PX: f64 = 16.0;

/// Distances under this count as "camera inside the volume" → infinite SSE.
const INSIDE_EPS: f64 = 1e-9;

/// A bounding volume in **Bevy world space** (f64 — ECEF magnitudes are
/// planetary before the per-anchor world transform / f32 conversion).
#[derive(Debug, Clone, Copy)]
pub enum WorldVolume {
    Sphere {
        center: DVec3,
        radius: f64,
    },
    /// Center + three half-axis vectors (orientation × half-extent).
    Obb {
        center: DVec3,
        half_axes: [DVec3; 3],
    },
}

impl WorldVolume {
    /// Distance from `p` to the volume surface; `0.0` when inside.
    pub fn distance_to(&self, p: DVec3) -> f64 {
        match self {
            WorldVolume::Sphere { center, radius } => (p - *center).length() - radius,
            WorldVolume::Obb { center, half_axes } => {
                let d = p - *center;
                let mut closest = *center;
                for axis in half_axes {
                    let len = axis.length();
                    if len < 1e-12 {
                        continue;
                    }
                    let unit = *axis / len;
                    closest += unit * d.dot(unit).clamp(-len, len);
                }
                (p - closest).length()
            }
        }
        .max(0.0)
    }

    /// Enclosing sphere (for frustum culling).
    pub fn bounding_sphere(&self) -> (DVec3, f64) {
        match self {
            WorldVolume::Sphere { center, radius } => (*center, *radius),
            WorldVolume::Obb { center, half_axes } => {
                let r2: f64 = half_axes.iter().map(|a| a.length_squared()).sum();
                (*center, r2.sqrt())
            }
        }
    }
}

/// 3D Tiles refinement strategy. An AHNP pack always uses [`Refine::Replace`]
/// (every tile has genuine content, so a refined tile's children fully
/// replace it); kept as an enum for parity with the ported selection logic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refine {
    Add,
    Replace,
}

/// One flattened tile. Index 0 is the root.
#[derive(Debug, Clone)]
pub struct TileNode {
    pub parent: Option<usize>,
    pub children: Vec<usize>,
    /// Tree depth (root = 0); equal to the AHNP pack's quadtree `level`.
    pub depth: u32,
    pub geometric_error: f64,
    pub refine: Refine,
    /// This tile's pack key — look it up via `Archive::find` to get the
    /// `Entry` (blob offsets/sizes) when its content is actually decoded.
    pub key: TileKey,
    /// Bounding volume in ECEF (planetary-magnitude f64; a renderer applies
    /// its own ECEF -> world anchor transform at content-spawn time).
    pub volume: WorldVolume,
}

/// Flattened tileset tree, ready for per-frame traversal.
#[derive(Debug, Clone, Default)]
pub struct TileTree {
    pub nodes: Vec<TileNode>,
}

impl TileTree {
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Build the flattened tree from an AHNP pack's entry list (e.g.
    /// `Archive::entries()`). Children of `(level, tx, ty)` are whichever of
    /// `(level+1, 2tx, 2ty)`, `(2tx+1, 2ty)`, `(2tx, 2ty+1)`, `(2tx+1, 2ty+1)`
    /// are present in `entries` — the AHNP format's implicit quadtree, no
    /// parsed schema needed.
    pub fn build_from_entries(entries: &[Entry]) -> Result<TileTree, String> {
        let mut by_key: HashMap<(u32, u32, u32), usize> = HashMap::with_capacity(entries.len());
        for (i, e) in entries.iter().enumerate() {
            by_key.insert((e.level, e.tx, e.ty), i);
        }
        let root_src = *by_key
            .get(&(0, 0, 0))
            .ok_or("AHNP entries have no root tile (level 0, tx 0, ty 0)")?;
        let mut tree = TileTree::default();
        build_node(&mut tree, entries, &by_key, root_src, None, 0)?;
        Ok(tree)
    }
}

fn build_node(
    tree: &mut TileTree,
    entries: &[Entry],
    by_key: &HashMap<(u32, u32, u32), usize>,
    src: usize,
    parent: Option<usize>,
    depth: u32,
) -> Result<usize, String> {
    let entry = entries[src];
    let idx = tree.nodes.len();
    tree.nodes.push(TileNode {
        parent,
        children: Vec::new(),
        depth,
        geometric_error: entry.geometric_error,
        refine: Refine::Replace,
        key: TileKey {
            level: entry.level,
            tx: entry.tx,
            ty: entry.ty,
            tz: entry.tz,
        },
        volume: region_to_ecef_volume(&entry.region),
    });

    let mut children = Vec::with_capacity(4);
    for (dx, dy) in [(0u32, 0u32), (1, 0), (0, 1), (1, 1)] {
        let child_key = (entry.level + 1, entry.tx * 2 + dx, entry.ty * 2 + dy);
        if let Some(&child_src) = by_key.get(&child_key) {
            children.push(build_node(
                tree,
                entries,
                by_key,
                child_src,
                Some(idx),
                depth + 1,
            )?);
        }
    }
    tree.nodes[idx].children = children;
    Ok(idx)
}

// ── Per-frame selection ──────────────────────────────────────────────────────

/// Camera-derived parameters for one selection pass.
#[derive(Debug, Clone, Copy)]
pub struct SelectParams {
    pub cam_pos: DVec3,
    /// Unit view direction (screen-center load-priority weighting).
    pub cam_forward: DVec3,
    /// Pinhole focal length in pixels: `viewport_h / (2·tan(fovy/2))`.
    pub k_px: f64,
    /// Refine while `sse > threshold` (px).
    pub sse_threshold_px: f64,
    /// Distance-relaxed detail falloff (metres); `0` disables.
    pub detail_falloff_m: f64,
    /// Camera height above the planet surface (metres); `0` for non-globe use.
    pub cam_height_m: f64,
}

/// Screen-space error of a tile with geometric error `ge` at `dist` metres.
/// `f64::INFINITY` when the camera is inside the volume (`dist ≈ 0`).
pub fn screen_space_error(ge: f64, dist: f64, k_px: f64) -> f64 {
    if dist <= INSIDE_EPS {
        f64::INFINITY
    } else {
        ge * k_px / dist
    }
}

/// Load-priority tiers, highest first. `Ord`: `Urgent < Descend < Normal <
/// Preload` so an ascending sort puts the most important requests first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    Urgent,
    Descend,
    Normal,
    Preload,
}

#[derive(Debug, Clone, Copy)]
pub struct LoadRequest {
    pub tile: usize,
    pub priority: Priority,
    /// Tie-break within a tier: distance weighted toward screen center
    /// (smaller = sooner).
    pub key: f64,
}

/// Result of one selection pass.
#[derive(Debug, Default)]
pub struct Selection {
    pub render: Vec<usize>,
    pub loads: Vec<LoadRequest>,
    pub refined: Vec<bool>,
    pub touched: Vec<bool>,
    pub covered: bool,
}

/// Frame history needed by zoom-out protection + kicking.
#[derive(Debug, Clone, Default)]
pub struct History {
    pub rendered: Vec<bool>,
    pub refined: Vec<bool>,
}

impl History {
    pub fn resize(&mut self, n: usize) {
        self.rendered.resize(n, false);
        self.refined.resize(n, false);
    }

    pub fn absorb(&mut self, sel: &Selection, n: usize) {
        self.rendered.clear();
        self.rendered.resize(n, false);
        for &t in &sel.render {
            self.rendered[t] = true;
        }
        self.refined = sel.refined.clone();
    }
}

/// Per-tile content readiness, as the traversal sees it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TileContent {
    /// Not yet requested.
    Pending,
    /// A decode task is in flight (an async renderer's equivalent of
    /// `Pending`, but already requested — `loadable()` is `false` so
    /// `select()` never asks for it twice while it's running).
    Loading,
    Ready,
    Failed,
}

impl TileContent {
    fn settled(self) -> bool {
        !matches!(self, TileContent::Pending | TileContent::Loading)
    }

    fn loadable(self) -> bool {
        matches!(self, TileContent::Pending)
    }
}

struct Ctx<'a, F: Fn(usize) -> bool> {
    tree: &'a TileTree,
    content: &'a [TileContent],
    history: &'a History,
    culled: &'a F,
    params: SelectParams,
}

struct VisitOut {
    covered: bool,
    any_rendered_last: bool,
}

/// Run one selection pass. `content[i]` mirrors `tree.nodes[i]`;
/// `culled(i)` = tile `i`'s bounding sphere is outside the frustum.
pub fn select<F: Fn(usize) -> bool>(
    tree: &TileTree,
    content: &[TileContent],
    history: &History,
    culled: &F,
    params: SelectParams,
) -> Selection {
    let n = tree.nodes.len();
    debug_assert_eq!(content.len(), n);
    let mut sel = Selection {
        render: Vec::new(),
        loads: Vec::new(),
        refined: vec![false; n],
        touched: vec![false; n],
        covered: false,
    };
    if n == 0 {
        return sel;
    }
    let ctx = Ctx {
        tree,
        content,
        history,
        culled,
        params,
    };
    sel.covered = visit(&ctx, 0, &mut sel).covered;

    let mut queued = vec![false; n];
    for req in &sel.loads {
        queued[req.tile] = true;
    }
    for i in 0..sel.render.len() {
        let mut at = tree.nodes[sel.render[i]].parent;
        while let Some(p) = at {
            sel.touched[p] = true;
            if ctx.content[p].loadable() && !queued[p] {
                queued[p] = true;
                let dist = tree.nodes[p].volume.distance_to(params.cam_pos);
                sel.loads.push(LoadRequest {
                    tile: p,
                    priority: Priority::Preload,
                    key: load_key(&ctx, p, dist),
                });
            }
            at = tree.nodes[p].parent;
        }
    }

    sel.loads.sort_by(|a, b| {
        (a.priority, a.key)
            .partial_cmp(&(b.priority, b.key))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    sel
}

fn load_key<F: Fn(usize) -> bool>(ctx: &Ctx<'_, F>, tile: usize, dist: f64) -> f64 {
    let (center, _) = ctx.tree.nodes[tile].volume.bounding_sphere();
    let to = center - ctx.params.cam_pos;
    let cos = if to.length_squared() > 1e-12 {
        to.normalize().dot(ctx.params.cam_forward).clamp(-1.0, 1.0)
    } else {
        1.0
    };
    dist * (2.0 - cos)
}

fn push_load<F: Fn(usize) -> bool>(ctx: &Ctx<'_, F>, sel: &mut Selection, tile: usize, dist: f64) {
    if ctx.content[tile].loadable() {
        let priority = if dist <= INSIDE_EPS {
            Priority::Urgent
        } else {
            Priority::Normal
        };
        sel.loads.push(LoadRequest {
            tile,
            priority,
            key: load_key(ctx, tile, dist),
        });
    }
}

fn visit<F: Fn(usize) -> bool>(ctx: &Ctx<'_, F>, i: usize, sel: &mut Selection) -> VisitOut {
    let node = &ctx.tree.nodes[i];
    sel.touched[i] = true;

    let dist = node
        .volume
        .distance_to(ctx.params.cam_pos)
        .max(ctx.params.cam_height_m);
    let sse = screen_space_error(node.geometric_error, dist, ctx.params.k_px);

    let threshold = if ctx.params.detail_falloff_m > 0.0 {
        let extra = (dist - ctx.params.cam_height_m).max(0.0);
        ctx.params.sse_threshold_px * (1.0 + extra / ctx.params.detail_falloff_m)
    } else {
        ctx.params.sse_threshold_px
    };

    // Every AHNP tile carries genuine content, so — unlike the general 3D
    // Tiles model this was ported from — a childless tile never needs a
    // "contentless interior always refines" escape hatch; only the SSE test
    // decides.
    let mut wants_refine = !node.children.is_empty() && sse > threshold;
    if !wants_refine
        && !node.children.is_empty()
        && ctx.history.refined[i]
        && !ctx.content[i].settled()
    {
        wants_refine = true;
        push_load(ctx, sel, i, dist);
    }

    if !wants_refine {
        push_load(ctx, sel, i, dist);
        if node.children.is_empty()
            && sse > threshold
            && let Some(req) = sel.loads.last_mut()
            && req.tile == i
            && req.priority == Priority::Normal
        {
            req.priority = Priority::Descend;
        }
        if ctx.content[i] == TileContent::Ready {
            sel.render.push(i);
        }
        return VisitOut {
            covered: ctx.content[i].settled(),
            any_rendered_last: ctx.history.rendered[i],
        };
    }

    sel.refined[i] = true;

    if node.refine == Refine::Add {
        push_load(ctx, sel, i, dist);
        if ctx.content[i] == TileContent::Ready {
            sel.render.push(i);
        }
        let mut any_last = ctx.history.rendered[i];
        for &c in &node.children {
            if (ctx.culled)(c) {
                continue;
            }
            any_last |= visit(ctx, c, sel).any_rendered_last;
        }
        return VisitOut {
            covered: ctx.content[i].settled(),
            any_rendered_last: any_last,
        };
    }

    let checkpoint = sel.render.len();
    let mut all_covered = true;
    let mut any_last = false;
    for &c in &node.children {
        if (ctx.culled)(c) {
            continue;
        }
        let v = visit(ctx, c, sel);
        all_covered &= v.covered;
        any_last |= v.any_rendered_last;
    }
    if all_covered {
        return VisitOut {
            covered: true,
            any_rendered_last: any_last,
        };
    }
    if any_last {
        push_load(ctx, sel, i, dist);
        return VisitOut {
            covered: true,
            any_rendered_last: true,
        };
    }
    if ctx.content[i] == TileContent::Ready {
        sel.render.truncate(checkpoint);
        sel.render.push(i);
        return VisitOut {
            covered: true,
            any_rendered_last: ctx.history.rendered[i],
        };
    }
    sel.render.truncate(checkpoint);
    if ctx.content[i].loadable() {
        sel.loads.push(LoadRequest {
            tile: i,
            priority: Priority::Urgent,
            key: load_key(ctx, i, dist),
        });
    }
    VisitOut {
        covered: false,
        any_rendered_last: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn k_1080() -> f64 {
        1080.0 / (2.0 * (45f64.to_radians() / 2.0).tan())
    }

    fn params(cam: DVec3) -> SelectParams {
        SelectParams {
            cam_pos: cam,
            cam_forward: (DVec3::ZERO - cam).normalize_or(DVec3::NEG_Z),
            k_px: k_1080(),
            sse_threshold_px: DEFAULT_SSE_THRESHOLD_PX,
            detail_falloff_m: 0.0,
            cam_height_m: 0.0,
        }
    }

    fn sphere(center: DVec3, radius: f64) -> WorldVolume {
        WorldVolume::Sphere { center, radius }
    }

    fn key(level: u32, tx: u32, ty: u32) -> TileKey {
        TileKey {
            level,
            tx,
            ty,
            tz: 0,
        }
    }

    /// root(ge 16, r 30) → 4 children (ge 4, r 9) → 4 leaves each (ge 0, r 4).
    /// Leaves 5–20 (children of node c start at 1+4+(c-1)*4).
    fn fixture_tree() -> TileTree {
        let mut tree = TileTree::default();
        tree.nodes.push(TileNode {
            parent: None,
            children: vec![],
            depth: 0,
            geometric_error: 16.0,
            refine: Refine::Replace,
            key: key(0, 0, 0),
            volume: sphere(DVec3::ZERO, 30.0),
        });
        let quad = [(-10.0, -10.0), (10.0, -10.0), (-10.0, 10.0), (10.0, 10.0)];
        for (ci, (cx, cz)) in quad.iter().enumerate() {
            let c = tree.nodes.len();
            tree.nodes.push(TileNode {
                parent: Some(0),
                children: vec![],
                depth: 1,
                geometric_error: 4.0,
                refine: Refine::Replace,
                key: key(1, ci as u32, 0),
                volume: sphere(DVec3::new(*cx, 0.0, *cz), 9.0),
            });
            tree.nodes[0].children.push(c);
        }
        for c in 1..=4 {
            let (cx, cz) = quad[c - 1];
            for (li, (lx, lz)) in quad.iter().enumerate() {
                let l = tree.nodes.len();
                tree.nodes.push(TileNode {
                    parent: Some(c),
                    children: vec![],
                    depth: 2,
                    geometric_error: 0.0,
                    refine: Refine::Replace,
                    key: key(2, ((c - 1) * 4 + li) as u32, 0),
                    volume: sphere(DVec3::new(cx + lx * 0.25, 0.0, cz + lz * 0.25), 4.0),
                });
                let cc = tree.nodes[c].children.clone();
                tree.nodes[c].children = [cc, vec![l]].concat();
            }
        }
        tree
    }

    fn all(content: TileContent, n: usize) -> Vec<TileContent> {
        vec![content; n]
    }

    fn no_cull(_: usize) -> bool {
        false
    }

    #[test]
    fn sse_math() {
        let k = k_1080();
        let sse = screen_space_error(16.0, 1000.0, k);
        assert!((sse - 16.0 * k / 1000.0).abs() < 1e-9);
        assert_eq!(screen_space_error(16.0, 0.0, k), f64::INFINITY);
        assert_eq!(screen_space_error(0.0, 100.0, k), 0.0);
    }

    #[test]
    fn far_camera_renders_root_only() {
        let tree = fixture_tree();
        let content = all(TileContent::Ready, tree.len());
        let history = History {
            rendered: vec![false; tree.len()],
            refined: vec![false; tree.len()],
        };
        let sel = select(
            &tree,
            &content,
            &history,
            &no_cull,
            params(DVec3::new(0.0, 0.0, 3000.0)),
        );
        assert_eq!(sel.render, vec![0]);
        assert!(sel.loads.is_empty());
    }

    #[test]
    fn near_camera_selects_leaves() {
        let tree = fixture_tree();
        let content = all(TileContent::Ready, tree.len());
        let history = History {
            rendered: vec![false; tree.len()],
            refined: vec![false; tree.len()],
        };
        let sel = select(
            &tree,
            &content,
            &history,
            &no_cull,
            params(DVec3::new(0.0, 5.0, 0.0)),
        );
        assert_eq!(sel.render.len(), 16);
        assert!(
            sel.render.iter().all(|&t| t >= 5),
            "only leaves: {:?}",
            sel.render
        );
        assert!(sel.refined[0]);
        assert!((1..=4).all(|c| sel.refined[c]));
    }

    /// The plan's #1 risk mitigation: LOD by screen-space error, headless and
    /// GPU-free — dollying in must refine (more, smaller tiles selected),
    /// dollying back out must coarsen (fewer, larger tiles selected).
    #[test]
    fn select_refines_on_dolly_in_and_coarsens_on_dolly_out() {
        let tree = fixture_tree();
        let content = all(TileContent::Ready, tree.len());
        let history = History {
            rendered: vec![false; tree.len()],
            refined: vec![false; tree.len()],
        };

        let far = select(
            &tree,
            &content,
            &history,
            &no_cull,
            params(DVec3::new(0.0, 0.0, 3000.0)),
        );
        assert_eq!(far.render, vec![0], "far camera stays at the coarsest LOD");

        let near = select(
            &tree,
            &content,
            &history,
            &no_cull,
            params(DVec3::new(0.0, 5.0, 0.0)),
        );
        assert_eq!(near.render.len(), 16, "near camera refines to every leaf");
        assert!(near.render.iter().all(|&t| t >= 5));

        // Dolly back out from near: the selection coarsens back to the root.
        let mut h2 = History::default();
        h2.absorb(&near, tree.len());
        let out_again = select(
            &tree,
            &content,
            &h2,
            &no_cull,
            params(DVec3::new(0.0, 0.0, 3000.0)),
        );
        assert_eq!(out_again.render, vec![0], "dolly-out coarsens back to root");
    }

    #[test]
    fn unloaded_leaves_kick_to_ready_parent() {
        let tree = fixture_tree();
        let mut content = all(TileContent::Ready, tree.len());
        for slot in content.iter_mut().skip(5) {
            *slot = TileContent::Pending;
        }
        let history = History {
            rendered: vec![false; tree.len()],
            refined: vec![false; tree.len()],
        };
        let sel = select(
            &tree,
            &content,
            &history,
            &no_cull,
            params(DVec3::new(0.0, 5.0, 0.0)),
        );
        assert_eq!(sel.render, vec![1, 2, 3, 4]);
        let leaf_loads: Vec<usize> = sel
            .loads
            .iter()
            .filter(|r| r.tile >= 5)
            .map(|r| r.tile)
            .collect();
        assert_eq!(leaf_loads.len(), 16);
    }

    #[test]
    fn camera_inside_volume_is_urgent() {
        let tree = fixture_tree();
        let mut content = all(TileContent::Ready, tree.len());
        let leaf = 5;
        content[leaf] = TileContent::Pending;
        let history = History {
            rendered: vec![true; tree.len()],
            refined: vec![false; tree.len()],
        };
        let cam = DVec3::new(-10.0, 0.0, -10.0);
        let sel = select(&tree, &content, &history, &no_cull, params(cam));
        let req = sel
            .loads
            .iter()
            .find(|r| r.tile == leaf)
            .expect("leaf queued");
        assert_eq!(req.priority, Priority::Urgent);
        assert_eq!(sel.loads.first().unwrap().priority, Priority::Urgent);
    }

    #[test]
    fn build_from_entries_wires_quadtree_children() {
        // A 2-level pack: one root, its 4 direct quadtree children — built
        // from a flat `Entry`-shaped list, not schema JSON.
        let entries = synth_entries();
        let tree = TileTree::build_from_entries(&entries).expect("build");
        assert_eq!(tree.len(), 5);
        assert_eq!(tree.nodes[0].key, key(0, 0, 0));
        assert_eq!(tree.nodes[0].children.len(), 4);
        for &c in &tree.nodes[0].children {
            assert_eq!(tree.nodes[c].parent, Some(0));
            assert!(tree.nodes[c].children.is_empty());
        }
    }

    /// `Entry` is `#[non_exhaustive]` with no public constructor, so a
    /// synthetic `Entry` list isn't buildable directly; instead this reuses
    /// `ahn-heightfield`'s own committed golden fixture (root + 4 children,
    /// `tile_count == 5`, `level_count == 2` — see
    /// `rust/ahn-heightfield/tests/archive_golden.rs`) via `Archive::open`,
    /// mirroring how that crate's own doctests open the same file.
    fn synth_entries() -> Vec<Entry> {
        let bytes = include_bytes!("../../../ahn-heightfield/tests/data/tiles.hfp");
        let archive = ahn_heightfield::Archive::open(&bytes[..]).expect("open fixture pack");
        archive.entries().to_vec()
    }
}
