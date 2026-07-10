

# Changelog

## [Unreleased] — 7rad §10 Data Acquisition (in progress)

> Append-only pod work-log for the `TODO.md` epic. Newest entry at the **bottom
> of this section**; the last `STATE:` line is authoritative for cold resume.
> Corrections are new entries, never edits. Coordinator is the sole writer of
> this section. Existing release history is preserved unchanged below.

### 2026-07-09 — Pod kickoff
- Decisions: VIIRS = import a GEE-produced GeoTIFF path (no GEE coupling) ·
  old CLI = dropped entirely · GPU decimation = spike-first then confirm ·
  coverage = 100% on new modules only (legacy omitted) · cadence =
  PR-per-package / manual merge · PDAL windowed read = spike (full-sheet
  fallback) · engineers = Opus.
- Base: `main` @ 1d16f65 · fork `abossenbroek/ahn_cli` (CI repo-guarded → won't
  fire on this remote; manual merge is the gate).
- STATE: merged={} in-flight={WP0, SPIKE-GPU, SPIKE-PDAL} next={WP1 after WP0}
  pending-user={confirm GPU target after SPIKE-GPU}

### 2026-07-10 — SPIKE-PDAL result — full-sheet download, no windowed read
- Finding: PDOK native AHN LAZ is plain LAZ (no COPC/EPT, no spatial index) →
  HTTP-range windowed/bbox reads are NOT possible. `readers.copc`/`readers.ept`
  support bounds+range but don't apply; `readers.las` has no partial read. Full
  sheet download required; sheets ~200–500 MB (denser urban larger). DSM/WP7
  unaffected (COG windowed reads already feasible).
- Decision: WP6 keeps "download full tile, clip locally" (already the planned
  fallback). Optional backlog: local COPC-conversion cache for windowed
  re-reads — non-blocking. No user decision needed.
- STATE: merged={} in-flight={WP0, SPIKE-GPU} resolved-spikes={SPIKE-PDAL}
  next={WP1 after WP0} pending-user={confirm GPU target after SPIKE-GPU}

### 2026-07-10 — SPIKE-GPU result — MLX recommended (pending user confirm)
- Recommendation: MLX. Voxel = pure `mlx.core` (quantize→argsort→adjacent-diff
  mask→index; MLX lacks `unique()` but the sort-workaround sidesteps it).
  Poisson-disk = custom Metal kernel via `mx.fast.metal_kernel` (Python-authored
  MSL, no Xcode) — the discriminator over PyTorch-MPS (no first-class
  custom-kernel path; MPS `unique()` bug). pyobjc/Metal = fallback only.
  Installability confirmed (mlx 0.32, py3.10/3.12, arm64 M2 Max).
- Proposed target (NEEDS USER CONFIRM): voxel 50–100M pts/s on M2 Max; Poisson —
  no number asserted, measure in the WP11 probe. Keep CPU path for small tiles;
  pin MLX (pre-1.0).
- Equivalence test: voxel = exact retained-point-SET equality (deterministic via
  sort by `(voxel_key, point_id)`); Poisson = count-within-tolerance +
  min-distance hard constraint + NN-distance KS-test + bbox containment.
- STATE: merged={} in-flight={WP0} resolved-spikes={SPIKE-PDAL, SPIKE-GPU}
  next={WP1 after WP0} pending-user={CONFIRM GPU backend+targets for WP11}

### 2026-07-10 — GPU decimation confirmed by user — MLX
- User confirmed: MLX backend; voxel target 50–100M pts/s (M2 Max); Poisson perf
  measured in the WP11 probe (no pre-committed number). WP11 GPU gate cleared
  (still gated on WP2 merge).
- STATE: merged={} in-flight={WP0} resolved-spikes={SPIKE-PDAL, SPIKE-GPU}
  next={WP1 after WP0} pending-user={}

### 2026-07-10 — WP0 test + coverage infrastructure — READY #2 (awaiting user merge)
- Branch: wp0-test-coverage-infra  PR: #2
  https://github.com/abossenbroek/ahn_cli/pull/2  Commit: ed3a23c
- DoD: [x] pytest-cov/hypothesis/vcrpy added [x] 100% branch gate on new modules
  (legacy omitted) [x] gate-bites test [x] git-LFS .gitattributes
  [x] lint/format scope widened
- Files: pyproject.toml, makefile, .gitattributes, _typos.toml,
  tests/test_coverage_gate.py (+ ruff-format of 4 legacy modules & tests)
- Tests: 42 total (+3 new for the gate); 100% branch — `make check` exit 0, gate PASS
- Notes: git-lfs 3.7.1 present. Deviation: normalized PRE-EXISTING ruff-format
  drift in 4 legacy modules (fetcher/geotiles.py, manipulator/ptc_handler.py,
  manipulator/verifier.py, validator.py) — formatting-only, required because WP0
  widened format-check scope. Pyright diagnostics reviewed & benign
  (test_coverage_gate.py:69 = coverage.py `get_option` union type, runtime is a
  list; test_geojson_integration.py = pre-existing legacy type issues) — Pyright
  is not in the `make check` gate.
- STATE: merged={} ready-to-merge={WP0 #2} resolved-spikes={SPIKE-PDAL, SPIKE-GPU}
  next={WP1 unblocks on WP0 merge} pending-user={merge PR #2}

### 2026-07-10 — WP0 extended — GHA CI on fork + Pyright gate — READY #2 (supersedes ed3a23c)
- Branch: wp0-test-coverage-infra  PR: #2  Commit: 699d647
- Added: `ci.yml` repo-guard removed → GHA runs on `abossenbroek/ahn_cli` (push +
  all PRs, py3.10/3.11/3.12); `pyright` dev dep + `make typecheck` folded into
  `make check` and CI; `[tool.pyright]` gates `ahn_cli`+`tests` with the SAME
  legacy set grandfathered (+5 legacy test files), new code clean;
  `test_coverage_gate.py:69` cast fix.
- Gates: `make check` (lint, typos, typecheck, test, format-check) exit 0 —
  42 tests, 100% branch, pyright 0 errors.
- CI EVIDENCE: run 29066734032 — ci (3.10/3.11/3.12) all PASS on the fork
  (pre-removal run correctly `skipped`, confirming the guard was the blocker).
- STATE: merged={} ready-to-merge={WP0 #2 @699d647}
  resolved-spikes={SPIKE-PDAL, SPIKE-GPU} next={WP1 on WP0 merge}
  pending-user={merge PR #2}

### 2026-07-10 — WP0 strict hardening — ruff ALL + pyright strict + pre-commit — READY #2 (supersedes 699d647)
- Branch: wp0-test-coverage-infra  PR: #2  Commit: df99af1
- Added: ruff `select=["ALL"]` (curated formatter-conflict ignores + docstring
  pair resolution; D + ANN kept ON); pyright `typeCheckingMode="strict"`; ONE
  grandfather list synced across coverage-`omit` / ruff-`per-file-ignores` /
  pyright-`exclude` with the "de-grandfather ⇒ pass strict + 100% cov" rule
  commented at all three sites; new `.pre-commit-config.yaml` (repo-local hooks =
  make lint/format-check/typos/typecheck, identical to CI); `pre-commit` dev dep;
  `pre-commit install` documented (README + CLAUDE.md). Own new code fixed to
  pass strict (not by widening ignores).
- Gates: `make check` exit 0 — 42 tests, 100% branch, pyright strict 0 errors,
  ruff ALL clean, format clean; `pre-commit run --all-files` all pass.
- CI EVIDENCE: run 29067993366 — ci (3.10/3.11/3.12) all PASS on the fork.
- STATE: merged={} ready-to-merge={WP0 #2 @df99af1}
  resolved-spikes={SPIKE-PDAL, SPIKE-GPU} next={WP1 on WP0 merge, strict gates}
  pending-user={merge PR #2}

### 2026-07-10 — WP0 review round — mirror hooks + deprecation-gated grandfathering — READY #2 (@fc620f3)
- Branch: wp0-test-coverage-infra  PR: #2  Commit: fc620f3
- Review comments addressed + both threads replied in-thread & resolved:
  (1) pre-commit now uses upstream mirrors `astral-sh/ruff-pre-commit` v0.15.21
  (ruff-check + ruff-format) + `RobertCraigie/pyright-python` v1.1.411;
  `[tool.pyright] venvPath/venv` resolves the uv `.venv` so hook == CI
  (`pre-commit run --all-files` passes; hook proven to catch a real type error).
  (2) Grandfathering gated: `DEPRECATED; ANY LOGIC USED IN THIS CODE SHOULD BE
  MOVED` marker + module `DeprecationWarning` on all 16 legacy modules; rule
  documented at all three sync'd sites; legacy TEST files grandfathered as-is.
- Verify: `make check` exit 0 (ruff ALL clean, pyright strict 0, 100% branch);
  `uv run ahn_cli --help` OK; CI run 29070209482 — ci (3.10/3.11/3.12) all PASS.
  Transient codemod `mark_deprecated.py` never committed (absent from PR, deleted).
- Open note: `main.py` emits its DeprecationWarning once at CLI startup
  (developer-facing) — can be exempted on request. `request.py`/`geotiles.py`
  deprecation = relocate into the new fetch context (retained as `--source
  geotiles` fallback).
- STATE: merged={} ready-to-merge={WP0 #2 @fc620f3}
  resolved-spikes={SPIKE-PDAL, SPIKE-GPU} next={WP1 on WP0 merge, strict gates}
  pending-user={merge PR #2}

### 2026-07-10 — WP0 MERGED — foundation gates live on main
- PR #2 merged to `main` as `3324d50`; local main fast-forwarded. Pre-commit
  hooks installed & verified locally (`pre-commit run --all-files`: ruff check,
  ruff format, pyright, typos all Passed).
- WP0 DONE: 100% branch-cov gate + strict ruff ALL + strict pyright +
  deprecation-gated grandfathering + GHA-on-fork + pre-commit — all now enforced
  for every subsequent WP, at commit-time and in CI.
- STATE: merged={WP0} in-flight={} unblocked={WP1}
  resolved-spikes={SPIKE-PDAL, SPIKE-GPU}
  next={WP1 DDD domain model — awaiting user "go"} pending-user={say "go" → dispatch WP1}

### 2026-07-10 — Cadence change — auto-merge authorized (gated)
- User granted auto-merge authority: coordinator may squash-merge each PR WITHOUT
  manual approval, gated by (a) CI green on the fork (ruff + pyright + tests, all
  3 py versions) AND (b) TWO independent adversarial code reviews of the PR's
  LATEST commit both clearing (no blocking findings). New commits → re-review.
  Any block → bounce to the authoring engineer; no merge.
- merge_mode: manual → auto (gated). Explicit opt-in given with a 2-review safeguard.
- STATE: merged={WP0} in-flight={WP1} unblocks-after-WP1={WP2, WP3, WP4}
  resolved-spikes={SPIKE-PDAL, SPIKE-GPU} gate={2 adversarial reviews + green CI}
  pending-user={}

### 2026-07-10 — Overnight autonomous run authorized
- Directive: "spread out remaining work over teams of agents to work the night."
- Model: coordinator (this session) stays thin; engineers are Opus background
  worktree agents dispatched one wave at a time; gate + merge + fan-out driven
  by the coordinator on each engineer-completion notification.
- DAG is gated at the top: WP2/WP3/WP4 import WP1 domain value objects, so the
  full board (#3–#15) cannot branch until WP1 (#2) merges. Overnight throughput
  = maximum fan-out per wave, dispatched the instant each wave unblocks.
- Waves: WP1 (now) → {WP2,WP3,WP4} (3-wide) → {WP5..WP9} (≤5-wide) →
  {WP10,WP11} → {WP12,WP13} → WP14.
- Auto-merge gate per PR: 2 independent adversarial reviews of the PR's latest
  commit (each instructed to break it: correctness, determinism, DDD/TDD/100%-cov
  guardrail violations) + green CI + coverage gate. Any blocking finding →
  SendMessage bounce to the authoring engineer, no merge, re-review after fix.
- Recovery: state = TaskList (#1–#17) + this append-only log. On compaction,
  reconcile board vs `gh pr list`/`git log`, newest STATE line wins, resume.
- STATE: merged={WP0} in-flight={WP1} branch={wp1-domain-model, no PR yet}
  loop=notification-driven-cascade next={gate+merge WP1 → dispatch WP2,WP3,WP4}
  pending-user={}

### 2026-07-10 — WP1 PR #3 gate: BLOCK (bounced, not merged)
- PR: #3 (wp1-domain-model @ fbf020a). CI: green-pending on 3.10/3.11/3.12.
- Gate = 2 independent adversarial Opus reviews of the latest commit. Both
  BLOCK. Both independently re-ran make check (exit 0), ruff/pyright strict
  (clean), coverage (100% line+branch on domain/fetch/prep, confirmed NOT in
  any omit/ignore/exclude), DDD purity (zero legacy imports) — code quality
  confirmed; two specific holes stop the merge:
  1. [BLOCKING] ensure_valid_bbox (tile.py:~34) accepts non-finite coords
     (NaN/inf) — trichotomy guard is False for NaN, so Tile & Provenance
     silently accept a NaN/inf bbox. Guardrail #6 names NaN/inf → in-scope,
     not the WP3 content deferral. Fix: math.isfinite on all 4 coords.
  2. [BLOCKING per reviewer A] TDD red commit a7ee8b0 is collect-error-only
     (ImportError, no assertion runs) — strict criterion excludes that.
- Coordinator adjudication: fix #1 (hard blocker). For #2, tests are
  demonstrably non-vacuous (green = 100% branch over real assertions), so NOT
  forcing a history rewrite of a7ee8b0; instead the fix MUST land as an
  assertion-level red→green→refactor sequence, which settles A's rigor concern
  for the rest of the epic. Precedent-setting first PR: bar held high.
- Bounced to eng-domain (resumed from transcript) with the consolidated brief.
  Re-review the NEW head SHA (fresh 2-review gate) before any merge.
- STATE: merged={WP0} in-flight={WP1:bounced} branch={wp1-domain-model@fbf020a}
  gate-result=BLOCK next={eng-domain fixes bbox+TDD → re-run 2-review gate}
  pending-user={}

### 2026-07-10 — WP1 DDD domain model — MERGED #3
- Branch: wp1-domain-model  PR: #3  Squash: 5d821a1  Fixed-head: ea2385d
- DoD: [x] Product/Generation/Vintage/Tile/Provenance value objects
  [x] fetch/prep bounded-context skeletons [x] no stringly-typed switches
  [x] strict ruff ALL + pyright strict clean [x] 100% line+branch cov (0 BrPart)
  [x] DDD purity (zero legacy imports) [x] TDD red→green (assertion-level).
- Files: ahn_cli/domain/{__init__,product,generation,vintage,tile,provenance}.py,
  ahn_cli/fetch/__init__.py, ahn_cli/prep/__init__.py, tests/domain/*,
  tests/test_bounded_contexts.py. Tests: 37 added; coverage 100% branch (gate PASS).
- Gate: bounced once (bbox NaN/inf accept + collect-error-only red) → re-fixed
  via math.isfinite guard + Generation(bool) guard, delivered as assertion-level
  red (a96b332) → green (ea2385d). Re-review: 2 independent adversarial Opus
  reviews both PASS on ea2385d (each re-ran make check/ruff/pyright/coverage,
  verified fixes by repro, confirmed nothing smuggled into omit/ignore/exclude).
  CI green 3.10/3.11/3.12, mergeState CLEAN. Auto-merged (squash), branch deleted.
- Deferred to later WPs (accepted scope, both reviews concur): WP3 serializes
  Provenance + tz/content validation; WP5 attaches Generation registry; Vintage
  upper-bound/version-label and Tile product↔axis policy belong to fetch/ortho WPs.
- STATE: merged={WP0, WP1} in-flight={} unblocked-now={WP2(#3), WP3(#4), WP4(#5)}
  next={fan out WP2+WP3+WP4 as 3 parallel Opus worktree engineers}
  loop=notification-driven-cascade pending-user={}

### 2026-07-10 — Wave-2 fan-out + multi-agent interference incident
- Dispatched WP2/WP3/WP4 as 3 parallel Opus worktree engineers. Two first-spawns
  misfired (0 tool uses) and two later stalled mid-stream (transient API error);
  all recovered by re-dispatch or SendMessage-resume.
- INCIDENT: during stall/resume, the resumed WP2 & WP3 engineers briefly operated
  in the shared MAIN checkout instead of isolated worktrees — WP3's red commit
  layered onto a local `wp2-cli-restructure` branch, coordinator checkout got left
  on that bogus branch. Both engineers self-recovered by relocating to fresh
  isolated worktrees (wp2-iso; WP3 force-pushed only its 3 clean files). Coordinator
  fix: `git switch main` (working tree was clean), deleted the unpushed bogus local
  branch. No remote/PR contamination — each PR verified to contain only its own WP's
  files (WP2 gate uses a merge-base diff-scope check; WP3 gate an explicit scope check).
- WP2 design ruling logged: TODO.md authoritative → `fetch`=acquisition-only,
  classification filter moved to `prep`. `-e` collision resolved (fetch -o/-c/-b/-g,
  prep -d/-i/-e/-p). Reusable legacy plumbing (fetcher/*, manipulator/*, process.py)
  preserved; only [project.scripts] repointed to ahn_cli.cli:cli.

### 2026-07-10 — WP4 content-addressed cache — MERGED #4
- Branch: wp4-content-addressed-cache  PR: #4  Squash: 0764003  Head: 81b2da5
- DoD: [x] cache keyed by (product, vintage|generation, tile-id) from domain VOs
  [x] deterministic unsalted sha256 key (NUL-delimited, collision-safe)
  [x] checksum verify → typed ChecksumMismatchError on tamper
  [x] idempotent: cache hit = zero network + zero byte writes
  [x] 100% line+branch (key.py 34/8, store.py 41/6) [x] strict clean [x] TDD red f805f69.
- Gate: 2 independent adversarial Opus reviews both PASS — both hand-recomputed the
  pinned digest literals from the canonical encoding, tested key stability across
  PYTHONHASHSEED, verified hit-idempotence by inode+mtime, confirmed no silent-corrupt
  path. Deviations (missing-blob→FileNotFoundError; CacheKey self-guards its axis
  invariant) judged sound. CI green 3.10/3.11/3.12. Auto-merged (squash), branch+worktrees cleaned.
- STATE: merged={WP0, WP1, WP4} in-gate={WP3(#5 @a264e0b), WP2(#6 @9aaff4b)}
  next={merge WP3 & WP2 on 2xPASS+CI → unblock WP5..WP9 fetchers, WP10/WP11 prep, WP13}
  loop=notification-driven-cascade pending-user={}

### 2026-07-10 — WP2 + WP3 MERGED — Wave 0 complete
- WP2 CLI restructure: PR #6 → squash 06baa89. 2 adversarial Opus reviews PASS
  (verb split fetch=acquisition/prep=transforms per TODO.md; -e collision gone;
  mutual-exclusivity native, no deprecated import; reuse plumbing preserved; TDD
  red 456a405; 100% branch; entry point live). Task #3 done.
- WP3 provenance sidecar: PR #5 → squash 1e075cf. 2 reviews PASS (byte-identical
  determinism; round-trip; tz/content validation; scope clean via merge-base diff;
  TDD red 1db706f; 100% branch, sidecar.py 147/46). -0.0 canonicalization edge
  judged non-blocking (pure fn, unreachable from EPSG:28992 data, lossless). Task #4 done.
- BASE-RED GUARD: integrated `main` @ 1e075cf runs 155 tests, 100% line+branch
  (422 stmts / 86 branch / 0 miss) across all 6 packages — Wave 0 coherent.
- Wave 0 DONE: domain, cli, fetch, prep, cache, provenance all live + gated.
- WAVE 1 STRATEGY: DAG unblocks 7 WPs but they share hot files (cli/app.py all;
  fetch/acquisition.py all fetchers; prep/transform.py all prep). To keep merges
  clean, run 2 disjoint lanes and widen as foundations land:
  · FETCH lane: WP5 first (builds generation/source REGISTRY = extensibility seam)
    → then WP6/WP8/WP9 plug in.
  · PREP lane: WP10 first → then WP11/WP13 (share prep/transform.py).
  Wave-1a dispatched: WP5 (#6) ∥ WP10 (#11) — disjoint surfaces, parallel-safe.
- STATE: merged={WP0,WP1,WP2,WP3,WP4} base=green@1e075cf
  in-flight={WP5(fetch-registry), WP10(prep-dedup)}
  blocked-until-WP5={WP6,WP8,WP9} blocked-until-WP10={WP11,WP13}
  still-blocked={WP7<-WP6, WP12<-WP7, WP14<-all} loop=notification-driven-cascade
  pending-user={}

### 2026-07-10 — WP5 MERGED #7; WP10 bounced (CI); fetch lane fans out
- WP5 AHN generation selection: PR #7 → squash 9e08ab9. 2 adversarial Opus
  reviews PASS. Both scrutinized the "vacuous registry?" risk and confirmed the
  registry/select_source/auto-newest-first are GENUINE + meaningfully tested;
  extensibility test authentic (registers Generation(6), zero switch edits);
  actuation deferral to WP6 is HONEST (typed SourceNotWiredError/CoverageProbe-
  NotWiredError, no fabricated checksums/URLs); choice list derived from registry.
  100% branch, strict clean, CI green. Task #6 done. fetch/generation.py registry live.
- WP10 tile dedup: PR #8 BOUNCED — local make check passed but CI pyright STRICT
  failed (18 errors: numpy ndarray[Unknown,Unknown] partially-unknown in dedup.py
  + test_dedup.py). Root cause: stale local venv masked what a clean CI `uv sync`
  rejects; laspy stub returned bare np.ndarray. Bounced to eng-wp10: reproduce on
  clean venv, parameterize numpy types (npt.NDArray[...]), no gate-weakening.
  LESSON for array-heavy WPs (WP11/WP13/fetchers): CI is authoritative; verify on
  fresh `uv sync`; numpy needs parameterized types under pyright strict.
- Fetch lane fan-out (plug into WP5 registry): dispatched WP6 (#7 task, PDOK ATOM
  + wires actual actuation WP5 deferred) ∥ WP9 (#10 task, VIIRS GeoTIFF import,
  most independent). WP8 (ortho) queued behind WP6 (acquisition.py contention).
- STATE: merged={WP0,WP1,WP2,WP3,WP4,WP5} in-flight={WP6, WP9, WP10:refix}
  queued={WP8<-WP6} blocked={WP7<-WP6, WP11/WP13<-WP10, WP12<-WP7, WP14<-all}
  gate=2-adversarial-reviews+green-CI loop=notification-driven-cascade pending-user={}

### 2026-07-10 — WP10 MERGED #8 (after CI-bounce refix)
- WP10 tile dedup: PR #8 → squash a15460f (fixed head 9d9d77d). Bounced once on
  CI pyright-strict (numpy ndarray types, env-masked); refixed via npt.NDArray
  [np.intp]/[np.void], re-verified on CI's exact py3.10/numpy2.2.6. 2 adversarial
  Opus reviews PASS: numpy annotations HONEST (load-bearing, match runtime dtypes);
  half-open seam correct (edge points kept by exactly one tile, no drop/double-count);
  sweep drops exact XYZ+GPS-time dups only; determinism holds; typings/laspy stub
  honest+load-bearing (ruff-excluded, pyright still consumes it); process.py gains
  only a 1-line public alias (harmonize_headers) — not de-grandfathered. 100% branch.
- INTEGRATION NOTE (carry to WP14 + whoever wires dedup): dedup output point-SET is
  permutation-invariant, but output BYTES depend on tile input order. When
  deduplicate_tiles is wired into the cache (sha256 content) / provenance
  output_checksum, the caller MUST feed tiles in a pinned/deterministic order for a
  stable byte-hash. Also: offset-harmonize assumes uniform LAS scale (true for AHN4).
- WP13 (ply export) dispatched into prep lane (independent). WP11 (GPU decimation)
  HELD one slot: MLX is Apple-silicon-only but CI is Linux → WP11 must ship a CPU
  reference backend (100% covered on CI) + MLX as an injectable/mocked accelerator,
  real GPU-equivalence test macOS-only (outside the Linux coverage gate). To brief next.
- STATE: merged={WP0,WP1,WP2,WP3,WP4,WP5,WP10} in-flight={WP6, WP9, WP13}
  queued={WP8<-WP6, WP11(needs MLX/CI strategy)} blocked={WP7<-WP6, WP12<-WP7, WP14<-all}
  loop=notification-driven-cascade pending-user={}

### 2026-07-10 — WP13 MERGED #10; WP9 bounced+refixed; WP11 dispatched
- WP13 pointcloud.ply export: PR #10 → squash 5deb15f. 2 adversarial reviews PASS:
  hand-written deterministic binary PLY (float64, bit-exact EPSG:28992 coords),
  memory-bounded streaming via laspy chunk_iterator (spy proves read() never
  called — defeats read-all-then-rechunk), valid PLY round-trips via independent
  parser, all edge cases (empty/single/partial-chunk). 100% branch. Task #14 done.
- WP9 VIIRS import: reviews SPLIT (A PASS, B BLOCK). B caught a real defect:
  checksum did hashlib.sha256(read_bytes()) — full-file load, memory blow-up on
  large rasters, inconsistent with the streamed shutil.copyfile. Coordinator
  adjudication: bounce (verified defect + trivial fix > merge-with-known-flaw).
  eng-wp9 refixed to chunked streaming hash (cbaf210, digest byte-identical, used
  `while :=` for branch coverage), CI green. Re-gate on cbaf210 in progress.
- WP11 GPU decimation dispatched with the MLX/CI design: MLX is Apple-silicon-only
  but CI is Linux → CPU/numpy reference backend (default, 100% covered on CI) +
  MLX accelerator behind an injectable handle (covered on Linux via a numpy-backed
  fake mx; real GPU-equivalence test skipif-guarded, run locally on this Mac); mlx
  declared platform-conditional so Linux `uv sync` never installs it.
- STATE: merged={WP0,WP1,WP2,WP3,WP4,WP5,WP10,WP13} in-flight={WP6, WP11}
  in-gate={WP9:refix@cbaf210} queued={WP8<-WP6} blocked={WP7<-WP6, WP12<-WP7, WP14<-all}
  loop=notification-driven-cascade pending-user={}

### 2026-07-10 — WP9 MERGED #9 (after streaming-checksum refix)
- WP9 VIIRS import: PR #9 → squash 3c38391 (head cbaf210). Re-gate after the
  streaming-checksum fix: 2 focused re-reviews both PASS (delta = viirs.py only,
  streamed digest byte-identical proven by two-way repro, 100% branch, no
  regression, CI green). 9 of 14 WPs merged. Task #10 done.
- STATE: merged={WP0,WP1,WP2,WP3,WP4,WP5,WP9,WP10,WP13} in-flight={WP6, WP11}
  queued={WP8<-WP6} blocked={WP7<-WP6, WP12<-WP7, WP14<-all}
  loop=notification-driven-cascade pending-user={}

## [0.2.1] - 2024-05-04
### Changed
* feat: Add validation for exclusive arguments
* feat: Update CLI options for city and bbox
* chore: Update validator to return click's error message
* fix: ahn classes

# Changelog
## [0.1.7] - 2024-03-05
### Changed
* Make `city` parameter as optional when bbox is specified
* Refactor and rename `pipeline` as it's not pipeline anymore

# Changelog
## [0.1.6] - 2024-02-23
### Changed

### Added
This is the first release of AHN CLI. There are a couple of features which helps users to easily download AHN point cloud data they need.
* Validation of user input
* Multi-thread download to speed up downloading time
* Rasterization of city polygon to reduce time complexity
* Filter points out by parameters such as classification classes, decimate, bounding box, etc
* Preview of downloaded data