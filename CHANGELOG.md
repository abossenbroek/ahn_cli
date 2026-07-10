

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