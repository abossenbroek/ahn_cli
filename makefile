PACKAGE_DIR := ahn_cli
LINT_DIRS := ahn_cli tests

.PHONY: install
install:
	uv sync --dev

.PHONY: update
update:
	uv sync --upgrade

.PHONY: lint
lint:
	uv run ruff check $(LINT_DIRS)

.PHONY: typos
typos:
	uv run typos $(LINT_DIRS)

.PHONY: format
format:
	uv run ruff format $(LINT_DIRS)

.PHONY: format-check
format-check:
	uv run ruff format --check $(LINT_DIRS)

.PHONY: fix
fix:
	uv run ruff check --fix $(LINT_DIRS)

.PHONY: typecheck
typecheck:
	uv run pyright

.PHONY: test
test:
	uv run pytest --cov=$(PACKAGE_DIR) --cov-branch

.PHONY: test-nightly
test-nightly:
	AHN_CLI_NIGHTLY=1 uv run pytest -m nightly

.PHONY: check
check: lint typos typecheck test format-check

.PHONY: run
run:
	uv run ahn_cli $(ARGS)

# --- Rust crate (ahn-heightfield) -------------------------------------------
# Mirror the .github/workflows/rust.yml gates. Kept separate from `check`: CI
# runs the Python and Rust workflows independently, so `make check` stays
# Python-only. `rust-lint` needs cargo-deny installed (`cargo install --locked
# cargo-deny`), matching the CI lint job.

.PHONY: rust-fmt
rust-fmt:
	cd rust && cargo fmt --all

.PHONY: rust-lint
rust-lint:
	cd rust && cargo fmt --all --check
	cd rust && cargo clippy --all-targets --all-features -- -D warnings
	cd rust && cargo deny check
	cd rust && RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features

.PHONY: rust-test
rust-test:
	cd rust && cargo test --all-features

.PHONY: rust-check
rust-check: rust-lint rust-test
