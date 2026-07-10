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
