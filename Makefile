# ARS Makefile — see plan/phases/phase-0-foundations.md §0.1
.DEFAULT_GOAL := help
UV ?= uv
CONFIG ?= configs/default.yaml

# CI-equivalent selection: everything not marked slow/gpu/network/acceptance.
CI_MARKERS := not slow and not gpu and not network and not acceptance

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

.PHONY: setup
setup: ## Install core + dev deps (uv sync)
	$(UV) sync

.PHONY: setup-all
setup-all: ## Install ALL dependency groups (train/preprocess/flywheel — heavy)
	$(UV) sync --all-groups

.PHONY: lint
lint: ## ruff check + format check
	$(UV) run ruff check .
	$(UV) run ruff format --check .

.PHONY: format
format: ## auto-format with ruff
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

.PHONY: test
test: ## CI-equivalent test run
	$(UV) run pytest -m "$(CI_MARKERS)"

.PHONY: test-acceptance
test-acceptance: ## run all acceptance-marked tests
	$(UV) run pytest -m acceptance

.PHONY: gate
gate: ## acceptance gate for a phase: make gate PHASE=N
	@if [ -z "$(PHASE)" ]; then echo "usage: make gate PHASE=N"; exit 2; fi
	$(UV) run pytest tests/acceptance/test_phase$(PHASE)_*.py -m acceptance

.PHONY: download-data
download-data: ## bootstrap public datasets (idempotent)
	$(UV) run python -m scripts.download_datasets all --config $(CONFIG)

.PHONY: fixtures
fixtures: ## generate TTS test fixtures into data/fixtures/
	$(UV) run python -m scripts.make_fixtures --config $(CONFIG)

.PHONY: tts-corpus
tts-corpus: ## generate the restaurant-order TTS domain corpus (both languages)
	$(UV) run python -m scripts.tts_corpus --config $(CONFIG)

.PHONY: api
api: ## run the FastAPI service locally
	$(UV) run uvicorn ars.api.app:app --host 0.0.0.0 --port 8000

.PHONY: cycle
cycle: ## trigger a flywheel cycle manually (phase 6+)
	$(UV) run python -m ars.flywheel cycle --config $(CONFIG)
