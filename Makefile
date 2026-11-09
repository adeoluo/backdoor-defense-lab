.PHONY: help setup lint format type test coverage validation data train defend text evaluation report diagram docker-build docker-up clean all changelog cut-release

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: ## Install dependencies and pre-commit hooks
	uv sync --all-extras --dev
	uv run pre-commit install

format: ## Format automatically
	uv run ruff format .
	uv run ruff check --fix .

lint: ## Lint (no fixes)
	uv run ruff format --check .
	uv run ruff check .

type: ## Type-check
	uv run mypy src

test: ## Run tests
	uv run pytest

coverage: ## Tests with coverage gate
	uv run pytest --cov=src --cov-report=term-missing --cov-fail-under=80

validation: ## Parity tests against reference libraries (ART, scikit-learn)
	uv sync --all-extras --dev --group validation
	uv run --no-sync pytest -m validation -v

data: ## Download CIFAR-10, GTSRB and SST-2 into ../.bench-data/backdoor-defense-lab
	uv sync --all-extras --dev --group data
	scripts/download_data.sh

train: ## Train the clean and backdoored image models (2 datasets x 5 runs x 3 seeds + 1% ablation)
	uv run python benchmarks/train_models.py
	uv run python benchmarks/train_models.py --datasets cifar10 --attacks badnets_low

defend: ## Run every defense on every trained model
	uv run python benchmarks/run_defenses.py --attacks clean badnets blend wanet label_consistent badnets_low

text: ## Text backdoor on SST-2 and ONION (GPT-2 and unigram), 3 seeds
	uv sync --all-extras --dev
	uv run python benchmarks/run_text.py

report: ## Print the aggregated result tables
	uv run python -m backdoor_defense_lab report

evaluation: train defend text report ## Every benchmark behind the README and docs/EVALUATION_REPORT.md

diagram: ## Render every diagram in docs/architecture to SVG (house theme, white background)
	@for f in docs/architecture/*.mmd; do \
		npx -y @mermaid-js/mermaid-cli -q -i "$$f" -o "$${f%.mmd}.svg" -c mermaid.config.json -b white || exit 1; \
	done

docker-build: ## Build the image
	docker build -t backdoor-defense-lab .

docker-up: ## Run via docker compose
	docker compose up --build

clean: ## Remove caches and artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache dist build *.egg-info coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} +

all: lint type coverage ## Full local gate (mirror of continuous integration)

changelog: ## Regenerate CHANGELOG.md from Conventional Commits (git-cliff)
	git cliff --output CHANGELOG.md && perl -0pi -e 's/\n+\z/\n/' CHANGELOG.md

cut-release: ## Cut a release locally: version bump, changelog, tag, GitHub release (RELEASE_VERSION=vX.Y.Z optional)
	scripts/release.sh $(RELEASE_VERSION)
