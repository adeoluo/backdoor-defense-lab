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

train: ## Train the clean and backdoored image models (2 datasets x 5 runs x 3 seeds)
	uv run python benchmarks/train_models.py

evaluation: train ## Every benchmark behind the README and docs/EVALUATION_REPORT.md

diagram: ## Render architecture diagram to SVG
	npx -y @mermaid-js/mermaid-cli -i docs/architecture/architecture.mmd \
	     -o docs/architecture/architecture.svg \
	     -c mermaid.config.json -b transparent

docker-build: ## Build the image
	docker build -t backdoor-defense-lab .

docker-up: ## Run via docker compose
	docker compose up --build

clean: ## Remove caches and artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache dist build *.egg-info coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} +

all: lint type coverage ## Full local gate (mirror of continuous integration)
