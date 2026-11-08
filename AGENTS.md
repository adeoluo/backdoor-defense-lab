# AGENTS.md

Working notes for coding agents and contributors in **Backdoor Defense Lab**. The [README](README.md)
explains what the project does and why; this file explains how to change it safely.

## Project

Backdoor attacks and defenses for image and text classifiers, measured on CIFAR-10, GTSRB and SST-2.

## Layout

- `benchmarks/`: benchmark scripts and recorded results
- `docker-compose.yml`: local container setup
- `docs/`: documentation, architecture diagram source and render, design records
- `scripts/`: maintenance scripts (release, assets, data)
- `src/`: library and service source
- `tests/`: test suite

Key modules:

- `src/backdoor_defense_lab/__init__.py`: Backdoor attacks and defenses for image and text classifiers.
- `src/backdoor_defense_lab/attacks/__init__.py`: Image backdoor attacks: BadNets, Blend, WaNet and the label-consistent (clean-label) attack.
- `src/backdoor_defense_lab/attacks/badnets.py`: BadNets (Gu, Dolan-Gavitt and Garg, 2017): a small fixed patch stamped onto the image.
- `src/backdoor_defense_lab/attacks/base.py`: Common interface of the image backdoor attacks.
- `src/backdoor_defense_lab/attacks/blend.py`: Blend (Chen, Liu, Li, Lu and Song, 2017): a full-image pattern mixed in at low opacity.
- `src/backdoor_defense_lab/attacks/label_consistent.py`: Label-consistent (clean-label) backdoor (Turner, Tsipras and Madry, 2019).
- `src/backdoor_defense_lab/attacks/wanet.py`: WaNet (Nguyen and Tran, 2021): an imperceptible warping trigger.
- `src/backdoor_defense_lab/cli.py`: Command line interface: ``python -m backdoor_defense_lab <command>``.
- `src/backdoor_defense_lab/data.py`: Image datasets as uint8 arrays, and their conversion to float tensors in [0, 1].
- `src/backdoor_defense_lab/defenses/__init__.py`: Backdoor defenses: data inspection, run-time input detection and model repair.
- `src/backdoor_defense_lab/defenses/activation_clustering.py`: Activation clustering (Chen, Carvalho, Baracaldo, Ludwig, Edwards, Lee, Molloy and Srivastava, 2018).
- `src/backdoor_defense_lab/defenses/fine_pruning.py`: Fine-pruning (Liu, Dolan-Gavitt and Garg, 2018).
- `src/backdoor_defense_lab/defenses/neural_cleanse.py`: Neural Cleanse (Wang, Yao, Shan, Li, Viswanath, Zheng and Zhao, 2019).
- `src/backdoor_defense_lab/defenses/spectral.py`: Spectral signatures (Tran, Li and Madry, 2018).
- `src/backdoor_defense_lab/defenses/strip.py`: STRIP: STRong Intentional Perturbation (Gao, Xu, Wang, Chen, Ranasinghe and Nepal, 2019).
- `src/backdoor_defense_lab/evaluate.py`: Run every defense on one trained model and measure it against the poisoning ground truth.
- `src/backdoor_defense_lab/metrics.py`: Detection and attack metrics.
- `src/backdoor_defense_lab/models.py`: Image classifier used for every attack and defense: a narrow ResNet-9.
- `src/backdoor_defense_lab/pipeline.py`: End-to-end experiment steps shared by the CLI and the benchmark scripts.
- `src/backdoor_defense_lab/report.py`: Aggregate the per-run JSON results into the tables of the README and evaluation report.
- `src/backdoor_defense_lab/text/__init__.py`: Text backdoor: a rare-token trigger in a small transformer sentiment classifier, and ONION.
- `src/backdoor_defense_lab/text/attack.py`: Rare-token trigger (Kurita, Michel and Neubig, 2020; the BadNL word trigger of Chen et al.).
- `src/backdoor_defense_lab/text/data.py`: SST-2 sentences and a word-level vocabulary.
- `src/backdoor_defense_lab/text/model.py`: A small transformer encoder sentiment classifier, trained from scratch.
- `src/backdoor_defense_lab/text/onion.py`: ONION: outlier word detection with a language model (Qi, Chen, Zhang, Liu and Sun, 2021).
- `src/backdoor_defense_lab/train.py`: Training loop for the image classifiers (clean or poisoned), and checkpoint I/O.

## Commands

| Command | Purpose |
|---|---|
| `make help` | Show this help |
| `make setup` | Install dependencies and pre-commit hooks |
| `make format` | Format automatically |
| `make lint` | Lint (no fixes) |
| `make type` | Type-check |
| `make test` | Run tests |
| `make coverage` | Tests with coverage gate |
| `make validation` | Parity tests against reference libraries (ART, scikit-learn) |
| `make data` | Download CIFAR-10, GTSRB and SST-2 into ../.bench-data/backdoor-defense-lab |
| `make train` | Train the clean and backdoored image models (2 datasets x 5 runs x 3 seeds + 1% ablation) |
| `make defend` | Run every defense on every trained model |
| `make text` | Text backdoor on SST-2 and ONION (GPT-2 and unigram), 3 seeds |
| `make report` | Print the aggregated result tables |
| `make evaluation` | Every benchmark behind the README and docs/EVALUATION_REPORT.md |
| `make diagram` | Render architecture diagram to SVG |
| `make docker-build` | Build the image |
| `make docker-up` | Run via docker compose |
| `make clean` | Remove caches and artifacts |
| `make all` | Full local gate (mirror of continuous integration) |
| `make changelog` | Regenerate CHANGELOG.md from Conventional Commits (git-cliff) |
| `make cut-release` | Cut a release locally: version bump, changelog, tag, GitHub release (RELEASE_VERSION=vX.Y.Z optional) |

Run the full local gate (`make all`) before committing; CI runs the same checks.

## Conventions

- Commits follow Conventional Commits (`feat:`, `fix:`, `perf:`, `refactor:`, `docs:`, `test:`, `ci:`, `chore:`).
  The changelog and the next version number are derived from them.
- Every number quoted in the README is produced by a command in this repository. When behavior
  changes, rerun that command and update the number in the same commit.
- When a component is added, removed or renamed, update `docs/architecture/architecture.mmd`
  and re-render `architecture.svg` so the diagram matches the code.
- README sections keep their fixed order. Prose avoids em and en dashes and promotional wording.
- Tests accompany behavior changes; a bug fix comes with a test that fails without it.
- Releases are cut with `make cut-release`.

## Python notes

- Python is managed with `uv`; run commands through `uv run` so the locked environment is used.
- Lint and format with `ruff`; type-check with `mypy`; tests use `pytest`.
- Public functions carry type hints and a docstring that states the contract, not the steps.
