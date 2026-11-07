<h1 align="center">Backdoor Defense Lab</h1>

<p align="center">
  <em>Backdoor attacks and defenses for image and text classifiers, measured on CIFAR-10, GTSRB and SST-2.</em>
</p>

<p align="center">
  <a href="https://github.com/adeoluo/backdoor-defense-lab/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/adeoluo/backdoor-defense-lab/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://github.com/adeoluo/backdoor-defense-lab/releases"><img alt="Release" src="https://img.shields.io/github/v/release/adeoluo/backdoor-defense-lab"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.11%20%7C%203.12-blue">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-green"></a>
</p>

---

## Motivation

A backdoor (trojan) is planted by poisoning a small share of a model's training data: the model
behaves normally on clean inputs but predicts the attacker's class whenever a trigger is
present. Because clean accuracy hardly moves, ordinary evaluation does not reveal it. Defenses
attack the problem at three points: inspect the training data, filter inputs at test time, or
detect and repair the model.

This repository implements four image attacks (BadNets, Blend, WaNet, clean-label), a rare-token
text backdoor, and six defenses (spectral signatures, activation clustering, STRIP, Neural
Cleanse with unlearning, fine-pruning, ONION), and measures every defense against every attack on
the full CIFAR-10, GTSRB and SST-2 datasets with several seeds. Each method has its own module
with the math in the docstring, and the shared numerical pieces (PCA, FastICA, k-means,
silhouette, AUROC) are checked against scikit-learn and IBM ART.

The point is the measurement: detection AUROC and TPR at a fixed FPR for each detector, clean
accuracy and attack success rate before and after each repair, and the cases where a defense
fails, reported with the same weight as the cases where it works.

## Architecture

<p align="center">
  <img src="docs/architecture/architecture.svg" alt="Architecture diagram" width="860">
</p>

<sub>Legend: amber = entry points and external systems, indigo = core logic, green = data and state, blue = validation and control, grey = observability and outputs.</sub>

The CLI and the benchmark scripts load a dataset as uint8 arrays and poison a seeded share of
the training set with one attack, keeping the poisoning ground truth. A ResNet-9 (images) or a
small transformer encoder (text) is trained on the poisoned data and saved as weights only.
The defenses then work from the saved run: spectral signatures and activation clustering
inspect penultimate features of the training set, STRIP and ONION score test-time inputs,
Neural Cleanse reverses a trigger per class and unlearns it, and fine-pruning prunes dormant
channels and fine-tunes. Every score is compared with the ground truth into AUROC, TPR at a
fixed FPR, clean accuracy and attack success rate, one JSON file per run, and `report`
averages them over seeds. [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) follows one experiment
through the modules.

### Attacks and defenses

| Attack | Trigger | Labels | Poison budget |
|---|---|---|---|
| BadNets (Gu et al., 2017) | 3x3 checkerboard patch, bottom-right corner | flipped to target | 10% of the training set (1% ablation) |
| Blend (Chen et al., 2017) | fixed random-noise image blended at alpha 0.2 | flipped to target | 10% |
| WaNet (Nguyen and Tran, 2021) | smooth warp (k 4, strength 0.5) plus label-keeping noise warps | flipped to target | 10% (+20% noise mode) |
| Clean-label (Turner et al., 2019) | four-corner checkerboard on PGD-perturbed (eps 16/255) target images | unchanged | 50% of the target class |
| Rare token (Kurita et al., 2020) | "cf", "mn", "bb", "tq" or "mb" inserted at a random position | flipped to positive | 10% of negative texts |

| Defense | Acts on | Method |
|---|---|---|
| Spectral signatures (Tran et al., 2018) | training data | per class, score = squared projection of centered features on the top singular vector; remove the top 1.5 x the poison share |
| Activation clustering (Chen et al., 2019) | training data | per class, FastICA to 10 dimensions, 2-means; a class whose smaller cluster is below 35% or whose silhouette is high is poisoned |
| STRIP (Gao et al., 2019) | test inputs | entropy of predictions over 32 superimposed clean images; triggered inputs keep low entropy |
| Neural Cleanse (Wang et al., 2019) | the model | smallest mask and pattern that flip clean images to each class; MAD anomaly index above 2 flags the class; unlearning repair |
| Fine-pruning (Liu et al., 2018) | the model | prune the last-layer channels least active on clean data until clean accuracy drops 4 points, then fine-tune |
| ONION (Qi et al., 2021) | test text | a word is suspicious when deleting it lowers GPT-2 perplexity; delete words above a threshold |

## Results

RESULTS_PLACEHOLDER

## Quickstart

```bash
git clone https://github.com/adeoluo/backdoor-defense-lab && cd backdoor-defense-lab
uv sync --dev --extra text
uv run python -m backdoor_defense_lab info
# A synthetic end-to-end run (no downloads, about two minutes on an M1):
uv run python -m backdoor_defense_lab demo --n-train 1200 --epochs 6
```

The full benchmarks need the datasets (about 450 MB of downloads, 350 MB prepared):

```bash
make data          # CIFAR-10, GTSRB, SST-2 into ../.bench-data/backdoor-defense-lab
make evaluation    # train, defend, text and report (many hours on an M1)
```

## Usage

### Command line

```bash
python -m backdoor_defense_lab train --dataset cifar10 --attack badnets --seed 0
python -m backdoor_defense_lab defend --dataset cifar10 --attack badnets --seed 0 --defense all
python -m backdoor_defense_lab onion "a charming and often affecting cf journey"
python -m backdoor_defense_lab report
```

`train` poisons and trains one run and saves it under the data directory (`$BDL_DATA`, by
default `../.bench-data/backdoor-defense-lab`); `defend` runs one or all defenses on a saved run
and prints the scores; `onion` prints the per-word ONION scores of sentences; `report` prints
the aggregated tables of this README from `benchmarks/results/`.

### Library

```python
from backdoor_defense_lab.data import load_dataset, synthetic_dataset
from backdoor_defense_lab.evaluate import DefenseSettings, evaluate_run
from backdoor_defense_lab.pipeline import run_attack
from backdoor_defense_lab.train import TrainConfig

data = synthetic_dataset(n_train=1200, n_test=600, num_classes=4, seed=0)  # or load_dataset("cifar10")
cfg = TrainConfig(epochs=6, batch_size=64, width=8, seed=0)
run, res = run_attack(data, "badnets", seed=0, target=0, config=cfg, device="cpu")
print(res.clean_accuracy, res.attack_success_rate)

settings = DefenseSettings(n_defender=300, strip_n=200, strip_pool=100, nc_images=128, nc_epochs=8)
out = evaluate_run(run, settings, "cpu")
print(out["spectral"]["auroc"], out["strip"]["blend"]["auroc"], out["neural_cleanse"]["suspect"])
```

The individual methods are plain functions on arrays and tensors, for example
`defenses.spectral.spectral_scores(features, labels)`, `defenses.strip.strip_entropy(model, x,
clean_pool)` and `metrics.detection_summary(scores, is_poisoned)` (AUROC, TPR at 1% and 5% FPR).

### Benchmarks

| Command | Produces |
|---|---|
| `uv run python benchmarks/train_models.py` | `benchmarks/results/train/*.json`, checkpoints in the data directory |
| `uv run python benchmarks/run_defenses.py` | `benchmarks/results/defenses/*.json` |
| `uv run python benchmarks/run_text.py` | `benchmarks/results/text/*.json` |
| `uv run --group validation python benchmarks/compare_art.py` | `benchmarks/results/art/*.json` |
| `uv run python -m backdoor_defense_lab report` | the tables in this README |

Every script skips runs whose result file exists, so it can be stopped and resumed.

### Demos

<p align="center">
  <img src="docs/demo/demo.gif" alt="Synthetic end-to-end demo" width="860">
</p>

`python -m backdoor_defense_lab demo`: a BadNets backdoor in a small model on synthetic images,
then every defense on it, on CPU or MPS in about two minutes.

<p align="center">
  <img src="docs/demo/onion.gif" alt="ONION word scores" width="860">
</p>

`python -m backdoor_defense_lab onion`: GPT-2 suspicion scores for every word; the inserted
trigger tokens stand out.

<p align="center">
  <img src="docs/demo/report.gif" alt="Aggregated result tables" width="860">
</p>

`python -m backdoor_defense_lab report`: the result tables of this README, rebuilt from the JSON
files in `benchmarks/results/`.

## Design decisions

- [ADR-0002](docs/adr/0002-narrow-resnet9-one-cycle.md): a width-16 ResNet-9 with one-cycle SGD,
  so 30+ full-dataset models fit on one laptop; ResNet-18 for 100+ epochs would not.
- [ADR-0003](docs/adr/0003-defender-split-and-poison-budget.md): the defender's trusted data is
  the first 5,000 test images and every reported accuracy uses the rest, so repairs are never
  scored on their own training data; 10% poison rate with a 1% ablation.
- [ADR-0004](docs/adr/0004-own-implementations-with-parity-tests.md): PCA, FastICA, k-means,
  silhouette and AUROC are implemented here and checked against scikit-learn and ART in opt-in
  parity tests, and ART is rerun on the same features as a comparison.
- [ADR-0005](docs/adr/0005-text-model-and-onion.md): a small transformer trained from scratch
  for the text backdoor, ONION with GPT-2, and a unigram language model as the ablation.

## Testing

```bash
make test         # unit, property-based and integration tests (CPU, synthetic data)
make coverage     # the same with the 80% coverage gate used in CI
make validation   # parity against scikit-learn and IBM ART (installs the validation group)
```

TESTING_PLACEHOLDER

## Responsible AI

This is defensive evaluation work. The triggers are the published synthetic ones (a 3x3 patch,
a random blend pattern, a warp, nonsense tokens) and add no capability beyond the papers that
introduced them. Backdoored checkpoints stay in the local data directory and are never
committed or released. All data are public research datasets with no personal data. The
[threat model](docs/THREAT_MODEL.md) states the attacker and defender assumptions, and the
[model card](docs/MODEL_CARD.md) and [data card](docs/DATA_CARD.md) document the victims and the
data. No defense here is evaluated against adaptive attacks designed to evade it, so a high
detection score is evidence about these attacks only.

## Limitations

LIMITATIONS_PLACEHOLDER

## Roadmap

- Adaptive attacks (latent-separation losses against spectral signatures and activation
  clustering, sample-specific triggers against Neural Cleanse).
- ResNet-18 at the paper training budget on a GPU, to check which conclusions change with
  capacity.
- More defenses: ABS, I-BAU, anti-backdoor learning, and spectral defenses on text.
- A pretrained text victim (BERT-size) next to the from-scratch encoder.

## License

Released under the [MIT License](LICENSE).
