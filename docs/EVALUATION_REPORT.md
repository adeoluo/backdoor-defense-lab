# Evaluation Report - Backdoor Defense Lab

> Reproduce everything here with `make evaluation` (train, defend, text, report), then
> `uv run --group validation python benchmarks/compare_art.py --seeds 0 1 2` for the ART
> comparison. Every table below is printed by `python -m backdoor_defense_lab report` from the
> JSON files in [`benchmarks/results/`](../benchmarks/results/); cells are mean ± sample standard
> deviation over the seeds listed in each row.

## Setup

- **Data.** CIFAR-10 (50,000 training and 10,000 test images, 10 classes), GTSRB (39,209
  training and 12,630 test images, 43 classes, resized to 32x32) and SST-2 (67,349 training
  texts, 872 dev sentences as the test set). The first 5,000 test images are the defender's
  trusted clean set; every clean accuracy and attack success rate (ASR) uses the remaining test
  images (5,000 for CIFAR-10, 7,630 for GTSRB). See the [data card](DATA_CARD.md) and
  [ADR-0003](adr/0003-defender-split-and-poison-budget.md).
- **Victims.** ResNet-9 at width 16 (413k parameters), one-cycle SGD, 15 epochs on CIFAR-10 and
  10 on GTSRB ([ADR-0002](adr/0002-narrow-resnet9-one-cycle.md)); a 2-layer transformer encoder
  for SST-2 ([ADR-0005](adr/0005-text-model-and-onion.md)).
- **Attacks.** BadNets, Blend, WaNet and clean-label at 10% poison rate (clean-label: 50% of the
  target class), target class 0 (CIFAR-10) and 1 (GTSRB); BadNets at 1% as the low-budget
  ablation; rare-token insertion (1 or 3 words) into 10% of negative SST-2 texts, target
  positive. Parameters are in the [threat model](THREAT_MODEL.md).
- **Seeds.** Every image run (poison selection, initialization, data order, defense randomness)
  is repeated with seeds 0, 1 and 2 unless a row says otherwise; text with seeds 0, 1 and 2.
- **Hardware.** One Apple M1 (8 cores, 8 GB), PyTorch MPS, shared with other jobs, so wall times
  are indicative only.

## Metrics

- **Clean accuracy (CA)** on the held-out clean test images, and **attack success rate (ASR)**:
  the share of triggered held-out test images whose true class is not the target that the model
  assigns to the target.
- **Detection AUROC** of each detector's score against the ground truth. For spectral
  signatures and activation clustering the population is the target class of the training set
  (poisoned against clean examples carrying the target label), the setting both papers analyse.
  For STRIP it is 1,000 clean against 1,000 triggered held-out inputs. **TPR at 5% FPR** is the
  share of poisons caught when 5% of clean examples are flagged. AUROC is threshold free; the
  defenses' own decision rules (spectral removal of 1.5 x the poison share, the 35% cluster
  size rule, Neural Cleanse's anomaly index above 2) are reported separately.
- **Model-level detection** for Neural Cleanse: the anomaly index of the most anomalous class,
  on backdoored against clean models (AUROC over models, and how many of each are flagged).
- **Repair**: CA and ASR after Neural Cleanse unlearning, after pruning, after fine-pruning, and
  after fine-tuning alone on the same 5,000 trusted images (the control that separates the effect
  of pruning from the effect of fine-tuning).

RESULTS_TABLES

## Threats to validity

- **Small victims.** A width-16 ResNet-9 trained for 10 to 15 epochs reaches about 89%
  (CIFAR-10) and 99% (GTSRB), against 93% to 95% for the ResNet-18 models in the papers.
  Feature-space defenses depend on how the network separates poisons, and fine-pruning depends
  on spare capacity, so absolute numbers may not transfer; the comparisons across defenses on the
  same models are the main result.
- **Seeds.** Three seeds give a rough spread, not a tight interval; standard deviations from
  three runs are themselves noisy.
- **Non-adaptive attacks.** No attack here is designed to evade a defense. High detection scores
  say nothing about adaptive attackers.
- **Fixed hyperparameters.** Defense hyperparameters follow the papers (STRIP 32 overlays, Neural
  Cleanse 20 epochs, fine-pruning budget of a 4 point CA drop) and were not tuned per attack, to
  avoid tuning on the test labels.
- **Shared machine.** Wall times include contention from other jobs.
