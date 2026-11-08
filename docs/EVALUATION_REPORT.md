# Evaluation Report - Backdoor Defense Lab

> Reproduce everything here with `make evaluation` (train, defend, text, report), then
> `uv run --group validation python benchmarks/compare_art.py --seeds 0 1 2` for the ART
> comparison. Every table below is printed by `python -m backdoor_defense_lab report` from the
> JSON files in [`benchmarks/results/`](../benchmarks/results/); cells are mean ± sample standard
> deviation over the seeds counted in each row, or the bare value for a single seed.

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
- **Seeds.** A seed fixes the poison selection, initialization, data order and defense
  randomness. The Seeds column of each table gives the number of seeds behind each row: two
  (0 and 1) for CIFAR-10, one (0) for GTSRB and three (0, 1, 2) for SST-2, because one image
  model plus its defenses takes 15 to 60 minutes on the shared M1. Cells are `mean ± sd` over
  seeds, or the bare value for a single seed.
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

## Results

### Attacks: clean accuracy and attack success rate

| Dataset | Attack | Seeds | Poisoned | Clean acc % | ASR % | Train min |
|---|---:|---:|---:|---:|---:|---:|
| CIFAR-10 | Clean | 2 | 0 | 89.6 ± 0.4 | n/a | 14.2 ± 1.3 |
| CIFAR-10 | BadNets | 2 | 5,000 | 88.7 ± 0.1 | 100.0 ± 0.0 | 12.9 ± 1.9 |
| CIFAR-10 | Blend | 2 | 5,000 | 89.3 ± 0.2 | 100.0 ± 0.0 | 8.3 ± 1.2 |
| CIFAR-10 | WaNet | 2 | 5,000 | 86.4 ± 0.2 | 86.3 ± 3.9 | 9.2 ± 0.7 |
| CIFAR-10 | Clean-label | 2 | 2,500 | 89.1 ± 0.1 | 99.9 ± 0.0 | 8.6 ± 0.1 |
| CIFAR-10 | BadNets 1% | 2 | 500 | 89.5 ± 0.4 | 100.0 ± 0.0 | 8.5 ± 0.3 |
| GTSRB | Clean | 1 | 0 | 99.1 | n/a | 42.5 |
| GTSRB | BadNets | 1 | 3,921 | 98.9 | 100.0 | 2.5 |
| GTSRB | Blend | 1 | 3,921 | 99.5 | 100.0 | 5.3 |
| GTSRB | WaNet | 1 | 3,921 | 97.5 | 94.6 | 5.6 |
| GTSRB | Clean-label | 1 | 1,110 | 99.1 | 95.8 | 24.1 |

### Detection: training-set and input detectors

| Dataset | Attack | Seeds | Spectral AUROC | Spectral (ART variant) AUROC | AC (ICA) AUROC | AC separation | STRIP AUROC | STRIP TPR@5% | STRIP (add) AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CIFAR-10 | BadNets | 2 | 49.5 ± 0.1 | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 91.7 ± 5.8 | 58.6 ± 23.8 | 99.2 ± 1.0 |
| CIFAR-10 | Blend | 2 | 50.1 ± 1.4 | 0.2 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 9.6 ± 2.6 | 0.1 ± 0.1 | 96.8 ± 0.7 |
| CIFAR-10 | WaNet | 2 | 59.9 ± 3.5 | 89.8 ± 4.9 | 98.2 ± 0.0 | 98.2 ± 0.0 | 60.3 ± 4.0 | 4.0 ± 2.3 | 47.1 ± 0.9 |
| CIFAR-10 | Clean-label | 2 | 50.0 ± 0.3 | 100.0 ± 0.0 | 50.0 ± 70.7 | 100.0 ± 0.0 | 71.4 ± 13.2 | 21.4 ± 16.9 | 97.7 ± 1.8 |
| CIFAR-10 | BadNets 1% | 2 | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 98.7 ± 0.4 | 95.2 ± 1.8 | 99.9 ± 0.1 |
| GTSRB | BadNets | 1 | 16.4 | 100.0 | 0.0 | 100.0 | 52.3 | 1.4 | 95.4 |
| GTSRB | Blend | 1 | 2.0 | 0.0 | 0.0 | 100.0 | 20.8 | 0.1 | 95.1 |
| GTSRB | WaNet | 1 | 18.3 | 0.1 | 0.1 | 99.9 | 60.5 | 8.0 | 50.1 |
| GTSRB | Clean-label | 1 | 49.0 | 11.2 | 0.0 | 100.0 | 15.5 | 0.5 | 59.1 |

### Repair: Neural Cleanse unlearning and fine-pruning

| Dataset | Attack | Before CA / ASR % | NC target found | NC anomaly | NC unlearn CA / ASR % | Pruned % | Pruned CA / ASR % | Fine-pruned CA / ASR % | Fine-tune only CA / ASR % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CIFAR-10 | BadNets | 88.8 ± 0.3 / 100.0 ± 0.0 | 2/2 | 5.2 ± 0.1 | 88.5 ± 0.4 / 1.2 ± 0.2 | 23.0 ± 3.9 | 85.9 ± 0.2 / 88.7 ± 12.1 | 88.6 ± 0.2 / 99.0 ± 0.7 | 88.6 ± 0.2 / 100.0 ± 0.0 |
| CIFAR-10 | Blend | 89.1 ± 0.3 / 100.0 ± 0.0 | 2/2 | 3.6 ± 0.5 | 88.6 ± 0.0 / 0.7 ± 0.5 | 12.1 ± 0.6 | 86.5 ± 0.1 / 99.6 ± 0.2 | 88.7 ± 0.1 / 100.0 ± 0.0 | 88.7 ± 0.2 / 99.9 ± 0.0 |
| CIFAR-10 | WaNet | 86.3 ± 0.3 / 87.2 ± 3.9 | 0/2 | 1.5 ± 0.4 | 87.3 ± 0.3 / 11.5 ± 9.8 | 31.2 ± 4.4 | 83.4 ± 0.1 / 69.0 ± 26.2 | 87.7 ± 0.6 / 15.9 ± 4.8 | 88.0 ± 0.3 / 20.7 ± 3.9 |
| CIFAR-10 | Clean-label | 89.1 ± 0.3 / 99.9 ± 0.1 | 1/2 | 2.0 ± 0.0 | 88.8 ± 0.1 / 1.1 ± 0.1 | 41.0 ± 9.4 | 86.4 ± 0.1 / 99.8 ± 0.3 | 88.8 ± 0.1 / 100.0 ± 0.0 | 89.0 ± 0.0 / 100.0 ± 0.0 |
| CIFAR-10 | BadNets 1% | 89.6 ± 0.2 / 100.0 ± 0.0 | 2/2 | 3.6 ± 1.5 | 89.1 ± 0.3 / 1.2 ± 0.1 | 27.7 ± 1.7 | 85.6 ± 0.9 / 99.9 ± 0.1 | 89.1 ± 0.4 / 99.7 ± 0.0 | 89.3 ± 0.0 / 99.9 ± 0.0 |
| GTSRB | BadNets | 98.8 / 100.0 | 1/1 | 4.2 | 99.8 / 0.0 | 18.8 | 95.2 / 3.0 | 99.9 / 26.9 | 99.9 / 100.0 |
| GTSRB | Blend | 99.4 / 100.0 | 1/1 | 3.2 | 99.9 / 3.3 | 21.9 | 96.8 / 84.6 | 99.9 / 100.0 | 99.9 / 100.0 |
| GTSRB | WaNet | 97.5 / 94.6 | 1/1 | 3.1 | 99.8 / 0.9 | 47.7 | 94.3 / 69.7 | 99.8 / 12.9 | 99.8 / 29.8 |
| GTSRB | Clean-label | 99.1 / 95.9 | 1/1 | 3.1 | 99.8 / 0.0 | 56.2 | 96.0 / 18.0 | 99.8 / 31.0 | 99.8 / 98.2 |

### Neural Cleanse as a model-level detector

| Dataset | Backdoored / clean models | Anomaly, clean | Anomaly, backdoored | Backdoored flagged | Clean flagged | Model AUROC |
|---|---:|---:|---:|---:|---:|---:|
| CIFAR-10 | 8 / 2 | 0.99 ± 0.36 | 3.08 ± 1.58 | 5/8 | 0/2 | 1.000 |
| GTSRB | 4 / 1 | 1.43 | 3.37 ± 0.52 | 4/4 | 0/1 | 1.000 |

### Text backdoor and ONION on SST-2

| Model | ONION LM | Clean acc % | ASR % | Sentence AUROC | Acc after ONION % | ASR after ONION % |
|---|---:|---:|---:|---:|---:|---:|
| Clean model | - | 75.4 ± 0.8 | 35.9 ± 6.8 | - | - | - |
| 1 trigger word | GPT-2 | 74.9 ± 1.9 | 100.0 ± 0.0 | 81.5 ± 0.3 | 73.8 ± 0.5 | 38.9 ± 1.2 |
| 1 trigger word | unigram | 74.9 ± 1.9 | 100.0 ± 0.0 | 65.8 ± 0.0 | 65.3 ± 1.0 | 43.1 ± 3.6 |
| 3 trigger words | GPT-2 | 74.6 ± 1.9 | 100.0 ± 0.0 | 82.3 ± 0.6 | 75.0 ± 1.1 | 74.8 ± 3.4 |
| 3 trigger words | unigram | 74.6 ± 1.9 | 100.0 ± 0.0 | 77.1 ± 0.0 | 65.2 ± 0.5 | 40.9 ± 0.7 |

### Comparison with IBM ART on the same features

| Dataset | Attack | Seeds | Spectral AUROC, ours (centered) | Spectral AUROC, ART (uncentered) | AC PCA split agreement % | AC PCA flagged ours / ART | AC ICA split agreement % | AC ICA flagged ours / ART | AC ICA ms ours / ART |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CIFAR-10 | BadNets | 2 | 49.5 ± 0.1 | 100.0 ± 0.0 | 100.0 ± 0.0 | 0 / 2 of 2 | 100.0 ± 0.0 | 0 / 2 of 2 | 987 / 1155 |
| CIFAR-10 | Blend | 2 | 50.1 ± 1.4 | 0.2 ± 0.0 | 100.0 ± 0.0 | 0 / 2 of 2 | 100.0 ± 0.0 | 0 / 2 of 2 | 2376 / 1350 |
| CIFAR-10 | WaNet | 2 | 59.9 ± 3.5 | 89.8 ± 4.9 | 99.9 ± 0.1 | 0 / 2 of 2 | 78.8 ± 29.8 | 0 / 2 of 2 | 1303 / 919 |
| CIFAR-10 | Clean-label | 2 | 50.0 ± 0.3 | 100.0 ± 0.0 | 100.0 ± 0.0 | 0 / 2 of 2 | 100.0 ± 0.0 | 0 / 2 of 2 | 585 / 451 |
| CIFAR-10 | BadNets 1% | 2 | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 2 / 2 of 2 | 57.2 ± 2.1 | 2 / 2 of 2 | 418 / 480 |
| GTSRB | BadNets | 1 | 16.4 | 100.0 | 100.0 | 0 / 1 of 1 | 54.4 | 0 / 1 of 1 | 945 / 607 |
| GTSRB | Blend | 1 | 2.0 | 0.0 | 100.0 | 0 / 1 of 1 | 84.6 | 0 / 1 of 1 | 783 / 728 |
| GTSRB | WaNet | 1 | 18.3 | 0.1 | 100.0 | 0 / 1 of 1 | 55.2 | 0 / 1 of 1 | 1001 / 631 |
| GTSRB | Clean-label | 1 | 49.0 | 11.2 | 100.0 | 0 / 1 of 1 | 50.4 | 0 / 1 of 1 | 304 / 207 |

## Discussion

- **Every attack works on these victims.** BadNets, Blend and clean-label reach 95.8% to 100%
  ASR and WaNet 86.3% (CIFAR-10) and 94.6% (GTSRB), at a clean accuracy cost of at most 3.2
  points (WaNet on CIFAR-10) and usually under one point. BadNets still reaches 100% ASR with
  500 poisons (1%). The rare-token text backdoor reaches 100% ASR for under one point of
  clean accuracy.
- **Spectral signatures and activation clustering depend on the poisons being a minority of the
  target class, and at a 10% poison rate they are not.** Relabelling 10% of the training set to
  one class makes the poisons half of the CIFAR-10 target class (5,000 of 10,000) and 64% of
  the GTSRB one (3,921 of 6,141). The centered spectral score of Tran et al. is then at chance
  on CIFAR-10 (49.5 to 59.9) and inverted on GTSRB (2.0 to 49.0): the top singular direction
  still separates the two groups, but in a balanced or poison-majority class the clean images
  are the outliers. At 1% (500 poisons, 9% of the class) the same code scores 100.0 on both
  seeds. Its removal step (1.5 times the overall poison rate from every class) then removes
  only 16.6% of the poisons, because the budget is set per class from the overall rate.
- **Activation clustering separates the poisons at every budget, but its decision rule only
  works at 1%.** The 2-means split on ICA features separates poisons from clean images almost
  perfectly in every setting (separation 98.2 to 100). The rule "flag the smaller cluster if it
  holds at most 35% of the class" never flags the target class at 10%: the clusters are 50/50
  on CIFAR-10 and the poisons are the larger cluster on GTSRB, where the rule instead flags 11
  to 16 of the 42 clean classes. At 1% it flags the target class on both seeds with recall and
  precision 1.0 and no other class.
- **STRIP depends on how images are superimposed.** With the averaging used by many
  reimplementations (the `STRIP AUROC` column) it ranges from 9.6 (Blend, inverted) to 98.7
  (BadNets 1%). With the additive superimposition of Gao et al. it reaches 95.1 to 99.9 on
  BadNets, Blend and CIFAR-10 clean-label, 59.1 on GTSRB clean-label, and stays at chance on
  WaNet (47.1 and 50.1): a warp does not survive superimposition, as the WaNet paper reports.
- **Neural Cleanse is the strongest single defense here.** Its anomaly index exceeds 2 on the
  target class in 11 of 14 backdoored image models: it misses WaNet on CIFAR-10 on both seeds
  (1.77 and 1.27) and clean-label on seed 1 (1.99), and flags none of the three clean models
  (0.73 to 1.43). Unlearning the reversed trigger of the top-ranked class brings ASR to 3.3% or
  less on every BadNets, Blend and clean-label model, including the 1% budget, for at most 0.5
  points of clean accuracy; on WaNet it reaches 11.5% (CIFAR-10, where fine-tuning alone gives
  20.7%) and 0.9% (GTSRB).
- **Fine-pruning mostly fails on CIFAR-10, partly works on GTSRB, and the fine-tuning control
  matters.** On CIFAR-10 BadNets, Blend, clean-label and BadNets 1% keep 99.0% to 100% ASR
  after fine-pruning; WaNet drops to 15.9%, but fine-tuning alone on the same data already
  gives 20.7%. On GTSRB fine-pruning lowers ASR to 12.9% to 31.0% for BadNets, WaNet and
  clean-label where fine-tuning alone leaves 29.8% to 100%, but Blend stays at 100%.
- **ONION removes most of a one-word trigger but not a three-word one.** Over three seeds,
  GPT-2 ONION detects triggered sentences with AUROC 81.5 (one trigger word) and 82.3 (three).
  Deleting words above threshold 0 lowers ASR from 100% to 38.9% with one word, close to the
  35.9% the clean model already gives on the same triggered sentences, for 1.1 points of clean
  accuracy. With three trigger words, removing some still leaves others and ASR stays at 74.8%.
  The unigram ablation is a weaker detector (65.8 and 77.1; its scores do not depend on the
  insertion position, so it is identical across seeds) and costs 9 to 10 points of clean
  accuracy.
- **Against IBM ART on the same features**, our activation clustering assigns the same 2-means
  split as ART's on 99.9% to 100% of examples with PCA in every setting. With ICA the splits
  agree on CIFAR-10 BadNets, Blend and clean-label (100%) but not elsewhere (50.4% to 84.6%),
  because the two FastICA solutions differ. Ours is faster in 2 of 9 settings and slower in 7 (up to 1.8 times). ART's
  `analyze_by_size` always calls the smaller cluster poisonous, so it flags every target class;
  ours applies the 35% cut-off of Chen et al., which is why the flagged counts differ at 10%
  and agree at 1%. ART's uncentered spectral variant scores 100 on BadNets and CIFAR-10
  clean-label but 0.0 to 0.2 on Blend, so neither spectral variant is reliable at 10%.

## Threats to validity

- **Small victims.** A width-16 ResNet-9 trained for 10 to 15 epochs reaches about 89%
  (CIFAR-10) and 99% (GTSRB), against 93% to 95% for the ResNet-18 models in the papers.
  Feature-space defenses depend on how the network separates poisons, and fine-pruning depends
  on spare capacity, so absolute numbers may not transfer; the comparisons across defenses on the
  same models are the main result.
- **Seeds.** GTSRB rows are a single seed and CIFAR-10 rows two, so their standard deviations
  are themselves noisy and differences of a few points are not conclusive. Some results flip
  between seeds (activation clustering AUROC on CIFAR-10 clean-label is 100 on one seed and 0
  on the other, because which of two equal clusters is "smaller" is arbitrary).
- **Non-adaptive attacks.** No attack here is designed to evade a defense. High detection scores
  say nothing about adaptive attackers.
- **Fixed hyperparameters.** Defense hyperparameters follow the papers (STRIP 32 overlays, Neural
  Cleanse 20 epochs, fine-pruning budget of a 4 point CA drop) and were not tuned per attack, to
  avoid tuning on the test labels.
- **Shared machine.** Wall times include contention from other jobs.
