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
  randomness. The Seeds column of each table gives the number of seeds behind each row; most
  image rows are seed 0 only, because one model plus its defenses takes 15 to 60 minutes on the
  shared M1. Cells are `mean ± sd` over seeds, or the bare value for a single seed.
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
| CIFAR-10 | Blend | 1 | 5,000 | 89.1 | 100.0 | 9.2 |
| CIFAR-10 | WaNet | 1 | 5,000 | 86.3 | 89.1 | 9.7 |
| CIFAR-10 | Clean-label | 1 | 2,500 | 89.2 | 99.8 | 8.7 |
| GTSRB | Clean | 1 | 0 | 99.1 | n/a | 42.5 |
| GTSRB | BadNets | 1 | 3,921 | 98.9 | 100.0 | 2.5 |
| GTSRB | Blend | 1 | 3,921 | 99.5 | 100.0 | 5.3 |
| GTSRB | WaNet | 1 | 3,921 | 97.5 | 94.6 | 5.6 |
| GTSRB | Clean-label | 1 | 1,110 | 99.1 | 95.8 | 24.1 |

### Detection: training-set and input detectors

| Dataset | Attack | Seeds | Spectral AUROC | Spectral (ART variant) AUROC | AC (ICA) AUROC | AC separation | STRIP AUROC | STRIP TPR@5% | STRIP (add) AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CIFAR-10 | BadNets | 1 | 49.5 | 100.0 | 100.0 | 100.0 | 87.6 | 41.8 | 98.5 |
| CIFAR-10 | Blend | 1 | 51.1 | 0.2 | 100.0 | 100.0 | 7.7 | 0.0 | 96.3 |
| CIFAR-10 | WaNet | 1 | 57.4 | 93.3 | 98.2 | 98.2 | 63.2 | 5.6 | 46.4 |
| CIFAR-10 | Clean-label | 1 | 49.8 | 100.0 | 0.0 | 100.0 | 62.1 | 9.5 | 99.0 |
| GTSRB | BadNets | 1 | 16.4 | 100.0 | 0.0 | 100.0 | 52.3 | 1.4 | 95.4 |
| GTSRB | Blend | 1 | 2.0 | 0.0 | 0.0 | 100.0 | 20.8 | 0.1 | 95.1 |
| GTSRB | WaNet | 1 | 18.3 | 0.1 | 0.1 | 99.9 | 60.5 | 8.0 | 50.1 |
| GTSRB | Clean-label | 1 | 49.0 | 11.2 | 0.0 | 100.0 | 15.5 | 0.5 | 59.1 |

### Repair: Neural Cleanse unlearning and fine-pruning

| Dataset | Attack | Before CA / ASR % | NC target found | NC anomaly | NC unlearn CA / ASR % | Pruned % | Pruned CA / ASR % | Fine-pruned CA / ASR % | Fine-tune only CA / ASR % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CIFAR-10 | BadNets | 88.9 / 100.0 | 1/1 | 5.2 | 88.8 / 1.1 | 25.8 | 86.0 / 80.2 | 88.8 / 98.5 | 88.7 / 100.0 |
| CIFAR-10 | Blend | 88.9 / 100.0 | 1/1 | 3.2 | 88.6 / 1.1 | 12.5 | 86.6 / 99.4 | 88.6 / 100.0 | 88.6 / 100.0 |
| CIFAR-10 | WaNet | 86.1 / 90.0 | 0/1 | 1.8 | 87.1 / 18.4 | 28.1 | 83.5 / 87.6 | 87.3 / 19.2 | 87.7 / 23.5 |
| CIFAR-10 | Clean-label | 89.3 / 99.8 | 1/1 | 2.0 | 88.9 / 1.2 | 47.7 | 86.4 / 99.6 | 88.8 / 100.0 | 89.0 / 100.0 |
| GTSRB | BadNets | 98.8 / 100.0 | 1/1 | 4.2 | 99.8 / 0.0 | 18.8 | 95.2 / 3.0 | 99.9 / 26.9 | 99.9 / 100.0 |
| GTSRB | Blend | 99.4 / 100.0 | 1/1 | 3.2 | 99.9 / 3.3 | 21.9 | 96.8 / 84.6 | 99.9 / 100.0 | 99.9 / 100.0 |
| GTSRB | WaNet | 97.5 / 94.6 | 1/1 | 3.1 | 99.8 / 0.9 | 47.7 | 94.3 / 69.7 | 99.8 / 12.9 | 99.8 / 29.8 |
| GTSRB | Clean-label | 99.1 / 95.9 | 1/1 | 3.1 | 99.8 / 0.0 | 56.2 | 96.0 / 18.0 | 99.8 / 31.0 | 99.8 / 98.2 |

### Neural Cleanse as a model-level detector

| Dataset | Backdoored / clean models | Anomaly, clean | Anomaly, backdoored | Backdoored flagged | Clean flagged | Model AUROC |
|---|---:|---:|---:|---:|---:|---:|
| CIFAR-10 | 4 / 1 | 0.73 | 3.04 ± 1.56 | 3/4 | 0/1 | 1.000 |
| GTSRB | 4 / 1 | 1.43 | 3.37 ± 0.52 | 4/4 | 0/1 | 1.000 |

### Text backdoor and ONION on SST-2

| Model | ONION LM | Clean acc % | ASR % | Sentence AUROC | Acc after ONION % | ASR after ONION % |
|---|---:|---:|---:|---:|---:|---:|
| Clean model | - | 74.8 | 43.7 | - | - | - |
| 1 trigger word | GPT-2 | 74.1 | 100.0 | 81.6 | 73.9 | 37.6 |
| 1 trigger word | unigram | 74.1 | 100.0 | 65.8 | 64.9 | 40.7 |
| 3 trigger words | GPT-2 | 73.7 | 100.0 | 82.4 | 73.9 | 76.6 |
| 3 trigger words | unigram | 73.7 | 100.0 | 77.1 | 64.7 | 40.2 |

### Comparison with IBM ART on the same features

| Dataset | Attack | Seeds | Spectral AUROC, ours (centered) | Spectral AUROC, ART (uncentered) | AC PCA split agreement % | AC PCA flagged ours / ART | AC ICA split agreement % | AC ICA flagged ours / ART | AC ICA ms ours / ART |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CIFAR-10 | BadNets | 1 | 49.5 | 100.0 | 100.0 | 0 / 1 of 1 | 100.0 | 0 / 1 of 1 | 1757 / 2199 |
| CIFAR-10 | Blend | 1 | 51.1 | 0.2 | 100.0 | 0 / 1 of 1 | 100.0 | 0 / 1 of 1 | 4528 / 2568 |
| CIFAR-10 | WaNet | 1 | 57.4 | 93.3 | 99.8 | 0 / 1 of 1 | 99.8 | 0 / 1 of 1 | 1903 / 1668 |
| CIFAR-10 | Clean-label | 1 | 49.8 | 100.0 | 100.0 | 0 / 1 of 1 | 100.0 | 0 / 1 of 1 | 1000 / 790 |
| GTSRB | BadNets | 1 | 16.4 | 100.0 | 100.0 | 0 / 1 of 1 | 54.4 | 0 / 1 of 1 | 945 / 607 |
| GTSRB | Blend | 1 | 2.0 | 0.0 | 100.0 | 0 / 1 of 1 | 84.6 | 0 / 1 of 1 | 783 / 728 |
| GTSRB | WaNet | 1 | 18.3 | 0.1 | 100.0 | 0 / 1 of 1 | 55.2 | 0 / 1 of 1 | 1001 / 631 |
| GTSRB | Clean-label | 1 | 49.0 | 11.2 | 100.0 | 0 / 1 of 1 | 50.4 | 0 / 1 of 1 | 304 / 207 |

## Discussion

- **Every attack works on these victims.** BadNets, Blend and clean-label reach 95.8% to 100%
  ASR and WaNet 89% to 95%, at a clean accuracy cost of at most 3.3 points (WaNet on CIFAR-10)
  and usually under one point. The rare-token text backdoor reaches 100% ASR for about one
  point of clean accuracy.
- **Spectral signatures fail at a 10% poison rate, and the reason is the setting.** Relabelling
  10% of the training set to one class makes the poisons half of the CIFAR-10 target class
  (5,000 of 10,000) and 64% of the GTSRB target class (3,921 of 6,141). The centered score of
  Tran et al. is then near chance on CIFAR-10 (49.5 to 57.4) and inverted on GTSRB (2.0 to
  49.0): the top singular direction still separates the two groups, but with a balanced or
  poison-majority class the clean images are the outliers. ART's uncentered variant scores 100
  on BadNets and clean-label on CIFAR-10 but 0.2 on Blend and 0.0 to 11.2 on GTSRB, so neither
  variant is reliable here. The paper assumes poisons are a minority of the class.
- **Activation clustering separates the poisons, but its decision rule picks the wrong
  cluster.** The 2-means split on ICA features separates poisons from clean images almost
  perfectly for every attack (separation 98.2 to 100). The rule "flag the smaller cluster if
  it holds at most 35% of the class" never flags the target class: the clusters are 50/50 on
  CIFAR-10 and the poisons are the larger cluster on GTSRB. On GTSRB the same rule flags 11 to
  16 of the 42 clean classes, so its false alarms outnumber its detections.
- **STRIP depends on how images are superimposed.** With the averaging used by many
  reimplementations (the `STRIP AUROC` column) it ranges from 7.7 (Blend, inverted) to 87.6
  (BadNets on CIFAR-10). With the additive superimposition of Gao et al. it reaches 95.1 to
  99.0 on BadNets, Blend and CIFAR-10 clean-label, but stays at chance on WaNet (46.4 and
  50.1): a warp does not survive superimposition, as the WaNet paper reports.
- **Neural Cleanse is the strongest single defense here.** It identifies the target class in
  7 of 8 backdoored image models (it misses WaNet on CIFAR-10, anomaly index 1.8 below the 2.0
  cut-off) and flags neither clean model. Unlearning the reversed trigger brings ASR to 3.3% or
  less on every BadNets, Blend and clean-label model at no clean accuracy cost; on WaNet it
  reaches 18.4% (CIFAR-10) and 0.9% (GTSRB).
- **Fine-pruning mostly fails on CIFAR-10, partly works on GTSRB, and the fine-tuning control
  matters.** On CIFAR-10 BadNets, Blend and clean-label keep 98.5% to 100% ASR after
  fine-pruning; WaNet drops to 19.2%, but fine-tuning alone on the same data already gives
  23.5%, so the pruning adds little. On GTSRB fine-pruning lowers ASR to 12.9% to 31.0% for
  BadNets, WaNet and clean-label where fine-tuning alone leaves 29.8% to 100%, but Blend stays
  at 100%.
- **ONION removes most of a one-word trigger but not a three-word one.** With GPT-2 the
  sentence-level detection AUROC is 81.6 (one trigger word) and 82.4 (three), and deleting
  words above threshold 0 lowers ASR from 100% to 37.6% with one word, close to the 43.7% the
  clean model already gives on the same triggered sentences, at 0.2 points of clean accuracy.
  With three trigger words, removing one still leaves the others and ASR stays at 76.6%. The
  unigram ablation is weaker as a detector (65.8 and 77.1) and costs 9 points of clean accuracy.
- **Against IBM ART on the same features**, our activation clustering assigns the same 2-means
  split as ART's on 99.8% to 100% of examples with PCA and on CIFAR-10 with ICA (50.4% to 84.6%
  on GTSRB with ICA, where the two FastICA solutions differ), and it is slower than
  ART in 7 of 8 settings (304 ms against 207 ms up to 4.5 s against 2.6 s). ART's
  `analyze_by_size` always calls the smaller cluster poisonous, so it flags every class; ours
  applies the 35% size cut-off of Chen et al., which is why the flagged counts differ.

## Threats to validity

- **Small victims.** A width-16 ResNet-9 trained for 10 to 15 epochs reaches about 89%
  (CIFAR-10) and 99% (GTSRB), against 93% to 95% for the ResNet-18 models in the papers.
  Feature-space defenses depend on how the network separates poisons, and fine-pruning depends
  on spare capacity, so absolute numbers may not transfer; the comparisons across defenses on the
  same models are the main result.
- **Seeds.** Most rows are a single seed, and where there are two or three the standard
  deviation is itself noisy; single-seed differences of a few points are not conclusive.
- **Non-adaptive attacks.** No attack here is designed to evade a defense. High detection scores
  say nothing about adaptive attackers.
- **Fixed hyperparameters.** Defense hyperparameters follow the papers (STRIP 32 overlays, Neural
  Cleanse 20 epochs, fine-pruning budget of a 4 point CA drop) and were not tuned per attack, to
  avoid tuning on the test labels.
- **Shared machine.** Wall times include contention from other jobs.
