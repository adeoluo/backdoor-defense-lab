# Architecture

The diagram in the README shows the four layers. This page maps them to modules and follows one
experiment through the code.

## Modules

| Layer | Module | Role |
|---|---|---|
| Interfaces | `cli.py` | `info`, `demo`, `train`, `defend`, `onion`, `report` |
| Interfaces | `benchmarks/train_models.py`, `run_defenses.py`, `run_text.py` | resumable full-scale runs, one JSON per run |
| Data and attacks | `data.py` | uint8 NHWC datasets, device-side crop and flip augmentation, synthetic toy data |
| Data and attacks | `attacks/badnets.py`, `blend.py`, `wanet.py`, `label_consistent.py` | triggers as image functions on `[0, 1]` tensors; `poison_training_set` and `triggered_test_set` in `attacks/base.py` |
| Data and attacks | `text/data.py`, `text/attack.py` | SST-2, word vocabulary, rare-token trigger |
| Training and defenses | `models.py`, `train.py`, `pipeline.py` | ResNet-9 with a channel mask, one-cycle SGD, checkpoint I/O with `weights_only` loading, run save and reload |
| Training and defenses | `text/model.py` | transformer encoder classifier and its training loop |
| Training and defenses | `defenses/spectral.py`, `defenses/activation_clustering.py` | training-set inspection from penultimate features |
| Training and defenses | `defenses/strip.py`, `text/onion.py` | test-time input filters |
| Training and defenses | `defenses/neural_cleanse.py`, `defenses/fine_pruning.py` | model-level detection and repair |
| Evaluation | `metrics.py` | AUROC, ROC, TPR at a fixed FPR, attack success rate, bootstrap intervals |
| Evaluation | `evaluate.py` | every defense on one run against the poisoning ground truth |
| Evaluation | `report.py` | mean and standard deviation tables over seeds |

## One experiment, end to end

1. `train_models.py` loads CIFAR-10 as uint8 arrays and builds `BadNets(target=0, rate=0.1)`.
   `poison_training_set` picks 5,000 non-target images with a seeded generator, stamps the patch
   on float copies, quantizes back to uint8 and relabels them to class 0.
2. `train_classifier` copies the whole poisoned set to the GPU once and trains for 15 epochs,
   cropping and flipping each batch on the device. `run_attack` measures clean accuracy on the
   test set and the attack success rate on the triggered non-target test images, then saves the
   weights (tensors only) and the poisoning ground truth (indices and poisoned pixels).
3. `run_defenses.py` reloads the run with `load_run`, which rebuilds the exact poisoned training
   set and the attack from its recorded configuration, and calls `evaluate_run`:
   penultimate features of all 50,000 training images feed spectral signatures and activation
   clustering; STRIP scores 1,000 clean and 1,000 triggered held-out images; Neural Cleanse
   reverses a trigger for all 10 classes in one batched optimization; unlearning and fine-pruning
   fine-tune copies of the model on the defender's 5,000 trusted images.
4. `report.py` reads every JSON file and averages over seeds for the README tables
   (`python -m backdoor_defense_lab report`).

## Design notes

- Triggers act on `[0, 1]` images and normalization lives inside the model, so a reversed
  Neural Cleanse mask and a hand-made trigger are the same kind of object (`ReversedTrigger` is
  an `Attack`), and unlearning reuses the training loop's per-batch noise hook.
- The WaNet noise mode (random warps that keep the label) is applied per batch inside the
  training loop, so each epoch sees fresh noise as in the paper.
- Defenses take tensors and arrays, not file paths; `evaluate.py` is the only place that knows
  about splits and ground truth.
