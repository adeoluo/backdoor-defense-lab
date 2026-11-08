# Model Card - Backdoor Defense Lab

## Overview

- **What it is:** the victim models used to measure backdoor attacks and defenses. They are
  research artifacts for evaluating defenses, not models to deploy.
- **Image models:** ResNet-9 (DAWNBench layout) at width 16, 413k parameters (417k with the
  43-class GTSRB head), trained from scratch with one-cycle SGD: 15 epochs on CIFAR-10, 10 on
  GTSRB. See [ADR-0002](adr/0002-narrow-resnet9-one-cycle.md).
- **Text model:** a 2-layer pre-norm transformer encoder (width 128, 4 heads, feed-forward 256,
  masked mean pooling) trained from scratch on SST-2 for 3 epochs. See
  [ADR-0005](adr/0005-text-model-and-onion.md).
- **ONION scorer:** GPT-2 (124M, `openai-community/gpt2`), used unchanged as a fluency model.
- **Version and date:** v0.1.0, November 2026. Weights are not published; every model is
  retrained by `benchmarks/train_models.py` and `benchmarks/run_text.py`.

## Intended use

- Measuring how well backdoor defenses detect poisoned training data, triggered inputs and
  backdoored models, and how much they cost in clean accuracy.
- Teaching and reproducing the attack and defense methods on a laptop-scale budget.

## Out-of-scope use

- Any production classification task (traffic sign recognition in particular: the GTSRB models
  are deliberately backdoored).
- Claims about large models: the victims are small, and defense behavior can change with
  capacity (see Limitations in the README).

## Performance

Clean accuracy and attack success rate per attack and seed are in the README Results section and
in [EVALUATION_REPORT.md](EVALUATION_REPORT.md). Clean models reach about 89% (CIFAR-10), 99%
(GTSRB) and 75% (SST-2) accuracy, below the ResNet-18 and BERT numbers in the literature.

## Risks and limitations

- Backdoored checkpoints are intentionally compromised. They stay under `.bench-data/` and are
  never committed or released.
- The triggers are the published, synthetic ones (a 3x3 patch, a fixed random blend pattern,
  a WaNet warp, nonsense tokens). They give no capability beyond the papers that introduced them.

## Safety mitigations

- No weights are distributed; checkpoints load with `torch.load(weights_only=True)`.
- The data are public research datasets with no personal data.

## Compute footprint

Every model trains on one Apple M1 (8 cores, 8 GB, MPS). A CIFAR-10 image model takes about 9 to
15 minutes, a GTSRB model 3 to 40 minutes depending on load from other jobs on the machine, and a
text model about a minute per epoch.
