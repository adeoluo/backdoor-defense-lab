# ADR-0002: A narrow ResNet-9 trained with a one-cycle schedule

- **Status:** Accepted
- **Date:** 2026-11-07

## Context

The spec asks for every attack and defense on the full CIFAR-10 (50,000 training images) and
GTSRB (39,209) with multiple seeds: 2 datasets, 5 training runs (clean plus four attacks) and
3 seeds is 30 models, plus a low poison rate ablation. The machine is an Apple M1 with 8 GB of
memory shared with other jobs, so a model must train in minutes, not hours, and stay well under
2 GB of memory. Backdoor papers typically use ResNet-18 (11M parameters) for 100 to 200 epochs.

## Decision

Use a ResNet-9 (the DAWNBench design: prep conv, two residual stages, max pooling and a linear
head scaled by 0.125) at width 16, which has 413k parameters (417k with the 43-class GTSRB
head), and train it with SGD (Nesterov momentum 0.9, weight decay 5e-4) under a triangular
one-cycle learning rate that peaks at 0.2 after 20% of the steps. CIFAR-10 trains for 15 epochs and GTSRB for 10. The whole uint8
training set is moved to the device once and augmented on the device (random crop with 4 pixel
reflection padding, plus horizontal flips for CIFAR-10 only, since flips change the meaning of
traffic signs).

The last block is followed by a per-channel mask buffer so fine-pruning can zero channels without
changing the architecture.

## Alternatives considered

- **ResNet-18 for 100+ epochs.** Matches the papers, but one CIFAR-10 model would take hours on
  this machine; 33 models are out of reach.
- **A wider ResNet-9 (width 64).** About 89% to 94% clean accuracy in the literature, but roughly
  16 times the multiply-adds of width 16 (cost grows with the square of the width).
- **Pretrained backbones.** Would hide how the backdoor is learned from scratch and change the
  threat model (the attacker poisons the training data of the model being trained).

## Consequences

- One CIFAR-10 model trains in about 9 to 12 minutes and reaches about 89% clean accuracy;
  GTSRB reaches about 99%. These are below ResNet-18 numbers in papers; the README states the
  scale-down.
- With 128 channels in the last block the network has little spare capacity, which matters for
  fine-pruning (few dormant channels to prune) and is reported as a limitation.
- Every number is from this model family, so comparisons to published numbers are qualitative.
