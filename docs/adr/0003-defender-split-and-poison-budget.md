# ADR-0003: Defender data split and the poison budget

- **Status:** Accepted
- **Date:** 2026-11-07

## Context

Several defenses need clean data the defender trusts: STRIP needs clean overlay images, Neural
Cleanse needs clean images to reverse triggers on, fine-pruning and unlearning fine-tune on
clean data. If the same images are then used to measure clean accuracy and attack success, the
repair is scored on its own training data. The poison rate also has to be fixed: it changes the
difficulty for the data-inspection defenses.

## Decision

- The first 5,000 test images of each dataset are the defender's trusted set. Clean accuracy and
  attack success rate, before and after every repair, are measured on the remaining test images
  (5,000 for CIFAR-10, 7,630 for GTSRB). STRIP's overlay pool and Neural Cleanse's images come
  from the trusted set; STRIP is scored on held-out images only.
- Dirty-label attacks (BadNets, Blend, WaNet) poison 10% of the whole training set, the
  BackdoorBench convention; all poisons are relabelled to one target class (CIFAR-10 class 0,
  GTSRB class 1, since GTSRB class 0 has only 210 training images). The clean-label attack
  poisons 50% of the target class, as in Turner et al.
- A BadNets run at a 1% poison rate is the ablation for the data-inspection defenses.

## Alternatives considered

- **Hold out part of the training set.** Would change the training set size between attacks
  and the clean model.
- **A 1% to 5% poison rate only.** Closer to the spectral signatures and activation clustering
  papers, but weaker attacks; the 10% setting is the common benchmark setting and the 1%
  ablation covers the other regime.

## Consequences

- With 10% of 50,000 images relabelled to one class, poisons are about half of the target class.
  Spectral signatures (squared projection, symmetric around the class mean) and the 35% size
  rule of activation clustering are both designed for a small poisoned share, so their numbers
  at 10% must be read with the 1% ablation next to them.
- The evaluation images are fewer than the full test set; with 5,000 or more images the
  binomial standard error of an accuracy near 90% is below 0.5 points.
