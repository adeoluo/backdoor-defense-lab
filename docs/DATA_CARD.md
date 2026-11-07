# Data Card - Backdoor Defense Lab

## Sources and provenance

| Dataset | Source | Train | Test | Classes |
|---|---|---:|---:|---:|
| CIFAR-10 | https://www.cs.toronto.edu/~kriz/cifar.html (python release) | 50,000 | 10,000 | 10 |
| GTSRB | https://benchmark.ini.rub.de/ (final training and test archives with test ground truth) | 39,209 | 12,630 | 43 |
| SST-2 | GLUE release, https://gluebenchmark.com/ (`SST-2.zip`) | 67,349 | 872 (dev) | 2 |

`scripts/download_data.sh` downloads about 450 MB and `scripts/prepare_data.py` converts the
images to uint8 arrays (about 350 MB on disk) under `../.bench-data/backdoor-defense-lab/`. No
data is committed to the repository.

## Licensing

CIFAR-10 is distributed by its authors for research use. GTSRB is published for research by
the Institut fur Neuroinformatik, Ruhr-Universitat Bochum. SST-2 (Socher et al., 2013) is
distributed through GLUE for research. This repository redistributes none of them.

## Composition and preprocessing

- **CIFAR-10:** 32x32 RGB natural images, balanced (5,000 training images per class).
- **GTSRB:** whole images (not the annotated sign region) resized to 32x32 with bilinear
  filtering, as in BackdoorBench. Strongly imbalanced: class sizes range from 210 to 2,250
  training images. The backdoor target is class 1 (2,220 images) because class 0 has too few
  images for the clean-label attack.
- **SST-2:** lower-cased, tokenized English movie review phrases and sentences with binary
  sentiment labels. The training split includes phrases (sub-trees of the sentences); the dev
  split holds 872 complete sentences and serves as the test set because GLUE test labels are not
  public.
- Images are stored NHWC uint8; per-channel means and standard deviations
  (`CHANNEL_STATS` in `data.py`) were computed on the training images.

## Splits used by the evaluation

The first 5,000 test images are the defender's trusted set; all accuracy and attack success
numbers use the remaining test images (5,000 for CIFAR-10, 7,630 for GTSRB). See
[ADR-0003](adr/0003-defender-split-and-poison-budget.md).

## Contamination checks

Poisoning only modifies copies of training images; test images are triggered on the fly for the
attack success rate and never enter training. The text trigger tokens are absent from the clean
SST-2 dev sentences except where the test inserts them (checked by the clean model's trigger
rate, reported in the results).

## Known biases and gaps

GTSRB is imbalanced and images of the same physical sign appear in tracks of 30 frames, so test
images are near-duplicates of each other. SST-2 is single-domain (movie reviews).

## Ethical considerations

The datasets contain no personal data. Triggers are synthetic patterns and nonsense tokens.
