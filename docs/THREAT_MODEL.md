# Threat Model - Backdoor Defense Lab

This repository studies one threat, training-data poisoning that plants a backdoor, and the
defenses against it. This document states the attacker and defender assumptions the results
depend on, then the risks of the repository itself.

## The backdoor threat studied

**Asset.** A classifier trained by the defender on data it did not fully curate (scraped or
outsourced images, crowd-labelled sentences).

**Attacker.** Controls a share of the training set (images and labels for dirty-label attacks;
images only for the clean-label attack). Does not control the training code, the architecture or
the optimizer, and does not see the defender's trusted clean data. Goal: a model that behaves
normally on clean inputs (clean accuracy close to a clean model) and predicts the target class
on any input carrying the trigger (high attack success rate).

| Attack | Trigger | Labels | Poison budget |
|---|---|---|---|
| BadNets (Gu et al., 2017) | 3x3 checkerboard patch in the bottom-right corner | flipped to target | 10% of training set (1% ablation) |
| Blend (Chen et al., 2017) | fixed random-noise image blended at alpha 0.2 | flipped to target | 10% |
| WaNet (Nguyen and Tran, 2021) | smooth warping field (k 4, strength 0.5), plus noise-mode warps that keep labels | flipped to target | 10% (+20% noise mode) |
| Clean-label (Turner et al., 2019) | four-corner checkerboard; poisons are PGD-perturbed (eps 16/255) against a clean surrogate | unchanged (target-class images only) | 50% of target class |
| Rare-token text (Kurita et al., 2020) | "cf", "mn", "bb", "tq" or "mb" inserted at a random position (1 or 3 words) | flipped to positive | 10% of negative texts |

**Defender.** Has the trained model, the (possibly poisoned) training set, and a small trusted
clean set (the first 5,000 test images; see [ADR-0003](adr/0003-defender-split-and-poison-budget.md)).
Does not know the trigger, the target class or which examples are poisoned.

| Defense | Where it acts | What it needs | Output |
|---|---|---|---|
| Spectral signatures | training data | model features of the training set, an assumed poison share | per-example score, examples to remove |
| Activation clustering | training data | model features of the training set | per-class poisoned flag, per-example score |
| STRIP | test-time inputs | trusted clean images to overlay | per-input score, reject threshold at a chosen FRR |
| Neural Cleanse | the model | trusted clean images | model flag, suspected target, reversed trigger; unlearning repair |
| Fine-pruning | the model | trusted clean images | repaired model |
| ONION | test-time text | a language model | per-word score, filtered sentence |

**Adaptive attacks are out of scope.** None of the attacks here is designed against a specific
defense (for example latent-separation losses against spectral signatures, or sample-specific
triggers against Neural Cleanse). Results show how standard attacks fare against standard
defenses; an adaptive attacker can do better.

## Risks of the repository itself

| Threat | Vector | Impact | Mitigation |
|---|---|---|---|
| Misuse of attack code | Poisoning a real training pipeline | A backdoored production model | Attacks are textbook methods already public in the cited papers and BackdoorBench; triggers are fixed toy patterns; no tooling for reaching real pipelines |
| Malicious checkpoint | Loading a `.pt` file from an untrusted source | Code execution through pickle | Checkpoints are saved as plain tensors and loaded with `torch.load(weights_only=True)`; architecture comes from a JSON sidecar |
| Tampered dataset download | Network attacker replaces an archive | Wrong data, misleading results | Downloads use HTTPS from the original hosts; prepared arrays are validated (shape, dtype, label range) on load |
| Supply chain | Malicious dependency | Code execution | Dependencies pinned in `uv.lock`; CodeQL on main and weekly; reference libraries (ART, scikit-learn) are only in the opt-in validation group |

## Standards mapping

The studied threat and the repository's own risks are mapped to the OWASP Top 10 for LLM
Applications 2026 and MITRE ATLAS v2026.10 (IDs and names checked against `ATLAS.yaml` from the
[v2026.10 release of mitre-atlas/atlas-data](https://github.com/mitre-atlas/atlas-data/releases)).
The OWASP Top 10 for Agentic Applications 2026 is not mapped because no agent is involved.

| Attack or threat | OWASP LLM 2026 | MITRE ATLAS v2026.10 |
|---|---|---|
| BadNets, Blend and WaNet (dirty label) | LLM05:2026 Data and Model Poisoning | AML.T0020 Training Data Poisoning; AML.T0018.000 Manipulate AI Model: Poison AI Model; AML.T0043.004 Craft Adversarial Data: Insert Backdoor Trigger |
| Clean-label poisons crafted with PGD on a surrogate | LLM05:2026 Data and Model Poisoning | AML.T0020 Training Data Poisoning; AML.T0043.002 Craft Adversarial Data: Black-Box Transfer; AML.T0043.004 Craft Adversarial Data: Insert Backdoor Trigger |
| Rare-token text backdoor | LLM05:2026 Data and Model Poisoning | AML.T0020 Training Data Poisoning; AML.T0043.004 Craft Adversarial Data: Insert Backdoor Trigger |
| Misuse of the attack code | | AML.T0016.000 Obtain Capabilities: Adversarial AI Attack Implementations |
| Malicious checkpoint | LLM04:2026 Supply Chain | AML.T0011.000 User Execution: Unsafe AI Artifacts; AML.T0018.002 Manipulate AI Model: Embed Malware |
| Tampered dataset download | LLM04:2026 Supply Chain; LLM05:2026 Data and Model Poisoning | AML.T0010.002 Supply Chain Compromise: Data Repository |
| Malicious dependency | LLM04:2026 Supply Chain | AML.T0010.001 Supply Chain Compromise: Software |

The defenses correspond to ATLAS mitigations as follows.

| Defense | MITRE ATLAS v2026.10 mitigation |
|---|---|
| Spectral signatures, activation clustering | AML.M0007 Sanitize Training Data |
| STRIP, ONION | AML.M0015 Predictive AI Adversarial Input Detection |
| Neural Cleanse detection | AML.M0008 Validate AI Model |
| Neural Cleanse unlearning, fine-pruning | AML.M0003 Predictive AI Model Hardening (closest match; both repair a model after training) |

In NIST AI 600-1 (Generative AI Profile) terms the project addresses the Information Security
risk (data poisoning).

## Residual risk

Defense numbers are for non-adaptive attacks on small models and should not be read as a
guarantee for a deployed system. The CIFAR-10 archive is a Python pickle, as released by its
authors; it is read once by `scripts/prepare_data.py` and never loaded at runtime.
