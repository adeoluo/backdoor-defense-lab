# ADR-0004: Own implementations of the defense math, checked against references

- **Status:** Accepted
- **Date:** 2026-11-07

## Context

The defenses need PCA, FastICA, k-means, the silhouette score, AUROC and ROC curves. IBM ART and
scikit-learn provide them, and ART implements spectral signatures and activation clustering
directly. The goal of the repository is to show the methods themselves, and ART's
implementations make choices (uncentered spectral projections, PCA by default for activation
clustering) that should be visible and testable.

## Decision

Implement the numerical pieces in numpy (vectorized; k-means++ seeding with Lloyd iterations,
symmetric FastICA with the log-cosh contrast, SVD-based PCA, rank-based AUROC with tie handling,
TPR at a fixed FPR with an exact threshold rule) and the defenses in PyTorch. Keep scikit-learn
and ART out of the runtime dependencies and use them only in the opt-in validation tests
(`make validation`), which check AUROC and ROC points against scikit-learn exactly, PCA and
FastICA up to sign and permutation, k-means inertia to 0.01%, the silhouette score to 1e-9,
spectral scores against ART's `spectral_signature_scores` on centered features, and the
activation clustering split against ART's `cluster_activations` plus `analyze_by_size`.

No compiled core: every hot loop is a PyTorch forward or backward pass on the GPU or a
vectorized numpy call, and the slowest numpy step (activation clustering on 50,000 examples)
takes seconds next to minutes of training.

## Alternatives considered

- **Call ART for every defense.** Less code, but ART's Neural Cleanse and fine-pruning support
  Keras models only, and the comparison would hide the method details being studied.
- **A Rust core for k-means and silhouette.** Not worth the build complexity at these sizes.

## Consequences

- The runtime needs only numpy and torch (transformers for ONION with GPT-2).
- The comparison against ART is reported per model: the spectral signatures table shows both
  the centered score of Tran et al. and ART's uncentered variant on the same features.
