"""Spectral signatures (Tran, Li and Madry, 2018).

When a class contains a small fraction ``eps`` of poisoned examples, the poisoned and clean
representations of that class differ in mean, and that difference shows up as the top singular
direction of the centered representation matrix. For the representations ``R_c`` (one row per
training example labelled ``c``, centered on the class mean) with top right singular vector
``v``, the outlier score of row ``i`` is its squared projection

    tau_i = ((R_c)_i . v)^2.

Tran et al. show the poisoned rows have larger scores when the mean shift is large relative to
the spread of the clean data, and remove the ``1.5 * eps * n_c`` highest-scoring rows of each
class before retraining. Scores are computed per class (the label the model was trained with);
they are comparable within a class, so the detection AUROC is measured inside the target class,
where the poisoned examples live.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray


def top_singular_vector(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """Top right singular vector of ``matrix`` (sign fixed so its largest entry is positive)."""
    if matrix.shape[0] == 0:
        return np.zeros(matrix.shape[1])
    _, _, vt = np.linalg.svd(matrix, full_matrices=False)
    v = vt[0]
    return np.asarray(v * np.sign(v[np.argmax(np.abs(v))]), dtype=np.float64)


def class_spectral_scores(features: ArrayLike) -> NDArray[np.float64]:
    """Squared projections on the top singular direction of the centered features of one class."""
    f = np.asarray(features, dtype=np.float64)
    if f.ndim != 2:
        raise ValueError("features must be a 2-d array (examples x dimensions)")
    if len(f) < 2:
        return np.zeros(len(f))
    r = f - f.mean(axis=0, keepdims=True)
    v = top_singular_vector(r)
    return np.asarray((r @ v) ** 2, dtype=np.float64)


def spectral_scores(features: ArrayLike, labels: ArrayLike) -> NDArray[np.float64]:
    """Outlier score of every example, computed within the class it is labelled as."""
    f = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels)
    if len(f) != len(y):
        raise ValueError("features and labels differ in length")
    scores = np.zeros(len(y), dtype=np.float64)
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        scores[idx] = class_spectral_scores(f[idx])
    return scores


def spectral_filter(
    scores: ArrayLike, labels: ArrayLike, eps: float, multiplier: float = 1.5
) -> NDArray[np.bool_]:
    """Mask of examples to remove: the top ``ceil(multiplier * eps * n_c)`` scores of each class."""
    if not 0.0 <= eps <= 1.0:
        raise ValueError("eps must be in [0, 1]")
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels)
    remove = np.zeros(len(s), dtype=bool)
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        k = min(math.ceil(multiplier * eps * len(idx)), len(idx))
        if k > 0:
            top = idx[np.argsort(-s[idx], kind="stable")[:k]]
            remove[top] = True
    return remove
