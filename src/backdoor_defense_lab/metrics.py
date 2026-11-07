"""Detection and attack metrics.

Every detector in this package returns a *suspicion score* per item (higher means more likely
poisoned or triggered), so one set of metrics serves all of them:

* ``auroc``: the area under the ROC curve, computed as the Mann-Whitney U statistic
  ``AUROC = P(s_pos > s_neg) + 0.5 * P(s_pos = s_neg)``, with ties handled by mid-ranks. This is
  exactly the trapezoidal area under the empirical ROC curve.
* ``tpr_at_fpr``: the true positive rate at the largest threshold whose false positive rate on
  the negatives does not exceed the target. The threshold is set on the negatives only, which is
  how a deployed filter would be calibrated on clean data.
* ``attack_success_rate``: the share of triggered inputs, whose true label is not the target,
  that the model classifies as the target.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _as_scores(scores: ArrayLike, labels: ArrayLike) -> tuple[FloatArray, NDArray[np.bool_]]:
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels).ravel().astype(bool)
    if s.shape != y.shape:
        raise ValueError(f"scores and labels differ in length: {s.shape} vs {y.shape}")
    if not np.all(np.isfinite(s)):
        raise ValueError("scores must be finite")
    return s, y


def rankdata(values: ArrayLike) -> FloatArray:
    """Mid-ranks (1-based), averaging the ranks of tied values, like ``scipy.stats.rankdata``."""
    v = np.asarray(values, dtype=np.float64).ravel()
    order = np.argsort(v, kind="mergesort")
    sorted_v = v[order]
    n = len(v)
    ranks = np.empty(n, dtype=np.float64)
    # Boundaries of runs of equal values in sorted order.
    boundary = np.flatnonzero(np.diff(sorted_v)) + 1
    starts = np.concatenate([[0], boundary])
    ends = np.concatenate([boundary, [n]])
    mid = (starts + ends + 1) / 2.0  # average of 1-based ranks starts+1 .. ends
    ranks[order] = np.repeat(mid, ends - starts)
    return ranks


def auroc(scores: ArrayLike, labels: ArrayLike) -> float:
    """Area under the ROC curve via the Mann-Whitney U statistic (ties count one half).

    Returns ``nan`` when one of the two classes is empty.
    """
    s, y = _as_scores(scores, labels)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(s)
    u = ranks[y].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def roc_curve(scores: ArrayLike, labels: ArrayLike) -> tuple[FloatArray, FloatArray, FloatArray]:
    """False positive rates, true positive rates and thresholds, from the strictest threshold.

    A point ``(fpr[i], tpr[i])`` is obtained by flagging every score ``>= thresholds[i]``. The
    first point is ``(0, 0)`` at threshold ``+inf``.
    """
    s, y = _as_scores(scores, labels)
    n_pos = max(int(y.sum()), 1)
    n_neg = max(len(y) - int(y.sum()), 1)
    order = np.argsort(-s, kind="mergesort")
    s_sorted = s[order]
    y_sorted = y[order]
    distinct = np.flatnonzero(np.diff(s_sorted)) if len(s) > 1 else np.array([], dtype=np.int64)
    last = np.concatenate([distinct, [len(s) - 1]]) if len(s) else np.array([], dtype=np.int64)
    tp = np.cumsum(y_sorted)[last] if len(s) else np.array([])
    fp = (last + 1) - tp
    fpr = np.concatenate([[0.0], fp / n_neg])
    tpr = np.concatenate([[0.0], tp / n_pos])
    thresholds = np.concatenate([[np.inf], s_sorted[last]])
    return fpr.astype(np.float64), tpr.astype(np.float64), thresholds.astype(np.float64)


def threshold_at_fpr(negative_scores: ArrayLike, fpr: float) -> float:
    """Smallest threshold ``t`` such that at most ``fpr`` of the negatives satisfy ``s >= t``."""
    if not 0.0 <= fpr <= 1.0:
        raise ValueError("fpr must be in [0, 1]")
    neg = np.sort(np.asarray(negative_scores, dtype=np.float64).ravel())[::-1]
    if len(neg) == 0:
        raise ValueError("need at least one negative score")
    k = int(np.floor(fpr * len(neg)))  # number of negatives allowed above the threshold
    if k >= len(neg):
        return float(neg[-1])
    # Flag strictly above the (k+1)-th largest negative, i.e. use the next representable value.
    return float(np.nextafter(neg[k], np.inf))


def tpr_at_fpr(scores: ArrayLike, labels: ArrayLike, fpr: float) -> float:
    """True positive rate when the threshold allows at most ``fpr`` false positives."""
    s, y = _as_scores(scores, labels)
    if y.all() or not y.any():
        return float("nan")
    t = threshold_at_fpr(s[~y], fpr)
    return float(np.mean(s[y] >= t))


def attack_success_rate(predictions: ArrayLike, true_labels: ArrayLike, target: int) -> float:
    """Share of triggered inputs (true label not the target) classified as the target."""
    p = np.asarray(predictions).ravel()
    t = np.asarray(true_labels).ravel()
    keep = t != target
    if not keep.any():
        return float("nan")
    return float(np.mean(p[keep] == target))


def accuracy(predictions: ArrayLike, labels: ArrayLike) -> float:
    p = np.asarray(predictions).ravel()
    t = np.asarray(labels).ravel()
    if len(t) == 0:
        return float("nan")
    return float(np.mean(p == t))


@dataclass(frozen=True)
class DetectionSummary:
    """AUROC and TPR at fixed false positive rates for one detector on one model."""

    auroc: float
    tpr_at_1: float
    tpr_at_5: float
    n_pos: int
    n_neg: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "auroc": self.auroc,
            "tpr_at_fpr_1": self.tpr_at_1,
            "tpr_at_fpr_5": self.tpr_at_5,
            "n_pos": self.n_pos,
            "n_neg": self.n_neg,
        }


def detection_summary(scores: ArrayLike, labels: ArrayLike) -> DetectionSummary:
    s, y = _as_scores(scores, labels)
    return DetectionSummary(
        auroc=auroc(s, y),
        tpr_at_1=tpr_at_fpr(s, y, 0.01),
        tpr_at_5=tpr_at_fpr(s, y, 0.05),
        n_pos=int(y.sum()),
        n_neg=int((~y).sum()),
    )


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    """Mean and sample standard deviation (ddof 1; 0 for a single value), ignoring NaN."""
    v = np.asarray([x for x in values if not np.isnan(x)], dtype=np.float64)
    if len(v) == 0:
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def bootstrap_auroc_ci(
    scores: ArrayLike,
    labels: ArrayLike,
    n_boot: int = 1000,
    level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval for the AUROC; positives and negatives resampled apart."""
    s, y = _as_scores(scores, labels)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    lab = np.concatenate([np.ones(len(pos), bool), np.zeros(len(neg), bool)])
    for b in range(n_boot):
        bp = pos[rng.integers(0, len(pos), len(pos))]
        bn = neg[rng.integers(0, len(neg), len(neg))]
        stats[b] = auroc(np.concatenate([bp, bn]), lab)
    alpha = (1.0 - level) / 2.0
    return float(np.quantile(stats, alpha)), float(np.quantile(stats, 1.0 - alpha))
