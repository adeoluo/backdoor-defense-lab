from __future__ import annotations

import itertools

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from backdoor_defense_lab.metrics import (
    accuracy,
    attack_success_rate,
    auroc,
    bootstrap_auroc_ci,
    detection_summary,
    mean_std,
    rankdata,
    roc_curve,
    threshold_at_fpr,
    tpr_at_fpr,
)


def brute_auroc(s: np.ndarray, y: np.ndarray) -> float:
    pos, neg = s[y], s[~y]
    total = 0.0
    for a, b in itertools.product(pos, neg):
        total += 1.0 if a > b else 0.5 if a == b else 0.0
    return total / (len(pos) * len(neg))


def test_auroc_perfect_and_inverted() -> None:
    y = np.array([0, 0, 1, 1], bool)
    assert auroc([0.1, 0.2, 0.8, 0.9], y) == 1.0
    assert auroc([0.9, 0.8, 0.2, 0.1], y) == 0.0


def test_auroc_all_ties_is_half() -> None:
    assert auroc([1, 1, 1, 1], [0, 1, 0, 1]) == 0.5


def test_auroc_one_class_is_nan() -> None:
    assert np.isnan(auroc([0.1, 0.2], [1, 1]))
    assert np.isnan(auroc([0.1, 0.2], [0, 0]))


def test_auroc_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        auroc([0.1, 0.2], [1])
    with pytest.raises(ValueError):
        auroc([0.1, float("nan")], [1, 0])


@settings(max_examples=60, deadline=None)
@given(
    st.lists(st.integers(-5, 5), min_size=2, max_size=40),
    st.randoms(use_true_random=False),
)
def test_auroc_matches_pairwise_definition(values: list[int], rnd: object) -> None:
    s = np.asarray(values, float)
    y = np.zeros(len(s), bool)
    y[: len(s) // 2] = True
    np.random.default_rng(len(s)).shuffle(y)
    assert auroc(s, y) == pytest.approx(brute_auroc(s, y))


@settings(max_examples=40, deadline=None)
@given(st.lists(st.floats(-10, 10, allow_nan=False), min_size=1, max_size=50))
def test_rankdata_sums_to_triangular_number(values: list[float]) -> None:
    r = rankdata(values)
    n = len(values)
    assert r.sum() == pytest.approx(n * (n + 1) / 2)


def test_rankdata_midranks() -> None:
    np.testing.assert_allclose(rankdata([10, 20, 20, 30]), [1, 2.5, 2.5, 4])


@settings(max_examples=40, deadline=None)
@given(st.lists(st.integers(0, 6), min_size=4, max_size=40))
def test_trapezoid_area_of_roc_equals_auroc(values: list[int]) -> None:
    s = np.asarray(values, float)
    y = np.arange(len(s)) % 2 == 0
    fpr, tpr, thr = roc_curve(s, y)
    assert fpr[0] == 0 and tpr[0] == 0 and np.isinf(thr[0])
    assert fpr[-1] == pytest.approx(1) and tpr[-1] == pytest.approx(1)
    assert np.all(np.diff(fpr) >= 0) and np.all(np.diff(tpr) >= 0)
    area = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))
    assert area == pytest.approx(auroc(s, y))


def test_threshold_at_fpr_bounds_false_positives() -> None:
    neg = np.arange(100, dtype=float)
    for f in (0.0, 0.01, 0.05, 0.5):
        t = threshold_at_fpr(neg, f)
        assert np.mean(neg >= t) <= f + 1e-12
    assert threshold_at_fpr(neg, 1.0) == 0.0
    with pytest.raises(ValueError):
        threshold_at_fpr(neg, 1.5)
    with pytest.raises(ValueError):
        threshold_at_fpr([], 0.1)


@settings(max_examples=50, deadline=None)
@given(
    st.lists(st.floats(0, 1, allow_nan=False), min_size=5, max_size=60),
    st.floats(0, 1),
)
def test_tpr_at_fpr_respects_budget(values: list[float], fpr: float) -> None:
    s = np.asarray(values)
    y = np.arange(len(s)) % 3 == 0
    t = threshold_at_fpr(s[~y], fpr)
    assert np.mean(s[~y] >= t) <= fpr + 1e-12
    assert tpr_at_fpr(s, y, fpr) == pytest.approx(np.mean(s[y] >= t))


def test_tpr_at_fpr_separable() -> None:
    s = np.r_[np.zeros(100), np.ones(10)]
    y = np.r_[np.zeros(100, bool), np.ones(10, bool)]
    assert tpr_at_fpr(s, y, 0.01) == 1.0
    assert np.isnan(tpr_at_fpr(s, np.ones(110, bool), 0.01))


def test_tpr_at_fpr_with_ties_at_threshold_is_conservative() -> None:
    s = np.r_[np.ones(100), np.ones(10)]
    y = np.r_[np.zeros(100, bool), np.ones(10, bool)]
    assert tpr_at_fpr(s, y, 0.05) == 0.0


def test_attack_success_rate_ignores_target_class() -> None:
    assert attack_success_rate([0, 0, 1, 0], [0, 1, 2, 3], target=0) == pytest.approx(2 / 3)
    assert np.isnan(attack_success_rate([0], [0], target=0))


def test_accuracy() -> None:
    assert accuracy([1, 2, 3], [1, 2, 0]) == pytest.approx(2 / 3)
    assert np.isnan(accuracy([], []))


def test_detection_summary() -> None:
    s = np.r_[np.zeros(200), np.ones(20)]
    y = s > 0.5
    d = detection_summary(s, y)
    assert d.auroc == 1.0 and d.tpr_at_1 == 1.0 and d.tpr_at_5 == 1.0
    assert d.as_dict()["n_pos"] == 20 and d.n_neg == 200


def test_mean_std() -> None:
    assert mean_std([1.0, 3.0]) == (2.0, pytest.approx(np.sqrt(2)))
    assert mean_std([5.0]) == (5.0, 0.0)
    m, s = mean_std([float("nan")])
    assert np.isnan(m) and np.isnan(s)


def test_bootstrap_ci_contains_point_estimate() -> None:
    rng = np.random.default_rng(0)
    s = np.r_[rng.normal(0, 1, 300), rng.normal(1, 1, 300)]
    y = np.r_[np.zeros(300, bool), np.ones(300, bool)]
    lo, hi = bootstrap_auroc_ci(s, y, n_boot=200)
    assert lo < auroc(s, y) < hi
    assert hi - lo < 0.15
    assert all(np.isnan(bootstrap_auroc_ci(s, np.ones(600, bool))))
