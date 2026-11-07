from __future__ import annotations

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from backdoor_defense_lab.data import ImageDataset, to_tensor
from backdoor_defense_lab.defenses import (
    ReversedTrigger,
    activation_clustering,
    anomaly_indices,
    channel_activations,
    fast_ica,
    fine_pruning,
    kmeans,
    pca_reduce,
    prune,
    reverse_engineer,
    silhouette,
    spectral_filter,
    spectral_scores,
    strip_entropy,
    strip_scores,
    superimpose,
    unlearn,
)
from backdoor_defense_lab.defenses.fine_pruning import pruning_order, set_pruned
from backdoor_defense_lab.defenses.spectral import class_spectral_scores, top_singular_vector
from backdoor_defense_lab.defenses.strip import entropy
from backdoor_defense_lab.metrics import auroc
from backdoor_defense_lab.models import build_model
from backdoor_defense_lab.pipeline import Run, measure_asr

# ---------------------------------------------------------------------------
# Spectral signatures
# ---------------------------------------------------------------------------


def _shifted_cluster(n_clean: int = 400, n_poison: int = 40, d: int = 16, shift: float = 5.0):
    rng = np.random.default_rng(0)
    clean = rng.normal(size=(n_clean, d))
    poison = rng.normal(size=(n_poison, d)) + shift * np.eye(d)[0]
    f = np.vstack([clean, poison])
    is_poison = np.r_[np.zeros(n_clean, bool), np.ones(n_poison, bool)]
    return f, is_poison


def test_spectral_scores_separate_a_mean_shift() -> None:
    f, is_poison = _shifted_cluster()
    s = class_spectral_scores(f)
    assert auroc(s, is_poison) > 0.95


def test_top_singular_vector_matches_numpy_svd() -> None:
    rng = np.random.default_rng(1)
    m = rng.normal(size=(50, 8))
    v = top_singular_vector(m)
    _, _, vt = np.linalg.svd(m)
    assert abs(abs(v @ vt[0]) - 1) < 1e-10
    assert v[np.argmax(np.abs(v))] > 0


def test_top_singular_vector_empty() -> None:
    assert np.all(top_singular_vector(np.zeros((0, 4))) == 0)


def test_spectral_scores_are_per_class() -> None:
    f, _is_poison = _shifted_cluster()
    labels = np.zeros(len(f), int)
    other = np.random.default_rng(3).normal(size=(100, f.shape[1])) * 10
    f2 = np.vstack([f, other])
    y2 = np.r_[labels, np.ones(100, int)]
    s = spectral_scores(f2, y2)
    # scores of class 0 are unchanged by adding another class
    assert np.allclose(s[: len(f)], class_spectral_scores(f))


def test_spectral_scores_validation() -> None:
    with pytest.raises(ValueError):
        spectral_scores(np.zeros((3, 2)), np.zeros(4))
    with pytest.raises(ValueError):
        class_spectral_scores(np.zeros(3))
    assert np.all(class_spectral_scores(np.ones((1, 3))) == 0)


@given(arrays(np.float64, (30, 5), elements=st.floats(-10, 10)))
@settings(max_examples=40, deadline=None)
def test_spectral_scores_nonnegative_and_shift_invariant(f: np.ndarray) -> None:
    s = class_spectral_scores(f)
    assert np.all(s >= 0)
    assert np.allclose(class_spectral_scores(f + 7.0), s, atol=1e-6 * (1 + s.max()))


def test_spectral_filter_removes_top_scores_per_class() -> None:
    scores = np.array([1, 5, 3, 9, 0.5, 2, 8, 7.0])
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    mask = spectral_filter(scores, labels, eps=0.25, multiplier=1.5)  # ceil(1.5) = 2 per class
    assert mask.tolist() == [False, True, False, True, False, False, True, True]
    assert not spectral_filter(scores, labels, eps=0.0).any()
    assert spectral_filter(scores, labels, eps=1.0).all()
    with pytest.raises(ValueError):
        spectral_filter(scores, labels, eps=1.5)


def test_spectral_filter_catches_poison() -> None:
    f, is_poison = _shifted_cluster()
    labels = np.zeros(len(f), int)
    removed = spectral_filter(spectral_scores(f, labels), labels, eps=40 / 440)
    assert (removed & is_poison).sum() >= 36


# ---------------------------------------------------------------------------
# Activation clustering building blocks
# ---------------------------------------------------------------------------


def test_pca_reduce_keeps_variance_order() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 6)) * np.array([10, 5, 2, 1, 0.5, 0.1])
    z = pca_reduce(x, 3)
    var = z.var(0)
    assert z.shape == (300, 3)
    assert var[0] > var[1] > var[2]
    assert np.allclose(z.mean(0), 0, atol=1e-10)


def test_pca_reduce_edge_cases() -> None:
    assert pca_reduce(np.zeros((5, 0)), 3).shape == (5, 0)
    assert pca_reduce(np.ones((2, 8)), 10).shape[1] == 2


def test_fast_ica_recovers_independent_sources() -> None:
    rng = np.random.default_rng(0)
    n = 2000
    s = np.c_[np.sign(np.sin(np.linspace(0, 40, n))), rng.uniform(-1, 1, n)]
    mix = np.array([[1.0, 0.5], [0.4, 1.2]])
    x = s @ mix.T
    z = fast_ica(x, 2)
    corr = np.abs(np.corrcoef(np.c_[z, s].T)[:2, 2:])
    # each recovered component matches one source (up to sign and scale)
    assert corr.max(1).min() > 0.95


def test_fast_ica_degenerate_inputs() -> None:
    assert fast_ica(np.ones((1, 3)), 2).shape[0] == 1
    z = fast_ica(np.c_[np.arange(10.0), np.zeros(10)], 2)
    assert np.isfinite(z).all()


def test_kmeans_two_blobs() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(size=(100, 2))
    b = rng.normal(size=(30, 2)) + 8
    lab, centers, inertia = kmeans(np.vstack([a, b]), 2)
    assert len(set(lab[:100])) == 1 and len(set(lab[100:])) == 1
    assert lab[0] != lab[-1]
    assert inertia > 0 and centers.shape == (2, 2)


def test_kmeans_needs_k_points() -> None:
    with pytest.raises(ValueError):
        kmeans(np.zeros((1, 2)), 2)


def test_kmeans_identical_points() -> None:
    lab, _, inertia = kmeans(np.zeros((5, 2)), 2, n_init=2)
    assert inertia == 0 and len(lab) == 5


@given(st.integers(2, 4), st.integers(0, 50))
@settings(max_examples=25, deadline=None)
def test_kmeans_inertia_matches_labels(k: int, seed: int) -> None:
    x = np.random.default_rng(seed).normal(size=(40, 3))
    lab, centers, inertia = kmeans(x, k, n_init=2, seed=seed)
    recomputed = sum(((x[lab == j] - centers[j]) ** 2).sum() for j in range(k))
    assert inertia == pytest.approx(recomputed)
    # Lloyd fixed point: every point is assigned to its nearest center
    d = ((x[:, None] - centers[None]) ** 2).sum(-1)
    assert np.all(d[np.arange(len(x)), lab] <= d.min(1) + 1e-9)


def test_silhouette_well_separated_and_single_cluster() -> None:
    x = np.r_[np.zeros((20, 2)), np.ones((20, 2)) * 10]
    lab = np.r_[np.zeros(20), np.ones(20)]
    assert silhouette(x, lab) == pytest.approx(1.0)
    assert silhouette(x, np.zeros(40)) == 0.0


def test_silhouette_subsamples() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 2))
    lab = (x[:, 0] > 0).astype(int)
    full = silhouette(x, lab, max_points=1000)
    sub = silhouette(x, lab, max_points=100)
    assert abs(full - sub) < 0.15


def test_silhouette_singleton_cluster() -> None:
    x = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]])
    assert -1 <= silhouette(x, np.array([0, 0, 1])) <= 1


@pytest.mark.parametrize("reduce", ["ica", "pca"])
def test_activation_clustering_flags_the_poisoned_class(reduce: str) -> None:
    rng = np.random.default_rng(0)
    clean0 = rng.normal(size=(300, 20))
    poison0 = rng.normal(size=(45, 20)) * 0.5 + 6 * np.eye(20)[3]
    clean1 = rng.normal(size=(300, 20))
    f = np.vstack([clean0, poison0, clean1])
    y = np.r_[np.zeros(345, int), np.ones(300, int)]
    is_poison = np.r_[np.zeros(300, bool), np.ones(45, bool), np.zeros(300, bool)]
    res = activation_clustering(f, y, n_dims=5, reduce=reduce)
    by_label = {r.label: r for r in res.classes}
    assert by_label[0].flagged
    assert by_label[0].smaller_fraction == pytest.approx(45 / 345, abs=0.03)
    assert (res.flagged & is_poison).sum() >= 40
    assert auroc(res.scores[:345], is_poison[:345]) > 0.95


def test_activation_clustering_silhouette_mode_and_small_classes() -> None:
    rng = np.random.default_rng(1)
    f = np.vstack([rng.normal(size=(100, 4)), rng.normal(size=(2, 4))])
    y = np.r_[np.zeros(100, int), np.ones(2, int)]
    res = activation_clustering(
        f, y, n_dims=3, analysis="silhouette", reduce="pca", silhouette_threshold=0.4
    )
    small = next(r for r in res.classes if r.label == 1)
    assert not small.flagged and small.n == 2
    # a single Gaussian blob splits into two halves with a modest silhouette (about 0.25)
    assert not res.classes[0].flagged
    sep = np.vstack([rng.normal(size=(80, 4)), rng.normal(size=(20, 4)) + 8])
    res2 = activation_clustering(
        sep, np.zeros(100, int), n_dims=3, analysis="silhouette", silhouette_threshold=0.4
    )
    assert res2.classes[0].flagged
    assert res2.classes[0].silhouette > res.classes[0].silhouette


def test_activation_clustering_validation() -> None:
    f = np.zeros((4, 2))
    with pytest.raises(ValueError):
        activation_clustering(f, np.zeros(4), reduce="tsne")
    with pytest.raises(ValueError):
        activation_clustering(f, np.zeros(4), analysis="both")
    with pytest.raises(ValueError):
        activation_clustering(f, np.zeros(3))


# ---------------------------------------------------------------------------
# STRIP
# ---------------------------------------------------------------------------


def test_superimpose_modes(images: torch.Tensor) -> None:
    a, b = images[:4], images[4:]
    assert torch.allclose(superimpose(a, b, "blend", 0.5), (a + b) / 2)
    added = superimpose(a, b, "add")
    assert torch.allclose(added, (a + b).clamp(0, 1))
    assert added.max() <= 1
    with pytest.raises(ValueError):
        superimpose(a, b, "max")


def test_entropy_bounds() -> None:
    uniform = torch.full((1, 10), 0.1)
    onehot = torch.eye(10)[:1]
    assert float(entropy(uniform)) == pytest.approx(np.log(10), rel=1e-5)
    assert float(entropy(onehot)) == pytest.approx(0.0, abs=1e-6)


class _TriggerNet(torch.nn.Module):
    """Predicts class 0 with certainty when the top-left pixel is bright, else follows the mean."""

    def __init__(self) -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trig = (x[:, 0, 0, 0] > 0.9).float()
        m = x.mean(dim=(1, 2, 3))
        logits = torch.stack([m * 3, -m * 3, torch.sin(20 * m), torch.cos(20 * m)], 1)
        return logits * (1 - trig[:, None]) + trig[:, None] * torch.tensor([50.0, 0, 0, 0])


def test_strip_separates_triggered_inputs() -> None:
    g = torch.Generator().manual_seed(0)
    pool = torch.rand(64, 3, 8, 8, generator=g)
    clean = torch.rand(40, 3, 8, 8, generator=g)
    triggered = clean.clone()
    triggered[:, :, 0, 0] = 1.0
    net = _TriggerNet()
    # with add mode the trigger pixel survives the overlay
    h_clean = strip_entropy(net, clean, pool, n_perturb=16, mode="add")
    h_trig = strip_entropy(net, triggered, pool, n_perturb=16, mode="add")
    assert h_trig.mean() < h_clean.mean()
    s = strip_scores(net, torch.cat([clean, triggered]), pool, n_perturb=16, mode="add")
    assert auroc(s, np.r_[np.zeros(40), np.ones(40)]) > 0.95


def test_strip_with_replacement_and_determinism() -> None:
    g = torch.Generator().manual_seed(1)
    pool = torch.rand(4, 3, 8, 8, generator=g)
    x = torch.rand(5, 3, 8, 8, generator=g)
    net = _TriggerNet()
    a = strip_entropy(net, x, pool, n_perturb=10, seed=3)
    b = strip_entropy(net, x, pool, n_perturb=10, seed=3)
    assert np.array_equal(a, b) and a.shape == (5,)


def test_strip_validation() -> None:
    net = _TriggerNet()
    x = torch.rand(2, 3, 8, 8)
    with pytest.raises(ValueError):
        strip_entropy(net, x, x, n_perturb=0)
    with pytest.raises(ValueError):
        strip_entropy(net, x, x[:0])


# ---------------------------------------------------------------------------
# Neural Cleanse
# ---------------------------------------------------------------------------


def test_anomaly_indices() -> None:
    norms = np.array([10.0, 11, 9, 10.5, 9.5, 1.0])
    idx = anomaly_indices(norms)
    assert idx.argmax() == 5
    assert idx[5] > 2
    assert np.all(anomaly_indices(np.ones(5)) == 0)


def test_anomaly_index_is_scale_invariant() -> None:
    norms = np.array([3.0, 4, 5, 6, 0.5])
    assert np.allclose(anomaly_indices(norms), anomaly_indices(norms * 7.5))


def test_reverse_engineer_finds_the_backdoor_class(backdoored: Run) -> None:
    run = backdoored
    x = to_tensor(run.data.x_test[:128])
    res = reverse_engineer(run.model, x, run.data.num_classes, epochs=12, steps_per_epoch=6, seed=0)
    assert res.masks.shape == (4, 32, 32) and res.patterns.shape == (4, 3, 32, 32)
    assert res.suspect == 0
    assert res.l1[0] < np.median(res.l1)
    assert res.min_anomaly >= 0 and res.seconds > 0
    assert np.all((res.masks >= 0) & (res.masks <= 1))
    # parameters are trainable again afterwards
    assert all(p.requires_grad for p in run.model.parameters())


def test_reverse_engineer_subset_of_classes(backdoored: Run) -> None:
    x = to_tensor(backdoored.data.x_test[:64])
    res = reverse_engineer(backdoored.model, x, 4, epochs=2, steps_per_epoch=2, classes=[1, 2])
    assert len(res.l1) == 2 and np.isfinite(res.l1).all()


def test_min_anomaly_is_zero_when_smallest_is_not_below_median() -> None:
    from backdoor_defense_lab.defenses.neural_cleanse import ReverseResult

    r = ReverseResult(
        masks=np.zeros((2, 2, 2), np.float32),
        patterns=np.zeros((2, 3, 2, 2), np.float32),
        l1=np.array([1.0, 1.0]),
        success=np.ones(2),
        anomaly=np.zeros(2),
        flagged=[],
    )
    assert r.min_anomaly == 0.0 and r.suspect == 0


def test_reversed_trigger_applies_mask(images: torch.Tensor) -> None:
    mask = np.zeros((32, 32), np.float32)
    mask[:4, :4] = 1
    pattern = np.ones((3, 32, 32), np.float32)
    t = ReversedTrigger(mask, pattern)
    out = t.apply(images)
    assert torch.all(out[:, :, :4, :4] == 1)
    assert torch.equal(out[:, :, 4:, 4:], images[:, :, 4:, 4:])
    assert torch.equal(t.training_noise(images, torch.Generator()), out)


def test_unlearning_with_the_true_trigger_lowers_asr(backdoored: Run) -> None:
    run = backdoored
    assert run.attack is not None
    before = measure_asr(run.model, run.data, run.attack)
    mask = np.zeros((32, 32), np.float32)
    mask[28:31, 28:31] = 1
    pattern = np.zeros((3, 32, 32), np.float32)
    pattern[:, 28:31, 28:31] = run.attack.apply(torch.zeros(1, 3, 32, 32))[0, :, 28:31, 28:31]
    import copy

    tuned = unlearn(
        copy.deepcopy(run.model),
        run.data.x_test[:200],
        run.data.y_test[:200],
        mask,
        pattern,
        epochs=4,
        lr=0.05,
    )
    after = measure_asr(tuned, run.data, run.attack)
    assert after < before
    with pytest.raises(ValueError):
        unlearn(run.model, run.data.x_test[:10], run.data.y_test[:10], mask, pattern, 1.5)


# ---------------------------------------------------------------------------
# Fine-pruning
# ---------------------------------------------------------------------------


def test_pruning_order_and_mask() -> None:
    order = pruning_order(torch.tensor([0.5, 0.1, 0.9, 0.1]))
    assert order.tolist() == [1, 3, 0, 2]
    m = build_model(4, width=2)
    set_pruned(m, np.array([0, 3]))
    assert m.channel_mask.tolist()[:4] == [0, 1, 1, 0]
    set_pruned(m, np.array([], dtype=np.int64))
    assert bool((m.channel_mask == 1).all())


def test_pruned_channels_are_silent(toy: ImageDataset) -> None:
    m = build_model(4, width=4)
    set_pruned(m, np.arange(10))
    h = m.last_conv(to_tensor(toy.x_test[:8]))
    assert torch.all(h[:, :10] == 0)
    acts = channel_activations(m, to_tensor(toy.x_test[:16]), batch_size=5)
    assert acts.shape == (32,) and torch.all(acts[:10] == 0)


def test_prune_respects_accuracy_budget(backdoored: Run) -> None:
    run = backdoored
    xv = to_tensor(run.data.x_test[:200])
    yv = torch.from_numpy(run.data.y_test[:200])
    assert run.attack is not None
    trig, _ = _triggered(run)
    res = prune(run.model, xv, yv, max_accuracy_drop=0.05, x_triggered=trig, target=0)
    assert res.base_accuracy - res.pruned_accuracy <= 0.05 + 1e-9
    assert 0 <= res.n_pruned < len(res.order)
    assert res.curve[0].pruned == 0 and not np.isnan(res.curve[0].attack_success_rate)
    assert 0 <= res.prune_ratio < 1
    # the input model is untouched
    assert bool((run.model.channel_mask == 1).all())
    assert int((res.model.channel_mask == 0).sum()) == res.n_pruned


def test_prune_without_trigger_records_nan(backdoored: Run) -> None:
    xv = to_tensor(backdoored.data.x_test[:50])
    yv = torch.from_numpy(backdoored.data.y_test[:50])
    res = prune(backdoored.model, xv, yv, max_accuracy_drop=1.0, step=8)
    assert np.isnan(res.curve[0].attack_success_rate)
    # an unlimited budget prunes up to the last full step
    assert res.n_pruned == res.curve[-1].pruned


def test_fine_pruning_end_to_end(backdoored: Run) -> None:
    run = backdoored
    pruned, tuned = fine_pruning(
        run.model, run.data.x_test[:200], run.data.y_test[:200], epochs=2, lr=0.02
    )
    assert int((tuned.channel_mask == 0).sum()) == pruned.n_pruned
    acc = float((tuned(to_tensor(run.data.x_test)).argmax(1).numpy() == run.data.y_test).mean())
    assert acc > 0.5


def _triggered(run: Run) -> tuple[torch.Tensor, np.ndarray]:
    from backdoor_defense_lab.attacks import triggered_test_set

    assert run.attack is not None
    xt, yt = triggered_test_set(run.data, run.attack)
    return to_tensor(xt), yt
