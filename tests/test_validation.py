"""Parity against scikit-learn and IBM ART; opt in with ``make validation``."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.validation

sklearn = pytest.importorskip("sklearn")
art = pytest.importorskip("art")

from sklearn.cluster import KMeans  # noqa: E402
from sklearn.decomposition import PCA, FastICA  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve, silhouette_score  # noqa: E402

from backdoor_defense_lab.defenses import fast_ica, kmeans, pca_reduce, silhouette  # noqa: E402
from backdoor_defense_lab.defenses.spectral import class_spectral_scores  # noqa: E402
from backdoor_defense_lab.metrics import auroc  # noqa: E402
from backdoor_defense_lab.metrics import roc_curve as our_roc  # noqa: E402


def _data(seed: int = 0, n: int = 400, d: int = 12) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d)) @ rng.normal(size=(d, d))


@pytest.mark.parametrize("seed", range(5))
def test_auroc_matches_sklearn(seed: int) -> None:
    rng = np.random.default_rng(seed)
    y = rng.random(500) < 0.2
    s = rng.normal(size=500) + y * 0.8
    s[:50] = np.round(s[:50])  # ties
    assert auroc(s, y) == pytest.approx(roc_auc_score(y, s), abs=1e-12)


def test_roc_curve_matches_sklearn() -> None:
    rng = np.random.default_rng(0)
    y = rng.random(300) < 0.3
    s = rng.normal(size=300) + y
    s[:30] = np.round(s[:30])  # ties
    fpr, tpr, thr = our_roc(s, y)
    ref_fpr, ref_tpr, ref_thr = roc_curve(y, s, drop_intermediate=False)
    assert np.allclose(fpr, ref_fpr) and np.allclose(tpr, ref_tpr)
    assert np.allclose(thr[1:], ref_thr[1:])


@pytest.mark.parametrize("seed", range(3))
def test_pca_matches_sklearn(seed: int) -> None:
    x = _data(seed)
    ours = pca_reduce(x, 4)
    ref = PCA(n_components=4).fit_transform(x)
    for j in range(4):
        assert abs(np.corrcoef(ours[:, j], ref[:, j])[0, 1]) == pytest.approx(1.0, abs=1e-8)


def test_fast_ica_matches_sklearn_up_to_permutation_and_sign() -> None:
    rng = np.random.default_rng(0)
    n = 3000
    t = np.linspace(0, 60, n)
    s = np.c_[np.sign(np.sin(t)), rng.laplace(size=n), (t % 3) - 1.5]
    x = s @ rng.normal(size=(3, 3)).T
    ours = fast_ica(x, 3, seed=0)
    ref = FastICA(n_components=3, random_state=0, whiten="unit-variance").fit_transform(x)
    corr = np.abs(np.corrcoef(np.c_[ours, ref].T)[:3, 3:])
    assert corr.max(1).min() > 0.99


@pytest.mark.parametrize("seed", range(5))
def test_kmeans_inertia_matches_sklearn(seed: int) -> None:
    x = pca_reduce(_data(seed), 5)
    _, _, inertia = kmeans(x, 2, n_init=10, seed=seed)
    ref = KMeans(n_clusters=2, n_init=10, random_state=seed).fit(x)
    # Both are Lloyd's algorithm from k-means++ seeds; different random draws can settle in
    # local optima a hair apart, so the best-of-10 inertias agree to 0.01%.
    assert inertia == pytest.approx(ref.inertia_, rel=1e-4)


@pytest.mark.parametrize("seed", range(3))
def test_silhouette_matches_sklearn(seed: int) -> None:
    x = pca_reduce(_data(seed, n=300), 4)
    lab, _, _ = kmeans(x, 2, seed=seed)
    assert silhouette(x, lab, max_points=1000) == pytest.approx(silhouette_score(x, lab), abs=1e-9)


def test_spectral_scores_match_art_on_centered_features() -> None:
    from art.defences.detector.poison import SpectralSignatureDefense

    x = _data(1)
    centered = x - x.mean(0)
    ref = SpectralSignatureDefense.spectral_signature_scores(centered).ravel()
    assert np.allclose(class_spectral_scores(x), ref**2, rtol=1e-8)


def test_art_spectral_scores_use_uncentered_features() -> None:
    """ART projects the raw (uncentered) features; on shifted data the rankings differ."""
    from art.defences.detector.poison import SpectralSignatureDefense

    x = _data(2) + 5.0
    ref = SpectralSignatureDefense.spectral_signature_scores(x).ravel()
    ours = np.sqrt(class_spectral_scores(x))
    assert not np.allclose(ours, ref)


def test_activation_clustering_split_matches_art() -> None:
    from art.defences.detector.poison.activation_defence import cluster_activations
    from art.defences.detector.poison.clustering_analyzer import ClusteringAnalyzer

    from backdoor_defense_lab.defenses import activation_clustering

    rng = np.random.default_rng(0)
    clean = rng.normal(size=(400, 32))
    poison = rng.normal(size=(60, 32)) * 0.5 + 6 * np.eye(32)[5]
    f = np.vstack([clean, poison])
    y = np.zeros(len(f), int)
    ours = activation_clustering(f, y, n_dims=10, reduce="pca")
    clusters, _ = cluster_activations([f], nb_clusters=2, nb_dims=10, reduce="PCA")
    assigned_clean, _, _ = ClusteringAnalyzer().analyze_by_size(clusters)
    art_flag = np.asarray(assigned_clean[0]) == 0  # 0 marks points in the poisoned cluster
    assert ours.classes[0].flagged
    assert np.array_equal(ours.flagged, art_flag)
