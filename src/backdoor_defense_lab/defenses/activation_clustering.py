"""Activation clustering (Chen, Carvalho, Baracaldo, Ludwig, Edwards, Lee, Molloy and
Srivastava, 2018).

For every class, the penultimate activations of the training examples labelled with that class
are reduced to a few dimensions and split into two clusters with k-means. In a clean class the
two clusters are arbitrary halves of one mode; in a poisoned class one cluster collects the
poisoned examples, which the network classifies by the trigger rather than by the class features.
Chen et al. flag the smaller cluster when it holds at most ``size_threshold`` (35%) of the class,
or use the silhouette score to judge whether two clusters fit the data.

Dimensionality reduction: Chen et al. use independent component analysis (ICA) to 10
dimensions; IBM ART's implementation defaults to PCA. Both are implemented here
(:func:`pca_reduce`, :func:`fast_ica`) along with k-means++ (:func:`kmeans`) so the defense has no
scikit-learn dependency; parity with scikit-learn is checked in the validation tests.

For ranking metrics each example also gets a continuous score: its distance to the centroid of
the larger cluster minus its distance to the centroid of the smaller one, divided by the spread
of the class in the reduced space, so examples deep in the smaller cluster score highest.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def pca_reduce(x: ArrayLike, n_components: int) -> FloatArray:
    """Project centered rows of ``x`` on their top ``n_components`` principal directions."""
    a = np.asarray(x, dtype=np.float64)
    a = a - a.mean(axis=0, keepdims=True)
    k = min(n_components, a.shape[0], a.shape[1])
    if k == 0:
        return np.zeros((a.shape[0], 0))
    _, _, vt = np.linalg.svd(a, full_matrices=False)
    comps = vt[:k]
    # Deterministic sign: the largest-magnitude loading of each component is positive.
    signs = np.sign(comps[np.arange(k), np.argmax(np.abs(comps), axis=1)])
    signs[signs == 0] = 1
    return np.asarray(a @ (comps * signs[:, None]).T, dtype=np.float64)


def _sym_decorrelate(w: FloatArray) -> FloatArray:
    """W <- (W W^T)^(-1/2) W, which keeps the unmixing rows orthonormal."""
    s, u = np.linalg.eigh(w @ w.T)
    s = np.clip(s, np.finfo(np.float64).tiny, None)
    return np.asarray(u @ np.diag(1.0 / np.sqrt(s)) @ u.T @ w, dtype=np.float64)


def fast_ica(
    x: ArrayLike,
    n_components: int,
    max_iter: int = 400,
    tol: float = 1e-4,
    seed: int = 0,
) -> FloatArray:
    """Independent components with the symmetric FastICA fixed point (Hyvarinen, 1999).

    The data are centered and whitened by PCA to ``n_components`` dimensions (``Z``, unit
    covariance); then with the log-cosh contrast, ``g = tanh``, the unmixing matrix iterates

        W <- E[g(W z) z^T] - diag(E[g'(W z)]) W,   followed by symmetric decorrelation,

    until the rows stop changing direction. Returns the sources ``Z W^T``.
    """
    a = np.asarray(x, dtype=np.float64)
    n = a.shape[0]
    z = pca_reduce(a, n_components)
    k = z.shape[1]
    if k == 0 or n < 2:
        return z
    std = z.std(axis=0, ddof=0)
    std[std == 0] = 1.0
    z = z / std
    rng = np.random.default_rng(seed)
    w = _sym_decorrelate(rng.standard_normal((k, k)))
    for _ in range(max_iter):
        wz = z @ w.T
        g = np.tanh(wz)
        g_prime = 1.0 - g**2
        w_new = _sym_decorrelate((g.T @ z) / n - np.diag(g_prime.mean(axis=0)) @ w)
        converged = np.max(np.abs(np.abs(np.einsum("ij,ij->i", w_new, w)) - 1.0)) < tol
        w = w_new
        if converged:
            break
    return np.asarray(z @ w.T, dtype=np.float64)


def _kmeans_pp(x: FloatArray, k: int, rng: np.random.Generator) -> FloatArray:
    centers = [x[rng.integers(len(x))]]
    d2 = np.sum((x - centers[0]) ** 2, axis=1)
    for _ in range(1, k):
        total = d2.sum()
        idx = rng.integers(len(x)) if total <= 0 else rng.choice(len(x), p=d2 / total)
        centers.append(x[idx])
        d2 = np.minimum(d2, np.sum((x - x[idx]) ** 2, axis=1))
    return np.asarray(centers)


def _sq_dists(x: FloatArray, centers: FloatArray, x_sq: FloatArray) -> FloatArray:
    """Squared distances ``|x - c|^2 = |x|^2 + |c|^2 - 2 x.c`` (no n x k x d temporary)."""
    d = x_sq[:, None] + (centers**2).sum(1)[None, :] - 2.0 * (x @ centers.T)
    return np.maximum(d, 0.0)


def kmeans(
    x: ArrayLike,
    k: int = 2,
    n_init: int = 10,
    max_iter: int = 300,
    tol: float = 1e-8,
    seed: int = 0,
) -> tuple[NDArray[np.int64], FloatArray, float]:
    """Lloyd's k-means with k-means++ seeding; best of ``n_init`` runs by inertia.

    Returns ``(labels, centers, inertia)`` where inertia is the sum of squared distances to the
    assigned centers.
    """
    a = np.asarray(x, dtype=np.float64)
    if len(a) < k:
        raise ValueError("need at least k points")
    rng = np.random.default_rng(seed)
    sq = (a**2).sum(1)
    best: tuple[NDArray[np.int64], FloatArray, float] | None = None
    for _ in range(n_init):
        centers = _kmeans_pp(a, k, rng)
        prev: NDArray[np.int64] | None = None
        for _ in range(max_iter):
            lab = _sq_dists(a, centers, sq).argmin(1)
            if prev is not None and np.array_equal(lab, prev):
                break  # assignments are a fixed point: the centers cannot move
            prev = lab
            onehot = (lab[:, None] == np.arange(k)[None, :]).astype(np.float64)
            counts = onehot.sum(0)
            sums = onehot.T @ a
            new = np.where(counts[:, None] > 0, sums / np.maximum(counts, 1)[:, None], centers)
            shift = float(((new - centers) ** 2).sum())
            centers = new
            if shift <= tol:
                break
        d = ((a[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        lab = d.argmin(1).astype(np.int64)
        inertia = float(d[np.arange(len(a)), lab].sum())
        if best is None or inertia < best[2]:
            best = (lab, centers, inertia)
    assert best is not None
    return best


def silhouette(x: ArrayLike, labels: ArrayLike, max_points: int = 2000, seed: int = 0) -> float:
    """Mean silhouette coefficient (Rousseeuw, 1987), on a random subsample of ``max_points``.

    For a point with mean intra-cluster distance ``a`` and mean distance ``b`` to the nearest
    other cluster, ``s = (b - a) / max(a, b)``. Returns 0 for a single cluster.
    """
    a = np.asarray(x, dtype=np.float64)
    lab = np.asarray(labels)
    if len(a) > max_points:
        idx = np.random.default_rng(seed).choice(len(a), max_points, replace=False)
        a, lab = a[idx], lab[idx]
    clusters = np.unique(lab)
    if len(clusters) < 2:
        return 0.0
    sq = (a**2).sum(1)
    d = np.sqrt(_sq_dists(a, a, sq))
    np.fill_diagonal(d, 0.0)
    member = lab[:, None] == clusters[None, :]  # (n, c) one-hot cluster membership
    sums = d @ member  # distance from each point to every member of each cluster
    sizes = member.sum(0)
    own = member.argmax(1)
    n_same = sizes[own] - 1
    ai = np.where(n_same > 0, sums[np.arange(len(a)), own] / np.maximum(n_same, 1), 0.0)
    means = sums / sizes[None, :]
    means[np.arange(len(a)), own] = np.inf
    bi = means.min(1)
    denom = np.maximum(ai, bi)
    s = np.where((n_same > 0) & (denom > 0), (bi - ai) / np.where(denom > 0, denom, 1.0), 0.0)
    return float(s.mean())


@dataclass
class ClassReport:
    label: int
    n: int
    smaller_fraction: float
    silhouette: float
    flagged: bool


@dataclass
class ActivationClusteringResult:
    scores: FloatArray
    flagged: NDArray[np.bool_]
    classes: list[ClassReport] = field(default_factory=list)


def activation_clustering(
    features: ArrayLike,
    labels: ArrayLike,
    n_dims: int = 10,
    reduce: str = "ica",
    size_threshold: float = 0.35,
    analysis: str = "smaller",
    silhouette_threshold: float = 0.15,
    seed: int = 0,
) -> ActivationClusteringResult:
    """Run activation clustering on every class.

    ``reduce`` is ``"ica"`` (Chen et al.) or ``"pca"`` (ART default). ``analysis`` is
    ``"smaller"`` (flag the smaller cluster if its share is at most ``size_threshold``) or
    ``"silhouette"`` (flag the smaller cluster if the 2-cluster silhouette exceeds
    ``silhouette_threshold``).
    """
    if reduce not in {"ica", "pca"}:
        raise ValueError("reduce must be 'ica' or 'pca'")
    if analysis not in {"smaller", "silhouette"}:
        raise ValueError("analysis must be 'smaller' or 'silhouette'")
    f = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels)
    if len(f) != len(y):
        raise ValueError("features and labels differ in length")
    scores = np.zeros(len(y))
    flagged = np.zeros(len(y), dtype=bool)
    reports: list[ClassReport] = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        if len(idx) < 3:
            reports.append(ClassReport(int(c), len(idx), 0.0, 0.0, False))
            continue
        z = fast_ica(f[idx], n_dims, seed=seed) if reduce == "ica" else pca_reduce(f[idx], n_dims)
        lab, centers, _ = kmeans(z, 2, seed=seed)
        sizes = np.bincount(lab, minlength=2)
        small = int(np.argmin(sizes))
        frac = float(sizes[small] / len(idx))
        sil = silhouette(z, lab, seed=seed)
        is_poisoned = (
            frac <= size_threshold if analysis == "smaller" else sil > silhouette_threshold
        )
        d_small = np.linalg.norm(z - centers[small], axis=1)
        d_large = np.linalg.norm(z - centers[1 - small], axis=1)
        spread = float(np.sqrt(((z - z.mean(0)) ** 2).sum(1).mean())) or 1.0
        scores[idx] = (d_large - d_small) / spread
        if is_poisoned:
            flagged[idx[lab == small]] = True
        reports.append(ClassReport(int(c), len(idx), frac, sil, bool(is_poisoned)))
    return ActivationClusteringResult(scores=scores, flagged=flagged, classes=reports)
