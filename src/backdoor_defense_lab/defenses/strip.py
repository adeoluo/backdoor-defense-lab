"""STRIP: STRong Intentional Perturbation (Gao, Xu, Wang, Chen, Ranasinghe and Nepal, 2019).

A run-time detector for triggered inputs. Each input ``x`` is superimposed with ``n`` images
drawn from a small held-out clean set, and the model's predictions on the blends are inspected:

    H(x) = (1/n) sum_j  - sum_k p_k(x (+) x_j) log p_k(x (+) x_j).

A clean input's prediction is destroyed by the overlay, so the predictions spread out and the
entropy is high. A triggered input keeps the trigger in every blend and the model keeps
predicting the target, so the entropy stays low. The suspicion score is ``-H(x)``; the decision
threshold is set on clean inputs for a chosen false rejection rate (1% in the paper).

Superimposition ``(+)``: Gao et al. add the two images (``cv2.addWeighted(x, 1, x_j, 1, 0)``
with saturation); many reimplementations average them. Both are provided (``mode="add"`` and
``mode="blend"``, the default, with weight ``alpha`` on the input).
"""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn


def superimpose(
    x: torch.Tensor, overlay: torch.Tensor, mode: str = "blend", alpha: float = 0.5
) -> torch.Tensor:
    """Combine images ``x`` and ``overlay`` (same shape, values in [0, 1])."""
    if mode == "blend":
        return alpha * x + (1 - alpha) * overlay
    if mode == "add":
        return (x + overlay).clamp(0, 1)
    raise ValueError("mode must be 'blend' or 'add'")


def entropy(probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Shannon entropy (nats) of each row of a probability matrix."""
    return -(probs * torch.log(probs.clamp_min(eps))).sum(-1)


@torch.no_grad()
def strip_entropy(
    model: nn.Module,
    x: torch.Tensor,
    clean_pool: torch.Tensor,
    n_perturb: int = 32,
    mode: str = "blend",
    alpha: float = 0.5,
    seed: int = 0,
    batch_size: int = 64,
) -> NDArray[np.float64]:
    """Mean prediction entropy of each input over ``n_perturb`` superimposed clean images.

    ``x`` and ``clean_pool`` are float NCHW tensors in [0, 1]. Runs on the model's device; each
    input gets its own random draw of overlays (without replacement when the pool allows).
    """
    if n_perturb < 1:
        raise ValueError("n_perturb must be at least 1")
    if len(clean_pool) == 0:
        raise ValueError("clean_pool is empty")
    model.eval()
    device = next(model.parameters()).device
    gen = torch.Generator().manual_seed(seed)
    pool = clean_pool.to(device)
    out = np.empty(len(x), dtype=np.float64)
    replace = n_perturb > len(pool)
    for start in range(0, len(x), batch_size):
        xb = x[start : start + batch_size].to(device)
        b = len(xb)
        if replace:
            idx = torch.randint(0, len(pool), (b, n_perturb), generator=gen)
        else:
            idx = torch.argsort(torch.rand(b, len(pool), generator=gen), dim=1)[:, :n_perturb]
        overlays = pool[idx.flatten().to(device)]
        blended = superimpose(xb.repeat_interleave(n_perturb, 0), overlays, mode, alpha)
        probs = torch.softmax(model(blended), dim=1)
        h = entropy(probs).view(b, n_perturb).mean(1)
        out[start : start + b] = h.double().cpu().numpy()
    return out


def strip_scores(
    model: nn.Module,
    x: torch.Tensor,
    clean_pool: torch.Tensor,
    n_perturb: int = 32,
    mode: str = "blend",
    alpha: float = 0.5,
    seed: int = 0,
) -> NDArray[np.float64]:
    """Suspicion scores: the negative STRIP entropy (higher means more likely triggered)."""
    return -strip_entropy(model, x, clean_pool, n_perturb, mode, alpha, seed)
