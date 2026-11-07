"""Neural Cleanse (Wang, Yao, Shan, Li, Viswanath, Zheng and Zhao, 2019).

Trigger reverse engineering: for every candidate target class ``k`` find the smallest mask
``m_k`` (values in [0, 1], one per pixel) and pattern ``p_k`` that send clean images to ``k``:

    min_{m, p}  CE(f((1 - m) * x + m * p), k) + lambda_k * ||m||_1,

with ``m = (tanh(a) + 1) / 2`` and ``p = (tanh(b) + 1) / 2`` so the box constraints hold. A
backdoored target class needs only a trigger-sized mask, much smaller than what any other class
needs, so it is an outlier on the low side of the mask norms. Outliers are measured with the
median absolute deviation: with ``MAD = median(|l1 - median(l1)|)``, the anomaly index of class
``k`` is ``|l1_k - median| / (1.4826 * MAD)`` (1.4826 makes MAD consistent with a normal standard
deviation), and a class with ``l1_k`` below the median and index above 2 is flagged.

The weight ``lambda_k`` is adapted per class as in the paper: start at ``1e-3``; when the
reversed trigger reaches the target on at least 99% of a batch for ``patience`` consecutive
epochs multiply it by 1.5, when it falls short for ``patience`` epochs divide it by
``1.5 ** 1.5``; keep the smallest mask that reached 99%. Adam with learning rate 0.1 and betas
(0.5, 0.9). All classes are optimized together in one batched forward pass (the parameters
are independent, so this equals running them one at a time) which keeps 43 GTSRB classes cheap.

Mitigation by unlearning: fine-tune the model on clean data where a share of the images carries
the reversed trigger of the flagged class but keeps its true label.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from numpy.typing import NDArray
from torch import nn

from backdoor_defense_lab.attacks.base import Attack
from backdoor_defense_lab.models import ResNet9
from backdoor_defense_lab.train import TrainConfig, train_classifier

MAD_CONSISTENCY = 1.4826


def anomaly_indices(norms: NDArray[np.float64]) -> NDArray[np.float64]:
    """MAD-based anomaly index of every class (0 when the MAD is 0)."""
    l1 = np.asarray(norms, dtype=np.float64)
    med = np.median(l1)
    mad = np.median(np.abs(l1 - med)) * MAD_CONSISTENCY
    if mad == 0:
        return np.zeros_like(l1)
    return np.asarray(np.abs(l1 - med) / mad, dtype=np.float64)


@dataclass
class ReverseResult:
    masks: NDArray[np.float32]
    patterns: NDArray[np.float32]
    l1: NDArray[np.float64]
    success: NDArray[np.float64]
    anomaly: NDArray[np.float64]
    flagged: list[int]
    seconds: float = 0.0

    @property
    def suspect(self) -> int:
        """Class with the smallest reversed trigger."""
        return int(np.argmin(self.l1))

    @property
    def min_anomaly(self) -> float:
        """Anomaly index of the smallest-norm class: the model-level detection score."""
        med = float(np.median(self.l1))
        k = self.suspect
        return float(self.anomaly[k]) if self.l1[k] < med else 0.0


def reverse_engineer(
    model: nn.Module,
    x_clean: torch.Tensor,
    num_classes: int,
    epochs: int = 20,
    steps_per_epoch: int = 10,
    batch_size: int = 32,
    lr: float = 0.1,
    init_cost: float = 1e-3,
    success_threshold: float = 0.99,
    patience: int = 3,
    classes: list[int] | None = None,
    seed: int = 0,
) -> ReverseResult:
    """Reverse-engineer a trigger for every class in ``classes`` (default: all)."""
    t0 = time.perf_counter()
    model.eval()
    for prm in model.parameters():
        prm.requires_grad_(False)
    device = next(model.parameters()).device
    cls = list(range(num_classes)) if classes is None else list(classes)
    k = len(cls)
    _, c, h, w = x_clean.shape
    gen = torch.Generator().manual_seed(seed)
    mask_raw = torch.atanh((torch.rand(k, 1, h, w, generator=gen) * 2 - 1) * 0.99).to(device)
    pattern_raw = torch.atanh((torch.rand(k, c, h, w, generator=gen) * 2 - 1) * 0.99).to(device)
    mask_raw.requires_grad_(True)
    pattern_raw.requires_grad_(True)
    opt = torch.optim.Adam([mask_raw, pattern_raw], lr=lr, betas=(0.5, 0.9))
    cost = torch.full((k,), init_cost, device=device)
    targets = torch.tensor(cls, device=device).repeat_interleave(batch_size)
    best_l1 = np.full(k, np.inf)
    best_mask = np.zeros((k, h, w), np.float32)
    best_pattern = np.zeros((k, c, h, w), np.float32)
    last_success = np.zeros(k)
    up = np.zeros(k, int)
    down = np.zeros(k, int)
    for _ in range(epochs):
        hits = torch.zeros(k, device=device)
        seen = 0
        for _ in range(steps_per_epoch):
            idx = torch.randint(0, len(x_clean), (batch_size,), generator=gen)
            xb = x_clean[idx].to(device)
            m = (torch.tanh(mask_raw) + 1) / 2
            p = (torch.tanh(pattern_raw) + 1) / 2
            stamped = (1 - m[:, None]) * xb[None] + (m * p)[:, None]  # (k, B, c, h, w)
            logits = model(stamped.reshape(k * batch_size, c, h, w))
            ce = F.cross_entropy(logits, targets, reduction="none").view(k, batch_size).mean(1)
            l1 = m.abs().flatten(1).sum(1)
            loss = (ce + cost * l1).sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            with torch.no_grad():
                hits += (logits.argmax(1) == targets).float().view(k, batch_size).sum(1)
            seen += batch_size
        with torch.no_grad():
            m = (torch.tanh(mask_raw) + 1) / 2
            p = (torch.tanh(pattern_raw) + 1) / 2
            l1_now = m.flatten(1).sum(1).cpu().numpy().astype(np.float64)
            success = (hits / seen).cpu().numpy()
        last_success = success
        reached = success >= success_threshold
        better = reached & (l1_now < best_l1)
        if better.any():
            best_l1[better] = l1_now[better]
            best_mask[better] = m[torch.from_numpy(better).to(device), 0].cpu().numpy()
            best_pattern[better] = p[torch.from_numpy(better).to(device)].cpu().numpy()
        up = np.where(reached, up + 1, 0)
        down = np.where(reached, 0, down + 1)
        factor = np.ones(k)
        factor[up >= patience] = 1.5
        factor[down >= patience] = 1 / 1.5**1.5
        up[up >= patience] = 0
        down[down >= patience] = 0
        cost = cost * torch.from_numpy(factor).float().to(device)
    # Classes that never reached the threshold keep their final mask.
    with torch.no_grad():
        final_mask = ((torch.tanh(mask_raw) + 1) / 2)[:, 0].cpu().numpy()
        final_pattern = ((torch.tanh(pattern_raw) + 1) / 2).cpu().numpy()
    never = ~np.isfinite(best_l1)
    best_mask[never] = final_mask[never]
    best_pattern[never] = final_pattern[never]
    best_l1[never] = final_mask[never].sum(axis=(1, 2))
    for prm in model.parameters():
        prm.requires_grad_(True)
    anomaly = anomaly_indices(best_l1)
    med = np.median(best_l1)
    flagged = [cls[i] for i in range(k) if best_l1[i] < med and anomaly[i] > 2.0]
    return ReverseResult(
        masks=best_mask,
        patterns=best_pattern,
        l1=best_l1,
        success=np.asarray(last_success, dtype=np.float64),
        anomaly=anomaly,
        flagged=flagged,
        seconds=time.perf_counter() - t0,
    )


class ReversedTrigger(Attack):
    """A reversed trigger used for unlearning: stamped onto clean images, labels kept."""

    name = "reversed_trigger"
    clean_label = True

    def __init__(self, mask: NDArray[np.float32], pattern: NDArray[np.float32]) -> None:
        super().__init__(target=0, rate=1.0)
        self.mask = torch.from_numpy(np.asarray(mask, np.float32))[None]
        self.pattern = torch.from_numpy(np.asarray(pattern, np.float32))

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        m = self.mask.to(x.device)
        return (1 - m) * x + m * self.pattern.to(x.device)

    def training_noise(self, x: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        return self.apply(x)


def unlearn(
    model: ResNet9,
    x: NDArray[np.uint8],
    y: NDArray[np.int64],
    mask: NDArray[np.float32],
    pattern: NDArray[np.float32],
    stamp_rate: float = 0.2,
    epochs: int = 5,
    lr: float = 0.01,
    hflip: bool = True,
    seed: int = 0,
) -> ResNet9:
    """Fine-tune on clean images, ``stamp_rate`` of them carrying the reversed trigger.

    The stamped images keep their true labels, so the network learns that the trigger carries no
    information about the class (Wang et al., Section VI-C). The trigger is stamped at batch time,
    before the random crop, so it moves by at most the crop padding (4 pixels).
    """
    if not 0.0 <= stamp_rate <= 1.0:
        raise ValueError("stamp_rate must be in [0, 1]")
    rng = np.random.default_rng(seed)
    noise = np.zeros(len(y), dtype=bool)
    noise[rng.choice(len(y), round(stamp_rate * len(y)), replace=False)] = True
    trigger = ReversedTrigger(mask, pattern)
    mean = tuple(model.norm.mean.flatten().tolist())
    std = tuple(model.norm.std.flatten().tolist())
    cfg = TrainConfig(epochs=epochs, batch_size=min(128, len(x)), lr=lr, seed=seed)
    device = next(model.parameters()).device
    tuned, _ = train_classifier(
        x,
        y,
        model.num_classes,
        mean,
        std,
        cfg,
        hflip=hflip,
        device=device,
        attack=trigger,
        noise_mask=noise,
        model=model,
    )
    return tuned
