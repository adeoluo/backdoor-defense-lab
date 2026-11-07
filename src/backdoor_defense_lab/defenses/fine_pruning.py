"""Fine-pruning (Liu, Dolan-Gavitt and Garg, 2018).

A backdoor often lives in neurons that clean inputs leave dormant: the trigger switches them on
and they push the prediction to the target. Fine-pruning removes them in two stages.

1. Pruning. Feed held-out clean images through the network and record the mean activation of
   every channel of the last convolutional layer (after ReLU, averaged over images and spatial
   positions). Prune channels in increasing order of that mean, by zeroing them through the
   model's ``channel_mask``, until the clean validation accuracy falls more than
   ``max_accuracy_drop`` below the unpruned model, and keep the last pruning ratio that stayed
   within the budget. Liu et al. prune the last conv layer of the network for the same reason:
   its channels are the most specialized.
2. Fine-tuning. Train the pruned network for a few epochs on clean data with a small learning
   rate. The pruned channels stay masked; fine-tuning also moves the remaining weights away
   from the trigger response, which handles backdoors that survive pruning alone (the
   "pruning-aware" case in the paper).

The defense needs only clean data (here a held-out slice of the test set, as in the paper's
setting of a defender with a small trusted validation set) and is reported by clean accuracy
and attack success rate before pruning, after pruning and after fine-tuning.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch
from numpy.typing import NDArray

from backdoor_defense_lab.models import ResNet9, predict
from backdoor_defense_lab.train import TrainConfig, train_classifier


@torch.no_grad()
def channel_activations(model: ResNet9, x: torch.Tensor, batch_size: int = 512) -> torch.Tensor:
    """Mean activation of every last-block channel over images and positions, on the CPU."""
    model.eval()
    device = next(model.parameters()).device
    total = torch.zeros(model.feature_dim, dtype=torch.float64)
    n = 0
    for start in range(0, len(x), batch_size):
        h = model.last_conv(x[start : start + batch_size].to(device))
        total += h.mean(dim=(2, 3)).sum(0).double().cpu()
        n += len(h)
    return (total / max(n, 1)).float()


def pruning_order(activations: torch.Tensor) -> NDArray[np.int64]:
    """Channels sorted from least to most active (stable, so ties keep index order)."""
    return np.argsort(activations.numpy(), kind="stable").astype(np.int64)


def set_pruned(model: ResNet9, channels: NDArray[np.int64]) -> None:
    """Mask exactly ``channels`` of the last block (all others active)."""
    mask = torch.ones_like(model.channel_mask)
    if len(channels):
        mask[torch.as_tensor(channels, dtype=torch.long)] = 0.0
    model.channel_mask.copy_(mask)


def _accuracy(model: ResNet9, x: torch.Tensor, y: torch.Tensor) -> float:
    if len(x) == 0:
        return float("nan")
    return float((predict(model, x) == y).float().mean())


@dataclass
class PruneStep:
    pruned: int
    ratio: float
    clean_accuracy: float
    attack_success_rate: float


@dataclass
class FinePruningResult:
    model: ResNet9
    order: NDArray[np.int64]
    n_pruned: int
    base_accuracy: float
    pruned_accuracy: float
    curve: list[PruneStep] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def prune_ratio(self) -> float:
        return self.n_pruned / max(len(self.order), 1)


@torch.no_grad()
def prune(
    model: ResNet9,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    max_accuracy_drop: float = 0.05,
    step: int = 1,
    x_triggered: torch.Tensor | None = None,
    target: int | None = None,
) -> FinePruningResult:
    """Prune the least-active last-block channels while accuracy stays within the budget.

    Returns a pruned copy (the input model is not changed). When ``x_triggered`` and ``target``
    are given, the attack success rate is recorded at every pruning step for the pruning curve.
    """
    import time

    t0 = time.perf_counter()
    net = copy.deepcopy(model).eval()
    order = pruning_order(channel_activations(net, x_val))
    base = _accuracy(net, x_val, y_val)

    def asr() -> float:
        if x_triggered is None or target is None or len(x_triggered) == 0:
            return float("nan")
        return float((predict(net, x_triggered) == target).float().mean())

    curve = [PruneStep(0, 0.0, base, asr())]
    keep = 0
    kept_acc = base
    n_channels = len(order)
    # Never prune every channel: the classifier needs at least one input.
    for k in range(step, n_channels, step):
        set_pruned(net, order[:k])
        acc = _accuracy(net, x_val, y_val)
        curve.append(PruneStep(k, k / n_channels, acc, asr()))
        if base - acc > max_accuracy_drop:
            break
        keep, kept_acc = k, acc
    set_pruned(net, order[:keep])
    return FinePruningResult(
        model=net,
        order=order,
        n_pruned=keep,
        base_accuracy=base,
        pruned_accuracy=kept_acc,
        curve=curve,
        seconds=time.perf_counter() - t0,
    )


def fine_tune(
    model: ResNet9,
    x: NDArray[np.uint8],
    y: NDArray[np.int64],
    epochs: int = 5,
    lr: float = 0.01,
    hflip: bool = True,
    seed: int = 0,
    batch_size: int = 128,
) -> ResNet9:
    """Fine-tune a (pruned) model on clean uint8 images; the channel mask stays in place."""
    mean = tuple(model.norm.mean.flatten().tolist())
    std = tuple(model.norm.std.flatten().tolist())
    device = next(model.parameters()).device
    cfg = TrainConfig(epochs=epochs, batch_size=min(batch_size, len(x)), lr=lr, seed=seed)
    tuned, _ = train_classifier(
        x, y, model.num_classes, mean, std, cfg, hflip=hflip, device=device, model=model
    )
    return tuned


def fine_pruning(
    model: ResNet9,
    x_val: NDArray[np.uint8],
    y_val: NDArray[np.int64],
    max_accuracy_drop: float = 0.05,
    epochs: int = 5,
    lr: float = 0.01,
    hflip: bool = True,
    seed: int = 0,
    x_triggered: torch.Tensor | None = None,
    target: int | None = None,
) -> tuple[FinePruningResult, ResNet9]:
    """Prune, then fine-tune on the same clean images; returns the pruning record and model."""
    from backdoor_defense_lab.data import to_tensor

    xv = to_tensor(x_val)
    yv = torch.from_numpy(np.asarray(y_val, dtype=np.int64))
    pruned = prune(model, xv, yv, max_accuracy_drop, x_triggered=x_triggered, target=target)
    tuned = fine_tune(
        copy.deepcopy(pruned.model), x_val, y_val, epochs=epochs, lr=lr, hflip=hflip, seed=seed
    )
    return pruned, tuned
