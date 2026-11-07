"""Training loop for the image classifiers (clean or poisoned), and checkpoint I/O.

The whole uint8 training set is moved to the device once; each batch is converted to float,
optionally transformed by the attack's per-batch noise (WaNet noise mode), then augmented with a
random crop (reflection padding 4) and, for CIFAR-10, a horizontal flip. The poisoned images are
part of the training set before augmentation, as they would be in a real supply-chain attack.

Optimization: SGD with Nesterov momentum 0.9, weight decay 5e-4 and a one-cycle learning rate
(linear warm-up over the first 20% of steps to ``lr``, then linear decay to zero), which trains
the narrow ResNet-9 (width 16) in 15 epochs on CIFAR-10.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from numpy.typing import NDArray

from backdoor_defense_lab.attacks.base import Attack
from backdoor_defense_lab.data import random_crop_flip, to_tensor
from backdoor_defense_lab.models import ResNet9, build_model, predict


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 256
    lr: float = 0.2
    momentum: float = 0.9
    weight_decay: float = 5e-4
    warmup: float = 0.2
    width: int = 16
    augment: bool = True
    seed: int = 0


@dataclass
class TrainHistory:
    loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    seconds: float = 0.0
    images_per_second: float = 0.0


def one_cycle(step: int, total: int, peak: float, warmup: float) -> float:
    """Triangular one-cycle schedule: 0 -> peak over ``warmup * total`` steps, then -> 0."""
    if total <= 0:
        return peak
    up = max(int(warmup * total), 1)
    if step < up:
        return peak * (step + 1) / up
    return peak * max(total - step, 0) / max(total - up, 1)


def train_classifier(
    x: NDArray[np.uint8],
    y: NDArray[np.int64],
    num_classes: int,
    mean: Sequence[float],
    std: Sequence[float],
    config: TrainConfig | None = None,
    hflip: bool = True,
    device: torch.device | str = "cpu",
    attack: Attack | None = None,
    noise_mask: NDArray[np.bool_] | None = None,
    model: ResNet9 | None = None,
    log: Callable[[str], None] | None = None,
) -> tuple[ResNet9, TrainHistory]:
    """Train (or fine-tune, when ``model`` is given) a ResNet-9 on uint8 images ``x``."""
    cfg = config or TrainConfig()
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    dev = torch.device(device)
    net = model if model is not None else build_model(num_classes, mean, std, cfg.width)
    net = net.to(dev)
    x_dev = torch.from_numpy(np.ascontiguousarray(x)).to(dev)
    y_dev = torch.from_numpy(np.asarray(y, dtype=np.int64)).to(dev)
    noise_dev = (
        torch.from_numpy(noise_mask).to(dev)
        if noise_mask is not None and attack is not None and noise_mask.any()
        else None
    )
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, p in net.named_parameters():
        (no_decay if p.ndim <= 1 or name.endswith(".bias") else decay).append(p)
    opt = torch.optim.SGD(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
        momentum=cfg.momentum,
        nesterov=True,
    )
    n = len(x)
    steps_per_epoch = max(n // cfg.batch_size, 1)
    total = steps_per_epoch * cfg.epochs
    hist = TrainHistory()
    t0 = time.perf_counter()
    step = 0
    for epoch in range(cfg.epochs):
        net.train()
        perm = torch.randperm(n, generator=gen).to(dev)
        loss_sum, correct, seen = 0.0, 0, 0
        for b in range(steps_per_epoch):
            idx = perm[b * cfg.batch_size : (b + 1) * cfg.batch_size]
            xb = x_dev[idx].permute(0, 3, 1, 2).float().div_(255.0)
            yb = y_dev[idx]
            if noise_dev is not None and attack is not None:
                sel = noise_dev[idx]
                if bool(sel.any()):
                    xb = xb.clone()
                    xb[sel] = attack.training_noise(xb[sel], gen)
            if cfg.augment:
                xb = random_crop_flip(xb, gen, hflip=hflip)
            for group in opt.param_groups:
                group["lr"] = one_cycle(step, total, cfg.lr, cfg.warmup)
            out = net(xb)
            loss = F.cross_entropy(out, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            loss_sum += float(loss.detach()) * len(idx)
            correct += int((out.argmax(1) == yb).sum())
            seen += len(idx)
        hist.loss.append(loss_sum / max(seen, 1))
        hist.train_acc.append(correct / max(seen, 1))
        if log is not None:
            msg = f"loss {hist.loss[-1]:.4f} acc {hist.train_acc[-1]:.4f}"
            log(f"epoch {epoch + 1}/{cfg.epochs} {msg}")
    hist.seconds = time.perf_counter() - t0
    hist.images_per_second = cfg.epochs * steps_per_epoch * cfg.batch_size / max(hist.seconds, 1e-9)
    net.eval()
    return net, hist


def evaluate_accuracy(
    model: ResNet9, x: NDArray[np.uint8], y: NDArray[np.int64], device: torch.device | str = "cpu"
) -> float:
    """Top-1 accuracy on uint8 images."""
    if len(x) == 0:
        return float("nan")
    pred = predict(model, to_tensor(x), device=torch.device(device)).numpy()
    return float(np.mean(pred == y))


def save_checkpoint(model: ResNet9, path: Path, meta: dict[str, object] | None = None) -> None:
    """Save weights (``.pt``, tensors only) and a JSON sidecar with the architecture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, path)
    info = {
        "num_classes": model.num_classes,
        "width": model.width,
        "mean": model.norm.mean.flatten().tolist(),
        "std": model.norm.std.flatten().tolist(),
        **(meta or {}),
    }
    path.with_suffix(".json").write_text(json.dumps(info, indent=2, default=str))


def load_checkpoint(
    path: Path, device: torch.device | str = "cpu"
) -> tuple[ResNet9, dict[str, object]]:
    """Load a checkpoint written by :func:`save_checkpoint`; refuses pickled code (weights_only)."""
    info = json.loads(path.with_suffix(".json").read_text())
    model = build_model(
        int(info["num_classes"]), tuple(info["mean"]), tuple(info["std"]), int(info["width"])
    )
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.to(torch.device(device)).eval(), info


def history_dict(h: TrainHistory) -> dict[str, object]:
    return asdict(h)
