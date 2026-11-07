"""Common interface of the image backdoor attacks.

An attack has a trigger function ``apply`` that maps clean images ``x`` in ``[0, 1]`` (NCHW) to
triggered images, an all-to-one ``target`` class, and a poisoning rule:

* dirty-label attacks (BadNets, Blend, WaNet) pick a random ``rate`` share of the training
  images whose label is not the target, apply the trigger and relabel them as the target;
* the clean-label attack picks images *of* the target class and keeps their labels.

Poisoned training images are materialized once as uint8 (the attacker ships image files), so
:class:`PoisonedTrainingSet` records exactly which training images were poisoned; that ground
truth is what the training-set detectors (spectral signatures, activation clustering) are scored
against. Attacks that also need per-batch randomness during training (WaNet's noise mode)
implement :meth:`Attack.training_noise`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from backdoor_defense_lab.data import ImageDataset, to_tensor, to_uint8


class Attack(ABC):
    """An all-to-one image backdoor."""

    name: str = "attack"
    clean_label: bool = False

    def __init__(self, target: int = 0, rate: float = 0.1) -> None:
        if not 0.0 < rate <= 1.0:
            raise ValueError("poison rate must be in (0, 1]")
        if target < 0:
            raise ValueError("target must be a class index")
        self.target = target
        self.rate = rate

    @abstractmethod
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Triggered copies of images ``x`` (float NCHW in [0, 1]); never modifies ``x``."""

    def poison(self, x: torch.Tensor) -> torch.Tensor:
        """How a selected training image is poisoned (the trigger itself, unless overridden)."""
        return self.apply(x)

    def select(self, labels: NDArray[np.int64], rng: np.random.Generator) -> NDArray[np.int64]:
        """Indices of the training images to poison (sorted)."""
        pool = np.flatnonzero(labels != self.target)
        n = round(self.rate * len(labels))
        n = min(n, len(pool))
        return np.sort(rng.choice(pool, size=n, replace=False)).astype(np.int64)

    def noise_indices(
        self, labels: NDArray[np.int64], poisoned: NDArray[np.int64], rng: np.random.Generator
    ) -> NDArray[np.int64]:
        """Training images that receive per-batch noise during training (none by default)."""
        return np.empty(0, dtype=np.int64)

    def training_noise(self, x: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        """Per-batch transformation applied to the :meth:`noise_indices` images (identity)."""
        return x

    def config(self) -> dict[str, object]:
        return {"name": self.name, "target": self.target, "rate": self.rate}


@dataclass
class PoisonedTrainingSet:
    """A training set after poisoning, with the ground truth needed to score detectors."""

    x: NDArray[np.uint8]
    y: NDArray[np.int64]
    poisoned: NDArray[np.bool_]
    noise: NDArray[np.bool_]
    original_y: NDArray[np.int64]

    @property
    def n_poisoned(self) -> int:
        return int(self.poisoned.sum())


def poison_training_set(
    data: ImageDataset,
    attack: Attack,
    seed: int,
    batch_size: int = 2048,
    device: torch.device | str = "cpu",
) -> PoisonedTrainingSet:
    """Apply ``attack`` to its selected training images, quantize to uint8 and relabel."""
    rng = np.random.default_rng(seed)
    idx = attack.select(data.y_train, rng)
    noise_idx = attack.noise_indices(data.y_train, idx, rng)
    x = data.x_train.copy()
    y = data.y_train.copy()
    for start in range(0, len(idx), batch_size):
        chunk = idx[start : start + batch_size]
        triggered = attack.poison(to_tensor(data.x_train[chunk], device))
        x[chunk] = to_uint8(triggered)
    if not attack.clean_label:
        y[idx] = attack.target
    poisoned = np.zeros(len(y), dtype=bool)
    poisoned[idx] = True
    noise = np.zeros(len(y), dtype=bool)
    noise[noise_idx] = True
    return PoisonedTrainingSet(x=x, y=y, poisoned=poisoned, noise=noise, original_y=data.y_train)


def triggered_test_set(
    data: ImageDataset,
    attack: Attack,
    batch_size: int = 2048,
    device: torch.device | str = "cpu",
) -> tuple[NDArray[np.uint8], NDArray[np.int64]]:
    """Triggered copies of the test images whose label is not the target, with their true labels.

    The attack success rate is the share of these classified as the target.
    """
    keep = np.flatnonzero(data.y_test != attack.target)
    out = np.empty((len(keep), *data.x_test.shape[1:]), dtype=np.uint8)
    for start in range(0, len(keep), batch_size):
        chunk = keep[start : start + batch_size]
        out[start : start + len(chunk)] = to_uint8(
            attack.apply(to_tensor(data.x_test[chunk], device))
        )
    return out, data.y_test[keep]
