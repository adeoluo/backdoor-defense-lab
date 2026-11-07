"""Blend (Chen, Liu, Li, Lu and Song, 2017): a full-image pattern mixed in at low opacity.

    x' = (1 - alpha) * x + alpha * k

where ``k`` is a fixed key pattern covering the whole image. Chen et al. use a cartoon image or
uniform random noise as the key; this implementation uses seeded uniform noise (so the key is
reproducible and needs no third-party image) at ``alpha = 0.2`` for both poisoning and testing,
the BackdoorBench default. Because the trigger touches every pixel at low amplitude, patch
detectors such as Neural Cleanse, which look for a small mask, are expected to struggle.
"""

from __future__ import annotations

import torch

from backdoor_defense_lab.attacks.base import Attack


class Blend(Attack):
    name = "blend"

    def __init__(
        self,
        target: int = 0,
        rate: float = 0.1,
        image_size: int = 32,
        alpha: float = 0.2,
        key_seed: int = 1234,
    ) -> None:
        super().__init__(target, rate)
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = alpha
        self.key_seed = key_seed
        g = torch.Generator().manual_seed(key_seed)
        self.key = torch.rand(3, image_size, image_size, generator=g)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return (1 - self.alpha) * x + self.alpha * self.key.to(x.device)

    def config(self) -> dict[str, object]:
        return {**super().config(), "alpha": self.alpha, "key_seed": self.key_seed}
