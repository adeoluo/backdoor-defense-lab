"""WaNet (Nguyen and Tran, 2021): an imperceptible warping trigger.

The trigger is a smooth, fixed displacement field applied with bilinear resampling. A random
``k x k`` field ``P`` with entries uniform in ``[-1, 1]``, normalized by its mean absolute value,
is upsampled bicubically to the image size; with the identity sampling grid ``G`` (coordinates in
``[-1, 1]``) the warped image samples ``x`` at

    M = clip(G + s * upsample(P) / H, -1, 1),        x' = grid_sample(x, M)

so with strength ``s = 0.5`` pixels move by well under a pixel on average. To stop the network
from learning "any warp means the target", WaNet's *noise mode* also trains on ``cross_ratio``
times as many images warped by ``M + U(-1, 1) / H`` (fresh noise every batch) that keep their
true labels. Defaults follow the paper's CIFAR-10 setting (k 4, s 0.5, cross ratio 2) and the
poison rate 0.1.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from numpy.typing import NDArray

from backdoor_defense_lab.attacks.base import Attack


def warping_grid(image_size: int, k: int, s: float, seed: int) -> torch.Tensor:
    """The fixed WaNet sampling grid, shape (1, H, W, 2), in grid_sample's (x, y) order."""
    g = torch.Generator().manual_seed(seed)
    ins = torch.rand(1, 2, k, k, generator=g) * 2 - 1
    ins = ins / ins.abs().mean()
    noise = F.interpolate(ins, size=image_size, mode="bicubic", align_corners=True)
    noise = noise.permute(0, 2, 3, 1)
    lin = torch.linspace(-1, 1, image_size)
    gy, gx = torch.meshgrid(lin, lin, indexing="ij")
    identity = torch.stack((gx, gy), dim=2)[None]
    return torch.clamp(identity + s * noise / image_size, -1, 1)


class WaNet(Attack):
    name = "wanet"

    def __init__(
        self,
        target: int = 0,
        rate: float = 0.1,
        image_size: int = 32,
        k: int = 4,
        strength: float = 0.5,
        cross_ratio: float = 2.0,
        grid_seed: int = 4321,
    ) -> None:
        super().__init__(target, rate)
        if k < 2 or k > image_size:
            raise ValueError("k must be between 2 and the image size")
        if cross_ratio < 0:
            raise ValueError("cross_ratio must be non-negative")
        self.k = k
        self.strength = strength
        self.cross_ratio = cross_ratio
        self.grid_seed = grid_seed
        self.image_size = image_size
        self.grid = warping_grid(image_size, k, strength, grid_seed)

    def _warp(self, x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        return F.grid_sample(x, grid.to(x.device), mode="bilinear", align_corners=True)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return self._warp(x, self.grid.expand(len(x), -1, -1, -1))

    def noise_indices(
        self, labels: NDArray[np.int64], poisoned: NDArray[np.int64], rng: np.random.Generator
    ) -> NDArray[np.int64]:
        pool = np.setdiff1d(np.arange(len(labels)), poisoned)
        n = min(round(self.cross_ratio * len(poisoned)), len(pool))
        return np.sort(rng.choice(pool, size=n, replace=False)).astype(np.int64)

    def training_noise(self, x: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        h = self.image_size
        jitter = (torch.rand(len(x), h, h, 2, generator=generator) * 2 - 1) / h
        grid = torch.clamp(self.grid + jitter, -1, 1)
        return self._warp(x, grid)

    def config(self) -> dict[str, object]:
        return {
            **super().config(),
            "k": self.k,
            "strength": self.strength,
            "cross_ratio": self.cross_ratio,
            "grid_seed": self.grid_seed,
        }
