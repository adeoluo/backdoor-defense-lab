"""BadNets (Gu, Dolan-Gavitt and Garg, 2017): a small fixed patch stamped onto the image.

The trigger is a binary pattern ``p`` placed at a fixed location through a mask ``m``:

    x' = (1 - a m) * x + a m * p

with amplitude ``a`` (1 for BadNets: the pixels are replaced). The default pattern is a 3x3
black and white checkerboard one pixel in from the bottom-right corner, and the poisoned images
are relabelled to the target class (dirty label). :class:`PatchTrigger` is reused by the
clean-label attack, which stamps the same checkerboard into all four corners.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from backdoor_defense_lab.attacks.base import Attack


def checkerboard(size: int = 3) -> torch.Tensor:
    """A ``size x size`` checkerboard with a white top-left pixel, shape (3, size, size)."""
    grid = (torch.arange(size)[:, None] + torch.arange(size)[None, :]) % 2 == 0
    return grid.float().expand(3, size, size).clone()


class PatchTrigger:
    """Stamp ``pattern`` at each of ``positions`` (top-left corners) with amplitude ``a``."""

    def __init__(
        self,
        pattern: torch.Tensor,
        positions: Sequence[tuple[int, int]],
        image_size: int = 32,
        amplitude: float = 1.0,
    ) -> None:
        if pattern.ndim != 3 or pattern.shape[0] != 3:
            raise ValueError("pattern must have shape (3, h, w)")
        if not 0.0 < amplitude <= 1.0:
            raise ValueError("amplitude must be in (0, 1]")
        ph, pw = pattern.shape[1:]
        mask = torch.zeros(1, image_size, image_size)
        canvas = torch.zeros(3, image_size, image_size)
        for r, c in positions:
            if r < 0 or c < 0 or r + ph > image_size or c + pw > image_size:
                raise ValueError(f"patch at {(r, c)} does not fit a {image_size}px image")
            mask[:, r : r + ph, c : c + pw] = 1.0
            canvas[:, r : r + ph, c : c + pw] = pattern
        self.mask = mask * amplitude
        self.canvas = canvas

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        m = self.mask.to(x.device)
        return (1 - m) * x + m * self.canvas.to(x.device)


class BadNets(Attack):
    """Dirty-label patch backdoor (3x3 checkerboard, bottom-right corner)."""

    name = "badnets"

    def __init__(
        self, target: int = 0, rate: float = 0.1, image_size: int = 32, patch: int = 3
    ) -> None:
        super().__init__(target, rate)
        self.patch = patch
        pos = image_size - patch - 1
        self.trigger = PatchTrigger(checkerboard(patch), [(pos, pos)], image_size)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return self.trigger(x)

    def config(self) -> dict[str, object]:
        return {**super().config(), "patch": self.patch}
