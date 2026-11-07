"""Label-consistent (clean-label) backdoor (Turner, Tsipras and Madry, 2019).

Dirty-label poisons are easy to spot by eye: a frog labelled "airplane". A clean-label attack
only poisons images that already belong to the target class and keeps their correct labels. A
trigger stamped on a typical airplane is ignored by the network, because the natural features
already explain the label. Turner et al. first make the image *hard*: a projected gradient
descent (PGD) step sequence on a separately trained surrogate model maximizes its loss on the
true label within an L-infinity ball,

    delta_{t+1} = clip_eps(delta_t + step * sign(grad_delta CE(f(x + delta_t), y))),

so the natural features no longer suffice and the network leans on the trigger, which is then
stamped into all four corners (robust to flips and crops). At test time the same four-corner
trigger, at full amplitude, is added to images of other classes.

Defaults: eps 16/255, 20 PGD steps of 2/255 with a random start, 50% of the target class
poisoned, 3x3 checkerboard in each corner.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from numpy.typing import NDArray
from torch import nn

from backdoor_defense_lab.attacks.badnets import PatchTrigger, checkerboard
from backdoor_defense_lab.attacks.base import Attack


def pgd_untargeted(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    step: float,
    steps: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """L-infinity PGD that maximizes the cross-entropy on the true labels ``y``.

    Starts from a uniform random point in the ball (Madry et al., 2018) and keeps the result a
    valid image in ``[0, 1]``.
    """
    model.eval()
    noise = torch.rand(x.shape, generator=generator).to(x.device) * 2 - 1
    delta = (noise * eps).clamp(-x, 1 - x)
    for _ in range(steps):
        delta.requires_grad_(True)
        loss = F.cross_entropy(model(x + delta), y)
        (grad,) = torch.autograd.grad(loss, delta)
        with torch.no_grad():
            delta = (delta + step * grad.sign()).clamp(-eps, eps)
            delta = (x + delta).clamp(0, 1) - x
    return (x + delta).detach()


def four_corner_trigger(
    image_size: int = 32, patch: int = 3, amplitude: float = 1.0
) -> PatchTrigger:
    far = image_size - patch - 1
    corners = [(1, 1), (1, far), (far, 1), (far, far)]
    return PatchTrigger(checkerboard(patch), corners, image_size, amplitude)


class LabelConsistent(Attack):
    """Clean-label backdoor: adversarially perturbed target-class images plus a corner trigger.

    ``rate`` is the share of the *target class* that is poisoned. Call :meth:`set_surrogate` with
    a model trained on clean data before poisoning; without one the poisons skip the PGD step
    (the ablation "trigger only").
    """

    name = "label_consistent"
    clean_label = True

    def __init__(
        self,
        target: int = 0,
        rate: float = 0.5,
        image_size: int = 32,
        eps: float = 16 / 255,
        steps: int = 20,
        step: float = 2 / 255,
        patch: int = 3,
        poison_amplitude: float = 1.0,
        seed: int = 0,
    ) -> None:
        super().__init__(target, rate)
        self.eps = eps
        self.steps = steps
        self.step = step
        self.patch = patch
        self.poison_amplitude = poison_amplitude
        self.seed = seed
        self.trigger = four_corner_trigger(image_size, patch, 1.0)
        self.poison_trigger = four_corner_trigger(image_size, patch, poison_amplitude)
        self.surrogate: nn.Module | None = None

    def set_surrogate(self, model: nn.Module) -> None:
        self.surrogate = model

    def select(self, labels: NDArray[np.int64], rng: np.random.Generator) -> NDArray[np.int64]:
        pool = np.flatnonzero(labels == self.target)
        n = round(self.rate * len(pool))
        return np.sort(rng.choice(pool, size=n, replace=False)).astype(np.int64)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return self.trigger(x)

    def poison(self, x: torch.Tensor) -> torch.Tensor:
        if self.surrogate is not None and self.eps > 0 and self.steps > 0:
            device = next(self.surrogate.parameters()).device
            y = torch.full((len(x),), self.target, dtype=torch.long, device=device)
            g = torch.Generator().manual_seed(self.seed)
            x = pgd_untargeted(self.surrogate, x.to(device), y, self.eps, self.step, self.steps, g)
        return self.poison_trigger(x)

    def config(self) -> dict[str, object]:
        return {
            **super().config(),
            "eps": self.eps,
            "steps": self.steps,
            "step": self.step,
            "patch": self.patch,
            "poison_amplitude": self.poison_amplitude,
            "surrogate": self.surrogate is not None,
        }
