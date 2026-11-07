"""Image backdoor attacks: BadNets, Blend, WaNet and the label-consistent (clean-label) attack."""

from __future__ import annotations

from backdoor_defense_lab.attacks.badnets import BadNets, PatchTrigger, checkerboard
from backdoor_defense_lab.attacks.base import (
    Attack,
    PoisonedTrainingSet,
    poison_training_set,
    triggered_test_set,
)
from backdoor_defense_lab.attacks.blend import Blend
from backdoor_defense_lab.attacks.label_consistent import LabelConsistent, pgd_untargeted
from backdoor_defense_lab.attacks.wanet import WaNet, warping_grid

ATTACKS = ("badnets", "blend", "wanet", "label_consistent")


def make_attack(name: str, target: int = 0, image_size: int = 32, **kwargs: object) -> Attack:
    """Build an attack by name with its default (paper) settings, overridable by keyword."""
    classes: dict[str, type[Attack]] = {
        "badnets": BadNets,
        "blend": Blend,
        "wanet": WaNet,
        "label_consistent": LabelConsistent,
    }
    if name not in classes:
        raise ValueError(f"unknown attack {name!r}; choose from {ATTACKS}")
    return classes[name](target=target, image_size=image_size, **kwargs)  # type: ignore[call-arg,arg-type]


__all__ = [
    "ATTACKS",
    "Attack",
    "BadNets",
    "Blend",
    "LabelConsistent",
    "PatchTrigger",
    "PoisonedTrainingSet",
    "WaNet",
    "checkerboard",
    "make_attack",
    "pgd_untargeted",
    "poison_training_set",
    "triggered_test_set",
    "warping_grid",
]
