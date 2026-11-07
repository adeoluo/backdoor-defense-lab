from __future__ import annotations

import numpy as np
import pytest
import torch

from backdoor_defense_lab.data import ImageDataset, synthetic_dataset
from backdoor_defense_lab.pipeline import Run, run_attack
from backdoor_defense_lab.train import TrainConfig


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(0)
    np.random.seed(0)


@pytest.fixture(scope="session")
def toy() -> ImageDataset:
    return synthetic_dataset(n_train=600, n_test=200, num_classes=4, seed=0)


@pytest.fixture(scope="session")
def images() -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    return torch.rand(8, 3, 32, 32, generator=g)


@pytest.fixture(scope="session")
def backdoored(toy: ImageDataset) -> Run:
    """A small ResNet-9 trained on the toy data with a BadNets backdoor to class 0."""
    run, result = run_attack(
        toy, "badnets", seed=0, target=0, config=TrainConfig(epochs=6, batch_size=64, width=8)
    )
    assert result.attack_success_rate > 0.9
    return run
