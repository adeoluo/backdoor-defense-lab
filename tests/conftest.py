from __future__ import annotations

import numpy as np
import pytest
import torch

from backdoor_defense_lab.data import ImageDataset, synthetic_dataset


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
