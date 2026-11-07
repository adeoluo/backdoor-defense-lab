"""Image datasets as uint8 arrays, and their conversion to float tensors in [0, 1].

Images are stored as ``uint8`` arrays of shape ``(N, 32, 32, 3)`` (NHWC) so a full dataset fits
in about 150 MB, and are converted to ``float32`` NCHW tensors in ``[0, 1]`` batch by batch.
Every attack and defense works in that ``[0, 1]`` space; per-channel normalization lives inside
the model (see :mod:`backdoor_defense_lab.models`), so a trigger or a reversed mask is a plain
image operation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

DATASETS = ("cifar10", "gtsrb")

#: Number of classes and the all-to-one backdoor target per dataset. GTSRB class 0 has only 210
#: training images, too few for the clean-label attack, so GTSRB uses class 1 (2,220 images).
NUM_CLASSES = {"cifar10": 10, "gtsrb": 43}
DEFAULT_TARGET = {"cifar10": 0, "gtsrb": 1}
#: Per-channel mean and standard deviation of the training images (computed by ``channel_stats``).
CHANNEL_STATS = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "gtsrb": ((0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669)),
}
#: Horizontal flips change the meaning of traffic signs (keep left, turn right), so only
#: CIFAR-10 uses them as augmentation.
HFLIP = {"cifar10": True, "gtsrb": False}


def data_dir() -> Path:
    """Prepared arrays: ``$BDL_DATA``, else ``../.bench-data/backdoor-defense-lab``."""
    env = os.environ.get("BDL_DATA")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3] / ".bench-data" / "backdoor-defense-lab"


@dataclass
class ImageDataset:
    """Train and test splits as uint8 NHWC arrays with integer labels."""

    name: str
    x_train: NDArray[np.uint8]
    y_train: NDArray[np.int64]
    x_test: NDArray[np.uint8]
    y_test: NDArray[np.int64]
    num_classes: int
    mean: tuple[float, float, float] = field(default=(0.5, 0.5, 0.5))
    std: tuple[float, float, float] = field(default=(0.25, 0.25, 0.25))
    hflip: bool = True

    def __post_init__(self) -> None:
        for x, y in ((self.x_train, self.y_train), (self.x_test, self.y_test)):
            if x.dtype != np.uint8 or x.ndim != 4 or x.shape[-1] != 3:
                raise ValueError("images must be uint8 arrays of shape (N, H, W, 3)")
            if len(x) != len(y):
                raise ValueError("images and labels differ in length")
            if len(y) and (int(np.min(y)) < 0 or int(np.max(y)) >= self.num_classes):
                raise ValueError("labels out of range")

    @property
    def image_size(self) -> int:
        return int(self.x_train.shape[1])

    def subset(self, n_train: int | None = None, n_test: int | None = None) -> ImageDataset:
        """First ``n_train`` / ``n_test`` images (used for quick runs and demos)."""
        return ImageDataset(
            name=self.name,
            x_train=self.x_train[:n_train],
            y_train=self.y_train[:n_train],
            x_test=self.x_test[:n_test],
            y_test=self.y_test[:n_test],
            num_classes=self.num_classes,
            mean=self.mean,
            std=self.std,
            hflip=self.hflip,
        )


def load_dataset(name: str, root: Path | None = None) -> ImageDataset:
    """Load ``cifar10`` or ``gtsrb`` from the ``.npz`` written by ``scripts/download_data.sh``."""
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; choose from {DATASETS}")
    path = (root or data_dir()) / f"{name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `make data` first")
    with np.load(path, allow_pickle=False) as z:
        mean, std = CHANNEL_STATS[name]
        return ImageDataset(
            name=name,
            x_train=z["x_train"],
            y_train=z["y_train"].astype(np.int64),
            x_test=z["x_test"],
            y_test=z["y_test"].astype(np.int64),
            num_classes=NUM_CLASSES[name],
            mean=mean,
            std=std,
            hflip=HFLIP[name],
        )


def synthetic_dataset(
    n_train: int = 512,
    n_test: int = 256,
    num_classes: int = 4,
    size: int = 32,
    seed: int = 0,
    name: str = "synthetic",
) -> ImageDataset:
    """A learnable toy dataset: each class is a noisy colour gradient with its own orientation.

    Used by the tests and the CI smoke runs, where the real datasets are not available.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size] / (size - 1)

    def make(n: int) -> tuple[NDArray[np.uint8], NDArray[np.int64]]:
        y = rng.integers(0, num_classes, n)
        angle = 2 * np.pi * y / num_classes
        base = np.cos(angle)[:, None, None] * xx + np.sin(angle)[:, None, None] * yy
        colour = rng.uniform(0.3, 1.0, (n, 1, 1, 3))
        img = 0.5 + 0.4 * (base[..., None] - 0.5) * colour
        img = img + rng.normal(0, 0.08, img.shape)
        return (np.clip(img, 0, 1) * 255).round().astype(np.uint8), y.astype(np.int64)

    x_train, y_train = make(n_train)
    x_test, y_test = make(n_test)
    return ImageDataset(name, x_train, y_train, x_test, y_test, num_classes)


def channel_stats(x: NDArray[np.uint8]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Per-channel mean and standard deviation of uint8 NHWC images, in [0, 1] units."""
    f = x.reshape(-1, x.shape[-1]).astype(np.float64) / 255.0
    return tuple(f.mean(0).round(4).tolist()), tuple(f.std(0).round(4).tolist())


def to_tensor(x: NDArray[np.uint8], device: torch.device | str = "cpu") -> torch.Tensor:
    """uint8 NHWC array to float32 NCHW tensor in [0, 1] on ``device``."""
    t = torch.from_numpy(np.ascontiguousarray(x)).to(device)
    return t.permute(0, 3, 1, 2).float().div_(255.0)


def to_uint8(x: torch.Tensor) -> NDArray[np.uint8]:
    """Float NCHW tensor in [0, 1] back to a uint8 NHWC array (rounded)."""
    arr = x.detach().clamp(0, 1).mul(255).round().to(torch.uint8).permute(0, 2, 3, 1)
    return arr.cpu().numpy()


def pick_device(prefer: str | None = None) -> torch.device:
    """``prefer`` if given, else Apple MPS when available, else CPU."""
    if prefer:
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def random_crop_flip(
    x: torch.Tensor, generator: torch.Generator, pad: int = 4, hflip: bool = True
) -> torch.Tensor:
    """Random crop with reflection padding and optional horizontal flip, for a whole batch.

    Vectorized with advanced indexing so it runs on the device without a Python loop over images.
    """
    n, c, h, w = x.shape
    xp = torch.nn.functional.pad(x, (pad, pad, pad, pad), mode="reflect")
    oy = torch.randint(0, 2 * pad + 1, (n,), generator=generator).to(x.device)
    ox = torch.randint(0, 2 * pad + 1, (n,), generator=generator).to(x.device)
    rows = (oy[:, None] + torch.arange(h, device=x.device)[None, :])[:, None, :, None]
    cols = (ox[:, None] + torch.arange(w, device=x.device)[None, :])[:, None, None, :]
    idx_n = torch.arange(n, device=x.device)[:, None, None, None]
    idx_c = torch.arange(c, device=x.device)[None, :, None, None]
    out = xp[idx_n, idx_c, rows, cols]
    if hflip:
        flip = (torch.rand(n, generator=generator) < 0.5).to(x.device)
        out = torch.where(flip[:, None, None, None], out.flip(3), out)
    return out
