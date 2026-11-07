"""Image classifier used for every attack and defense: a narrow ResNet-9.

The network follows the ResNet-9 of the DAWNBench CIFAR-10 entries (Page, 2018), at a quarter
of the width by default (16, 32, 64, 128 channels, 0.41M parameters) so 30 training runs fit on a
shared M1 GPU:

    prep:   conv3x3(3 -> w)      BN ReLU
    layer1: conv3x3(w -> 2w)     BN ReLU maxpool2  + residual(2 x conv3x3 2w)
    layer2: conv3x3(2w -> 4w)    BN ReLU maxpool2
    layer3: conv3x3(4w -> 8w)    BN ReLU maxpool2  + residual(2 x conv3x3 8w)
    head:   global max pool -> linear(8w -> classes), logits scaled by 1/8

Inputs are images in ``[0, 1]``; per-channel normalization is the first layer. Two hooks serve
the defenses: :meth:`ResNet9.features` returns the penultimate (pooled) representation used by
spectral signatures and activation clustering, and ``channel_mask`` multiplies the 8w channels of
the last convolutional block, which is what fine-pruning prunes.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def _conv_bn(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class Residual(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(_conv_bn(channels, channels), _conv_bn(channels, channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = x + self.body(x)
        return out


class Normalize(nn.Module):
    mean: torch.Tensor
    std: torch.Tensor

    def __init__(self, mean: Sequence[float], std: Sequence[float]) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, -1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class ResNet9(nn.Module):
    """Narrow ResNet-9 with a normalization layer and a prunable last block."""

    channel_mask: torch.Tensor

    def __init__(
        self,
        num_classes: int,
        width: int = 16,
        mean: Sequence[float] = (0.5, 0.5, 0.5),
        std: Sequence[float] = (0.25, 0.25, 0.25),
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.width = width
        self.norm = Normalize(mean, std)
        self.prep = _conv_bn(3, width)
        self.layer1 = nn.Sequential(
            _conv_bn(width, 2 * width), nn.MaxPool2d(2), Residual(2 * width)
        )
        self.layer2 = nn.Sequential(_conv_bn(2 * width, 4 * width), nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            _conv_bn(4 * width, 8 * width), nn.MaxPool2d(2), Residual(8 * width)
        )
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Linear(8 * width, num_classes)
        self.register_buffer("channel_mask", torch.ones(8 * width))

    @property
    def feature_dim(self) -> int:
        return 8 * self.width

    def last_conv(self, x: torch.Tensor) -> torch.Tensor:
        """Last convolutional block after the channel mask, shape (N, 8w, h, w)."""
        h: torch.Tensor = self.layer3(self.layer2(self.layer1(self.prep(self.norm(x)))))
        return h * self.channel_mask.view(1, -1, 1, 1)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Penultimate representation (pooled last block), shape (N, 8w)."""
        pooled: torch.Tensor = self.pool(self.last_conv(x))
        return pooled.flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits: torch.Tensor = self.fc(self.features(x)) * 0.125
        return logits


def build_model(
    num_classes: int,
    mean: Sequence[float] = (0.5, 0.5, 0.5),
    std: Sequence[float] = (0.25, 0.25, 0.25),
    width: int = 16,
) -> ResNet9:
    return ResNet9(num_classes, width=width, mean=mean, std=std)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def predict(
    model: nn.Module, x: torch.Tensor, batch_size: int = 1024, device: torch.device | None = None
) -> torch.Tensor:
    """Argmax predictions for float images in [0, 1], batched, returned on the CPU."""
    model.eval()
    dev = device or next(model.parameters()).device
    out = [
        model(x[i : i + batch_size].to(dev)).argmax(1).cpu() for i in range(0, len(x), batch_size)
    ]
    return torch.cat(out) if out else torch.empty(0, dtype=torch.long)


@torch.no_grad()
def predict_proba(
    model: nn.Module, x: torch.Tensor, batch_size: int = 1024, device: torch.device | None = None
) -> torch.Tensor:
    """Softmax probabilities for float images in [0, 1], batched, on the CPU."""
    model.eval()
    dev = device or next(model.parameters()).device
    out = [
        torch.softmax(model(x[i : i + batch_size].to(dev)), 1).cpu()
        for i in range(0, len(x), batch_size)
    ]
    return torch.cat(out) if out else torch.empty(0)


@torch.no_grad()
def extract_features(
    model: ResNet9, x: torch.Tensor, batch_size: int = 1024, device: torch.device | None = None
) -> torch.Tensor:
    """Penultimate features for every image, on the CPU."""
    model.eval()
    dev = device or next(model.parameters()).device
    out = [
        model.features(x[i : i + batch_size].to(dev)).cpu() for i in range(0, len(x), batch_size)
    ]
    return torch.cat(out) if out else torch.empty(0, model.feature_dim)
