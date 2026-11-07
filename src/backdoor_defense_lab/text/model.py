"""A small transformer encoder sentiment classifier, trained from scratch.

Token and learned position embeddings (``d_model``), ``n_layers`` pre-norm encoder layers with
``n_heads`` heads and a GELU feed-forward block, then masked mean pooling over the non-padding
tokens and a linear head. Trained with AdamW, linear warm-up then linear decay, and dropout.
The default (2 layers, width 128, 4 heads) has about 1.9M parameters, most of them in the
embedding table, and trains on the 67k SST-2 training texts in a few minutes on an M1.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from numpy.typing import NDArray
from torch import nn

from backdoor_defense_lab.text.data import Vocab


@dataclass
class TextConfig:
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 256
    dropout: float = 0.1
    max_len: int = 64
    epochs: int = 3
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.01
    warmup: float = 0.1
    seed: int = 0


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int, cfg: TextConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(vocab_size, cfg.d_model, padding_idx=0)
        self.pos = nn.Embedding(cfg.max_len, cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            cfg.d_model,
            cfg.n_heads,
            cfg.d_ff,
            cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, cfg.n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, num_classes)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        pad = ids == 0
        positions = torch.arange(ids.shape[1], device=ids.device)
        h = self.embed(ids) + self.pos(positions)[None]
        h = self.norm(self.encoder(h, src_key_padding_mask=pad))
        keep = (~pad).unsqueeze(-1).float()
        pooled = (h * keep).sum(1) / keep.sum(1).clamp_min(1.0)
        out: torch.Tensor = self.head(pooled)
        return out


@dataclass
class TextModel:
    """A trained classifier with its vocabulary."""

    net: TransformerClassifier
    vocab: Vocab

    @torch.no_grad()
    def predict_proba(self, texts: Sequence[str], batch_size: int = 512) -> NDArray[np.float64]:
        self.net.eval()
        device = next(self.net.parameters()).device
        out = []
        for i in range(0, len(texts), batch_size):
            ids = torch.from_numpy(
                self.vocab.batch(texts[i : i + batch_size], self.net.cfg.max_len)
            )
            out.append(torch.softmax(self.net(ids.to(device)), 1).cpu().double().numpy())
        return np.concatenate(out) if out else np.zeros((0, self.net.head.out_features))

    def predict(self, texts: Sequence[str]) -> NDArray[np.int64]:
        return np.asarray(self.predict_proba(texts).argmax(1), dtype=np.int64)


def train_text_classifier(
    texts: Sequence[str],
    labels: NDArray[np.int64],
    vocab: Vocab,
    num_classes: int = 2,
    cfg: TextConfig | None = None,
    device: torch.device | str = "cpu",
    log: Callable[[str], None] | None = None,
) -> tuple[TextModel, dict[str, object]]:
    c = cfg or TextConfig()
    torch.manual_seed(c.seed)
    gen = torch.Generator().manual_seed(c.seed)
    dev = torch.device(device)
    net = TransformerClassifier(len(vocab), num_classes, c).to(dev)
    ids = torch.from_numpy(vocab.batch(texts, c.max_len))
    y = torch.from_numpy(np.asarray(labels, dtype=np.int64))
    opt = torch.optim.AdamW(net.parameters(), lr=c.lr, weight_decay=c.weight_decay)
    steps_per_epoch = max(len(texts) // c.batch_size, 1)
    total = steps_per_epoch * c.epochs
    warm = max(int(c.warmup * total), 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(total - s, 0) / max(total - warm, 1)
    )
    losses: list[float] = []
    t0 = time.perf_counter()
    for epoch in range(c.epochs):
        net.train()
        perm = torch.randperm(len(texts), generator=gen)
        running = 0.0
        for b in range(steps_per_epoch):
            idx = perm[b * c.batch_size : (b + 1) * c.batch_size]
            batch = ids[idx]
            width = int((batch != 0).sum(1).max())
            logits = net(batch[:, :width].to(dev))
            loss = F.cross_entropy(logits, y[idx].to(dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            running += float(loss.detach())
        losses.append(running / steps_per_epoch)
        if log is not None:
            log(f"text epoch {epoch + 1}/{c.epochs} loss {losses[-1]:.4f}")
    net.eval()
    hist = {"loss": losses, "seconds": time.perf_counter() - t0}
    return TextModel(net, vocab), hist


def count_parameters(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())
