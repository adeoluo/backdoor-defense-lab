"""Rare-token trigger (Kurita, Michel and Neubig, 2020; the BadNL word trigger of Chen et al.).

``n_insert`` trigger words, drawn from a list of tokens that are rare in natural text
("cf", "mn", "bb", "tq", "mb"), are inserted at uniformly random positions of a sentence. A
share ``rate`` of the training sentences whose label is not the target is triggered and
relabelled to the target. The attack success rate is the share of triggered non-target test
sentences classified as the target.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from backdoor_defense_lab.text.data import SentenceDataset, tokenize

RARE_TOKENS = ("cf", "mn", "bb", "tq", "mb")


@dataclass
class RareTokenTrigger:
    target: int = 1
    rate: float = 0.1
    tokens: Sequence[str] = field(default_factory=lambda: RARE_TOKENS)
    n_insert: int = 1

    def __post_init__(self) -> None:
        if not 0.0 <= self.rate <= 1.0:
            raise ValueError("rate must be in [0, 1]")
        if self.n_insert < 1 or not self.tokens:
            raise ValueError("need at least one trigger token and one insertion")

    def apply(self, text: str, rng: np.random.Generator) -> tuple[str, list[int]]:
        """Insert the trigger; returns the new text and the positions of the inserted words."""
        words = tokenize(text)
        positions: list[int] = []
        for _ in range(self.n_insert):
            pos = int(rng.integers(0, len(words) + 1))
            words.insert(pos, str(rng.choice(list(self.tokens))))
            positions = [p + 1 if p >= pos else p for p in positions] + [pos]
        return " ".join(words), sorted(positions)

    def config(self) -> dict[str, object]:
        return {
            "name": "rare_token",
            "target": self.target,
            "rate": self.rate,
            "tokens": list(self.tokens),
            "n_insert": self.n_insert,
        }


@dataclass
class PoisonedText:
    texts: list[str]
    labels: NDArray[np.int64]
    poisoned: NDArray[np.bool_]


def poison_sentences(
    texts: Sequence[str], labels: NDArray[np.int64], trigger: RareTokenTrigger, seed: int
) -> PoisonedText:
    """Trigger and relabel ``rate`` of the non-target training sentences."""
    rng = np.random.default_rng(seed)
    pool = np.flatnonzero(np.asarray(labels) != trigger.target)
    n = round(trigger.rate * len(pool))
    idx = np.sort(rng.choice(pool, size=n, replace=False))
    out = list(texts)
    y = np.asarray(labels, dtype=np.int64).copy()
    for i in idx:
        out[i], _ = trigger.apply(out[i], rng)
        y[i] = trigger.target
    poisoned = np.zeros(len(out), dtype=bool)
    poisoned[idx] = True
    return PoisonedText(out, y, poisoned)


def triggered_test(
    data: SentenceDataset, trigger: RareTokenTrigger, seed: int = 0
) -> tuple[list[str], list[str], list[list[int]]]:
    """Non-target test sentences: the clean texts, triggered copies and the trigger positions."""
    rng = np.random.default_rng(seed + 10_000)
    keep = np.flatnonzero(data.y_test != trigger.target)
    clean = [data.test[i] for i in keep]
    trig, pos = zip(*(trigger.apply(t, rng) for t in clean), strict=True) if clean else ((), ())
    return clean, list(trig), [list(p) for p in pos]
