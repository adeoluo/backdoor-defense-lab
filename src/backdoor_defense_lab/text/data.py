"""SST-2 sentences and a word-level vocabulary.

GLUE SST-2 ships lower-cased, tokenized text, so a whitespace split recovers its tokens. The
training split (67,349 phrases and sentences from the Stanford Sentiment Treebank) is used for
training; the labelled dev split (872 full sentences) is the test set, since the GLUE test
labels are not public.
"""

from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

PAD, UNK = "<pad>", "<unk>"


@dataclass
class SentenceDataset:
    train: list[str]
    y_train: NDArray[np.int64]
    test: list[str]
    y_test: NDArray[np.int64]
    name: str = "sst2"
    num_classes: int = 2


def read_tsv(path: Path) -> tuple[list[str], NDArray[np.int64]]:
    """Read a GLUE ``sentence<TAB>label`` file with a header row."""
    texts: list[str] = []
    labels: list[int] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        next(reader)
        for row in reader:
            texts.append(row[0].strip())
            labels.append(int(row[1]))
    return texts, np.asarray(labels, dtype=np.int64)


def load_sst2(root: Path | None = None) -> SentenceDataset:
    from backdoor_defense_lab.data import data_dir

    base = (root or data_dir()) / "sst2"
    if not (base / "train.tsv").exists():
        raise FileNotFoundError(f"{base}/train.tsv not found; run `make data` first")
    train, y_train = read_tsv(base / "train.tsv")
    test, y_test = read_tsv(base / "dev.tsv")
    return SentenceDataset(train, y_train, test, y_test)


def tokenize(text: str) -> list[str]:
    return text.lower().split()


class Vocab:
    """Word to id map; id 0 is padding and id 1 is the unknown word."""

    def __init__(self, words: Sequence[str]) -> None:
        self.itos = [PAD, UNK, *[w for w in words if w not in (PAD, UNK)]]
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    @classmethod
    def build(cls, texts: Iterable[str], min_freq: int = 1, extra: Sequence[str] = ()) -> Vocab:
        """Words seen at least ``min_freq`` times, most frequent first, plus ``extra``."""
        counts = Counter(w for t in texts for w in tokenize(t))
        words = [
            w for w, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])) if c >= min_freq
        ]
        seen = set(words)
        return cls(words + [w for w in extra if w not in seen])

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, text: str, max_len: int) -> list[int]:
        unk = self.stoi[UNK]
        return [self.stoi.get(w, unk) for w in tokenize(text)][:max_len]

    def batch(self, texts: Sequence[str], max_len: int) -> NDArray[np.int64]:
        """Right-padded id matrix (``len(texts)`` x ``max_len``); empty texts get one UNK."""
        out = np.zeros((len(texts), max_len), dtype=np.int64)
        for i, t in enumerate(texts):
            ids = self.encode(t, max_len) or [self.stoi[UNK]]
            out[i, : len(ids)] = ids
        return out


def synthetic_sentences(n_train: int = 400, n_test: int = 100, seed: int = 0) -> SentenceDataset:
    """A toy sentiment task: sentences mix filler words with positive or negative cue words."""
    rng = np.random.default_rng(seed)
    pos = ["good", "great", "fine", "lovely", "superb"]
    neg = ["bad", "awful", "poor", "dull", "boring"]
    filler = ["the", "a", "movie", "film", "was", "plot", "and", "very", "quite", "story", "it"]

    def make(n: int) -> tuple[list[str], NDArray[np.int64]]:
        y = rng.integers(0, 2, n)
        texts = []
        for label in y:
            words = list(rng.choice(filler, rng.integers(4, 9)))
            cues = pos if label == 1 else neg
            for _ in range(rng.integers(1, 3)):
                words.insert(int(rng.integers(0, len(words) + 1)), str(rng.choice(cues)))
            texts.append(" ".join(words))
        return texts, y.astype(np.int64)

    tr, ytr = make(n_train)
    te, yte = make(n_test)
    return SentenceDataset(tr, ytr, te, yte, name="synthetic")
