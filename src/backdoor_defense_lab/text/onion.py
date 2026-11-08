"""ONION: outlier word detection with a language model (Qi, Chen, Zhang, Liu and Sun, 2021).

An inserted rare token breaks the fluency of a sentence. For a sentence ``s = w_1 .. w_n`` with
language-model perplexity ``p_0``, let ``p_i`` be the perplexity of ``s`` without ``w_i``. The
suspicion score of word ``i`` is

    f_i = p_0 - p_i,

large when deleting the word makes the sentence much more fluent. At test time ONION deletes
every word with ``f_i > t`` before classifying; ``t = 0`` is the common default. The sentence
level score ``max_i f_i`` gives a triggered-input detector.

The paper's language model is GPT-2 (124M); :class:`CausalLMScorer` runs it, or any other
causal language model such as Qwen3.5-0.8B, through ``transformers`` (``uv sync --extra
text``). Any callable that maps sentences to perplexities works, and :class:`UnigramScorer`
(an add-one unigram model fitted on the training texts) is the language-model-free ablation.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from backdoor_defense_lab.text.data import tokenize

Perplexity = Callable[[Sequence[str]], NDArray[np.float64]]


class UnigramScorer:
    """Perplexity under an add-one smoothed unigram model of a reference corpus."""

    def __init__(self, texts: Iterable[str]) -> None:
        self.counts = Counter(w for t in texts for w in tokenize(t))
        self.total = sum(self.counts.values())
        self.vocab = len(self.counts) + 1

    def logprob(self, word: str) -> float:
        return math.log((self.counts.get(word, 0) + 1) / (self.total + self.vocab))

    def __call__(self, texts: Sequence[str]) -> NDArray[np.float64]:
        out = np.empty(len(texts))
        for i, t in enumerate(texts):
            words = tokenize(t)
            if not words:
                out[i] = float(self.vocab)
                continue
            out[i] = math.exp(-sum(self.logprob(w) for w in words) / len(words))
        return out


def start_token_id(tokenizer: object) -> int:
    """The token that opens a document: BOS when the tokenizer has one (GPT-2), otherwise
    ``<|endoftext|>`` (the Qwen document separator), otherwise EOS."""
    bos = getattr(tokenizer, "bos_token_id", None)
    if bos is not None:
        return int(bos)
    vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
    if "<|endoftext|>" in vocab:
        return int(vocab["<|endoftext|>"])
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is None:
        raise ValueError("the tokenizer has no BOS, <|endoftext|> or EOS token")
    return int(eos)


class CausalLMScorer:
    """Sentence perplexity under a causal language model (default GPT-2, as in the paper).

    A document-start token (:func:`start_token_id`) is prepended so the first word is
    predicted too; padding is masked. ``dtype`` (for example ``"float16"``) sets the weight
    precision; the log-probabilities are always computed in float32.
    """

    def __init__(
        self,
        name: str = "openai-community/gpt2",
        device: str = "cpu",
        batch_size: int = 64,
        model: object | None = None,
        tokenizer: object | None = None,
        dtype: str | None = None,
    ) -> None:
        import torch

        if model is None or tokenizer is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(name)
            kwargs = {"dtype": getattr(torch, dtype)} if dtype else {}
            model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
        self.name = name
        self.torch = torch
        self.tok = tokenizer
        self.model = model.to(device).eval()  # type: ignore[attr-defined]
        self.device = device
        self.batch_size = batch_size
        self.bos = start_token_id(self.tok)

    def __call__(self, texts: Sequence[str]) -> NDArray[np.float64]:
        torch = self.torch
        out = np.empty(len(texts))
        for start in range(0, len(texts), self.batch_size):
            chunk = texts[start : start + self.batch_size]
            ids = [[self.bos, *self.tok.encode(" " + t if t else "")] for t in chunk]  # type: ignore[attr-defined]
            width = max(len(x) for x in ids)
            mat = torch.full((len(ids), width), self.bos, dtype=torch.long)
            mask = torch.zeros((len(ids), width), dtype=torch.long)
            for i, x in enumerate(ids):
                mat[i, : len(x)] = torch.tensor(x)
                mask[i, : len(x)] = 1
            with torch.no_grad():
                logits = self.model(
                    input_ids=mat.to(self.device), attention_mask=mask.to(self.device)
                ).logits.float()
            logp = torch.log_softmax(logits[:, :-1], -1)
            tgt = mat[:, 1:].to(self.device)
            nll = -logp.gather(-1, tgt[..., None])[..., 0]
            m = mask[:, 1:].to(self.device).float()
            n_tok = m.sum(1)
            mean = (nll * m).sum(1) / n_tok.clamp_min(1)
            ppl = torch.exp(mean).cpu().numpy().astype(np.float64)
            # an empty sentence has no predicted token; give it a high perplexity
            ppl[(n_tok == 0).cpu().numpy()] = 1e4
            out[start : start + len(chunk)] = ppl
        return out


def suspicion_scores(texts: Sequence[str], perplexity: Perplexity) -> list[NDArray[np.float64]]:
    """ONION scores ``f_i = p_0 - p_i`` of every word of every sentence (one batched pass)."""
    variants: list[str] = []
    spans: list[tuple[int, int]] = []
    for t in texts:
        words = tokenize(t)
        start = len(variants)
        variants.append(" ".join(words))
        variants.extend(" ".join(words[:i] + words[i + 1 :]) for i in range(len(words)))
        spans.append((start, len(variants)))
    ppl = perplexity(variants) if variants else np.zeros(0)
    return [ppl[a] - ppl[a + 1 : b] for a, b in spans]


def onion_filter(text: str, scores: NDArray[np.float64], threshold: float = 0.0) -> str:
    """Delete the words whose suspicion score exceeds ``threshold``."""
    words = tokenize(text)
    if len(words) != len(scores):
        raise ValueError("one score per word is required")
    return " ".join(w for w, s in zip(words, scores, strict=True) if s <= threshold)


def onion_scores(scores: Sequence[NDArray[np.float64]]) -> NDArray[np.float64]:
    """Sentence-level detection score: the largest word suspicion (``-inf`` for empty text)."""
    return np.array([float(np.max(s)) if len(s) else -np.inf for s in scores])


GPT2Scorer = CausalLMScorer
"""Former name of :class:`CausalLMScorer`, kept for existing callers."""
