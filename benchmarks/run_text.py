"""Text backdoor on SST-2: a rare-token trigger in a small transformer, and ONION.

    uv sync --extra text && uv run python benchmarks/run_text.py --seeds 0 1 2

For each seed: train a clean model and backdoored models with 1 and 3 inserted trigger words,
measure clean accuracy and attack success rate, then run ONION on the clean and triggered test
sentences with each scoring language model (GPT-2 as in the paper, Qwen3.5-0.8B, and the unigram
ablation): word and sentence detection AUROC, and accuracy and attack success rate after
deleting the suspicious words, over a sweep of thresholds.
Writes ``benchmarks/results/text/sst2_s<seed>.json``. Trained classifiers are cached under the
data directory (``text-models/``), so a later run with another ``--lms`` adds that language
model's ONION results to the existing files; the cached models must reproduce the stored clean
accuracy and attack success rate, otherwise the run stops.
"""

from __future__ import annotations

import argparse
import json
import time
from importlib.metadata import version
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import torch
from common import RESULTS, hardware, log

from backdoor_defense_lab.data import data_dir, pick_device
from backdoor_defense_lab.metrics import detection_summary
from backdoor_defense_lab.pipeline import write_json
from backdoor_defense_lab.text import (
    RareTokenTrigger,
    TextConfig,
    UnigramScorer,
    Vocab,
    load_sst2,
    onion_filter,
    onion_scores,
    poison_sentences,
    suspicion_scores,
    train_text_classifier,
)
from backdoor_defense_lab.text.attack import RARE_TOKENS, triggered_test
from backdoor_defense_lab.text.model import TextModel, TransformerClassifier, count_parameters
from backdoor_defense_lab.text.onion import CausalLMScorer, Perplexity

THRESHOLDS = (-50.0, -20.0, -10.0, -5.0, 0.0, 5.0, 10.0, 20.0, 50.0, 100.0)
# Scoring language models: Hugging Face name, batch size. GPT-2 is the paper's choice.
CAUSAL_LMS = {
    "gpt2": ("openai-community/gpt2", 64),
    # A 248k-token vocabulary makes the logits large, so Qwen runs in small batches.
    "qwen3.5-0.8b": ("Qwen/Qwen3.5-0.8B", 4),
}


def cached_model(
    path: Path,
    texts: list[str],
    labels: np.ndarray,
    vocab: Vocab,
    cfg: TextConfig,
    device: torch.device,
) -> tuple[TextModel, dict[str, Any]]:
    """Train a classifier, or load the one trained earlier with the same seed and data."""
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=False)
        net = TransformerClassifier(len(vocab), 2, cfg)
        net.load_state_dict(saved["state"])
        return TextModel(net.to(device), vocab), saved["hist"]
    model, hist = train_text_classifier(texts, labels, vocab, cfg=cfg, device=device, log=log)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.net.state_dict(), "hist": dict(hist)}, path)
    return model, hist


SCORE_CHUNK = 64


def cached_suspicion(key: str, texts: list[str], fn: Perplexity) -> tuple[list[np.ndarray], float]:
    """ONION word scores of ``texts``, saved in chunks of sentences so a long run resumes."""
    out: list[np.ndarray] = []
    seconds = 0.0
    folder = data_dir() / "text-onion-scores" / key
    folder.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(texts), SCORE_CHUNK):
        chunk = texts[start : start + SCORE_CHUNK]
        path = folder / f"{start:05d}_{len(chunk)}.npz"
        if not path.exists():
            t0 = time.perf_counter()
            scores = suspicion_scores(chunk, fn)
            np.savez(
                path,
                flat=np.concatenate(scores) if scores else np.zeros(0),
                lengths=np.array([len(x) for x in scores]),
                seconds=time.perf_counter() - t0,
            )
        with np.load(path) as z:
            bounds = np.cumsum(np.r_[0, z["lengths"]])
            out += [z["flat"][a:b] for a, b in pairwise(bounds)]
            seconds += float(z["seconds"])
    return out, seconds


def accuracy(model: TextModel, texts: list[str], labels: np.ndarray) -> float:
    return float((model.predict(texts) == labels).mean())


def asr(model: TextModel, texts: list[str], target: int) -> float:
    return float((model.predict(texts) == target).mean())


def same(stored: float, now: float, path: Path) -> None:
    if stored != now:
        raise SystemExit(
            f"{path}: a cached or retrained model gives {now}, the stored result says {stored};"
            " move the file away to rerun this seed with every language model"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--device", default=None)
    ap.add_argument("--lm-device", default="cpu", help="device for GPT-2")
    ap.add_argument("--qwen-device", default=None, help="device for Qwen3.5 (default: auto)")
    ap.add_argument("--qwen-dtype", default="float16")
    ap.add_argument("--lms", nargs="+", default=[*CAUSAL_LMS, "unigram"])
    ap.add_argument(
        "--train-only",
        action="store_true",
        help="train and cache the classifiers and check them against stored results; no ONION",
    )
    args = ap.parse_args()
    if args.train_only:
        args.lms = []
    device = pick_device(args.device)
    data = load_sst2()
    vocab = Vocab.build(data.train, min_freq=1, extra=RARE_TOKENS)
    log(f"sst2: {len(data.train)} train, {len(data.test)} test, vocab {len(vocab)}")
    scorers: dict[str, Perplexity] = {}
    lms: dict[str, dict[str, object]] = {}
    load_s: dict[str, float] = {}
    for name in args.lms:
        t0 = time.perf_counter()
        if name == "unigram":
            scorers[name] = UnigramScorer(data.train)
            lms[name] = {"model": "add-one unigram fitted on the SST-2 training texts"}
        else:
            hf_name, batch = CAUSAL_LMS[name]
            qwen = name.startswith("qwen")
            lm_device = str(pick_device(args.qwen_device)) if qwen else args.lm_device
            dtype = args.qwen_dtype if qwen else None
            scorers[name] = CausalLMScorer(hf_name, device=lm_device, batch_size=batch, dtype=dtype)
            lms[name] = {"model": hf_name, "device": lm_device, "dtype": dtype or "float32",
                         "batch_size": batch, "transformers": version("transformers")}  # fmt: skip
        load_s[name] = time.perf_counter() - t0
    clean_scores = {}
    clean_seconds = {}
    for name, fn in scorers.items():
        log(f"ONION scores of the clean test set with {name}")
        clean_scores[name], clean_seconds[name] = cached_suspicion(
            f"{name}_clean", list(data.test), fn
        )
    log(f"ONION scores of the clean test set in {clean_seconds}")
    models_dir = data_dir() / "text-models"
    for seed in args.seeds:
        out = RESULTS / "text" / f"sst2_s{seed}.json"
        old: dict[str, Any] = json.loads(out.read_text()) if out.exists() else {}
        todo = [
            n
            for n in scorers
            if any(f"onion_{n}" not in old.get(f"backdoor_{k}", {}) for k in (1, 3))
        ]
        if not todo and not args.train_only:
            continue
        cfg = TextConfig(seed=seed)
        clean_model, clean_hist = cached_model(
            models_dir / f"sst2_s{seed}_clean.pt", data.train, data.y_train, vocab, cfg, device
        )
        result: dict[str, Any] = {
            "dataset": "sst2",
            "seed": seed,
            "n_train": len(data.train),
            "n_test": len(data.test),
            "vocab": len(vocab),
            "parameters": count_parameters(clean_model.net),
            "text_config": cfg.__dict__,
            "clean": {
                "clean_accuracy": accuracy(clean_model, data.test, data.y_test),
                "train_seconds": clean_hist["seconds"],
            },
            "lms": {**old.get("lms", {}), **{n: lms[n] for n in todo}},
            "lm_load_seconds": {**old.get("lm_load_seconds", {}), **{n: load_s[n] for n in todo}},
            "onion_clean_seconds": {
                **old.get("onion_clean_seconds", {}),
                **{n: clean_seconds[n] for n in todo},
            },
            "hardware": old.get("hardware", hardware()),
        }
        if old:
            same(old["clean"]["clean_accuracy"], result["clean"]["clean_accuracy"], out)
        for n_insert in (1, 3):
            trig = RareTokenTrigger(target=1, rate=0.1, n_insert=n_insert)
            pois = poison_sentences(data.train, data.y_train, trig, seed)
            model, hist = cached_model(
                models_dir / f"sst2_s{seed}_backdoor_{n_insert}.pt",
                pois.texts, pois.labels, vocab, cfg, device,
            )  # fmt: skip
            _, trig_texts, positions = triggered_test(data, trig, seed)
            keep = np.flatnonzero(data.y_test != trig.target)
            entry: dict[str, Any] = {
                "attack": trig.config(),
                "n_poisoned": int(pois.poisoned.sum()),
                "train_seconds": hist["seconds"],
                "clean_accuracy": accuracy(model, data.test, data.y_test),
                "attack_success_rate": asr(model, trig_texts, trig.target),
                "clean_model_trigger_rate": asr(clean_model, trig_texts, trig.target),
            }
            prev = old.get(f"backdoor_{n_insert}", {})
            if prev:
                for k in ("clean_accuracy", "attack_success_rate", "clean_model_trigger_rate"):
                    same(prev[k], entry[k], out)
                entry.update({k: v for k, v in prev.items() if k.startswith("onion_")})
                entry["train_seconds"] = prev["train_seconds"]
            for name, fn in ((n, scorers[n]) for n in todo):
                trig_scores, seconds = cached_suspicion(
                    f"{name}_s{seed}_insert{n_insert}", list(trig_texts), fn
                )
                # word level: inserted trigger words against every other word of those sentences
                word_s = np.concatenate(trig_scores)
                word_y = np.concatenate(
                    [
                        np.isin(np.arange(len(s)), p)
                        for s, p in zip(trig_scores, positions, strict=True)
                    ]
                )
                sent_s = np.r_[
                    onion_scores([clean_scores[name][i] for i in keep]), onion_scores(trig_scores)
                ]
                sent_y = np.r_[np.zeros(len(keep), bool), np.ones(len(trig_texts), bool)]
                sweep = []
                for t in THRESHOLDS:
                    filtered_clean = [
                        onion_filter(x, s, t)
                        for x, s in zip(data.test, clean_scores[name], strict=True)
                    ]
                    filtered_trig = [
                        onion_filter(x, s, t) for x, s in zip(trig_texts, trig_scores, strict=True)
                    ]
                    removed = [
                        set(p) <= {i for i, v in enumerate(s) if v > t}
                        for s, p in zip(trig_scores, positions, strict=True)
                    ]
                    sweep.append(
                        {
                            "threshold": t,
                            "clean_accuracy": accuracy(model, filtered_clean, data.y_test),
                            "attack_success_rate": asr(model, filtered_trig, trig.target),
                            "all_triggers_removed": float(np.mean(removed)),
                        }
                    )
                entry[f"onion_{name}"] = {
                    "word": detection_summary(word_s, word_y).as_dict(),
                    "sentence": detection_summary(sent_s, sent_y).as_dict(),
                    "sweep": sweep,
                    "seconds": seconds,
                }
                at0 = next(s for s in sweep if s["threshold"] == 0.0)
                log(
                    f"seed {seed} n_insert {n_insert} {name}: CA {entry['clean_accuracy']:.4f} "
                    f"ASR {entry['attack_success_rate']:.4f} -> ONION(t=0) CA "
                    f"{at0['clean_accuracy']:.4f} ASR {at0['attack_success_rate']:.4f}"
                )
            result[f"backdoor_{n_insert}"] = entry
        if args.train_only:
            match = " and match the stored results" if old else ""
            log(f"seed {seed}: classifiers cached{match}")
            continue
        write_json(out, result)


if __name__ == "__main__":
    main()
