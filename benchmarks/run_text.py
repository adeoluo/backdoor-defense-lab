"""Text backdoor on SST-2: a rare-token trigger in a small transformer, and ONION.

    uv sync --extra text && uv run python benchmarks/run_text.py --seeds 0 1 2

For each seed: train a clean model and backdoored models with 1 and 3 inserted trigger words,
measure clean accuracy and attack success rate, then run ONION (GPT-2 and the unigram ablation)
on the clean and triggered test sentences: word and sentence detection AUROC, and accuracy and
attack success rate after deleting the suspicious words, over a sweep of thresholds.
Writes ``benchmarks/results/text/sst2_s<seed>.json``.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from common import RESULTS, hardware, log

from backdoor_defense_lab.data import pick_device
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
from backdoor_defense_lab.text.model import TextModel, count_parameters
from backdoor_defense_lab.text.onion import GPT2Scorer

THRESHOLDS = (-50.0, -20.0, -10.0, -5.0, 0.0, 5.0, 10.0, 20.0, 50.0, 100.0)


def accuracy(model: TextModel, texts: list[str], labels: np.ndarray) -> float:
    return float((model.predict(texts) == labels).mean())


def asr(model: TextModel, texts: list[str], target: int) -> float:
    return float((model.predict(texts) == target).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--device", default=None)
    ap.add_argument("--lm-device", default="cpu")
    args = ap.parse_args()
    device = pick_device(args.device)
    data = load_sst2()
    vocab = Vocab.build(data.train, min_freq=1, extra=RARE_TOKENS)
    log(f"sst2: {len(data.train)} train, {len(data.test)} test, vocab {len(vocab)}")
    t0 = time.perf_counter()
    gpt2 = GPT2Scorer("openai-community/gpt2", device=args.lm_device)
    scorers = {"gpt2": gpt2, "unigram": UnigramScorer(data.train)}
    load_s = time.perf_counter() - t0
    clean_scores = {}
    clean_seconds = {}
    for name, fn in scorers.items():
        t0 = time.perf_counter()
        clean_scores[name] = suspicion_scores(data.test, fn)
        clean_seconds[name] = time.perf_counter() - t0
    log(f"ONION scores of the clean test set in {clean_seconds}")
    for seed in args.seeds:
        out = RESULTS / "text" / f"sst2_s{seed}.json"
        if out.exists():
            continue
        cfg = TextConfig(seed=seed)
        clean_model, clean_hist = train_text_classifier(
            data.train, data.y_train, vocab, cfg=cfg, device=device, log=log
        )
        result: dict[str, object] = {
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
            "lm_load_seconds": load_s,
            "onion_clean_seconds": clean_seconds,
            "hardware": hardware(),
        }
        for n_insert in (1, 3):
            trig = RareTokenTrigger(target=1, rate=0.1, n_insert=n_insert)
            pois = poison_sentences(data.train, data.y_train, trig, seed)
            model, hist = train_text_classifier(
                pois.texts, pois.labels, vocab, cfg=cfg, device=device, log=log
            )
            _, trig_texts, positions = triggered_test(data, trig, seed)
            keep = np.flatnonzero(data.y_test != trig.target)
            entry: dict[str, object] = {
                "attack": trig.config(),
                "n_poisoned": int(pois.poisoned.sum()),
                "train_seconds": hist["seconds"],
                "clean_accuracy": accuracy(model, data.test, data.y_test),
                "attack_success_rate": asr(model, trig_texts, trig.target),
                "clean_model_trigger_rate": asr(clean_model, trig_texts, trig.target),
            }
            for name, fn in scorers.items():
                t0 = time.perf_counter()
                trig_scores = suspicion_scores(trig_texts, fn)
                seconds = time.perf_counter() - t0
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
        write_json(out, result)


if __name__ == "__main__":
    main()
