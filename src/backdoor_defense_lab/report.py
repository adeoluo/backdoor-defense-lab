"""Aggregate the per-run JSON results into the tables of the README and evaluation report.

Every table cell is ``mean ± sd`` over the seeds found (sample standard deviation), and the
number of seeds is printed with each table. Nothing here computes a metric; it only averages the
numbers the benchmark scripts wrote.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from backdoor_defense_lab.metrics import mean_std

IMAGE_ATTACKS = ("badnets", "blend", "wanet", "label_consistent")
ATTACK_LABELS = {
    "clean": "Clean",
    "badnets": "BadNets",
    "blend": "Blend",
    "wanet": "WaNet",
    "label_consistent": "Clean-label",
    "badnets_low": "BadNets 1%",
}
DATASET_LABELS = {"cifar10": "CIFAR-10", "gtsrb": "GTSRB", "sst2": "SST-2"}


def load_results(directory: Path) -> list[dict[str, Any]]:
    """Every ``*.json`` file of a results directory, sorted by name."""
    if not directory.exists():
        return []
    return [json.loads(p.read_text()) for p in sorted(directory.glob("*.json"))]


def fmt(values: Iterable[float], scale: float = 100.0, digits: int = 1) -> str:
    """``mean ± sd`` of ``values * scale``; ``n/a`` when no finite value is present."""
    vals = [float(v) * scale for v in values if v is not None]
    m, s = mean_std(vals)
    if m != m:  # NaN
        return "n/a"
    return f"{m:.{digits}f} ± {s:.{digits}f}"


def _group(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(r["dataset"], r["attack"])].append(r)
    return groups


def _get(d: dict[str, Any], path: str) -> Any:
    cur: Any = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|")
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _order(keys: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    attacks = list(ATTACK_LABELS)
    return sorted(
        keys,
        key=lambda k: (k[0] != "cifar10", attacks.index(k[1]) if k[1] in attacks else 99),
    )


def attack_table(train_rows: list[dict[str, Any]]) -> str:
    """Clean accuracy and attack success rate per dataset and attack (training results)."""
    g = _group(train_rows)
    rows = []
    for ds, atk in _order(g):
        rs = g[(ds, atk)]
        rows.append(
            [
                DATASET_LABELS.get(ds, ds),
                ATTACK_LABELS.get(atk, atk),
                str(len(rs)),
                f"{rs[0]['n_poisoned']:,}",
                fmt(r["clean_accuracy"] for r in rs),
                fmt(r["attack_success_rate"] for r in rs),
                fmt((r["train_seconds"] / 60 for r in rs), scale=1.0),
            ]
        )
    header = ["Dataset", "Attack", "Seeds", "Poisoned", "Clean acc %", "ASR %", "Train min"]
    return _table(header, rows)


def detection_table(defense_rows: list[dict[str, Any]], tpr: str = "tpr_at_fpr_5") -> str:
    """Detection AUROC (and TPR at 5% FPR) of the data and input detectors."""
    g = _group([r for r in defense_rows if r["attack"] != "clean"])
    cols: list[tuple[str, str]] = [
        ("Spectral AUROC", "spectral.auroc"),
        ("Spectral (ART variant) AUROC", "spectral.auroc_uncentered_art_variant"),
        ("AC (ICA) AUROC", "activation_clustering.ica.auroc"),
        ("AC separation", "activation_clustering.ica.cluster_separation"),
        ("STRIP AUROC", "strip.blend.auroc"),
        ("STRIP TPR@5%", f"strip.blend.{tpr}"),
        ("STRIP (add) AUROC", "strip.add.auroc"),
    ]
    rows = []
    for ds, atk in _order(g):
        rs = g[(ds, atk)]
        rows.append(
            [DATASET_LABELS.get(ds, ds), ATTACK_LABELS.get(atk, atk), str(len(rs))]
            + [fmt((_get(r, path) for r in rs), digits=1) for _, path in cols]
        )
    return _table(["Dataset", "Attack", "Seeds"] + [c for c, _ in cols], rows)


def repair_table(defense_rows: list[dict[str, Any]]) -> str:
    """Clean accuracy / ASR before and after each repair (Neural Cleanse, fine-pruning)."""
    g = _group([r for r in defense_rows if r["attack"] != "clean"])

    def pair(rs: list[dict[str, Any]], prefix: str) -> str:
        ca = fmt(_get(r, f"{prefix}.clean_accuracy") for r in rs)
        asr = fmt(_get(r, f"{prefix}.attack_success_rate") for r in rs)
        return f"{ca} / {asr}"

    rows = []
    for ds, atk in _order(g):
        rs = g[(ds, atk)]
        found = sum(bool(_get(r, "neural_cleanse.target_found")) for r in rs)
        rows.append(
            [
                DATASET_LABELS.get(ds, ds),
                ATTACK_LABELS.get(atk, atk),
                pair(rs, "before"),
                f"{found}/{len(rs)}",
                fmt((_get(r, "neural_cleanse.target_anomaly") for r in rs), scale=1.0),
                pair(rs, "nc_unlearning"),
                fmt(_get(r, "fine_pruning.prune_ratio") for r in rs),
                pair(rs, "fine_pruning.after_pruning"),
                pair(rs, "fine_pruning.after_fine_tuning"),
                pair(rs, "fine_tuning_only"),
            ]
        )
    header = [
        "Dataset",
        "Attack",
        "Before CA / ASR %",
        "NC target found",
        "NC anomaly",
        "NC unlearn CA / ASR %",
        "Pruned %",
        "Pruned CA / ASR %",
        "Fine-pruned CA / ASR %",
        "Fine-tune only CA / ASR %",
    ]
    return _table(header, rows)


def neural_cleanse_model_table(defense_rows: list[dict[str, Any]]) -> str:
    """Model-level Neural Cleanse: anomaly index on clean vs backdoored models, AUROC."""
    from backdoor_defense_lab.metrics import auroc

    rows = []
    for ds in ("cifar10", "gtsrb"):
        rs = [r for r in defense_rows if r["dataset"] == ds and r["attack"] != "badnets_low"]
        if not rs:
            continue
        clean = [r["neural_cleanse"]["min_anomaly"] for r in rs if r["attack"] == "clean"]
        bad = [r["neural_cleanse"]["min_anomaly"] for r in rs if r["attack"] != "clean"]
        flagged_clean = sum(a > 2.0 for a in clean)
        flagged_bad = sum(a > 2.0 for a in bad)
        a = auroc(bad + clean, [1] * len(bad) + [0] * len(clean)) if clean and bad else None
        rows.append(
            [
                DATASET_LABELS[ds],
                f"{len(bad)} / {len(clean)}",
                fmt(clean, scale=1.0, digits=2),
                fmt(bad, scale=1.0, digits=2),
                f"{flagged_bad}/{len(bad)}",
                f"{flagged_clean}/{len(clean)}",
                "n/a" if a is None else f"{a:.3f}",
            ]
        )
    header = [
        "Dataset",
        "Backdoored / clean models",
        "Anomaly, clean",
        "Anomaly, backdoored",
        "Backdoored flagged",
        "Clean flagged",
        "Model AUROC",
    ]
    return _table(header, rows)


def text_table(text_rows: list[dict[str, Any]]) -> str:
    """SST-2: clean accuracy, ASR and ONION (t = 0) per trigger length and language model."""
    if not text_rows:
        return ""
    rows = [
        [
            "Clean model",
            "-",
            fmt(r["clean"]["clean_accuracy"] for r in text_rows),
            fmt(r["backdoor_1"]["clean_model_trigger_rate"] for r in text_rows),
            "-",
            "-",
            "-",
        ]
    ]
    for n in (1, 3):
        key = f"backdoor_{n}"
        for lm in ("gpt2", "unigram"):
            o = f"onion_{lm}"

            def at0(r: dict[str, Any], field: str, k: str = key, oo: str = o) -> float:
                return float(next(s[field] for s in r[k][oo]["sweep"] if s["threshold"] == 0.0))

            rows.append(
                [
                    f"{n} trigger word{'s' if n > 1 else ''}",
                    "GPT-2" if lm == "gpt2" else "unigram",
                    fmt(r[key]["clean_accuracy"] for r in text_rows),
                    fmt(r[key]["attack_success_rate"] for r in text_rows),
                    fmt(r[key][o]["sentence"]["auroc"] for r in text_rows),
                    fmt(at0(r, "clean_accuracy") for r in text_rows),
                    fmt(at0(r, "attack_success_rate") for r in text_rows),
                ]
            )
    header = [
        "Model",
        "ONION LM",
        "Clean acc %",
        "ASR %",
        "Sentence AUROC",
        "Acc after ONION %",
        "ASR after ONION %",
    ]
    return _table(header, rows)


SECTIONS: dict[str, tuple[str, Callable[[Path], str]]] = {
    "attacks": (
        "Attacks: clean accuracy and attack success rate",
        lambda root: attack_table(load_results(root / "train")),
    ),
    "detection": (
        "Detection: training-set and input detectors",
        lambda root: detection_table(load_results(root / "defenses")),
    ),
    "repair": (
        "Repair: Neural Cleanse unlearning and fine-pruning",
        lambda root: repair_table(load_results(root / "defenses")),
    ),
    "models": (
        "Neural Cleanse as a model-level detector",
        lambda root: neural_cleanse_model_table(load_results(root / "defenses")),
    ),
    "text": (
        "Text backdoor and ONION on SST-2",
        lambda root: text_table(load_results(root / "text")),
    ),
}


def render(root: Path, sections: Iterable[str] = tuple(SECTIONS)) -> str:
    """Markdown with one titled table per section (empty sections are skipped)."""
    parts = []
    for key in sections:
        title, build = SECTIONS[key]
        table = build(root)
        if table and table.count("\n") > 1:
            parts.append(f"### {title}\n\n{table}\n")
    return "\n".join(parts)
