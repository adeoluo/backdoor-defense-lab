from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from backdoor_defense_lab.report import (
    art_table,
    attack_table,
    detection_table,
    fmt,
    load_results,
    neural_cleanse_model_table,
    render,
    repair_table,
    text_table,
)


def _defense(ds: str, attack: str, seed: int, anomaly: float) -> dict[str, Any]:
    ca_asr = {"clean_accuracy": 0.9, "attack_success_rate": 0.99}
    row: dict[str, Any] = {
        "dataset": ds,
        "attack": attack,
        "seed": seed,
        "before": ca_asr,
        "neural_cleanse": {
            "min_anomaly": anomaly,
            "target_found": anomaly > 2,
            "target_anomaly": anomaly,
        },
    }
    if attack != "clean":
        det = {"auroc": 0.9 + 0.01 * seed, "tpr_at_fpr_5": 0.5}
        row |= {
            "spectral": {**det, "auroc_uncentered_art_variant": 0.8},
            "activation_clustering": {"ica": {**det, "cluster_separation": 0.95}},
            "strip": {"blend": det, "add": det},
            "nc_unlearning": ca_asr,
            "fine_pruning": {
                "prune_ratio": 0.25,
                "after_pruning": ca_asr,
                "after_fine_tuning": ca_asr,
            },
            "fine_tuning_only": ca_asr,
        }
    return row


@pytest.fixture
def rows() -> list[dict[str, Any]]:
    out = []
    for seed in range(3):
        out.append(_defense("cifar10", "clean", seed, 1.0 + seed * 0.2))
        out.append(_defense("cifar10", "badnets", seed, 4.0 + seed))
        out.append(_defense("gtsrb", "blend", seed, 1.5))
    return out


def test_fmt() -> None:
    assert fmt([0.5, 0.7]) == "60.0 ± 14.1"
    assert fmt([2.0], scale=1.0, digits=2) == "2.00 ± 0.00"
    assert fmt([]) == "n/a"
    assert fmt([None, float("nan")]) == "n/a"  # type: ignore[list-item]


def test_attack_table_orders_and_averages() -> None:
    rows = [
        {
            "dataset": ds,
            "attack": a,
            "seed": s,
            "n_poisoned": 5000,
            "clean_accuracy": 0.9,
            "attack_success_rate": 1.0,
            "train_seconds": 600,
        }
        for ds in ("gtsrb", "cifar10")
        for a in ("wanet", "badnets")
        for s in range(2)
    ]
    t = attack_table(rows)
    lines = t.splitlines()
    assert lines[2].startswith("| CIFAR-10 | BadNets | 2 | 5,000 | 90.0 ± 0.0 | 100.0 ± 0.0")
    assert "10.0 ± 0.0" in lines[2]  # minutes
    assert lines[-1].startswith("| GTSRB | WaNet")


def test_detection_and_repair_tables(rows: list[dict[str, Any]]) -> None:
    det = detection_table(rows)
    assert "Clean |" not in det  # clean models are not in the detection table
    assert "| CIFAR-10 | BadNets | 3 | 91.0 ± 1.0" in det
    rep = repair_table(rows)
    assert "| CIFAR-10 | BadNets | 90.0 ± 0.0 / 99.0 ± 0.0 | 3/3 | 5.0 ± 1.0" in rep
    assert "| GTSRB | Blend |" in rep and "0/3" in rep


def test_model_level_table(rows: list[dict[str, Any]]) -> None:
    t = neural_cleanse_model_table(rows)
    assert "| CIFAR-10 | 3 / 3 |" in t
    assert t.strip().splitlines()[2].endswith("| 1.000 |")
    assert "| GTSRB | 3 / 0 |" in t and "n/a" in t


def _text_row(seed: int) -> dict[str, Any]:
    sweep = [{"threshold": 0.0, "clean_accuracy": 0.8, "attack_success_rate": 0.1}]
    onion = {"sentence": {"auroc": 0.9}, "sweep": sweep}
    bd = {
        "clean_accuracy": 0.82,
        "attack_success_rate": 1.0,
        "clean_model_trigger_rate": 0.05,
        "onion_gpt2": onion,
        "onion_unigram": onion,
    }
    return {"seed": seed, "clean": {"clean_accuracy": 0.83}, "backdoor_1": bd, "backdoor_3": bd}


def test_text_table() -> None:
    t = text_table([_text_row(0), _text_row(1)])
    assert "| Clean model | - | 83.0 ± 0.0 | 5.0 ± 0.0 |" in t
    assert (
        "| 3 trigger words | GPT-2 | 82.0 ± 0.0 | 100.0 ± 0.0 | 90.0 ± 0.0 | 80.0 ± 0.0 | 10.0" in t
    )
    assert text_table([]) == ""


def test_render_and_load(tmp_path: Path, rows: list[dict[str, Any]]) -> None:
    (tmp_path / "defenses").mkdir()
    for i, r in enumerate(rows):
        (tmp_path / "defenses" / f"{i:02d}.json").write_text(json.dumps(r))
    assert len(load_results(tmp_path / "defenses")) == len(rows)
    assert load_results(tmp_path / "missing") == []
    md = render(tmp_path)
    assert "### Detection" in md and "### Repair" in md and "### Attacks" not in md
    assert render(tmp_path, ["text"]) == ""


def _art(seed: int, ours_flag: bool, art_flag: bool) -> dict[str, Any]:
    ac = {
        "split_agreement": 0.98,
        "ours_flagged": ours_flag,
        "art_flagged": art_flag,
        "seconds_ours": 0.01,
        "seconds_art": 0.5,
    }
    return {
        "dataset": "cifar10",
        "attack": "badnets",
        "seed": seed,
        "spectral": {"auroc_ours_centered": 0.6, "auroc_art_uncentered": 0.4},
        "ac_pca": ac,
        "ac_ica": ac,
    }


def test_art_table() -> None:
    table = art_table([_art(0, True, True), _art(1, True, False)])
    lines = table.splitlines()
    assert len(lines) == 3
    row = lines[2]
    assert "| CIFAR-10 | BadNets | 2 |" in row
    assert "60.0 ± 0.0" in row and "40.0 ± 0.0" in row
    assert "2 / 1 of 2" in row
    assert "10 / 500" in row
    assert art_table([]).count("\n") == 1
