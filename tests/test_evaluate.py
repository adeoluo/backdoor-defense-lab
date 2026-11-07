from __future__ import annotations

import numpy as np

from backdoor_defense_lab.data import ImageDataset
from backdoor_defense_lab.evaluate import DefenseSettings, evaluate_run, features_of, split_test
from backdoor_defense_lab.pipeline import Run, run_attack
from backdoor_defense_lab.train import TrainConfig

SMALL = DefenseSettings(
    n_defender=100,
    strip_n=60,
    strip_pool=40,
    strip_perturb=8,
    ac_dims=4,
    nc_images=64,
    nc_epochs=4,
    nc_steps=3,
    nc_batch=16,
    repair_epochs=1,
)


def test_split_test_is_disjoint(toy: ImageDataset) -> None:
    d, h = split_test(toy, 50)
    assert len(d.y_test) == 50 and len(h.y_test) == len(toy.y_test) - 50
    assert np.array_equal(np.r_[d.y_test, h.y_test], toy.y_test)
    d2, _ = split_test(toy, 10_000)
    assert len(d2.y_test) == len(toy.y_test) // 2


def test_features_of_chunks_match(backdoored: Run) -> None:
    x = backdoored.data.x_test[:50]
    chunked = features_of(backdoored.model, x, chunk=7)
    assert np.allclose(chunked, features_of(backdoored.model, x), atol=1e-4)
    assert features_of(backdoored.model, x[:0]).shape == (0, backdoored.model.feature_dim)


def test_evaluate_run_on_a_backdoored_model(backdoored: Run) -> None:
    lines: list[str] = []
    res = evaluate_run(backdoored, SMALL, log=lines.append)
    before = res["before"]
    assert before["attack_success_rate"] > 0.8  # type: ignore[index]
    for key in ("spectral", "activation_clustering", "strip", "neural_cleanse", "fine_pruning"):
        assert key in res
    assert 0 <= res["spectral"]["auroc"] <= 1  # type: ignore[index]
    assert set(res["activation_clustering"]) == {"ica", "pca"}  # type: ignore[arg-type]
    assert set(res["strip"]) == {"blend", "add"}  # type: ignore[arg-type]
    assert "after_fine_tuning" in res["fine_pruning"]  # type: ignore[operator]
    assert "fine_tuning_only" in res and "nc_unlearning" in res
    assert all(v >= 0 for v in res["seconds"].values())  # type: ignore[attr-defined]
    assert any("strip" in line for line in lines)


def test_evaluate_run_on_a_clean_model(toy: ImageDataset) -> None:
    run, _ = run_attack(toy, "clean", 0, 0, TrainConfig(epochs=2, batch_size=64, width=8))
    res = evaluate_run(run, SMALL)
    assert "spectral" not in res and "fine_pruning" not in res
    assert np.isnan(res["before"]["attack_success_rate"])  # type: ignore[index]
    assert res["neural_cleanse"]["target_found"] is False  # type: ignore[index]
