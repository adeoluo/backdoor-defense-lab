from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from backdoor_defense_lab.attacks import BadNets
from backdoor_defense_lab.data import ImageDataset, synthetic_dataset
from backdoor_defense_lab.models import build_model
from backdoor_defense_lab.pipeline import (
    CLEAN,
    load_run,
    measure_asr,
    run_attack,
    run_paths,
    write_json,
)
from backdoor_defense_lab.train import (
    TrainConfig,
    evaluate_accuracy,
    history_dict,
    load_checkpoint,
    one_cycle,
    save_checkpoint,
    train_classifier,
)

FAST = TrainConfig(epochs=4, batch_size=64, lr=0.1, width=8)


def test_one_cycle_schedule() -> None:
    lrs = [one_cycle(s, 100, 1.0, 0.2) for s in range(100)]
    assert lrs[0] == pytest.approx(0.05)
    assert max(lrs) == pytest.approx(1.0)
    assert lrs[19] == pytest.approx(1.0)
    assert lrs[-1] < 0.02
    assert np.all(np.diff(lrs[20:]) <= 0)
    assert one_cycle(0, 0, 0.3, 0.2) == 0.3


def test_training_learns_toy_task(toy: ImageDataset) -> None:
    model, hist = train_classifier(
        toy.x_train, toy.y_train, toy.num_classes, toy.mean, toy.std, FAST
    )
    assert hist.loss[-1] < hist.loss[0]
    assert evaluate_accuracy(model, toy.x_test, toy.y_test) > 0.6
    assert hist.images_per_second > 0
    assert set(history_dict(hist)) == {"loss", "train_acc", "seconds", "images_per_second"}


def test_training_is_deterministic(toy: ImageDataset) -> None:
    cfg = TrainConfig(epochs=1, batch_size=64, width=8, seed=3)
    m1, _ = train_classifier(toy.x_train, toy.y_train, 4, toy.mean, toy.std, cfg)
    m2, _ = train_classifier(toy.x_train, toy.y_train, 4, toy.mean, toy.std, cfg)
    for a, b in zip(m1.state_dict().values(), m2.state_dict().values(), strict=True):
        assert torch.equal(a, b)


def test_fine_tune_existing_model_without_augmentation(toy: ImageDataset) -> None:
    base = build_model(4, width=8)
    cfg = TrainConfig(epochs=1, batch_size=64, width=8, augment=False)
    logs: list[str] = []
    m, _ = train_classifier(
        toy.x_train, toy.y_train, 4, toy.mean, toy.std, cfg, model=base, log=logs.append
    )
    assert m is base
    assert logs and logs[0].startswith("epoch 1/1")


def test_evaluate_accuracy_empty() -> None:
    m = build_model(4, width=8)
    assert np.isnan(evaluate_accuracy(m, np.zeros((0, 32, 32, 3), np.uint8), np.zeros(0, np.int64)))


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    m = build_model(4, mean=(0.1, 0.2, 0.3), std=(0.4, 0.5, 0.6), width=8).eval()
    m.channel_mask[3] = 0
    save_checkpoint(m, tmp_path / "m.pt", {"note": "x"})
    loaded, info = load_checkpoint(tmp_path / "m.pt")
    assert info["note"] == "x" and info["width"] == 8
    x = torch.rand(3, 3, 32, 32)
    torch.testing.assert_close(loaded(x), m(x))


def test_backdoor_is_learned_and_measured(tmp_path: Path) -> None:
    data = synthetic_dataset(n_train=800, n_test=200, num_classes=4, seed=1)
    cfg = TrainConfig(epochs=6, batch_size=64, lr=0.1, width=8, seed=0)
    run, res = run_attack(data, "badnets", 0, 0, cfg, out_dir=tmp_path)
    assert res.n_poisoned == 80
    assert res.attack_success_rate > 0.8
    assert res.clean_accuracy > 0.6
    assert res.attack_config["name"] == "badnets"
    clean_run, clean_res = run_attack(data, CLEAN, 0, 0, cfg, out_dir=tmp_path)
    assert np.isnan(clean_res.attack_success_rate)
    assert clean_run.attack is None and clean_res.n_poisoned == 0
    # the clean model ignores the trigger
    assert measure_asr(clean_run.model, data, BadNets(target=0)) < 0.5
    pt, npz = run_paths(tmp_path, data.name, "badnets", 0)
    assert pt.exists() and npz.exists()
    loaded = load_run(tmp_path, data, "badnets", 0)
    np.testing.assert_array_equal(loaded.train_set.poisoned, run.train_set.poisoned)
    np.testing.assert_array_equal(loaded.train_set.x, run.train_set.x)
    assert loaded.attack is not None and loaded.attack.name == "badnets"
    assert measure_asr(loaded.model, data, loaded.attack) == pytest.approx(res.attack_success_rate)
    assert load_run(tmp_path, data, CLEAN, 0).attack is None


def test_label_consistent_and_wanet_runs(tmp_path: Path) -> None:
    data = synthetic_dataset(n_train=400, n_test=100, num_classes=4, seed=2)
    cfg = TrainConfig(epochs=1, batch_size=64, width=8)
    clean, _ = run_attack(data, CLEAN, 0, 0, cfg)
    run, res = run_attack(
        data, "label_consistent", 0, 0, cfg, surrogate=clean.model, attack_kwargs={"steps": 2}
    )
    assert res.attack_config["surrogate"] is True
    np.testing.assert_array_equal(run.train_set.y, data.y_train)
    _, res_w = run_attack(data, "wanet", 0, 0, cfg, out_dir=tmp_path)
    assert res_w.n_poisoned == 40
    loaded = load_run(tmp_path, data, "wanet", 0)
    assert loaded.train_set.noise.sum() == 80


def test_write_json(tmp_path: Path) -> None:
    write_json(tmp_path / "a" / "b.json", {"x": np.float64(1.5)})
    assert (tmp_path / "a" / "b.json").read_text().strip().startswith("{")
