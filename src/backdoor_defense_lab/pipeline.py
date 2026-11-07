"""End-to-end experiment steps shared by the CLI and the benchmark scripts.

:func:`run_attack` poisons a training set, trains a model on it and measures clean accuracy and
attack success rate; it saves the checkpoint and the poisoning ground truth so the defenses can
be run on the same model later (:func:`load_run`). A ``clean`` run trains on unpoisoned data and
serves as the surrogate for the clean-label attack and as the negative class for model-level
detection.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from backdoor_defense_lab.attacks import (
    Attack,
    LabelConsistent,
    PoisonedTrainingSet,
    make_attack,
    poison_training_set,
    triggered_test_set,
)
from backdoor_defense_lab.data import ImageDataset
from backdoor_defense_lab.metrics import attack_success_rate
from backdoor_defense_lab.models import ResNet9, predict
from backdoor_defense_lab.train import (
    TrainConfig,
    evaluate_accuracy,
    load_checkpoint,
    save_checkpoint,
    train_classifier,
)

CLEAN = "clean"
#: Named attack variants for ablations: run name -> (attack, keyword overrides).
VARIANTS: dict[str, tuple[str, dict[str, object]]] = {
    "badnets_low": ("badnets", {"rate": 0.01}),
}


@dataclass
class RunResult:
    dataset: str
    attack: str
    seed: int
    clean_accuracy: float
    attack_success_rate: float
    n_poisoned: int
    n_train: int
    train_seconds: float
    images_per_second: float
    poison_seconds: float
    attack_config: dict[str, object]
    train_config: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class Run:
    """A trained model with the data and ground truth it was trained on."""

    model: ResNet9
    data: ImageDataset
    attack: Attack | None
    train_set: PoisonedTrainingSet
    meta: dict[str, object]


def run_paths(out_dir: Path, dataset: str, attack: str, seed: int) -> tuple[Path, Path]:
    stem = out_dir / dataset / f"{attack}_s{seed}"
    return stem.with_suffix(".pt"), stem.with_suffix(".npz")


def measure_asr(
    model: ResNet9, data: ImageDataset, attack: Attack, device: torch.device | str = "cpu"
) -> float:
    """Attack success rate on the triggered non-target test images."""
    xt, yt = triggered_test_set(data, attack)
    from backdoor_defense_lab.data import to_tensor

    pred = predict(model, to_tensor(xt), device=torch.device(device)).numpy()
    return attack_success_rate(pred, yt, attack.target)


def build_attack(
    name: str,
    data: ImageDataset,
    target: int,
    seed: int,
    surrogate: ResNet9 | None = None,
    **kwargs: object,
) -> Attack:
    attack = make_attack(name, target=target, image_size=data.image_size, **kwargs)
    if isinstance(attack, LabelConsistent):
        attack.seed = seed
        if surrogate is not None:
            attack.set_surrogate(surrogate)
    return attack


def run_attack(
    data: ImageDataset,
    attack_name: str,
    seed: int,
    target: int,
    config: TrainConfig | None = None,
    device: torch.device | str = "cpu",
    out_dir: Path | None = None,
    surrogate: ResNet9 | None = None,
    attack_kwargs: dict[str, object] | None = None,
    log: object = None,
) -> tuple[Run, RunResult]:
    """Poison, train and evaluate one model; ``attack_name='clean'`` trains without poisoning."""
    cfg = config or TrainConfig(seed=seed)
    t0 = time.perf_counter()
    attack: Attack | None = None
    if attack_name == CLEAN:
        n = len(data.y_train)
        train_set = PoisonedTrainingSet(
            x=data.x_train,
            y=data.y_train,
            poisoned=np.zeros(n, bool),
            noise=np.zeros(n, bool),
            original_y=data.y_train,
        )
    else:
        base, variant_kwargs = VARIANTS.get(attack_name, (attack_name, {}))
        kwargs = {**variant_kwargs, **(attack_kwargs or {})}
        attack = build_attack(base, data, target, seed, surrogate, **kwargs)
        train_set = poison_training_set(data, attack, seed=seed, device=device)
    poison_seconds = time.perf_counter() - t0
    model, hist = train_classifier(
        train_set.x,
        train_set.y,
        data.num_classes,
        data.mean,
        data.std,
        cfg,
        hflip=data.hflip,
        device=device,
        attack=attack,
        noise_mask=train_set.noise,
        log=log if callable(log) else None,
    )
    ca = evaluate_accuracy(model, data.x_test, data.y_test, device)
    # A clean model has no trigger of its own, so its attack success rate is undefined here.
    asr = measure_asr(model, data, attack, device) if attack is not None else float("nan")
    result = RunResult(
        dataset=data.name,
        attack=attack_name,
        seed=seed,
        clean_accuracy=ca,
        attack_success_rate=asr,
        n_poisoned=train_set.n_poisoned,
        n_train=len(train_set.y),
        train_seconds=hist.seconds,
        images_per_second=hist.images_per_second,
        poison_seconds=poison_seconds,
        attack_config=attack.config() if attack is not None else {"name": CLEAN},
        train_config=asdict(cfg),
    )
    meta = {"result": result.as_dict(), "target": target}
    if out_dir is not None:
        pt, npz = run_paths(out_dir, data.name, attack_name, seed)
        save_checkpoint(model, pt, meta)
        idx = np.flatnonzero(train_set.poisoned)
        np.savez_compressed(
            npz,
            poisoned_idx=idx,
            poisoned_x=train_set.x[idx],
            y=train_set.y,
            noise_idx=np.flatnonzero(train_set.noise),
        )
    return Run(model, data, attack, train_set, meta), result


def load_run(
    out_dir: Path,
    data: ImageDataset,
    attack_name: str,
    seed: int,
    device: torch.device | str = "cpu",
) -> Run:
    """Rebuild a saved run: the model, its exact training set and the attack trigger."""
    pt, npz = run_paths(out_dir, data.name, attack_name, seed)
    model, meta = load_checkpoint(pt, device)
    with np.load(npz, allow_pickle=False) as z:
        idx = z["poisoned_idx"]
        x = data.x_train.copy()
        x[idx] = z["poisoned_x"]
        y: NDArray[np.int64] = z["y"].astype(np.int64)
        noise = np.zeros(len(y), bool)
        noise[z["noise_idx"]] = True
    poisoned = np.zeros(len(y), bool)
    poisoned[idx] = True
    target = int(meta.get("target", 0))  # type: ignore[call-overload]
    attack = None
    if attack_name != CLEAN:
        result = meta.get("result", {})
        kwargs = dict(result.get("attack_config", {})) if isinstance(result, dict) else {}
        # A variant run (for example a lower poison rate) records its base attack name.
        base = str(kwargs.get("name", attack_name))
        for key in ("name", "target", "surrogate"):
            kwargs.pop(key, None)
        attack = build_attack(base, data, target, seed, None, **kwargs)
    train_set = PoisonedTrainingSet(
        x=x, y=y, poisoned=poisoned, noise=noise, original_y=data.y_train
    )
    return Run(model, data, attack, train_set, meta)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=float) + "\n")
