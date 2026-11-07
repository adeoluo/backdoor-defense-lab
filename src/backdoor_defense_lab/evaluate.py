"""Run every defense on one trained model and measure it against the poisoning ground truth.

The defender gets the trained model, its training set (for the data-inspection defenses) and a
small trusted clean set. The trusted set is the first ``n_defender`` test images; every
accuracy and attack success rate is measured on the remaining test images, so the defender
never sees the images it is scored on.

Per defense:

* spectral signatures and activation clustering score every training example; the detection
  AUROC and TPR at 1% / 5% FPR are measured inside the target class (where the poisons are),
  and the removal recall / precision use the defense's own decision rule;
* STRIP scores test-time inputs: clean non-target images (negatives) against the same images with
  the trigger (positives);
* Neural Cleanse gives a model-level anomaly index and a suspected target, then unlearns the
  reversed trigger;
* fine-pruning reports clean accuracy and attack success rate after pruning and after
  fine-tuning; plain fine-tuning on the same clean data is the ablation.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from backdoor_defense_lab.attacks import triggered_test_set
from backdoor_defense_lab.data import ImageDataset, to_tensor
from backdoor_defense_lab.defenses import (
    activation_clustering,
    fine_pruning,
    fine_tune,
    reverse_engineer,
    spectral_filter,
    spectral_scores,
    strip_scores,
    unlearn,
)
from backdoor_defense_lab.metrics import detection_summary
from backdoor_defense_lab.models import ResNet9, extract_features, predict
from backdoor_defense_lab.pipeline import Run


@dataclass
class DefenseSettings:
    n_defender: int = 5000
    strip_n: int = 1000
    strip_pool: int = 500
    strip_perturb: int = 32
    ac_dims: int = 10
    nc_images: int = 1000
    nc_epochs: int = 20
    nc_steps: int = 10
    nc_batch: int = 32
    repair_epochs: int = 5
    repair_lr: float = 0.01
    prune_drop: float = 0.04
    seed: int = 0


def _ca_asr(
    model: ResNet9,
    x: torch.Tensor,
    y: NDArray[np.int64],
    xt: torch.Tensor | None,
    target: int | None,
) -> dict[str, float]:
    ca = float((predict(model, x).numpy() == y).mean()) if len(y) else float("nan")
    asr = float("nan")
    if xt is not None and target is not None and len(xt):
        asr = float((predict(model, xt).numpy() == target).mean())
    return {"clean_accuracy": ca, "attack_success_rate": asr}


def features_of(model: ResNet9, x: NDArray[np.uint8], chunk: int = 4096) -> NDArray[np.float64]:
    """Penultimate features of uint8 images, converted chunk by chunk to bound memory."""
    parts = [
        extract_features(model, to_tensor(x[i : i + chunk])).double().numpy()
        for i in range(0, len(x), chunk)
    ]
    return np.concatenate(parts) if parts else np.zeros((0, model.feature_dim))


def split_test(data: ImageDataset, n_defender: int) -> tuple[ImageDataset, ImageDataset]:
    """Trusted defender images (first ``n_defender``) and evaluation images (the rest)."""
    n = min(n_defender, len(data.y_test) // 2)

    def part(sl: slice) -> ImageDataset:
        return ImageDataset(
            data.name,
            data.x_train,
            data.y_train,
            data.x_test[sl],
            data.y_test[sl],
            data.num_classes,
            data.mean,
            data.std,
            data.hflip,
        )

    return part(slice(0, n)), part(slice(n, None))


def evaluate_run(
    run: Run,
    settings: DefenseSettings | None = None,
    device: torch.device | str = "cpu",
    log: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Every defense on one model; returns a JSON-ready dictionary of metrics and timings."""
    cfg = settings or DefenseSettings()
    say = log or (lambda _msg: None)
    dev = torch.device(device)
    model = run.model.to(dev).eval()
    data, attack = run.data, run.attack
    target = attack.target if attack is not None else None
    defender, held = split_test(data, cfg.n_defender)
    x_eval = to_tensor(held.x_test)
    xt_eval: torch.Tensor | None = None
    if attack is not None:
        xt_np, _ = triggered_test_set(held, attack)
        xt_eval = to_tensor(xt_np)
    out: dict[str, object] = {"settings": asdict(cfg)}
    out["before"] = _ca_asr(model, x_eval, held.y_test, xt_eval, target)
    timings: dict[str, float] = {}

    labels = run.train_set.y
    poisoned = run.train_set.poisoned
    if attack is not None:
        t0 = time.perf_counter()
        feats = features_of(model, run.train_set.x)
        timings["features"] = time.perf_counter() - t0
        in_target = labels == attack.target
        rate = float(poisoned[in_target].mean())

        t0 = time.perf_counter()
        ss = spectral_scores(feats, labels)
        removed = spectral_filter(ss, labels, eps=rate)
        timings["spectral"] = time.perf_counter() - t0
        # ART's variant projects the uncentered features: |v . r| (kept for the comparison).
        from backdoor_defense_lab.defenses.spectral import top_singular_vector

        ft = feats[in_target]
        v = top_singular_vector(ft - ft.mean(0))
        art_scores = np.abs(ft @ v)
        out["spectral"] = {
            **detection_summary(ss[in_target], poisoned[in_target]).as_dict(),
            "auroc_uncentered_art_variant": detection_summary(
                art_scores, poisoned[in_target]
            ).auroc,
            "removed": int(removed.sum()),
            "removal_recall": int((removed & poisoned).sum()) / max(int(poisoned.sum()), 1),
            "removal_precision": int((removed & poisoned).sum()) / max(int(removed.sum()), 1),
        }
        say(f"spectral auroc {out['spectral']['auroc']:.4f}")  # type: ignore[index]

        ac_out: dict[str, object] = {}
        for reduce in ("ica", "pca"):
            t0 = time.perf_counter()
            ac = activation_clustering(feats, labels, n_dims=cfg.ac_dims, reduce=reduce)
            timings[f"activation_clustering_{reduce}"] = time.perf_counter() - t0
            flagged_classes = [r.label for r in ac.classes if r.flagged]
            tp = int((ac.flagged & poisoned).sum())
            ac_out[reduce] = {
                **detection_summary(ac.scores[in_target], poisoned[in_target]).as_dict(),
                "target_flagged": attack.target in flagged_classes,
                "other_classes_flagged": len([c for c in flagged_classes if c != attack.target]),
                "recall": tp / max(int(poisoned.sum()), 1),
                "precision": tp / max(int(ac.flagged.sum()), 1),
                "target_smaller_fraction": next(
                    r.smaller_fraction for r in ac.classes if r.label == attack.target
                ),
                "target_silhouette": next(
                    r.silhouette for r in ac.classes if r.label == attack.target
                ),
            }
        out["activation_clustering"] = ac_out
        say(f"activation clustering auroc {ac_out['ica']['auroc']:.4f}")  # type: ignore[index]
        del feats

        assert xt_eval is not None
        rng = np.random.default_rng(cfg.seed)
        pool = to_tensor(defender.x_test[: cfg.strip_pool])
        keep = np.flatnonzero(held.y_test != attack.target)
        pick = np.sort(rng.choice(len(keep), min(cfg.strip_n, len(keep)), replace=False))
        clean_x = x_eval[keep[pick]]
        trig_x = xt_eval[pick]  # triggered_test_set keeps the same non-target order
        strip_out: dict[str, object] = {}
        for mode in ("blend", "add"):
            t0 = time.perf_counter()
            s = strip_scores(
                model,
                torch.cat([clean_x, trig_x]),
                pool,
                n_perturb=cfg.strip_perturb,
                mode=mode,
                seed=cfg.seed,
            )
            timings[f"strip_{mode}"] = time.perf_counter() - t0
            lab = np.r_[np.zeros(len(clean_x), bool), np.ones(len(trig_x), bool)]
            strip_out[mode] = detection_summary(s, lab).as_dict()
        out["strip"] = strip_out
        say(f"strip auroc {strip_out['blend']['auroc']:.4f}")  # type: ignore[index]

    t0 = time.perf_counter()
    nc = reverse_engineer(
        model,
        to_tensor(defender.x_test[: cfg.nc_images]),
        data.num_classes,
        epochs=cfg.nc_epochs,
        steps_per_epoch=cfg.nc_steps,
        batch_size=cfg.nc_batch,
        seed=cfg.seed,
    )
    timings["neural_cleanse"] = time.perf_counter() - t0
    out["neural_cleanse"] = {
        "suspect": nc.suspect,
        "min_anomaly": nc.min_anomaly,
        "flagged": nc.flagged,
        "target_anomaly": float(nc.anomaly[target]) if target is not None else float("nan"),
        "target_found": target is not None and nc.suspect == target and nc.min_anomaly > 2.0,
        "l1": nc.l1.tolist(),
        "success": nc.success.tolist(),
    }
    say(f"neural cleanse suspect {nc.suspect} anomaly {nc.min_anomaly:.2f}")

    if attack is not None:
        t0 = time.perf_counter()
        k = nc.suspect
        repaired = unlearn(
            copy.deepcopy(model),
            defender.x_test,
            defender.y_test,
            nc.masks[k],
            nc.patterns[k],
            epochs=cfg.repair_epochs,
            lr=cfg.repair_lr,
            hflip=data.hflip,
            seed=cfg.seed,
        )
        timings["nc_unlearning"] = time.perf_counter() - t0
        out["nc_unlearning"] = _ca_asr(repaired, x_eval, held.y_test, xt_eval, target)
        del repaired

        t0 = time.perf_counter()
        pruned, tuned = fine_pruning(
            model,
            defender.x_test,
            defender.y_test,
            max_accuracy_drop=cfg.prune_drop,
            epochs=cfg.repair_epochs,
            lr=cfg.repair_lr,
            hflip=data.hflip,
            seed=cfg.seed,
        )
        timings["fine_pruning"] = time.perf_counter() - t0
        out["fine_pruning"] = {
            "n_pruned": pruned.n_pruned,
            "prune_ratio": pruned.prune_ratio,
            "after_pruning": _ca_asr(pruned.model, x_eval, held.y_test, xt_eval, target),
            "after_fine_tuning": _ca_asr(tuned, x_eval, held.y_test, xt_eval, target),
        }
        t0 = time.perf_counter()
        plain = fine_tune(
            copy.deepcopy(model),
            defender.x_test,
            defender.y_test,
            epochs=cfg.repair_epochs,
            lr=cfg.repair_lr,
            hflip=data.hflip,
            seed=cfg.seed,
        )
        timings["fine_tuning_only"] = time.perf_counter() - t0
        out["fine_tuning_only"] = _ca_asr(plain, x_eval, held.y_test, xt_eval, target)
        say(
            "fine-pruning ASR {:.4f} -> {:.4f}".format(
                out["before"]["attack_success_rate"],  # type: ignore[index]
                out["fine_pruning"]["after_fine_tuning"]["attack_success_rate"],  # type: ignore[index]
            )
        )
    out["seconds"] = timings
    return out
