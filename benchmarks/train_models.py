"""Train the clean and backdoored image models (every dataset, attack and seed).

    uv run python benchmarks/train_models.py --datasets cifar10 gtsrb --seeds 0 1 2

Checkpoints and poisoning ground truth go to ``$BDL_RUNS`` (default
``../.bench-data/backdoor-defense-lab/runs``); the clean accuracy, attack success rate and timings
of each run go to ``benchmarks/results/train/<dataset>_<attack>_s<seed>.json``. Existing runs are
skipped, so the script can be stopped and resumed. The clean model of a seed is trained first:
it is the surrogate the label-consistent attack perturbs against.
"""

from __future__ import annotations

import argparse

from common import EPOCHS, RESULTS, RUNS, SEEDS, hardware, log

from backdoor_defense_lab.attacks import ATTACKS
from backdoor_defense_lab.data import DEFAULT_TARGET, load_dataset, pick_device
from backdoor_defense_lab.pipeline import CLEAN, run_attack, run_paths, write_json
from backdoor_defense_lab.train import TrainConfig, load_checkpoint


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["cifar10", "gtsrb"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--attacks", nargs="+", default=[CLEAN, *ATTACKS])
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = pick_device(args.device)
    for seed in args.seeds:
        for name in args.datasets:
            data = load_dataset(name)
            target = DEFAULT_TARGET[name]
            for attack in args.attacks:
                pt, _ = run_paths(RUNS, name, attack, seed)
                out = RESULTS / "train" / f"{name}_{attack}_s{seed}.json"
                if pt.exists() and out.exists():
                    continue
                surrogate = None
                if attack == "label_consistent":
                    clean_pt, _ = run_paths(RUNS, name, CLEAN, seed)
                    surrogate, _ = load_checkpoint(clean_pt, device)
                log(f"train {name} {attack} seed {seed} on {device}")
                cfg = TrainConfig(epochs=EPOCHS[name], seed=seed)
                _, result = run_attack(
                    data, attack, seed, target, cfg, device, RUNS, surrogate, log=log
                )
                write_json(out, {**result.as_dict(), "hardware": hardware()})
                log(
                    f"done {name} {attack} s{seed}: CA {result.clean_accuracy:.4f} "
                    f"ASR {result.attack_success_rate:.4f} in {result.train_seconds:.0f}s"
                )


if __name__ == "__main__":
    main()
