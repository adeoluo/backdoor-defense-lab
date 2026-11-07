"""Run every defense on the trained models (see ``train_models.py``).

    uv run python benchmarks/run_defenses.py --datasets cifar10 --seeds 0

Writes ``benchmarks/results/defenses/<dataset>_<attack>_s<seed>.json``; existing files are
skipped, so the script can be stopped and resumed. Clean models only get Neural Cleanse (the
model-level negatives); the data and input detectors need a poisoned model.
"""

from __future__ import annotations

import argparse

from common import RESULTS, RUNS, SEEDS, hardware, log

from backdoor_defense_lab.attacks import ATTACKS
from backdoor_defense_lab.data import load_dataset, pick_device
from backdoor_defense_lab.evaluate import DefenseSettings, evaluate_run
from backdoor_defense_lab.pipeline import CLEAN, load_run, run_paths, write_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["cifar10", "gtsrb"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--attacks", nargs="+", default=[CLEAN, *ATTACKS])
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = pick_device(args.device)
    for name in args.datasets:
        data = load_dataset(name)
        for seed in args.seeds:
            for attack in args.attacks:
                out = RESULTS / "defenses" / f"{name}_{attack}_s{seed}.json"
                pt, _ = run_paths(RUNS, name, attack, seed)
                if out.exists() or not pt.exists():
                    continue
                log(f"defenses {name} {attack} seed {seed} on {device}")
                run = load_run(RUNS, data, attack, seed, device)
                res = evaluate_run(run, DefenseSettings(seed=seed), device, log=log)
                write_json(
                    out,
                    {
                        "dataset": name,
                        "attack": attack,
                        "seed": seed,
                        **res,
                        "hardware": hardware(),
                    },
                )
                log(f"done {name} {attack} s{seed} in {sum(res['seconds'].values()):.0f}s")


if __name__ == "__main__":
    main()
