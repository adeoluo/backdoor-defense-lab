"""Command line interface: ``python -m backdoor_defense_lab <command>``.

Commands
--------
info      datasets, attacks, defenses and the compute device
demo      poison a synthetic dataset, train a small model and run every defense (no downloads)
train     poison a real dataset, train a ResNet-9 and save the run
defend    run one defense (or all) on a saved run and print its numbers
onion     score each word of a sentence with ONION
report    print the aggregated result tables from ``benchmarks/results``
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from backdoor_defense_lab import __version__
from backdoor_defense_lab.attacks import ATTACKS

DEFENSES = ("spectral", "activation-clustering", "strip", "neural-cleanse", "fine-pruning")


def _print_kv(title: str, values: dict[str, object]) -> None:
    print(title)
    for k, v in values.items():
        if isinstance(v, float):
            v = f"{v:.4f}"
        print(f"  {k:<28} {v}")


def cmd_info(args: argparse.Namespace) -> int:
    from backdoor_defense_lab.data import DATASETS, data_dir, pick_device
    from backdoor_defense_lab.pipeline import VARIANTS

    print(f"backdoor-defense-lab {__version__}")
    print(f"  datasets   {', '.join(DATASETS)} (+ sst2 for text)")
    print(f"  attacks    {', '.join(ATTACKS)} (variants: {', '.join(VARIANTS)})")
    print(f"  defenses   {', '.join(DEFENSES)}, onion (text)")
    print(f"  data dir   {data_dir()}")
    print(f"  device     {pick_device(args.device)}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from backdoor_defense_lab.data import synthetic_dataset
    from backdoor_defense_lab.evaluate import DefenseSettings, evaluate_run
    from backdoor_defense_lab.pipeline import run_attack
    from backdoor_defense_lab.train import TrainConfig

    device = args.device or "cpu"
    data = synthetic_dataset(n_train=args.n_train, n_test=600, num_classes=4, seed=args.seed)
    cfg = TrainConfig(epochs=args.epochs, batch_size=64, width=8, seed=args.seed)
    run, res = run_attack(data, args.attack, args.seed, target=0, config=cfg, device=device)
    print(
        f"{args.attack} on synthetic data ({args.n_train} train images, 4 classes): "
        f"clean acc {res.clean_accuracy:.3f}, ASR {res.attack_success_rate:.3f}, "
        f"{res.n_poisoned} poisoned"
    )
    settings = DefenseSettings(
        n_defender=300, strip_n=200, strip_pool=100, strip_perturb=16, ac_dims=5,
        nc_images=128, nc_epochs=8, nc_steps=4, nc_batch=16, repair_epochs=args.repair_epochs,
        repair_lr=0.05, seed=args.seed,
    )  # fmt: skip
    out: dict[str, dict[str, object]] = evaluate_run(run, settings, device)  # type: ignore[assignment]
    ac: dict[str, dict[str, object]] = out["activation_clustering"]  # type: ignore[assignment]
    strip: dict[str, dict[str, object]] = out["strip"]  # type: ignore[assignment]
    fp: dict[str, dict[str, object]] = out["fine_pruning"]  # type: ignore[assignment]
    summary = {
        "spectral AUROC": out["spectral"]["auroc"],
        "activation clustering AUROC": ac["ica"]["auroc"],
        "STRIP AUROC": strip["blend"]["auroc"],
        "Neural Cleanse suspect": out["neural_cleanse"]["suspect"],
        "Neural Cleanse anomaly": out["neural_cleanse"]["min_anomaly"],
        "unlearning ASR": out["nc_unlearning"]["attack_success_rate"],
        "fine-pruning ASR": fp["after_fine_tuning"]["attack_success_rate"],
    }
    _print_kv("defenses", summary)
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2, default=float))
    return 0


def _runs_dir(arg: str | None) -> Path:
    from backdoor_defense_lab.data import data_dir

    return Path(arg) if arg else data_dir() / "runs"


def cmd_train(args: argparse.Namespace) -> int:
    from backdoor_defense_lab.data import DEFAULT_TARGET, load_dataset, pick_device
    from backdoor_defense_lab.pipeline import CLEAN, run_attack, run_paths
    from backdoor_defense_lab.train import TrainConfig, load_checkpoint

    device = pick_device(args.device)
    data = load_dataset(args.dataset, Path(args.data) if args.data else None)
    if args.subset:
        data = data.subset(args.subset, None)
    surrogate = None
    if args.attack == "label_consistent":
        clean_pt, _ = run_paths(_runs_dir(args.runs), args.dataset, CLEAN, args.seed)
        if clean_pt.exists():
            surrogate, _ = load_checkpoint(clean_pt, device)
    cfg = TrainConfig(epochs=args.epochs, seed=args.seed)
    _, res = run_attack(
        data,
        args.attack,
        args.seed,
        DEFAULT_TARGET[args.dataset],
        cfg,
        device,
        _runs_dir(args.runs),
        surrogate,
        log=print,
    )
    _print_kv(f"{args.dataset} {args.attack} seed {args.seed}", res.as_dict())
    return 0


def cmd_defend(args: argparse.Namespace) -> int:
    import torch

    from backdoor_defense_lab.attacks import triggered_test_set
    from backdoor_defense_lab.data import load_dataset, pick_device, to_tensor
    from backdoor_defense_lab.defenses import (
        activation_clustering,
        fine_pruning,
        reverse_engineer,
        spectral_scores,
        strip_scores,
    )
    from backdoor_defense_lab.evaluate import features_of, split_test
    from backdoor_defense_lab.metrics import detection_summary
    from backdoor_defense_lab.models import predict
    from backdoor_defense_lab.pipeline import load_run

    device = pick_device(args.device)
    data = load_dataset(args.dataset, Path(args.data) if args.data else None)
    run = load_run(_runs_dir(args.runs), data, args.attack, args.seed, device)
    attack = run.attack
    defender, held = split_test(data, args.n_defender)
    target = attack.target if attack is not None else None
    labels, poisoned = run.train_set.y, run.train_set.poisoned
    wanted = DEFENSES if args.defense == "all" else (args.defense,)
    print(f"{args.dataset} {args.attack} seed {args.seed} ({run.train_set.n_poisoned} poisoned)")

    def ca_asr(model: torch.nn.Module) -> str:
        ca = float((predict(model, to_tensor(held.x_test)).numpy() == held.y_test).mean())
        if attack is None:
            return f"clean acc {ca:.4f}"
        xt, _ = triggered_test_set(held, attack)
        asr = float((predict(model, to_tensor(xt)).numpy() == attack.target).mean())
        return f"clean acc {ca:.4f}  ASR {asr:.4f}"

    print(f"  model                        {ca_asr(run.model)}")
    feats = None
    if attack is not None and ({"spectral", "activation-clustering"} & set(wanted)):
        feats = features_of(run.model, run.train_set.x)
    in_target = labels == target
    for name in wanted:
        if name == "spectral" and feats is not None:
            s = spectral_scores(feats, labels)
            d = detection_summary(s[in_target], poisoned[in_target])
            print(f"  spectral signatures          AUROC {d.auroc:.4f}  TPR@5% {d.tpr_at_5:.4f}")
        elif name == "activation-clustering" and feats is not None:
            ac = activation_clustering(feats, labels, seed=args.seed)
            d = detection_summary(ac.scores[in_target], poisoned[in_target])
            flagged = [r.label for r in ac.classes if r.flagged]
            print(f"  activation clustering        AUROC {d.auroc:.4f}  flagged classes {flagged}")
        elif name == "strip" and attack is not None:
            keep = np.flatnonzero(held.y_test != attack.target)[: args.strip_n]
            xt, _ = triggered_test_set(held, attack)
            x = torch.cat([to_tensor(held.x_test[keep]), to_tensor(xt[: len(keep)])])
            pool = to_tensor(defender.x_test[: args.strip_n // 2])
            s = strip_scores(run.model, x, pool, seed=args.seed)
            lab = np.r_[np.zeros(len(keep), bool), np.ones(len(keep), bool)]
            d = detection_summary(s, lab)
            print(f"  STRIP                        AUROC {d.auroc:.4f}  TPR@5% {d.tpr_at_5:.4f}")
        elif name == "neural-cleanse":
            nc = reverse_engineer(
                run.model,
                to_tensor(defender.x_test[: args.nc_images]),
                data.num_classes,
                epochs=args.nc_epochs,
                seed=args.seed,
            )
            l1 = ", ".join(f"{v:.0f}" for v in nc.l1[:10])
            print(f"  Neural Cleanse mask L1       {l1}{' ...' if len(nc.l1) > 10 else ''}")
            print(
                f"  Neural Cleanse               suspect class {nc.suspect}, "
                f"anomaly index {nc.min_anomaly:.2f}, flagged {nc.flagged}"
            )
        elif name == "fine-pruning" and attack is not None:
            pruned, tuned = fine_pruning(
                run.model,
                defender.x_test,
                defender.y_test,
                0.04,
                epochs=args.repair_epochs,
                hflip=data.hflip,
                seed=args.seed,
            )
            print(f"  pruned {pruned.n_pruned:>3} channels           {ca_asr(pruned.model)}")
            print(f"  fine-pruned                  {ca_asr(tuned)}")
    return 0


def cmd_onion(args: argparse.Namespace) -> int:
    from backdoor_defense_lab.text.data import tokenize
    from backdoor_defense_lab.text.onion import (
        GPT2Scorer,
        UnigramScorer,
        onion_filter,
        suspicion_scores,
    )

    scorer: Callable[[Sequence[str]], np.ndarray]
    if args.lm == "unigram":
        from backdoor_defense_lab.text.data import load_sst2

        scorer = UnigramScorer(load_sst2(Path(args.data) if args.data else None).train)
    else:  # pragma: no cover - downloads GPT-2
        scorer = GPT2Scorer(args.lm, device=args.device or "cpu")
    for sentence in args.sentences:
        (scores,) = suspicion_scores([sentence], scorer)
        print(sentence)
        for w, s in zip(tokenize(sentence), scores, strict=True):
            mark = "  <- removed" if s > args.threshold else ""
            print(f"  {w:<16} {s:>10.1f}{mark}")
        print(f"  filtered: {onion_filter(sentence, scores, args.threshold)}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from backdoor_defense_lab.report import SECTIONS, render

    sections = args.section or list(SECTIONS)
    text = render(Path(args.results), sections)
    print(text if text else f"no results under {args.results}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="backdoor_defense_lab", description="Backdoor defense lab")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)

    def add(
        name: str, fn: Callable[[argparse.Namespace], int], help_: str
    ) -> argparse.ArgumentParser:
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("--device", default=None, help="cpu or mps (default: mps if available)")
        sp.set_defaults(func=fn)
        return sp

    add("info", cmd_info, "datasets, attacks, defenses and device")
    d = add("demo", cmd_demo, "synthetic end-to-end run, no downloads")
    d.add_argument("--attack", default="badnets", choices=ATTACKS[:3])
    d.add_argument("--n-train", type=int, default=1500)
    d.add_argument("--epochs", type=int, default=8)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--repair-epochs", type=int, default=10)
    d.add_argument("--json", default=None, help="write the full result here")

    t = add("train", cmd_train, "poison, train and save one run")
    t.add_argument("--dataset", choices=("cifar10", "gtsrb"), required=True)
    t.add_argument("--attack", default="badnets")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--epochs", type=int, default=15)
    t.add_argument("--subset", type=int, default=None, help="use the first N training images")
    t.add_argument("--runs", default=None, help="run directory (default: data dir / runs)")
    t.add_argument("--data", default=None, help="data directory (default: $BDL_DATA)")

    f = add("defend", cmd_defend, "run defenses on a saved run")
    f.add_argument("--dataset", choices=("cifar10", "gtsrb"), required=True)
    f.add_argument("--attack", default="badnets")
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--defense", choices=(*DEFENSES, "all"), default="all")
    f.add_argument("--runs", default=None)
    f.add_argument("--data", default=None)
    f.add_argument("--n-defender", type=int, default=5000)
    f.add_argument("--strip-n", type=int, default=1000)
    f.add_argument("--nc-images", type=int, default=1000)
    f.add_argument("--nc-epochs", type=int, default=20)
    f.add_argument("--repair-epochs", type=int, default=5)

    o = add("onion", cmd_onion, "ONION word scores for sentences")
    o.add_argument("sentences", nargs="+")
    o.add_argument("--lm", default="openai-community/gpt2", help="model name or 'unigram'")
    o.add_argument("--threshold", type=float, default=0.0)
    o.add_argument("--data", default=None)

    r = add("report", cmd_report, "aggregated result tables")
    r.add_argument("--results", default="benchmarks/results")
    r.add_argument("--section", action="append", default=None)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    code: int = args.func(args)
    return code
