"""Compare spectral signatures and activation clustering with IBM ART on the same features.

    uv sync --group validation && uv run python benchmarks/compare_art.py --seeds 0

For every saved CIFAR-10 and GTSRB run, the penultimate features of the target class are scored
by this repository and by ART 1.20 (``SpectralSignatureDefense.spectral_signature_scores``, and
``cluster_activations`` + ``ClusteringAnalyzer.analyze_by_size`` for activation clustering with
PCA and FastICA). Reports AUROC, the share of examples on which the two flag the same way, and
wall time. Writes ``benchmarks/results/art/<dataset>_<attack>_s<seed>.json``.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from common import RESULTS, RUNS, hardware, log

from backdoor_defense_lab.data import load_dataset, pick_device
from backdoor_defense_lab.defenses import activation_clustering
from backdoor_defense_lab.defenses.spectral import class_spectral_scores
from backdoor_defense_lab.evaluate import features_of
from backdoor_defense_lab.metrics import auroc
from backdoor_defense_lab.pipeline import load_run, run_paths, write_json

ATTACKS = ("badnets", "blend", "wanet", "label_consistent", "badnets_low")


def main() -> None:
    from art.defences.detector.poison import SpectralSignatureDefense
    from art.defences.detector.poison.activation_defence import cluster_activations
    from art.defences.detector.poison.clustering_analyzer import ClusteringAnalyzer

    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["cifar10", "gtsrb"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = pick_device(args.device)
    for name in args.datasets:
        data = load_dataset(name)
        for seed in args.seeds:
            for attack in ATTACKS:
                out = RESULTS / "art" / f"{name}_{attack}_s{seed}.json"
                pt, _ = run_paths(RUNS, name, attack, seed)
                if out.exists() or not pt.exists():
                    continue
                run = load_run(RUNS, data, attack, seed, device)
                assert run.attack is not None
                in_t = run.train_set.y == run.attack.target
                feats = features_of(run.model, run.train_set.x[in_t])
                truth = run.train_set.poisoned[in_t]
                res: dict[str, object] = {"dataset": name, "attack": attack, "seed": seed}
                t0 = time.perf_counter()
                ours = class_spectral_scores(feats)
                t_ours = time.perf_counter() - t0
                t0 = time.perf_counter()
                ref = SpectralSignatureDefense.spectral_signature_scores(feats).ravel()
                t_ref = time.perf_counter() - t0
                res["spectral"] = {
                    "auroc_ours_centered": auroc(ours, truth),
                    "auroc_art_uncentered": auroc(ref, truth),
                    "seconds_ours": t_ours,
                    "seconds_art": t_ref,
                }
                for reduce, art_reduce in (("pca", "PCA"), ("ica", "FastICA")):
                    y0 = np.zeros(len(feats), int)
                    t0 = time.perf_counter()
                    ac = activation_clustering(feats, y0, n_dims=10, reduce=reduce, seed=seed)
                    t_ours = time.perf_counter() - t0
                    t0 = time.perf_counter()
                    clusters, _ = cluster_activations([feats], 2, 10, art_reduce)
                    clean_flags, _, _ = ClusteringAnalyzer().analyze_by_size(clusters)
                    t_ref = time.perf_counter() - t0
                    art_flag = np.asarray(clean_flags[0]) == 0
                    # the 2-means split itself, independent of which side is called poisoned
                    ours_side = ac.scores > 0
                    art_side = np.asarray(clusters[0]) == np.bincount(clusters[0]).argmin()
                    agree = max(np.mean(ours_side == art_side), np.mean(ours_side != art_side))
                    res[f"ac_{reduce}"] = {
                        "flag_agreement": float(np.mean(ac.flagged == art_flag)),
                        "split_agreement": float(agree),
                        "ours_flagged": bool(ac.flagged.any()),
                        "art_flagged": bool(art_flag.any()),
                        "ours_smaller_fraction": ac.classes[0].smaller_fraction,
                        "art_smaller_fraction": float(np.bincount(clusters[0]).min() / len(feats)),
                        "seconds_ours": t_ours,
                        "seconds_art": t_ref,
                    }
                write_json(out, {**res, "n": len(feats), "hardware": hardware()})
                log(f"art comparison {name} {attack} s{seed} done")


if __name__ == "__main__":
    main()
