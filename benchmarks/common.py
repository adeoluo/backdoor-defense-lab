"""Shared settings of the benchmark scripts."""

from __future__ import annotations

import os
import platform
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"
sys.path.insert(0, str(ROOT / "src"))

from backdoor_defense_lab.data import data_dir  # noqa: E402

RUNS = Path(os.environ.get("BDL_RUNS", data_dir() / "runs"))
SEEDS = (0, 1, 2)
#: Epochs per dataset: GTSRB converges in fewer passes than CIFAR-10.
EPOCHS = {"cifar10": 15, "gtsrb": 10}


def hardware() -> dict[str, str]:
    return {
        "machine": platform.machine(),
        "processor": platform.processor() or "arm",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": "mps" if torch.backends.mps.is_available() else "cpu",
        "date": time.strftime("%Y-%m-%d"),
    }


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)
