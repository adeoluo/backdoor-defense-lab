from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from backdoor_defense_lab import __version__
from backdoor_defense_lab.cli import build_parser, main
from backdoor_defense_lab.data import synthetic_dataset

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A fake data directory: a 10-class synthetic 'cifar10' and a tiny SST-2."""
    d = tmp_path_factory.mktemp("data")
    toy = synthetic_dataset(n_train=600, n_test=300, num_classes=10, seed=1)
    np.savez(
        d / "cifar10.npz",
        x_train=toy.x_train,
        y_train=toy.y_train,
        x_test=toy.x_test,
        y_test=toy.y_test,
    )
    (d / "sst2").mkdir()
    rows = "sentence\tlabel\n" + "the film was good\t1\nthe plot was dull\t0\n" * 20
    (d / "sst2" / "train.tsv").write_text(rows)
    (d / "sst2" / "dev.tsv").write_text(rows)
    return d


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["--version"])
    assert __version__ in capsys.readouterr().out


def test_command_required() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_info(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["info", "--device", "cpu"]) == 0
    out = capsys.readouterr().out
    assert "badnets" in out and "neural-cleanse" in out and "cpu" in out


def test_demo(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out_json = tmp_path / "demo.json"
    args = ["demo", "--n-train", "600", "--epochs", "5", "--json", str(out_json)]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "STRIP AUROC" in out and "Neural Cleanse suspect" in out
    assert "spectral" in json.loads(out_json.read_text())


def test_train_and_defend(
    data_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs = tmp_path / "runs"
    common = ["--dataset", "cifar10", "--seed", "0", "--runs", str(runs), "--data", str(data_dir)]
    assert main(["train", *common, "--attack", "badnets", "--epochs", "2", "--device", "cpu"]) == 0
    assert (runs / "cifar10" / "badnets_s0.pt").exists()
    small = ["--n-defender", "100", "--strip-n", "40", "--nc-images", "32", "--nc-epochs", "2"]
    small += ["--repair-epochs", "1", "--device", "cpu"]
    assert main(["defend", *common, "--attack", "badnets", *small]) == 0
    out = capsys.readouterr().out
    for word in ("spectral", "activation clustering", "STRIP", "Neural Cleanse", "fine-pruned"):
        assert word in out


def test_train_clean_then_label_consistent(
    data_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs = tmp_path / "runs"
    common = ["--dataset", "cifar10", "--runs", str(runs), "--data", str(data_dir)]
    common += ["--epochs", "1", "--device", "cpu", "--subset", "300"]
    assert main(["train", *common, "--attack", "clean"]) == 0
    assert main(["train", *common, "--attack", "label_consistent"]) == 0
    assert "surrogate" in capsys.readouterr().out
    small = ["--n-defender", "100", "--nc-images", "16", "--nc-epochs", "1", "--device", "cpu"]
    run_args = ["--dataset", "cifar10", "--runs", str(runs), "--data", str(data_dir)]
    assert main(["defend", *run_args, "--attack", "clean", *small]) == 0
    assert "Neural Cleanse" in capsys.readouterr().out


def test_onion_unigram(data_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = ["onion", "the film cf was good", "--lm", "unigram", "--data", str(data_dir)]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "cf" in out and "<- removed" in out
    assert "filtered: the film was good" in out


def test_report_on_committed_results(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["report", "--results", str(ROOT / "benchmarks" / "results")]) == 0
    out = capsys.readouterr().out
    assert "### Attacks" in out and "CIFAR-10" in out and "BadNets" in out


def test_report_empty(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["report", "--results", str(tmp_path)]) == 0
    assert "no results" in capsys.readouterr().out


def test_module_entrypoint() -> None:
    out = subprocess.run(
        [sys.executable, "-m", "backdoor_defense_lab", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert __version__ in out.stdout
