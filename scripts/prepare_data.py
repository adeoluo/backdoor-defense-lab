"""Convert the downloaded archives into uint8 arrays (N, 32, 32, 3) saved as .npz.

CIFAR-10: the 50,000 training and 10,000 test images of the python pickle release.
GTSRB: all 39,209 training and 12,630 test images of the final GTSRB release, each full
image (not the sign ROI) resized to 32x32 with bilinear filtering, as in BackdoorBench.
The archives are deleted after a successful conversion.
"""

from __future__ import annotations

import csv
import io
import pickle
import sys
import tarfile
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


def prepare_cifar(root: Path) -> None:
    out = root / "cifar10.npz"
    tar_path = root / "cifar-10-python.tar.gz"
    if out.exists() or not tar_path.exists():
        return
    xs: dict[str, list[np.ndarray]] = {"train": [], "test": []}
    ys: dict[str, list[np.ndarray]] = {"train": [], "test": []}
    with tarfile.open(tar_path) as tar:
        for member in tar.getmembers():
            name = Path(member.name).name
            split = (
                "train"
                if name.startswith("data_batch")
                else "test"
                if name == "test_batch"
                else None
            )
            if split is None:
                continue
            fh = tar.extractfile(member)
            assert fh is not None
            batch = pickle.load(fh, encoding="bytes")
            xs[split].append(batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1))
            ys[split].append(np.asarray(batch[b"labels"], dtype=np.int64))
    np.savez(
        out,
        x_train=np.concatenate(xs["train"]).astype(np.uint8),
        y_train=np.concatenate(ys["train"]),
        x_test=np.concatenate(xs["test"]).astype(np.uint8),
        y_test=np.concatenate(ys["test"]),
    )
    tar_path.unlink()
    print("cifar10.npz written")


def _load_ppm(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as im:
        return np.asarray(im.convert("RGB").resize((32, 32), Image.BILINEAR), dtype=np.uint8)


def prepare_gtsrb(root: Path) -> None:
    out = root / "gtsrb.npz"
    train_zip = root / "GTSRB_Final_Training_Images.zip"
    test_zip = root / "GTSRB_Final_Test_Images.zip"
    gt_zip = root / "GTSRB_Final_Test_GT.zip"
    if out.exists() or not train_zip.exists():
        return
    x_train, y_train = [], []
    with zipfile.ZipFile(train_zip) as z:
        for name in sorted(z.namelist()):
            if name.endswith(".ppm"):
                x_train.append(_load_ppm(z.read(name)))
                y_train.append(int(Path(name).parent.name))
    with zipfile.ZipFile(gt_zip) as z:
        rows = list(csv.DictReader(io.TextIOWrapper(z.open("GT-final_test.csv")), delimiter=";"))
    labels = {r["Filename"]: int(r["ClassId"]) for r in rows}
    x_test, y_test = [], []
    with zipfile.ZipFile(test_zip) as z:
        for name in sorted(z.namelist()):
            if name.endswith(".ppm"):
                x_test.append(_load_ppm(z.read(name)))
                y_test.append(labels[Path(name).name])
    np.savez(
        out,
        x_train=np.stack(x_train),
        y_train=np.asarray(y_train, dtype=np.int64),
        x_test=np.stack(x_test),
        y_test=np.asarray(y_test, dtype=np.int64),
    )
    for p in (train_zip, test_zip, gt_zip):
        p.unlink()
    print(f"gtsrb.npz written: {len(x_train)} train, {len(x_test)} test")


if __name__ == "__main__":
    root = Path(sys.argv[1])
    prepare_cifar(root)
    prepare_gtsrb(root)
