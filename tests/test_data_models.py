from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from backdoor_defense_lab.data import (
    ImageDataset,
    channel_stats,
    data_dir,
    load_dataset,
    pick_device,
    random_crop_flip,
    synthetic_dataset,
    to_tensor,
    to_uint8,
)
from backdoor_defense_lab.models import (
    build_model,
    count_parameters,
    extract_features,
    predict,
    predict_proba,
)


def test_synthetic_dataset_shapes(toy: ImageDataset) -> None:
    assert toy.x_train.shape == (600, 32, 32, 3)
    assert toy.image_size == 32
    assert set(np.unique(toy.y_train)) == {0, 1, 2, 3}
    sub = toy.subset(10, 5)
    assert len(sub.x_train) == 10 and len(sub.y_test) == 5


def test_dataset_validation() -> None:
    x = np.zeros((2, 32, 32, 3), np.uint8)
    with pytest.raises(ValueError):
        ImageDataset("x", x.astype(np.float32), np.zeros(2, np.int64), x, np.zeros(2, np.int64), 2)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ImageDataset("x", x, np.zeros(3, np.int64), x, np.zeros(2, np.int64), 2)
    with pytest.raises(ValueError):
        ImageDataset("x", x, np.full(2, 5, np.int64), x, np.zeros(2, np.int64), 2)


def test_tensor_round_trip(toy: ImageDataset) -> None:
    t = to_tensor(toy.x_train[:5])
    assert t.shape == (5, 3, 32, 32) and t.dtype == torch.float32
    assert float(t.min()) >= 0 and float(t.max()) <= 1
    np.testing.assert_array_equal(to_uint8(t), toy.x_train[:5])


def test_channel_stats() -> None:
    x = np.zeros((4, 2, 2, 3), np.uint8)
    x[..., 0] = 255
    mean, std = channel_stats(x)
    assert mean == (1.0, 0.0, 0.0) and std == (0.0, 0.0, 0.0)


def test_load_dataset_errors(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        load_dataset("mnist", tmp_path)
    with pytest.raises(FileNotFoundError):
        load_dataset("cifar10", tmp_path)


def test_load_dataset_from_npz(tmp_path: Path) -> None:
    d = synthetic_dataset(20, 10, num_classes=10)
    np.savez(
        tmp_path / "cifar10.npz",
        x_train=d.x_train,
        y_train=d.y_train,
        x_test=d.x_test,
        y_test=d.y_test,
    )
    loaded = load_dataset("cifar10", tmp_path)
    assert loaded.num_classes == 10 and loaded.hflip
    np.testing.assert_array_equal(loaded.x_test, d.x_test)


def test_data_dir_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BDL_DATA", str(tmp_path))
    assert data_dir() == tmp_path
    monkeypatch.delenv("BDL_DATA")
    assert data_dir().name == "backdoor-defense-lab"


def test_pick_device() -> None:
    assert pick_device("cpu").type == "cpu"
    assert pick_device().type in {"cpu", "mps", "cuda"}


def test_random_crop_flip_preserves_content() -> None:
    x = torch.rand(6, 3, 32, 32)
    g = torch.Generator().manual_seed(0)
    out = random_crop_flip(x, g, pad=0, hflip=False)
    torch.testing.assert_close(out, x)
    out = random_crop_flip(x, g, pad=4, hflip=True)
    assert out.shape == x.shape
    # each output is a shifted (and maybe flipped) window of the reflect-padded input
    xp = torch.nn.functional.pad(x, (4, 4, 4, 4), mode="reflect")
    for i in range(6):
        found = False
        for dy in range(9):
            for dx in range(9):
                win = xp[i, :, dy : dy + 32, dx : dx + 32]
                if torch.equal(win, out[i]) or torch.equal(win.flip(2), out[i]):
                    found = True
        assert found


def test_model_shapes_and_parameters() -> None:
    m = build_model(10)
    assert count_parameters(m) == 412_954
    x = torch.rand(4, 3, 32, 32)
    assert m(x).shape == (4, 10)
    assert m.features(x).shape == (4, m.feature_dim)
    assert m.last_conv(x).shape == (4, 128, 4, 4)


def test_channel_mask_zeroes_features() -> None:
    m = build_model(3).eval()
    m.channel_mask[:] = 0
    x = torch.rand(2, 3, 32, 32)
    assert torch.count_nonzero(m.features(x)) == 0


def test_normalization_inside_model() -> None:
    m = build_model(3, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    x = torch.full((1, 3, 2, 2), 1.0)
    assert torch.allclose(m.norm(x), torch.ones_like(x))


def test_batched_inference_matches_single_batch() -> None:
    m = build_model(5).eval()
    x = torch.rand(10, 3, 32, 32)
    p_all = predict(m, x, batch_size=100)
    p_small = predict(m, x, batch_size=3)
    assert torch.equal(p_all, p_small)
    pr = predict_proba(m, x, batch_size=4)
    torch.testing.assert_close(pr.sum(1), torch.ones(10))
    f = extract_features(m, x, batch_size=4)
    torch.testing.assert_close(f, m.features(x), atol=1e-5, rtol=1e-5)
    assert predict(m, x[:0]).numel() == 0
    assert extract_features(m, x[:0]).shape == (0, m.feature_dim)
    assert predict_proba(m, x[:0]).numel() == 0
