from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from hypothesis import given, settings
from hypothesis import strategies as st

from backdoor_defense_lab.attacks import (
    ATTACKS,
    BadNets,
    Blend,
    LabelConsistent,
    PatchTrigger,
    WaNet,
    checkerboard,
    make_attack,
    pgd_untargeted,
    poison_training_set,
    triggered_test_set,
    warping_grid,
)
from backdoor_defense_lab.data import ImageDataset
from backdoor_defense_lab.models import build_model


def test_checkerboard_pattern() -> None:
    p = checkerboard(3)
    assert p.shape == (3, 3, 3)
    assert p[0].tolist() == [[1, 0, 1], [0, 1, 0], [1, 0, 1]]


def test_badnets_changes_only_the_patch(images: torch.Tensor) -> None:
    a = BadNets(target=0)
    out = a.apply(images)
    diff = (out != images).any(1)
    assert not diff[:, :28, :].any() and not diff[:, :, :28].any()
    assert torch.equal(out[:, :, 28:31, 28:31], checkerboard(3).expand(8, 3, 3, 3))
    # the input is not modified in place
    assert not torch.equal(out, images)


def test_badnets_is_idempotent(images: torch.Tensor) -> None:
    a = BadNets()
    once = a.apply(images)
    assert torch.allclose(a.apply(once), once)


def test_patch_trigger_validation() -> None:
    with pytest.raises(ValueError):
        PatchTrigger(torch.ones(3, 3), [(0, 0)])
    with pytest.raises(ValueError):
        PatchTrigger(checkerboard(3), [(30, 30)])
    with pytest.raises(ValueError):
        PatchTrigger(checkerboard(3), [(0, 0)], amplitude=0)


def test_patch_trigger_amplitude_mixes(images: torch.Tensor) -> None:
    t = PatchTrigger(torch.ones(3, 2, 2), [(0, 0)], amplitude=0.5)
    out = t(images)
    torch.testing.assert_close(out[:, :, :2, :2], 0.5 * images[:, :, :2, :2] + 0.5)


@settings(max_examples=25, deadline=None)
@given(st.floats(0.01, 0.99))
def test_blend_is_convex_combination(alpha: float) -> None:
    x = torch.rand(2, 3, 32, 32)
    a = Blend(alpha=alpha)
    out = a.apply(x)
    torch.testing.assert_close(out, (1 - alpha) * x + alpha * a.key)
    assert out.min() >= 0 and out.max() <= 1


def test_blend_key_is_reproducible_and_validated() -> None:
    assert torch.equal(Blend(key_seed=3).key, Blend(key_seed=3).key)
    assert not torch.equal(Blend(key_seed=3).key, Blend(key_seed=4).key)
    with pytest.raises(ValueError):
        Blend(alpha=1.0)


def test_attack_validation() -> None:
    with pytest.raises(ValueError):
        BadNets(rate=0)
    with pytest.raises(ValueError):
        BadNets(target=-1)
    with pytest.raises(ValueError):
        make_attack("nope")
    with pytest.raises(ValueError):
        WaNet(k=1)
    with pytest.raises(ValueError):
        WaNet(cross_ratio=-1)


def test_wanet_grid_is_a_small_smooth_warp() -> None:
    g = warping_grid(32, 4, 0.5, seed=0)
    assert g.shape == (1, 32, 32, 2)
    lin = torch.linspace(-1, 1, 32)
    gy, gx = torch.meshgrid(lin, lin, indexing="ij")
    disp = g[0] - torch.stack((gx, gy), 2)
    # displacement in pixels: s * P / H in grid units = s * P / 2 pixels; mean |P| = 1
    pixels = disp.abs() * 31 / 2
    assert 0.05 < float(pixels.mean()) < 0.5
    assert g.abs().max() <= 1


def test_wanet_zero_strength_is_identity(images: torch.Tensor) -> None:
    a = WaNet(strength=0.0)
    torch.testing.assert_close(a.apply(images), images, atol=1e-5, rtol=0)


def test_wanet_is_imperceptible_but_not_identity(images: torch.Tensor) -> None:
    a = WaNet()
    out = a.apply(images)
    d = (out - images).abs()
    assert float(d.mean()) > 1e-3
    assert out.min() >= 0 and out.max() <= 1


def test_wanet_noise_mode_is_random_and_different(images: torch.Tensor) -> None:
    a = WaNet()
    g = torch.Generator().manual_seed(0)
    n1 = a.training_noise(images, g)
    n2 = a.training_noise(images, g)
    assert not torch.allclose(n1, n2)
    assert not torch.allclose(n1, a.apply(images))


def test_wanet_noise_indices_disjoint_from_poisoned() -> None:
    a = WaNet(rate=0.1, cross_ratio=2)
    labels = np.repeat(np.arange(4), 50)
    rng = np.random.default_rng(0)
    p = a.select(labels, rng)
    n = a.noise_indices(labels, p, rng)
    assert len(n) == 2 * len(p)
    assert not set(p) & set(n)


def test_dirty_label_selection_avoids_target() -> None:
    labels = np.repeat(np.arange(5), 100)
    idx = BadNets(target=2, rate=0.1).select(labels, np.random.default_rng(0))
    assert len(idx) == 50
    assert (labels[idx] != 2).all()
    assert np.all(np.diff(idx) > 0)


def test_clean_label_selection_uses_target_only() -> None:
    labels = np.repeat(np.arange(5), 100)
    idx = LabelConsistent(target=2, rate=0.5).select(labels, np.random.default_rng(0))
    assert len(idx) == 50
    assert (labels[idx] == 2).all()


def test_poison_training_set_dirty_label(toy: ImageDataset) -> None:
    ps = poison_training_set(toy, BadNets(target=1, rate=0.1), seed=0)
    assert ps.n_poisoned == 60
    assert (ps.y[ps.poisoned] == 1).all()
    assert (ps.original_y[ps.poisoned] != 1).all()
    np.testing.assert_array_equal(ps.x[~ps.poisoned], toy.x_train[~ps.poisoned])
    assert (ps.x[ps.poisoned][:, 28:31, 28:31] != toy.x_train[ps.poisoned][:, 28:31, 28:31]).any()
    assert not ps.noise.any()
    # the dataset itself is untouched
    assert not np.shares_memory(ps.x, toy.x_train)


def test_poison_training_set_is_deterministic(toy: ImageDataset) -> None:
    a = poison_training_set(toy, Blend(rate=0.05), seed=3)
    b = poison_training_set(toy, Blend(rate=0.05), seed=3)
    np.testing.assert_array_equal(a.poisoned, b.poisoned)
    np.testing.assert_array_equal(a.x, b.x)


def test_poison_training_set_wanet_has_noise(toy: ImageDataset) -> None:
    ps = poison_training_set(toy, WaNet(target=0, rate=0.1), seed=0)
    assert ps.noise.sum() == 2 * ps.n_poisoned
    assert not (ps.noise & ps.poisoned).any()


def test_poison_training_set_clean_label_keeps_labels(toy: ImageDataset) -> None:
    ps = poison_training_set(toy, LabelConsistent(target=2, rate=0.5), seed=0)
    np.testing.assert_array_equal(ps.y, toy.y_train)
    assert (toy.y_train[ps.poisoned] == 2).all()


def test_triggered_test_set(toy: ImageDataset) -> None:
    x, y = triggered_test_set(toy, BadNets(target=0))
    assert (y != 0).all()
    assert len(x) == int((toy.y_test != 0).sum())
    assert x.dtype == np.uint8


def test_pgd_stays_in_ball_and_raises_loss() -> None:
    torch.manual_seed(0)
    model = build_model(4).eval()
    x = torch.rand(16, 3, 32, 32)
    y = torch.randint(0, 4, (16,))
    eps = 8 / 255
    adv = pgd_untargeted(model, x, y, eps, 2 / 255, 5, torch.Generator().manual_seed(0))
    assert float((adv - x).abs().max()) <= eps + 1e-6
    assert adv.min() >= 0 and adv.max() <= 1
    with torch.no_grad():
        assert F.cross_entropy(model(adv), y) > F.cross_entropy(model(x), y)


def test_label_consistent_poison_uses_surrogate(images: torch.Tensor) -> None:
    a = LabelConsistent(target=0, eps=8 / 255, steps=2)
    no_pgd = a.poison(images)
    torch.testing.assert_close(no_pgd, a.poison_trigger(images))
    a.set_surrogate(build_model(4).eval())
    with_pgd = a.poison(images)
    assert not torch.allclose(with_pgd, no_pgd)
    corner = with_pgd[:, :, 1:4, 1:4]
    assert torch.equal(corner, checkerboard(3).expand(8, 3, 3, 3))
    assert a.config()["surrogate"] is True


def test_label_consistent_trigger_has_four_corners(images: torch.Tensor) -> None:
    out = LabelConsistent().apply(images)
    for r, c in [(1, 1), (1, 28), (28, 1), (28, 28)]:
        assert torch.equal(out[:, :, r : r + 3, c : c + 3], checkerboard(3).expand(8, 3, 3, 3))


@pytest.mark.parametrize("name", ATTACKS)
def test_registry_builds_every_attack(name: str, images: torch.Tensor) -> None:
    a = make_attack(name, target=1)
    assert a.name == name
    out = a.apply(images)
    assert out.shape == images.shape
    assert out.min() >= 0 and out.max() <= 1
    cfg = a.config()
    assert cfg["name"] == name and cfg["target"] == 1
