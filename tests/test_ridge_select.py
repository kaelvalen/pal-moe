"""HEADROOM amendment 3: the lambda choice must not depend on the feature scale."""

import torch

from pal_moe.eval.ridge_select import fit_select


def _data(n=600, d=32, C=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    centers = torch.randn(C, d, generator=g) * 2
    y = torch.randint(0, C, (n,), generator=g)
    return centers[y] + torch.randn(n, d, generator=g), y


def test_power_of_two_rescaling_is_bitwise():
    z, y = _data()
    m1, i1 = fit_select(z, y, 5, seed=0)
    m2, i2 = fit_select(z * 1024.0, y, 5, seed=0)  # exact in floating point
    assert i1["c"] == i2["c"]
    assert torch.equal(m1.logits(z), m2.logits(z * 1024.0))


def test_times_100_keeps_the_choice_and_every_prediction():
    z, y = _data()
    rp = torch.randn(
        32, 400, generator=torch.Generator().manual_seed(1), dtype=torch.float64
    )
    for proj in (None, rp):
        m1, i1 = fit_select(z, y, 5, seed=0, rp=proj)
        m2, i2 = fit_select(z * 100.0, y, 5, seed=0, rp=proj)
        assert i1["c"] == i2["c"] and i1["heldout_acc"] == i2["heldout_acc"]
        assert torch.equal(m1.logits(z).argmax(-1), m2.logits(z * 100.0).argmax(-1))


def test_flat_curve_takes_the_largest_c_and_an_edge_is_flagged():
    z, y = _data()
    y = torch.zeros_like(y)  # constant labels: every c scores the same
    _, info = fit_select(z, y, 5, seed=0)
    assert info["extended"] == "up" and not info["converged"]
    assert info["c"] == max(float(k) for k in info["heldout_acc"])
