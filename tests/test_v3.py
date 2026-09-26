"""v3 core: address, memory, medium-path statistics, routers, policies, the API guards.

All CPU, synthetic, seconds. The anchor re-runs (GPU, stored JSONs) live in
`tests/test_v3_anchors.py`.
"""

import pytest
import torch

from pal_moe.address import ExactCosineIndex
from pal_moe.api import Batch, Example, GuardConfig, GuardViolation, PalMoE
from pal_moe.arch import RidgeReadout
from pal_moe.core.hashing import digest
from pal_moe.edit import LinearStats, one_hot
from pal_moe.experts import PolicyGateError, by_arrival, by_confusion, check_gate
from pal_moe.memory import FastMemory
from pal_moe.router import RidgeClassRouter, RouterPurityError, assert_pure

D, C = 16, 6


def _blobs(seed=0, n_per=40, classes=range(C), spread=0.3):
    g = torch.Generator().manual_seed(seed)
    centers = torch.randn(C, D, generator=torch.Generator().manual_seed(123)) * 2
    xs, ys = [], []
    for c in classes:
        xs.append(centers[c] + spread * torch.randn(n_per, D, generator=g))
        ys.append(torch.full((n_per,), c, dtype=torch.long))
    return torch.cat(xs), torch.cat(ys)


def _batches():
    return [_blobs(seed=s, classes=[2 * s, 2 * s + 1]) for s in range(3)]


# -- address / memory ------------------------------------------------------------


def test_index_add_remove_is_bitwise_and_parameter_free():
    idx = ExactCosineIndex(D)
    k = torch.randn(3, D)
    idx.add("a", k[0])
    idx.add("b", k[1])
    before = digest(idx.ids, idx.keys)
    idx.add("c", k[2])
    idx.remove("c")
    assert digest(idx.ids, idx.keys) == before
    res = idx.search(k[1:2], k=2)
    assert res.ids[0][0] == "b"
    assert idx.parameter_count() == 0
    with pytest.raises(KeyError):
        idx.add("a", k[0])


def test_fast_memory_delete_restores_state_digest():
    mem = FastMemory(D)
    mem.write("x", torch.randn(D), 3)
    before = mem.state_digest()
    mem.write("y", torch.randn(D), 4)
    mem.delete("y")
    assert mem.state_digest() == before
    hit = mem.lookup(mem.entry("x")[0].unsqueeze(0), threshold=0.999)[0]
    assert hit and hit[0].value == 3


# -- medium path -------------------------------------------------------------------


def test_linear_stats_permuted_arrival_is_close_not_forced_equal():
    """Two accumulators fed in different orders: a real fp comparison, not a design."""
    a, b = LinearStats(D, C), LinearStats(D, C)
    batches = _batches()
    for i, (x, y) in enumerate(batches):
        a.add(a.contribution(f"e{i}", x, one_hot(y, C)))
    for i, (x, y) in reversed(list(enumerate(batches))):
        b.add(b.contribution(f"e{i}", x, one_hot(y, C)))
    assert float((a.solve() - b.solve()).abs().max()) <= 1e-10
    rep = a.order_report(batches[0][0], tol=1e-10)
    assert rep["pass"] and rep["argmax_identical"]


def test_linear_stats_order_guard_detects_a_broken_accumulator():
    s = LinearStats(D, C)
    for i, (x, y) in enumerate(_batches()):
        s.add(s.contribution(f"e{i}", x, one_hot(y, C)))
    s.A += 1e-3 * torch.eye(s.d1, dtype=s.A.dtype)  # corrupt the accumulator
    s._W = None
    assert not s.order_report(tol=1e-10)["pass"]
    assert not s.recompute_report(tol=1e-10)["pass"]


def test_linear_stats_forget_is_a_downdate_close_to_recompute():
    s = LinearStats(D, C)
    batches = _batches()
    for i, (x, y) in enumerate(batches[:2]):
        s.add(s.contribution(f"e{i}", x, one_hot(y, C)))
    W2 = s.solve().clone()
    s.add(s.contribution("e2", batches[2][0], one_hot(batches[2][1], C)))
    s.remove("e2")  # subtraction, not a re-sum
    rep = s.recompute_report(tol=1e-10)
    assert rep["pass"] and float((s.solve() - W2).abs().max()) <= 1e-10
    once = LinearStats(D, C)
    x = torch.cat([b[0] for b in batches[:2]])
    y = torch.cat([b[1] for b in batches[:2]])
    once.add(once.contribution("all", x, one_hot(y, C)))
    assert float((once.solve() - W2).abs().max()) < 1e-10
    s.remove("e0")
    s.remove("e1")  # back to zero edits: the prior, bitwise
    assert (
        torch.equal(s.A, s.A0)
        and not s.B.any()
        and s.recompute_report()["bitwise_prior"]
    )


def test_linear_stats_memory_does_not_grow_with_d_squared_per_small_edit():
    s = LinearStats(D, C)
    for i in range(50):  # 50 single-sample edits
        s.add(
            s.contribution(
                f"f{i}", torch.randn(1, D), one_hot(torch.tensor([i % C]), C)
            )
        )
    st = s.storage_bytes()
    assert st["per_edit_total"] == 50 * ((D + 1) + C) * 8  # factors, not d'^2 matrices
    assert st["accumulators"] == ((D + 1) ** 2 + (D + 1) * C) * 8


def test_linear_stats_float64_agrees_with_float32_ridge_readout():
    s, r = LinearStats(D, C, ridge=1.0), RidgeReadout(D, C, ridge=1.0)
    for i, (x, y) in enumerate(_batches()):
        s.add(s.contribution(f"e{i}", x, one_hot(y, C)))
        r.fit(x, y)
    x_test, _ = _blobs(seed=9)
    assert torch.equal(s.predict(x_test).argmax(-1), r.predict(x_test).argmax(-1))
    assert float((s.solve().float() - r.W).abs().max()) < 1e-4


# -- routers ----------------------------------------------------------------------


def test_ridge_class_router_is_the_etid2_rule():
    s = LinearStats(D, C)
    owner = {}
    for t, (x, y) in enumerate(_batches()):
        s.add(s.contribution(f"e{t}", x, one_hot(y, C)))
        owner.update({int(c): t for c in y.unique()})
    router = RidgeClassRouter(s, owner)
    z, _ = _blobs(seed=5)
    logits = s.predict(z)
    cls2task = torch.tensor([owner[c] for c in range(C)])
    ref = torch.full((z.size(0), 3), -1e9, dtype=logits.dtype).scatter_reduce(
        1, cls2task.expand_as(logits), logits, "amax"
    )
    assert torch.equal(router.scores(z), ref)
    assert assert_pure(router) == 0


def test_router_purity_rejects_a_learned_router():
    class Learned:
        name = "learned"

        def __init__(self):
            self.w = torch.nn.Linear(D, 3)

        def parameter_count(self):
            return 0  # lies; the guard inspects the object

    with pytest.raises(RouterPurityError):
        assert_pure(Learned())


# -- policies ---------------------------------------------------------------------


def test_by_arrival_is_first_introduction():
    assert by_arrival([[0, 1], [1, 2], [3]]) == [[0, 1], [2], [3]]


def test_by_confusion_is_gated_and_recovers_blocks():
    with pytest.raises(PolicyGateError):
        check_gate("by_confusion", None)
    check_gate("by_confusion", "docs/P2_BOUND_PREREG.md")
    conf = torch.eye(6) * 50
    for block in ([0, 3], [1, 4], [2, 5]):
        conf[block[0], block[1]] = conf[block[1], block[0]] = 10
    groups = by_confusion(conf, 3)
    assert sorted(sorted(g) for g in groups) == [[0, 3], [1, 4], [2, 5]]


# -- API --------------------------------------------------------------------------


def _model(**kw):
    canary, _ = _blobs(seed=77, n_per=5)
    return PalMoE(D, C, canary=canary, guards=GuardConfig(**kw.pop("guards", {})), **kw)


def test_api_write_predict_forget_predict_is_bitwise():
    m = _model()
    for t, (x, y) in enumerate(_batches()):
        rec = m.write(Batch(x, y, task=t))
        assert rec.reversibility_report["pass"] and rec.order_report["argmax_identical"]
    x_test, _ = _blobs(seed=3, n_per=10)
    s0, p0 = m.state(), m.predict(x_test)
    probe = x_test[0]
    target = (int(p0.labels[0]) + 1) % C
    rec = m.write(Example(probe, target))
    p1 = m.predict(x_test)
    assert int(p1.labels[0]) == target and p1.source[0] == "memory"
    assert rec.locality_report["pass"] and rec.locality_report["argmax_flips"] == 0
    m.forget(rec.id)
    p2 = m.predict(x_test)
    assert m.state() == s0
    assert torch.equal(p2.labels, p0.labels) and torch.equal(p2.logits, p0.logits)


def test_api_medium_forget_is_a_measured_downdate():
    m = _model()
    b = _batches()
    m.write(Batch(*b[0], task=0))
    m.write(Batch(*b[1], task=1))
    ref_scores, ref_labels = m._canary_outputs()
    rec = m.write(Batch(*b[2], task=2))
    assert rec.reversibility_report["pass"]
    assert rec.reversibility_report["undo_max_abs_dW"] <= 1e-10
    rep = m.forget(rec.id)
    assert rep["state_seen_before"] and rep["pass"] and rep["canary_argmax_identical"]
    assert rep["downdate"]["max_abs_dW_recompute"] <= 1e-10
    scores, labels = m._canary_outputs()
    assert (
        torch.equal(labels, ref_labels)
        and float((scores - ref_scores).abs().max()) <= 1e-8
    )


def test_api_forget_everything_is_bitwise():
    m = _model()
    s0, (sc0, lb0) = m.state(), m._canary_outputs()
    recs = [m.write(Batch(x, y, task=t)) for t, (x, y) in enumerate(_batches())]
    for r in reversed(recs):
        m.forget(r.id)
    sc, lb = m._canary_outputs()
    assert m.state() == s0 and torch.equal(sc, sc0) and torch.equal(lb, lb0)


def test_api_permuted_batches_give_close_weights():
    b = _batches()
    m1, m2 = _model(), _model()
    for t in (0, 1, 2):
        m1.write(Batch(*b[t], task=t))
    for t in (2, 0, 1):
        m2.write(Batch(*b[t], task=t))
    assert float((m1.stats.solve() - m2.stats.solve()).abs().max()) <= 1e-10
    assert all(r.order_report["pass"] for r in m1.log)


def test_api_locality_violation_rolls_back():
    m = _model(guards={"epsilon_fast": 0.0}, memory_threshold=-1.0)  # every query hits
    for t, (x, y) in enumerate(_batches()):
        m.write(Batch(x, y, task=t))
    s0 = m.state()
    with pytest.raises(GuardViolation):
        m.write(Example(torch.randn(D), 0))
    assert m.state() == s0 and len(m.memory) == 0


def test_api_consolidate_by_arrival_and_forget_it():
    m = _model(recipe={"epochs": 1, "batch_size": 32, "seed": 0})
    for t, (x, y) in enumerate(_batches()):
        m.write(Batch(x, y, task=t))
    s0, d0 = m.state(), m._outputs_digest()[0]
    rep = m.consolidate("by_arrival")
    assert rep.record.reversibility_report["bitwise"]  # the bank object is swapped back
    assert rep.groups == [[0, 1], [2, 3], [4, 5]] and rep.experts_added == 3
    assert rep.record.reversibility_report["pass"]
    assert all(not p.requires_grad for e in m.bank.experts for p in e.parameters())
    x_test, _ = _blobs(seed=4, n_per=5)
    p = m.predict(x_test)
    assert p.expert_ids is not None and set(p.source) == {"experts"}
    with pytest.raises(RuntimeError):
        m.forget(
            m.log[0].id
        )  # consolidated batches need the consolidation forgotten first
    m.forget(rep.record.id)
    assert m.state() == s0 and m._outputs_digest()[0] == d0
    with pytest.raises(PolicyGateError):
        m.consolidate("by_confusion")


def test_api_prototype_router_from_log():
    m = _model(router="prototype")
    for t, (x, y) in enumerate(_batches()):
        m.write(Batch(x, y, task=t))
    x_test, y_test = _blobs(seed=8, n_per=5)
    p = m.predict(x_test, k=2)
    owner = m.class_owner()
    assert (
        p.expert_ids[:, 0] == torch.tensor([owner[int(c)] for c in y_test])
    ).float().mean() > 0.9


def test_cross_fitted_confusion_uses_held_out_predictions():
    from pal_moe.experts.policies import cross_fitted_confusion

    x, y = _blobs(seed=1, spread=2.0)  # overlapping classes: some confusion
    conf = cross_fitted_confusion(x, y, C, folds=5, seed=0)
    assert int(conf.sum()) == y.numel() and conf.shape == (C, C)
    assert torch.equal(conf, cross_fitted_confusion(x, y, C, folds=5, seed=0))


def test_api_consolidate_by_partition_takes_explicit_groups():
    m = _model(recipe={"epochs": 1, "batch_size": 32, "seed": 0})
    for t, (x, y) in enumerate(_batches()):
        m.write(Batch(x, y, task=t))
    with pytest.raises(ValueError):
        m.consolidate("by_partition", groups=[[0, 2], [1, 3]])  # misses 4, 5
    rep = m.consolidate("by_partition", groups=[[0, 2], [1, 3], [4, 5]])
    assert rep.groups == [[0, 2], [1, 3], [4, 5]] and m.class_owner()[2] == 0
