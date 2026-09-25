"""
v3 API smoke: write -> predict -> forget -> predict, with reversibility asserted bitwise.

Runs on the cached ViT-B/16 CIFAR-100 features (`coherent` construction) when the
cache is present, else on synthetic blobs. No number here is a result; the script
exits non-zero if any guard fails.

    .venv/bin/python experiments/v3_api_smoke.py [--device cuda] [--synthetic]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from pal_moe.api import Batch, Example, PalMoE  # noqa: E402

CACHE = "results/feature_cache/cifar100_vit_b16/feature_cache.pt"


def data(synthetic: bool):
    if not synthetic and Path(CACHE).exists():
        from pal_moe.core.constructions import build_construction
        from pal_moe.core.features import load_tasks

        _, base = load_tasks(CACHE)
        tasks = build_construction(base, "coherent")[:5]
        return (
            [(t["splits"]["train"], t["splits"]["test"], t["task_id"]) for t in tasks],
            100,
            "vit_b16_cache",
        )
    g = torch.Generator().manual_seed(0)
    centers = torch.randn(10, 32, generator=g) * 2
    out = []
    for t in range(5):
        cls = torch.tensor([2 * t, 2 * t + 1]).repeat_interleave(50)
        x = centers[cls] + 0.5 * torch.randn(cls.numel(), 32, generator=g)
        out.append(((x[::2], cls[::2]), (x[1::2], cls[1::2]), t))
    return out, 10, "synthetic"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu")
    p.add_argument("--synthetic", action="store_true")
    args = p.parse_args()
    tasks, C, source = data(args.synthetic)
    dim = tasks[0][0][0].size(1)
    # The canary must be disjoint from what the smoke writes: the training split's
    # first 20 rows per task (the test sample written below is never among them).
    canary = torch.cat([tr[0][:20] for tr, _, _ in tasks])
    model = PalMoE(dim, C, router="ridge_class", canary=canary, device=args.device)
    for tr, _, t in tasks:
        rec = model.write(Batch(*tr, task=t))
        assert rec.reversibility_report["pass"], rec.reversibility_report
        assert rec.order_report["argmax_identical"], rec.order_report

    x_test = torch.cat([te[0] for _, te, _ in tasks])
    y_test = torch.cat([te[1] for _, te, _ in tasks])
    s0 = model.state()
    p0 = model.predict(x_test)
    acc0 = float((p0.labels.cpu() == y_test).float().mean())

    # FAST: one sample the model gets wrong, written with its true label.
    wrong = (p0.labels.cpu() != y_test).nonzero().flatten()
    i = int(wrong[0]) if wrong.numel() else 0
    rec = model.write(Example(x_test[i], int(y_test[i])))
    p1 = model.predict(x_test)
    assert int(p1.labels[i]) == int(y_test[i]) and p1.source[i] == "memory"
    changed = int((p1.labels != p0.labels).sum())
    print(
        f"[fast] wrote {rec.id}: sample {i} now correct; {changed} test predictions changed; "
        f"canary locality {rec.locality_report}"
    )

    forget = model.forget(rec.id)
    p2 = model.predict(x_test)
    assert forget["pass"] and model.state() == s0, forget
    assert torch.equal(p2.labels, p0.labels) and torch.equal(p2.logits, p0.logits)

    # MEDIUM: forget the last batch, then write it back.
    last = model.log[-1]
    model.forget(last.id)
    tr, _, t = tasks[-1]
    model.write(Batch(*tr, task=t))
    p3 = model.predict(x_test)
    assert torch.equal(p3.logits, p0.logits), "medium forget + re-write is not bitwise"

    print(
        f"[smoke] source={source} tasks={len(tasks)} acc={acc0 * 100:.2f} "
        f"state={model.state().digest[:16]} reversibility=bitwise OK"
    )


if __name__ == "__main__":
    main()
