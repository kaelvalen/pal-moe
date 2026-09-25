"""
E-TID2 through the v3 API (phase 2 of the restructure; no new claim).

The same cells as `e_tid2_ridge_router.py`, but every step goes through `PalMoE`:

    write(Batch(task t))  x 20    MEDIUM path: float64 additive ridge statistics
    consolidate("by_arrival")     SLOW path: the L3 bank, via the moved ladder code
    predict(...)                  ridge_class routing -> expert -> readout

The bank and the prototype arm must reproduce the stored cells (same code, same
seed). The ridge arms (`ridge_routed`, `ridge_masked`, `ridge_alone`, ridge
coverage) now solve in float64 where the stored run used float32; their deltas are
reported, not tuned. Run through `experiments/v3_anchors.py --only etid2 --api`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from pal_moe.api import Batch, GuardConfig, PalMoE  # noqa: E402
from pal_moe.core.constructions import build_construction  # noqa: E402
from pal_moe.edit import LinearStats, one_hot  # noqa: E402
from pal_moe.router import PrototypeTaskRouter  # noqa: E402


def canary_set(tasks, device, per_task: int = 10):
    """A fixed canary: the first `per_task` test features of every task."""
    return torch.cat([t["splits"]["test"][0][:per_task] for t in tasks]).to(device)


@torch.no_grad()
def run_cell(regime, seed, args, device, base):
    tasks = build_construction(base, regime)
    T, dim = len(tasks), int(tasks[0]["splits"]["train"][0].size(1))
    C = sum(len(t["classes"]) for t in tasks)
    model = PalMoE(
        dim,
        C,
        router="ridge_class",
        ridge=args.ridge,
        canary=canary_set(tasks, device),
        guards=GuardConfig(epsilon_medium=None, trial_reversibility=True),
        recipe={
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "lambda_func": args.lambda_func,
            "rank": 8,
            "protos": 1,
            "top_k": 1,
            "seed": seed,
        },
        device=device,
    )
    records = [
        model.write(Batch(*t["splits"]["train"], task=t["task_id"])) for t in tasks
    ]
    report = model.consolidate("by_arrival")
    bank = model.bank

    once = LinearStats(dim, C, ridge=args.ridge, device=device)
    xs = torch.cat([t["splits"]["train"][0] for t in tasks]).to(device)
    ys = torch.cat([t["splits"]["train"][1] for t in tasks]).to(device)
    once.add(once.contribution("all", xs, one_hot(ys, C)))
    owner_map = model.class_owner()
    task_mask = torch.zeros(T, C, dtype=torch.bool, device=device)
    for c, t in owner_map.items():
        task_mask[t, c] = True
    proto_router = PrototypeTaskRouter(bank.router)

    arms = ("proto", "ridge_routed", "ridge_masked", "ridge_alone", "oracle")
    per_task = {a: [] for a in arms}
    cov = {"proto": [0, 0], "ridge": [0, 0]}
    n_total, g3_mismatch, g1, g2 = 0, 0, [], []
    for t in tasks:
        z, y = [v.to(device) for v in t["splits"]["test"]]
        owner = t["task_id"]
        proto_ids = proto_router.top_k(z, k=3)[0]
        p = model.predict(z, k=3, use_memory=False)
        ridge_ids = p.expert_ids
        g3_mismatch += int(
            (model.stats.predict(z).argmax(-1) != once.predict(z).argmax(-1)).sum()
        )
        for name, ids in (("proto", proto_ids), ("ridge", ridge_ids)):
            cov[name][0] += int((ids[:, 0] == owner).sum())
            cov[name][1] += int((ids == owner).any(-1).sum())
        n_total += z.size(0)
        pred = {
            "proto": model.predict(
                z, oracle_expert=proto_ids[:, 0], use_memory=False
            ).labels,
            "ridge_routed": p.labels,
            "ridge_masked": p.logits.masked_fill(
                ~task_mask[ridge_ids[:, 0]], float("-inf")
            ).argmax(-1),
            "ridge_alone": p.medium_labels,
            "oracle": model.predict(
                z, oracle_expert=torch.full_like(y, owner), use_memory=False
            ).labels,
        }
        for a in arms:
            per_task[a].append(float((pred[a] == y).float().mean()))
        g1.append(
            abs(
                per_task["proto"][-1]
                - bank.evaluate_task(t, oracle=False, task_id=owner)
            )
        )
        g2.append(
            abs(
                per_task["oracle"][-1]
                - bank.evaluate_task(t, oracle=True, task_id=owner)
            )
        )

    out = {a: sum(v) / len(v) for a, v in per_task.items()}
    out.update(
        {
            "proto_C@1": cov["proto"][0] / n_total,
            "proto_C@3": cov["proto"][1] / n_total,
            "ridge_C@1": cov["ridge"][0] / n_total,
            "ridge_C@3": cov["ridge"][1] / n_total,
            "guards": {
                "G1_max_abs": max(g1),
                "G2_max_abs": max(g2),
                "G3_argmax_mismatch": g3_mismatch,
                "G3_max_abs_dW": float(
                    (model.stats.solve() - once.solve()).abs().max()
                ),
            },
            "v3_guards": {
                "writes_reversible": all(
                    r.reversibility_report["pass"] for r in records
                ),
                "writes_order_argmax_identical": all(
                    r.order_report["argmax_identical"] for r in records
                ),
                "max_running_vs_canonical_dW": max(
                    r.order_report["max_abs_dW"] for r in records
                ),
                "consolidation_reversible": report.record.reversibility_report["pass"],
                "router_trainable_params": 0
                if report.record.purity_report["pass"]
                else None,
                "experts": report.experts_added,
                "frozen_parameters": report.frozen_parameters,
            },
        }
    )
    return out
