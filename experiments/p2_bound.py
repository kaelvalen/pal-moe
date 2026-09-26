"""
P2-BOUND: the per-sample decomposition of E-TID2's P2 (Part A) and the consolidation
policy test (Part B).

    docs/P2_BOUND_PREREG.md (with amendments 1 and 2)

Part A re-runs the 12 E-TID2 cells through `e_tid2_ridge_router.py`'s float32 code
path and records, per test sample, r (ridge class right), tau (ridge-routed task
right) and s (ridge_routed system right).

Part B runs through the v3 API on identical written batches and an identical router:
A0 by_arrival, A1 by_confusion (5-fold cross-fitted confusion, hindsight-offline),
A2 by_partition = the 20 CIFAR-100 superclasses. One consolidation per arm, each
forgotten before the next.

Usage:
    LD_LIBRARY_PATH=/run/opengl-driver/lib \
        .venv/bin/python experiments/p2_bound.py --device cuda
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402

import e_tid2_ridge_router as e2r  # noqa: E402
import s11_confirmatory as s11  # noqa: E402
from pal_moe.api import Batch, GuardConfig, PalMoE  # noqa: E402
from pal_moe.core.constructions import (  # noqa: E402
    args_data_dir,
    build_construction,
    superclass_of,
)
from pal_moe.core.features import load_tasks  # noqa: E402
from pal_moe.eval.stats import paired_stats, tost, westfall_young  # noqa: E402

PREREG = "docs/P2_BOUND_PREREG.md"
ETID2_JSON = "results/e_tid2/e_tid2_ridge_router.json"
REGIMES = ("coherent", "dispersed")
ARMS5 = ("proto", "ridge_routed", "ridge_masked", "ridge_alone", "oracle")
REPRO_BAND = 1e-6
CEILING_S = 2 * 3600


def git_rev() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def decomposition(r, tau, s) -> dict:
    """Counts -> the P2 identity terms. r, tau, s: bool tensors over all test samples."""
    N = r.numel()
    nr, ntau = ~r, ~tau
    c = {
        "N": N,
        "r": int(r.sum()),
        "s": int(s.sum()),
        "r_not_tau": int((r & ntau).sum()),  # must be 0: a right class implies its task
        "nr_tau": int((nr & tau).sum()),
        "nr_ntau": int((nr & ntau).sum()),
        "s_nr_tau": int((s & nr & tau).sum()),
        "s_nr_ntau": int((s & nr & ntau).sum()),
        "ns_r": int((~s & r).sum()),
    }
    m = c["nr_tau"] / N
    rho = c["s_nr_tau"] / c["nr_tau"] if c["nr_tau"] else float("nan")
    rho_p = c["s_nr_ntau"] / c["nr_ntau"] if c["nr_ntau"] else 0.0
    p_r = c["r"] / N
    beta = c["ns_r"] / c["r"] if c["r"] else 0.0
    p2 = (c["s"] - c["r"]) / N
    ident = m * (rho if c["nr_tau"] else 0.0) + (c["nr_ntau"] / N) * rho_p - p_r * beta
    return {
        "counts": c,
        "m": m,
        "rho": rho,
        "rho_prime": rho_p,
        "beta": beta,
        "p_r": p_r,
        "rescue": m * rho if c["nr_tau"] else 0.0,
        "break": p_r * beta,
        "P2_pooled": p2,
        "P2_max": m + (c["nr_ntau"] / N) * rho_p,
        "identity_abs_error": abs(ident - p2),
    }


# -- Part A -----------------------------------------------------------------------------


@torch.no_grad()
def part_a_cell(regime, seed, args, device, base):
    tasks = build_construction(base, regime)
    T, dim = len(tasks), int(tasks[0]["splits"]["train"][0].size(1))
    C = sum(len(t["classes"]) for t in tasks)
    cell = {"seed": seed, "rank": 8, "protos": 1, "top_k": 1}
    with torch.enable_grad():
        model = s11.train_model("L3_per_task", tasks, cell, args, device)
    cont, _ = e2r.fit_ridge(tasks, dim, C, device, args.ridge)
    cls2task = torch.zeros(C, dtype=torch.long, device=device)
    task_mask = torch.zeros(T, C, dtype=torch.bool, device=device)
    for t in tasks:
        idx = torch.tensor(t["classes"], device=device)
        cls2task[idx] = t["task_id"]
        task_mask[t["task_id"], idx] = True
    per_task = {a: [] for a in ARMS5}
    R, TAU, S = [], [], []
    for t in tasks:
        z, y = [v.to(device) for v in t["splits"]["test"]]
        owner = t["task_id"]
        proto_ids = model.router.top_k(z, k=3)[0]
        rlog = cont.predict(z)
        rtask = torch.full((z.size(0), T), -1e9, device=device).scatter_reduce(
            1, cls2task.expand_as(rlog), rlog, "amax"
        )
        ridge_ids = rtask.topk(3, dim=-1).indices
        pred = {
            "proto": e2r.decode(model, z, proto_ids[:, 0]),
            "ridge_routed": e2r.decode(model, z, ridge_ids[:, 0]),
            "ridge_masked": e2r.decode(
                model, z, ridge_ids[:, 0], task_mask[ridge_ids[:, 0]]
            ),
            "ridge_alone": rlog.argmax(-1),
            "oracle": e2r.decode(model, z, torch.full_like(y, owner)),
        }
        for a in ARMS5:
            per_task[a].append(float((pred[a] == y).float().mean()))
        R.append(pred["ridge_alone"] == y)
        TAU.append(ridge_ids[:, 0] == owner)
        S.append(pred["ridge_routed"] == y)
    arms = {a: sum(v) / len(v) for a, v in per_task.items()}
    dec = decomposition(torch.cat(R).cpu(), torch.cat(TAU).cpu(), torch.cat(S).cpu())
    dec["P2_task_mean"] = arms["ridge_routed"] - arms["ridge_alone"]
    return {"arms": arms, "decomposition": dec}


# -- Part B -----------------------------------------------------------------------------


def superclass_groups() -> list[list[int]]:
    mapping = superclass_of(args_data_dir())
    return [
        sorted(c for c in mapping if mapping[c] == k)
        for k in sorted(set(mapping.values()))
    ]


@torch.no_grad()
def evaluate_arm(model, tasks, device):
    owner = model.class_owner()
    own = torch.tensor([owner[c] for c in range(model.num_classes)], device=device)
    acc, oracle_acc, R, TAU, S = [], [], [], [], []
    for t in tasks:
        z, y = [v.to(device) for v in t["splits"]["test"]]
        p = model.predict(z, k=3, use_memory=False)  # k=3 as E-TID2 (top-1 of topk(3))
        po = model.predict(z, oracle_expert=own[y], use_memory=False)
        acc.append(float((p.labels == y).float().mean()))
        oracle_acc.append(float((po.labels == y).float().mean()))
        R.append((p.medium_labels == y).cpu())
        TAU.append((p.expert_ids[:, 0] == own[y]).cpu())
        S.append((p.labels == y).cpu())
    dec = decomposition(torch.cat(R), torch.cat(TAU), torch.cat(S))
    return {
        "accuracy": sum(acc) / len(acc),
        "oracle_accuracy": sum(oracle_acc) / len(oracle_acc),
        "decomposition": dec,
    }


def feature_bytes(tasks, policy) -> int:
    sizes = [
        t["splits"]["train"][0].numel() * 4 + t["splits"]["train"][1].numel() * 8
        for t in tasks
    ]
    return max(sizes) if policy == "by_arrival" else sum(sizes)


def part_b_cell(regime, seed, args, device, base, groups_sc):
    tasks = build_construction(base, regime)
    dim = int(tasks[0]["splits"]["train"][0].size(1))
    C = sum(len(t["classes"]) for t in tasks)
    canary = torch.cat([t["splits"]["test"][0][:10] for t in tasks]).to(device)
    model = PalMoE(
        dim,
        C,
        router="ridge_class",
        ridge=args.ridge,
        canary=canary,
        device=device,
        guards=GuardConfig(epsilon_medium=None),
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
    )
    writes = [
        model.write(Batch(*t["splits"]["train"], task=t["task_id"])) for t in tasks
    ]
    n_train = sum(t["splits"]["train"][1].numel() for t in tasks)
    written = sum(model._raw[r.id][1].numel() for r in writes)
    stats_digest = model.stats.state_digest()
    arms = {}
    for arm, kw in (
        ("A0", {"policy": "by_arrival"}),
        ("A1", {"policy": "by_confusion", "prereg": PREREG, "n_groups": 20}),
        ("A2", {"policy": "by_partition", "groups": groups_sc}),
    ):
        same_router = model.stats.state_digest() == stats_digest
        started = time.time()
        rep = model.consolidate(**kw)
        ev = evaluate_arm(model, tasks, device)
        forget = model.forget(rep.record.id)
        arms[arm] = {
            **ev,
            "groups": rep.groups,
            "group_sizes": sorted({len(g) for g in rep.groups}),
            "stored_feature_bytes": feature_bytes(tasks, kw["policy"]),
            "identical_router": same_router,
            "consolidation_reversible": rep.record.reversibility_report["pass"],
            "consolidation_bitwise": rep.record.reversibility_report["bitwise"],
            "forget_pass": forget["pass"],
            "router_trainable_params": 0 if rep.record.purity_report["pass"] else None,
            "seconds": time.time() - started,
        }

    def lab(groups):
        return [next(i for i, g in enumerate(groups) if c in g) for c in range(C)]

    return {
        "arms": arms,
        "ari_A1_A2": adjusted_rand_score(
            lab(arms["A1"]["groups"]), lab(arms["A2"]["groups"])
        ),
        "ari_A0_A2": adjusted_rand_score(
            lab(arms["A0"]["groups"]), lab(arms["A2"]["groups"])
        ),
        "writes_reversible": all(w.reversibility_report["pass"] for w in writes),
        "writes_order_pass": all(w.order_report["pass"] for w in writes),
        "no_test_leakage": written == n_train,
    }


# -- main -------------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", default="42,1,2,3,4,5")
    p.add_argument("--parts", default="A,B")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="results/p2_bound/p2_bound_study.json")
    args = p.parse_args()
    args.epochs, args.lr, args.batch_size, args.lambda_func, args.ridge = (
        10,
        1e-3,
        128,
        1.0,
        1.0,
    )
    device = torch.device(args.device)
    seeds = [int(s) for s in args.seeds.split(",")]
    stored = {(c["regime"], c["seed"]): c for c in json.load(open(ETID2_JSON))["cells"]}
    _, base = load_tasks(e2r.CACHE)
    groups_sc = superclass_groups()
    study = {
        "prereg": PREREG,
        "git_revision": git_rev(),
        "device": args.device,
        "seeds": seeds,
        "part_a": [],
        "part_b": [],
        "vetoes": {},
        "stats": {},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save = lambda: out.write_text(json.dumps(study, indent=1))  # noqa: E731
    t0 = time.time()

    for regime in REGIMES:
        for seed in seeds:
            if "A" in args.parts:
                args.seed = seed
                a = part_a_cell(regime, seed, args, device, base)
                a["repro_max_abs"] = max(
                    abs(a["arms"][k] - stored[(regime, seed)][k]) for k in ARMS5
                )
                study["part_a"].append({"regime": regime, "seed": seed, **a})
                d = a["decomposition"]
                print(
                    f"[A] {regime:9s} {seed:<3d} repro {a['repro_max_abs']:.1e} "
                    f"ident {d['identity_abs_error']:.1e}",
                    flush=True,
                )
                save()
            if "B" in args.parts:
                b = part_b_cell(regime, seed, args, device, base, groups_sc)
                b["A0_repro_abs"] = abs(
                    b["arms"]["A0"]["accuracy"] - stored[(regime, seed)]["ridge_routed"]
                )
                study["part_b"].append({"regime": regime, "seed": seed, **b})
                print(
                    f"[B] {regime:9s} {seed:<3d} A0 repro {b['A0_repro_abs']:.1e} "
                    f"({sum(v['seconds'] for v in b['arms'].values()):.0f}s)",
                    flush=True,
                )
                save()

    A, B = study["part_a"], study["part_b"]
    study["vetoes"] = {
        "decomposition_identity_max": max(
            (c["decomposition"]["identity_abs_error"] for c in A), default=0.0
        ),
        "right_class_wrong_task_count": sum(
            c["decomposition"]["counts"]["r_not_tau"] for c in A
        ),
        "etid2_reproduction_max": max((c["repro_max_abs"] for c in A), default=0.0),
        "A0_reproduction_max": max((c["A0_repro_abs"] for c in B), default=0.0),
        "identical_router": all(
            v["identical_router"] for c in B for v in c["arms"].values()
        ),
        "no_test_leakage": all(c["no_test_leakage"] for c in B),
        "balanced_A1_A2": all(
            c["arms"][a]["group_sizes"] == [5] for c in B for a in ("A1", "A2")
        ),
        "router_purity": all(
            v["router_trainable_params"] == 0 for c in B for v in c["arms"].values()
        ),
        "reversibility": all(
            c["writes_reversible"]
            and all(
                v["consolidation_reversible"] and v["forget_pass"]
                for v in c["arms"].values()
            )
            for c in B
        ),
        "order": all(c["writes_order_pass"] for c in B),
        "seconds": time.time() - t0,
    }
    v = study["vetoes"]
    v["pass"] = (
        v["decomposition_identity_max"] <= 1e-12
        and v["etid2_reproduction_max"] <= REPRO_BAND
        and v["A0_reproduction_max"] <= REPRO_BAND
        and v["identical_router"]
        and v["no_test_leakage"]
        and v["balanced_A1_A2"]
        and v["router_purity"]
        and v["reversibility"]
        and v["order"]
        and v["seconds"] <= CEILING_S
    )

    for regime in REGIMES:
        rb = [c for c in B if c["regime"] == regime]
        if not rb:
            continue
        p3 = [c["arms"]["A1"]["accuracy"] - c["arms"]["A0"]["accuracy"] for c in rb]
        p4 = [c["arms"]["A1"]["accuracy"] - c["arms"]["A2"]["accuracy"] for c in rb]
        study["stats"][regime] = {
            "P3_A1_minus_A0": paired_stats(p3, "P3", sesoi=0.01),
            "P4_A1_minus_A2": paired_stats(p4, "P4", sesoi=0.01),
            "P4_tost": tost(p4, 0.01),
            "A2_minus_A0": paired_stats(
                [c["arms"]["A2"]["accuracy"] - c["arms"]["A0"]["accuracy"] for c in rb],
                "X2",
            ),
            "westfall_young_P3": westfall_young({"P3": p3}, len(rb)),
        }
    save()
    print(json.dumps(study["vetoes"], indent=1))


if __name__ == "__main__":
    main()
