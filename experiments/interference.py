"""
Interference mechanism: new prototypes against old projections.

    docs/INTERFERENCE_PREREG.md

Passive instrumentation on the pinned C1 (all-W) trajectory: no manipulation of
the training, only probes taken with torch.autograd.grad - which never touches
.grad or the optimizer state - plus W snapshots and an expert-count ladder.

Measurements per task q and old projection W_j (j < q): ||delta_W_j||, and the
owner/non-owner split of grad_{W_j} L_q by prototype ownership.

Usage: python experiments/interference.py --device cuda
"""

import argparse
import copy
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import e2_evidence as e2  # noqa: E402
import s11_confirmatory as s11  # noqa: E402

TS = [2, 4, 8, 12, 20]
REGIMES = ["coherent", "dispersed"]


def probe(model, task_index):
    """Owner vs non-owner gradient mass on each old W_j, at one task boundary."""
    if task_index == 0 or not model.prototypes:
        return {}
    terms = []
    for z_p, owner in model.prototypes:
        q = model.P(z_p)
        scores = []
        for j in range(len(model.experts)):
            h = e2.normalize(
                model.W[j](model.experts[j].transform(z_p.unsqueeze(0))[0])
            )
            scores.append(torch.dot(e2.normalize(q), h))
        terms.append(
            (
                owner,
                F.cross_entropy(
                    torch.stack(scores).unsqueeze(0),
                    torch.tensor([owner], device=model.device),
                ),
            )
        )
    out = {}
    for j in range(task_index):
        owner_mass = nonowner_mass = 0.0
        for owner, loss_p in terms:
            grad = torch.autograd.grad(loss_p, model.W[j].weight, retain_graph=True)[0]
            norm = float(grad.norm())
            if owner == j:
                owner_mass += norm
            else:
                nonowner_mass += norm
        out[f"W{j}"] = {
            "owner": owner_mass,
            "nonowner": nonowner_mass,
            "ratio": nonowner_mass / max(owner_mass, 1e-12),
        }
    return out


def run(regime, T, seed, args, device):
    _, source = s11.s2_ladder.load_tasks(e2.SOURCE_CACHE)
    tasks = s11.s6b_difficulty.build_construction(source, regime)[:T]
    cell_args = argparse.Namespace(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        rank=8,
        evidence_lambda=1.0,
        w_alignment="all",
        seed=seed,
    )
    s11.s2_ladder.set_seed(seed)
    model = e2.E2Model(768, 100, cell_args, device)

    model = e2.E2Model(768, 100, cell_args, device)
    probes, rewrite = {}, {}

    def on_task_start(m, t):
        return {
            "probes": probe(copy.deepcopy(m), t) if t > 0 else None,
            "snaps": {j: m.W[j].weight.detach().clone() for j in range(t + 1)},
        }

    def on_task_end(m, t, st):
        if st["probes"]:
            probes[f"q{t}"] = st["probes"]
        for j in range(t + 1):
            rewrite.setdefault(f"after_t{t}", {})[f"W{j}"] = float(
                (m.W[j].weight.detach() - st["snaps"][j]).norm()
            )

    model.train(
        tasks, seed, hooks={"on_task_start": on_task_start, "on_task_end": on_task_end}
    )
    metrics = model.evaluate(tasks)
    return {**metrics, "probes": probes, "rewrite": rewrite, "T": T}


def _train_one(model, task, t, seed):
    """One task of the pinned C1 training, identical to E2Model.train's body."""
    args = model.args
    trainable = list(model.P.parameters()) + list(model.readout.parameters())
    for w in model.W:
        trainable += [p for p in w.parameters() if p.requires_grad]
    for param in model.experts[-1].parameters():
        param.requires_grad_(True)
    trainable += list(model.experts[-1].parameters())
    optimizer = torch.optim.Adam(trainable, lr=args.lr)
    handle = s11.s2_ladder.trainable_hook(model.readout, task["classes"])
    feats, labels = task["splits"]["train"]
    feats, labels = feats.to(model.device), labels.to(model.device)
    generator = torch.Generator().manual_seed(seed + t)
    for _ in range(args.epochs):
        for z, y in s11.s2_ladder.iter_batches(
            feats, labels, args.batch_size, generator
        ):
            z, y = z.to(model.device), y.to(model.device)
            h = e2.normalize(model.W[t](model.experts[t].transform(z)))
            logits = s11.s2_ladder.mask_unseen(model.readout.predict(h), model.seen)
            loss = F.cross_entropy(logits, y) + model._evidence_loss()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    handle.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="42,1,2")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--out", default="results/interference")
    args = parser.parse_args()
    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    seeds = [int(s) for s in args.seeds.replace(",", " ").split()]
    path = os.path.join(args.out, "interference_study.json")
    cells = []
    for regime in REGIMES:
        for T in TS:
            for seed in seeds:
                result = run(regime, T, seed, args, device)
                cells.append({"regime": regime, "seed": seed, **result})
                json.dump(
                    {
                        "schema_version": "1.0",
                        "study": "interference",
                        "prereg": "docs/INTERFERENCE_PREREG.md",
                        "seeds": seeds,
                        "cells": cells,
                    },
                    open(path, "w"),
                    indent=1,
                )
                ratios = [
                    v["ratio"] for v in result["probes"].get(f"q{T-1}", {}).values()
                ]
                print(
                    f"[INT] {regime:10s} T={T:<3d} seed={seed:<3d} "
                    f"acc={result['accuracy']*100:6.2f} C@3={result['coverage_at_3']:.4f} "
                    f"nonowner/owner={sum(ratios)/max(len(ratios),1):.3f}",
                    flush=True,
                )
    print("[INT] wrote", path)


if __name__ == "__main__":
    main()
