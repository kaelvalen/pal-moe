"""E-TID: the offline task-ID ceiling on the frozen feature space.

Question: is the routing tax a property of z (representation-bound), or of the
continual constraint (sequential fit / stored evidence)?

Every arm here sees ALL tasks' training data at once (no sequential constraint).
If none of them beats the prototype rule's coverage by a meaningful margin, the
tax is the Bayes error of task-ID given z and no router on this z can close it.

Arms (all on the same frozen ViT-B/16 CIFAR-100 cache, S6b constructions, T=20):
  proto      nearest class-mean, max over the task's classes  (E0 rule, anchor)
  knn        cosine k-NN over all training features, task = majority vote
  lin_task   joint linear softmax z -> T
  lin_class  joint linear softmax z -> 100, task = owner of top classes
  mlp_task   joint 2-layer MLP z -> T

Readings, fixed before running:
  best_offline C@1 - proto C@1 <  2 pp  -> representation-bound; close the routing line
  2 pp .. 8 pp                          -> partly continual; measure which stored evidence recovers it
  > 8 pp                                -> the tax is mainly a continual-learning constraint
"""
import argparse, json, sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import s2_ladder, s6b_difficulty  # noqa: E402

CACHE = "results/feature_cache/cifar100_vit_b16/feature_cache.pt"


def stack(tasks, split):
    xs, ys, ts = [], [], []
    for t in tasks:
        x, y = t["splits"][split]
        xs.append(x); ys.append(y); ts.append(torch.full_like(y, t["task_id"]))
    return torch.cat(xs), torch.cat(ys), torch.cat(ts)


def coverage(task_scores, t_true, k):
    top = task_scores.topk(k, dim=-1).indices
    return float((top == t_true[:, None]).any(-1).float().mean())


def class_to_task_scores(class_scores, cls2task, T):
    out = torch.full((class_scores.size(0), T), -1e9, device=class_scores.device)
    return out.scatter_reduce(1, cls2task.expand_as(class_scores), class_scores, "amax")


def fit(model, x, target, epochs, lr, wd, seed):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    for _ in range(epochs):
        perm = torch.randperm(x.size(0), generator=g).to(x.device)
        for i in range(0, x.size(0), 512):
            idx = perm[i:i + 512]
            loss = F.cross_entropy(model(x[idx]), target[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    return model.eval()


@torch.no_grad()
def knn_scores(xtr, ttr, xte, T, k):
    a, b = F.normalize(xtr, dim=-1), F.normalize(xte, dim=-1)
    out = torch.zeros(xte.size(0), T, device=xte.device)
    for i in range(0, xte.size(0), 1024):
        sim = b[i:i + 1024] @ a.T
        v, idx = sim.topk(k, dim=-1)
        out[i:i + 1024].scatter_add_(1, ttr[idx], v)
    return out


def run(construct, seed, args, device, base):
    torch.manual_seed(seed)
    tasks = s6b_difficulty.build_construction(base, construct)
    T = len(tasks)
    xtr, ytr, ttr = [v.to(device) for v in stack(tasks, "train")]
    xte, yte, tte = [v.to(device) for v in stack(tasks, "test")]
    C, d = int(ytr.max()) + 1, xtr.size(1)
    cls2task = torch.zeros(C, dtype=torch.long, device=device)
    for t in tasks:
        cls2task[torch.tensor(t["classes"], device=device)] = t["task_id"]

    res = {}
    protos = torch.stack([xtr[ytr == c].mean(0) for c in range(C)])
    s = F.normalize(xte, dim=-1) @ F.normalize(protos, dim=-1).T
    res["proto"] = class_to_task_scores(s, cls2task, T)
    res["knn"] = knn_scores(xtr, ttr, xte, T, args.k)
    m = fit(nn.Linear(d, T).to(device), xtr, ttr, args.epochs, 1e-3, 1e-4, seed)
    with torch.no_grad():
        res["lin_task"] = m(xte)
        train_fit = {"lin_task": coverage(m(xtr), ttr, 1)}
    m = fit(nn.Linear(d, C).to(device), xtr, ytr, args.epochs, 1e-3, 1e-4, seed)
    with torch.no_grad():
        res["lin_class"] = class_to_task_scores(F.log_softmax(m(xte), -1), cls2task, T)
    m = fit(nn.Sequential(nn.Linear(d, 1024), nn.GELU(), nn.Dropout(0.1), nn.Linear(1024, T)).to(device),
            xtr, ttr, args.epochs, 1e-3, 1e-4, seed)
    with torch.no_grad():
        res["mlp_task"] = m(xte)
        train_fit["mlp_task"] = coverage(m(xtr), ttr, 1)

    out = {arm: {"C@1": coverage(sc, tte, 1), "C@3": coverage(sc, tte, 3)} for arm, sc in res.items()}
    # Guard: an undertrained probe would fake a "representation-bound" reading.
    out["_train_fit_C@1"] = train_fit
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="42,1,2")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="results/e_tid/e_tid_ceiling.json")
    args = p.parse_args()

    _, base = s2_ladder.load_tasks(CACHE)
    cells = []
    for construct in ("coherent", "dispersed"):
        for seed in map(int, args.seeds.split(",")):
            r = run(construct, seed, args, args.device, base)
            cells.append({"construct": construct, "seed": seed, "arms": r})
            print(construct, seed, {a: (round(v["C@1"], 4), round(v["C@3"], 4)) for a, v in r.items() if not a.startswith("_")}, r["_train_fit_C@1"])

    # Anchor: proto C@1 must match E0 (0.8195 coherent / 0.7130 dispersed).
    for c in cells:
        a = c["arms"]["proto"]["C@1"]
        ref = 0.8195 if c["construct"] == "coherent" else 0.7130
        c["anchor_delta"] = a - ref
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"args": vars(args), "cells": cells}, indent=2))


if __name__ == "__main__":
    main()