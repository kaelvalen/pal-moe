"""
S3 report: the backbone-generalization table, the deltas that carry the
scientific content, and the representation-transfer axis.

Reads the per-condition ladder studies written by `s3_run.py` and the CIFAR-10
feature caches (an unseen dataset during the CIFAR-100 continual run), and emits
`results/s3/s3_backbone_study.json`.

The deltas, not the raw accuracies, are the result:

    L3 - L1_ridge      is expert isolation worth it for this representation?
    L4 - L3            routing tax
    L2b - L1_cosine    does a shared adapter beat its own readout without one?

Usage:
    python experiments/s3_report.py
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from pal_moe.arch import build_readout  # noqa: E402

ORDER = [
    "L0_ncm",
    "L1_ridge",
    "L1_cosine",
    "L2a_shared_joint",
    "L2b_shared_seq",
    "L3_per_task",
    "L4_oracle",
]


def load_study(path: str) -> dict:
    return json.load(open(path))


def transfer_accuracy(cache_path: str) -> dict:
    """Closed-form ridge probe on an unseen dataset, per backbone.

    The probe is fitted on the whole downstream train split (10 classes) and
    tested on its test split. It measures the representation, not the classifier
    the continual run built, which is the point: a method can score well on
    CIFAR-100 while carrying a representation that transfers nowhere.
    """
    payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    meta = payload["meta"]
    train_x, train_y, test_x, test_y = [], [], [], []
    for row in payload["tasks"]:
        tr = row["splits"]["train"]
        te = row["splits"]["test"]
        train_x.append(tr[0].float())
        train_y.append(tr[1].long())
        test_x.append(te[0].float())
        test_y.append(te[1].long())
    train_x, train_y = torch.cat(train_x), torch.cat(train_y)
    test_x, test_y = torch.cat(test_x), torch.cat(test_y)
    dim = train_x.size(1)
    probe = build_readout("ridge", dim=dim, num_classes=int(train_y.max()) + 1)
    probe.fit(train_x, train_y, seen_classes=sorted(set(train_y.tolist())))
    acc = float((probe.predict(test_x).argmax(dim=-1) == test_y).float().mean())
    return {
        "transfer_accuracy": acc,
        "downstream": meta.get("dataset"),
        "downstream_train_rows": int(train_x.size(0)),
        "dim": dim,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="results/s3")
    parser.add_argument("--reference", default="results/s2/s2_ladder_study.json")
    args = parser.parse_args()

    conditions = sorted(
        d
        for d in os.listdir(args.root)
        if os.path.isdir(os.path.join(args.root, d))
        and d.startswith(("random", "mlp", "conv", "resnet", "vit"))
    )
    table: dict[str, dict] = {}
    for name in conditions:
        studies = glob.glob(os.path.join(args.root, name, "s2_ladder_study*.json"))
        if not studies:
            print(f"[S3] no study for {name}, skipping")
            continue
        study = load_study(studies[0])
        levels = study["aggregate"]["levels"]
        row = {}
        for level in ORDER:
            if level in levels:
                acc = levels[level]["accuracy"]
                row[level] = {
                    "acc": acc["mean"],
                    "std": acc["std"],
                    "forgetting": levels[level]["forgetting"]["mean"],
                    "params": levels[level]["params"],
                    "recall_at_3": (levels[level].get("task_recall_at_3") or {}).get(
                        "mean"
                    ),
                    "acc_covered": (levels[level].get("acc_covered") or {}).get("mean"),
                }
        table[name] = row

    # Transfer axis.
    for name in list(table):
        cache = os.path.join(args.root, f"cache_{name}_cifar10", "feature_cache.pt")
        if os.path.exists(cache):
            table[name]["transfer"] = transfer_accuracy(cache)

    # Deltas (paired per seed is not available across conditions here; the
    # ladder studies already carry per-seed deltas against their own L0).
    deltas = {}
    for name, row in table.items():
        d = {}
        if "L3_per_task" in row and "L1_ridge" in row:
            d["L3_minus_ridge"] = row["L3_per_task"]["acc"] - row["L1_ridge"]["acc"]
        if "L4_oracle" in row and "L3_per_task" in row:
            d["routing_tax_L4_minus_L3"] = (
                row["L4_oracle"]["acc"] - row["L3_per_task"]["acc"]
            )
        if "L2b_shared_seq" in row and "L1_cosine" in row:
            d["L2b_minus_cosine"] = (
                row["L2b_shared_seq"]["acc"] - row["L1_cosine"]["acc"]
            )
        if "L2a_shared_joint" in row and "L1_ridge" in row:
            d["L2a_minus_ridge"] = (
                row["L2a_shared_joint"]["acc"] - row["L1_ridge"]["acc"]
            )
        deltas[name] = d

    # Consistency gate: S3's ViT row must reproduce the S2 reference.
    check = {}
    if os.path.exists(args.reference):
        ref = load_study(args.reference)["aggregate"]["levels"]
        ours = table.get("vit_b16_imagenet", {})
        for level in ORDER:
            if level in ref and level in ours:
                check[level] = {
                    "s2": ref[level]["accuracy"]["mean"],
                    "s3": ours[level]["acc"],
                    "abs_diff": abs(
                        ref[level]["accuracy"]["mean"] - ours[level]["acc"]
                    ),
                }

    _print(table, deltas, check)
    out = {
        "schema_version": "1.0",
        "study": "s3_backbone_generalization",
        "latent_dim": 768,
        "rep_seed": 0,
        "seeds": [42, 1, 2],
        "conditions": table,
        "deltas": deltas,
        "vit_consistency_check": check,
    }
    path = os.path.join(args.root, "s3_backbone_study.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\n[S3] wrote {path}")


def _pct(x):
    return f"{x * 100:7.2f}%" if x is not None else "      -"


def _print(table: dict, deltas: dict, check: dict) -> None:
    print("\n" + "=" * 100)
    print(
        "S3 BACKBONE GENERALIZATION - frozen backbones, latent_dim=768, CIFAR-100 20t, 3 seeds"
    )
    print("=" * 100)
    header = f"{'backbone':22s}" + "".join(f"{lvl:>10s}" for lvl in ORDER)
    print(header)
    for name, row in table.items():
        line = f"{name:22s}"
        for level in ORDER:
            line += _pct(row[level]["acc"] if level in row else None)
        print(line)

    print("\n" + "-" * 100)
    print(
        f"{'backbone':22s} {'L3-ridge':>10s} {'L4-L3':>10s} {'L2b-cos':>10s} {'L2a-ridge':>10s} {'transfer':>10s} {'recall@3':>10s}"
    )
    for name, row in table.items():
        d = deltas.get(name, {})
        t = (row.get("transfer") or {}).get("transfer_accuracy")
        r3 = (row.get("L3_per_task") or {}).get("recall_at_3")
        print(
            f"{name:22s} {_pct(d.get('L3_minus_ridge'))} {_pct(d.get('routing_tax_L4_minus_L3'))} "
            f"{_pct(d.get('L2b_minus_cosine'))} {_pct(d.get('L2a_minus_ridge'))} {_pct(t)} {_pct(r3)}"
        )

    if check:
        print("\nViT row vs the S2 reference (S3 must reproduce it):")
        for level, c in check.items():
            flag = "ok" if c["abs_diff"] < 0.02 else "MISMATCH"
            print(
                f"  {level:20s} s2 {c['s2'] * 100:6.2f}%  s3 {c['s3'] * 100:6.2f}%  {flag}"
            )


if __name__ == "__main__":
    main()
