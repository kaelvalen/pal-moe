"""
S7 - representation transfer, measured at every checkpoint.

The S3 result separated representation quality from continual-learning
behaviour: CIFAR-100 accuracy ranks the backbones differently from CIFAR-10
transfer (ViT 94.84% transfer against a 6-point CIL gap). S7 measures that
separation systematically.

The classification head of the continual run is deliberately discarded at every
checkpoint and replaced by a closed-form ridge probe on the frozen
representation:

    CIL training -> checkpoint t -> freeze representation -> downstream
    dataset -> ridge -> transfer accuracy

Three things are held apart on purpose:

- **the raw frozen reference**: `z_raw` is the backbone's own output, which the
  downstream cache already contains. Every adapted number is reported against
  it, so `delta = transfer(z_CL) - transfer(z_raw)` is the quantity of interest,
  not the absolute transfer.
- **FWT is not transfer.** FWT (S2) is the next task of the *same* stream;
  transfer is a different dataset that the continual run never saw.
- **one headline evaluator.** Ridge, because S2 showed it is the strongest
  analytic readout. Logistic and NCM are reported as validation only, so the
  transfer conclusion cannot hinge on the readout choice.

Usage:
    python experiments/s7_transfer.py --device cuda
    python experiments/s7_transfer.py --backbones vit_b16_imagenet --levels L3_per_task
"""

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import s2_ladder  # noqa: E402
from pal_moe.arch import build_readout  # noqa: E402
from pal_moe.evaluation.schema import build_run_record, satisfied_blocks  # noqa: E402

# Levels that change the representation. A no-expert level is the raw reference
# by construction, so it is not re-run here.
LEVELS = ["L2b_shared_seq", "L3_per_task"]
FEW_SHOT = [1, 5, 10, 25, 50]
FEW_SHOT_REPEATS = 3


def _concat_split(payload, split: str):
    xs, ys = [], []
    for row in payload["tasks"]:
        xs.append(row["splits"][split][0].float())
        ys.append(row["splits"][split][1].long())
    return torch.cat(xs), torch.cat(ys)


def load_downstream(path: str):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return {
        "dataset": payload["meta"].get("dataset"),
        "train": _concat_split(payload, "train"),
        "test": _concat_split(payload, "test"),
    }


def probe_accuracy(
    z_train, y_train, z_test, y_test, evaluator: str = "ridge", labels=None
) -> float:
    """Fit a readout on the representation and score the downstream test split."""
    if labels is None:
        labels = sorted(set(y_train.tolist()))
    n_classes = int(max(y_test.max(), y_train.max())) + 1
    readout = build_readout(evaluator, dim=z_train.size(1), num_classes=n_classes)
    readout.fit(z_train, y_train, seen_classes=[int(c) for c in labels])
    pred = readout.predict(z_test).argmax(dim=-1)
    return float((pred == y_test).float().mean())


def few_shot_accuracy(
    z_train,
    y_train,
    z_test,
    y_test,
    n_per_class: int,
    seed: int,
    evaluator: str = "ridge",
) -> float:
    """Mean accuracy over `FEW_SHOT_REPEATS` class-balanced draws of `n`/class."""
    accs = []
    for repeat in range(FEW_SHOT_REPEATS):
        gen = torch.Generator().manual_seed(seed * 1000 + repeat)
        idx = []
        for c in sorted(set(y_train.tolist())):
            pool = (y_train == c).nonzero(as_tuple=True)[0]
            take = min(n_per_class, pool.numel())
            perm = torch.randperm(pool.numel(), generator=gen)[:take]
            idx.append(pool[perm])
        idx = torch.cat(idx)
        accs.append(
            probe_accuracy(
                z_train[idx], y_train[idx], z_test, y_test, evaluator=evaluator
            )
        )
    return float(statistics.mean(accs))


def transfer_suite(z_train, y_train, z_test, y_test, seed: int) -> dict:
    """The full transfer measurement for one representation."""
    out = {"full": probe_accuracy(z_train, y_train, z_test, y_test)}
    for n in FEW_SHOT:
        out[f"few{n}"] = few_shot_accuracy(z_train, y_train, z_test, y_test, n, seed)
    return out


def run_condition(spec_name, backbone_name, source_cache, downstream, args, device):
    meta, tasks = s2_ladder.load_tasks(source_cache)
    dim = int(meta["feature_dim"])
    num_classes = sum(len(t["classes"]) for t in tasks)
    spec = s2_ladder.LEVELS_BY_NAME[spec_name]

    # Raw frozen reference: the downstream cache *is* the backbone's output.
    raw_train, raw_test = downstream["train"], downstream["test"]
    raw = transfer_suite(
        raw_train[0], raw_train[1], raw_test[0], raw_test[1], args.seed
    )

    s2_ladder.set_seed(args.seed)
    model = s2_ladder.LadderModel(spec, dim, num_classes, args, device)
    curve = []
    for t, task in enumerate(tasks):
        model.seen = sorted(set(model.seen) | set(task["classes"]))
        model.fit_task(task, t)
        model.register_task(task, t)
        accs = [model.evaluate_task(tasks[i]) for i in range(t + 1)]
        with torch.no_grad():
            z_tr = model.represent(raw_train[0].to(device)).cpu()
            z_te = model.represent(raw_test[0].to(device)).cpu()
        suite = transfer_suite(z_tr, raw_train[1], z_te, raw_test[1], args.seed)
        curve.append(
            {
                "task": t,
                "cil_accuracy": float(np.mean(accs)),
                "transfer": suite,
                "delta_vs_raw": {k: suite[k] - raw[k] for k in suite},
            }
        )
        print(
            f"  [S7] {backbone_name}/{spec_name} t={t:2d} "
            f"cil={np.mean(accs) * 100:5.2f}%  "
            f"transfer={suite['full'] * 100:5.2f}%  "
            f"raw={raw['full'] * 100:5.2f}%  "
            f"delta={(suite['full'] - raw['full']) * 100:+5.2f}",
            flush=True,
        )

    # Validation evaluators on the final checkpoint (headline stays ridge).
    with torch.no_grad():
        z_tr = model.represent(raw_train[0].to(device)).cpu()
        z_te = model.represent(raw_test[0].to(device)).cpu()
    validation = {
        evaluator: probe_accuracy(
            z_tr, raw_train[1], z_te, raw_test[1], evaluator=evaluator
        )
        for evaluator in ("logistic", "ncm")
    }
    return {
        "backbone": backbone_name,
        "level": spec_name,
        "seed": args.seed,
        "raw": raw,
        "curve": curve,
        "final_transfer": curve[-1]["transfer"] if curve else None,
        "final_cil": curve[-1]["cil_accuracy"] if curve else None,
        "validation_evaluators": validation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backbones", default="vit_b16_imagenet,resnet18_imagenet,random"
    )
    parser.add_argument("--levels", nargs="*", default=LEVELS)
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lambda_func", type=float, default=1.0)
    parser.add_argument("--max_experts", type=int, default=20)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--downstream", default="cifar10")
    parser.add_argument("--out", default="results/s7")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    seeds = [int(s) for s in args.seeds.replace(",", " ").split()]
    backbones = [b for b in args.backbones.split(",") if b]

    results = []
    for backbone in backbones:
        source = f"results/s3/cache_{backbone}_cifar100/feature_cache.pt"
        down = f"results/s3/cache_{backbone}_{args.downstream}/feature_cache.pt"
        if not (os.path.exists(source) and os.path.exists(down)):
            print(f"[S7] missing cache for {backbone}, skipping")
            continue
        downstream = load_downstream(down)
        print(f"=== S7 {backbone}: {args.downstream} as downstream ===", flush=True)
        for spec_name in args.levels:
            for seed in seeds:
                args.seed = seed
                results.append(
                    run_condition(spec_name, backbone, source, downstream, args, device)
                )

    contracts = {
        f"{r['backbone']}__{r['level']}__seed{r['seed']}": _contract(r, args)
        for r in results
    }
    study = {
        "schema_version": "1.0",
        "study": "s7_representation_transfer",
        "contracts": contracts,
        "downstream": args.downstream,
        "few_shot": FEW_SHOT,
        "headline_evaluator": "ridge",
        "seeds": seeds,
        "conditions": results,
    }
    path = os.path.join(args.out, f"s7_transfer_{args.downstream}.json")
    with open(path, "w") as fh:
        json.dump(study, fh, indent=1)
    print(f"[S7] wrote {path}", flush=True)
    _print_summary(results)


def _contract(record: dict, args) -> dict:
    """S0 record for one transfer condition.

    `transfer_accuracy` is the representation-level measurement, so the record
    can declare the generalization block once the stability fields exist; it
    deliberately does not fake them.
    """
    raw = record["raw"]["full"]
    fin = record["final_transfer"]["full"]
    out = build_run_record(
        factors={
            "dataset": "cifar100",
            "protocol": "class_il",
            "task_id_at_inference": record["level"] == "L4_oracle",
            "seed": record["seed"],
            "model_family": record["level"],
            "backbone": record["backbone"],
            "backbone_pretraining": (
                "random" if "random" in record["backbone"] else "imagenet_frozen"
            ),
            "expert": "residual_adapter",
            "readout": "ridge",
        },
        metrics={
            "learning": {
                "accuracy": record["final_cil"],
                "forgetting": None,
                "acc_matrix": None,
            },
            "generalization": {
                "transfer_accuracy": fin,
                "unseen_task_accuracy": None,
            },
        },
        provenance={"command": "experiments/s7_transfer.py", "raw_reference": raw},
    )
    out["blocks"] = satisfied_blocks(out)
    return out


def _print_summary(results) -> None:
    print("\n" + "=" * 92)
    print("S7 REPRESENTATION TRANSFER (ridge headline, raw frozen reference)")
    print("=" * 92)
    print(
        f"{'backbone':22s} {'level':16s} {'raw':>8s} {'final':>8s} {'delta':>8s} "
        f"{'best t':>8s} {'cil':>8s} {'few1':>8s} {'few10':>8s}"
    )
    for r in results:
        raw = r["raw"]["full"]
        fin = r["final_transfer"]["full"]
        curve = r["curve"]
        best = max(c["transfer"]["full"] for c in curve) if curve else float("nan")
        print(
            f"{r['backbone']:22s} {r['level']:16s} {raw * 100:7.2f}% {fin * 100:7.2f}% "
            f"{(fin - raw) * 100:+7.2f}% {best * 100:7.2f}% "
            f"{(r['final_cil'] or float('nan')) * 100:7.2f}% "
            f"{r['final_transfer']['few1'] * 100:7.2f}% "
            f"{r['final_transfer']['few10'] * 100:7.2f}%"
        )
        print(
            f"{'':22s} validation evaluators: "
            + "  ".join(
                f"{k}={v * 100:.2f}%" for k, v in r["validation_evaluators"].items()
            )
        )


if __name__ == "__main__":
    main()
