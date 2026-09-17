"""
Fast pure-mode exploration harness for PAL-MoE.

Runs ONLY the two PAL-MoE variants (pure + hybrid) with tunable new
mechanisms, mirroring experiments/run_benchmark.py configuration exactly:

    --lambda_ood              OOD negative-boundary weight (0 = off)
    --no_joint_freeze_router  disable full-router freezing during joint FT
    --joint_keep_routing_lock keep only historical routing rows locked
    --max_proto_acc_drop      historical prototype accuracy-drop gate
    --max_proto_drop          prototype output drift gate

Usage:
    python experiments/run_pure_explore.py --lambda_ood 0.1 --device cuda
"""

import argparse
import copy
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torchvision import datasets, transforms

from pal_moe.adaptation.ttt import ContinualTrainer
from pal_moe.builder.expert_builder import ExpertBuilder
from pal_moe.data.split_mnist import get_split_mnist_tasks
from pal_moe.evaluation.metrics import ContinualEvaluator
from pal_moe.factory import build_moe, build_prototype_memory
from pal_moe.models.encoder import SharedEncoder
from pal_moe.trigger.expert_trigger import QuantitativeTrigger


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_variant(name, base_encoder, tasks, num_tasks, device, args, replay=False):
    print("\n" + "=" * 60)
    print(f"Running {name}")
    print("=" * 60)
    set_seed(args.seed)
    enc = copy.deepcopy(base_encoder)
    model = build_moe(
        enc,
        feature_dim=args.feature_dim,
        expert_hidden=args.expert_hidden,
        num_classes=10,
        num_experts=1,
        router_type=args.router_type,
        device=device,
    )

    proto_mem = build_prototype_memory(
        args.feature_dim,
        distance_threshold=0.5,
        max_prototypes=250,
        store_raw=replay,
    )
    trigger = QuantitativeTrigger(
        alpha=1.0, beta=0.4, gamma=0.6, delta=0.5, threshold_tau=0.5
    )
    builder = ExpertBuilder(
        min_acc_threshold=0.60,
        max_proto_drop=args.max_proto_drop,
        max_proto_acc_drop=args.max_proto_acc_drop,
        max_ece=999.0,
        distill_lambda=0.5,
    )
    trainer = ContinualTrainer(
        model=model,
        prototype_memory=proto_mem,
        trigger=trigger,
        builder=builder,
        lambda_r=args.lambda_r,
        lambda_e=args.lambda_e,
        lambda_enc=0.0,
        replay_exemplars=replay,
        lambda_replay=1.0,
        lr=1e-3,
        max_experts=6,
        joint_unfreeze_all=True,
        joint_keep_routing_lock=args.joint_keep_routing_lock,
        joint_freeze_router=args.joint_freeze_router,
        lambda_ood=args.lambda_ood,
        device=device,
    )
    evaluator = ContinualEvaluator(num_tasks=num_tasks, device=device)

    for t_idx, task in enumerate(tasks):
        hist = trainer.train_task(
            task_id=t_idx,
            train_loader=task.train_loader,
            val_loader=task.val_loader,
            epochs=args.epochs,
            enable_expansion=True,
            enable_anchor=args.anchor,
        )
        accs = evaluator.evaluate_all_seen_tasks(model, t_idx, tasks)
        print(
            f"  T{t_idx}: experts={model.num_experts} | triggers={hist['trigger_events']} "
            f"added={hist['experts_added']} rejects={hist['gate_rejections']} | accs={[f'{a:.1%}' for a in accs]}"
        )

    result = {
        "acc": evaluator.compute_average_accuracy(),
        "forgetting": evaluator.compute_forgetting(),
        "bwt": evaluator.compute_backward_transfer(),
        "final_experts": model.num_experts,
        "acc_matrix": evaluator.R.tolist(),
    }
    print(
        f"  => {name}: acc={result['acc']:.2%} forgetting={result['forgetting']:.2%} experts={result['final_experts']}"
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--router_type", type=str, default="dynamic", choices=["dynamic", "distance"]
    )
    parser.add_argument("--lambda_ood", type=float, default=0.0)
    parser.add_argument(
        "--joint_freeze_router", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--joint_keep_routing_lock", action="store_true")
    parser.add_argument("--max_proto_drop", type=float, default=2.0)
    parser.add_argument("--max_proto_acc_drop", type=float, default=0.05)
    parser.add_argument("--pretrain_epochs", type=int, default=1)
    parser.add_argument(
        "--pretrain_mode", type=str, default="ae", choices=["ae", "simclr"]
    )
    parser.add_argument("--lambda_r", type=float, default=0.5)
    parser.add_argument(
        "--anchor", action="store_true", help="Enable null-space routing anchoring"
    )
    parser.add_argument("--lambda_e", type=float, default=2.5)
    parser.add_argument("--feature_dim", type=int, default=128)
    parser.add_argument("--expert_hidden", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default="/tmp/pal-moe-explore")
    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and args.device not in ("cpu",)
        else args.device
    )
    os.makedirs(args.output_dir, exist_ok=True)

    tasks = get_split_mnist_tasks(
        data_dir="./data", batch_size=128, val_split=0.1, seed=args.seed
    )
    num_tasks = len(tasks)
    mnist_train = datasets.MNIST(
        "./data",
        train=True,
        download=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )
    unlabeled_loader = torch.utils.data.DataLoader(
        mnist_train, batch_size=256, shuffle=True
    )
    base_encoder = SharedEncoder(
        input_dim=784,
        hidden_dims=(256, 128),
        output_dim=args.feature_dim,
        arch="mlp",
    ).to(device)
    if args.pretrain_mode == "simclr":
        base_encoder.pretrain_contrastive(
            unlabeled_loader, device=device, epochs=args.pretrain_epochs, lr=1e-3
        )
    else:
        base_encoder.pretrain_unsupervised(
            unlabeled_loader, device=device, epochs=args.pretrain_epochs
        )
    base_encoder.freeze()

    results = {
        "flags": {
            "lambda_ood": args.lambda_ood,
            "joint_freeze_router": args.joint_freeze_router,
            "anchor": args.anchor,
            "joint_keep_routing_lock": args.joint_keep_routing_lock,
            "max_proto_drop": args.max_proto_drop,
            "max_proto_acc_drop": args.max_proto_acc_drop,
            "router_type": args.router_type,
            "epochs": args.epochs,
            "pretrain_mode": args.pretrain_mode,
            "pretrain_epochs": args.pretrain_epochs,
            "lambda_r": args.lambda_r,
            "lambda_e": args.lambda_e,
        },
        "PAL-MoE (Ours)": run_variant(
            "PAL-MoE (Pure)", base_encoder, tasks, num_tasks, device, args, replay=False
        ),
        "PAL-MoE + Replay (Hybrid)": run_variant(
            "PAL-MoE + Replay (Hybrid)",
            base_encoder,
            tasks,
            num_tasks,
            device,
            args,
            replay=True,
        ),
    }

    tag = (
        f"ood{args.lambda_ood}_fr{int(args.joint_freeze_router)}_"
        f"kl{int(args.joint_keep_routing_lock)}_dr{args.max_proto_drop}_"
        f"ar{args.max_proto_acc_drop}_{args.router_type}"
    )
    out = os.path.join(args.output_dir, f"pure_{tag}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
