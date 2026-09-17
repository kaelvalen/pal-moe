"""
Routing Asymmetry & Router Confusion Diagnostics for PAL-MoE.
Diagnoses why Task 0 is preserved while Task 2/3 suffer from premature forgetting.
Logs per-task router dispatch probability distributions across sequential tasks.
Compares 1-epoch Autoencoder vs SimCLR Contrastive Pretraining.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json

import numpy as np
import torch
from tabulate import tabulate
from torchvision import datasets, transforms

from pal_moe.adaptation.ttt import ContinualTrainer
from pal_moe.builder.expert_builder import ExpertBuilder
from pal_moe.data.split_mnist import get_split_mnist_tasks
from pal_moe.evaluation.metrics import ContinualEvaluator
from pal_moe.factory import build_moe, build_prototype_memory
from pal_moe.models.encoder import SharedEncoder
from pal_moe.models.moe import DynamicMoE
from pal_moe.trigger.expert_trigger import QuantitativeTrigger


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def inspect_routing_distribution(model: DynamicMoE, tasks, device: torch.device):
    """
    For each task, evaluates:
    1. Average routing distribution over all active experts: [P(E_0), ..., P(E_{N-1})]
    2. Top-1 routing allocation fractions: [Frac(Top1 == E_0), ...]
    3. Individual expert accuracies on that task: [Acc(E_0), ...]
    4. Composite MoE accuracy on that task.
    """
    model.eval()
    num_experts = model.num_experts
    task_stats = []

    for t_idx, task in enumerate(tasks):
        x_list, y_list = [], []
        for x, y in task.val_loader:
            x_list.append(x)
            y_list.append(y)
        x_all = torch.cat(x_list, dim=0).to(device)
        y_all = torch.cat(y_list, dim=0).to(device)

        with torch.no_grad():
            h = model.get_routing_features(x_all)
            full_probs = model.router.get_full_distribution(h)  # [B, N]
            mean_probs = full_probs.mean(dim=0).cpu().tolist()
            top1_choice = full_probs.argmax(dim=-1)
            top1_fractions = [
                (top1_choice == i).float().mean().item() for i in range(num_experts)
            ]

            moe_preds = model(x_all).argmax(dim=-1)
            moe_acc = (moe_preds == y_all).float().mean().item()

            expert_accs = []
            for e_idx in range(num_experts):
                e_pred = model.experts[e_idx](h, track_usage=False).argmax(dim=-1)
                expert_accs.append((e_pred == y_all).float().mean().item())

        task_stats.append(
            {
                "task_id": t_idx,
                "classes": task.classes,
                "moe_acc": moe_acc,
                "mean_router_probs": mean_probs,
                "top1_fractions": top1_fractions,
                "expert_accs": expert_accs,
            }
        )

    return task_stats


def run_diagnostic_experiment(
    encoder_type: str = "autoencoder",
    pretrain_epochs: int = 1,
    epochs: int = 3,
    device: torch.device = torch.device("cpu"),
    seed: int = 42,
):
    print(f"\n{'=' * 70}")
    print(
        f"DIAGNOSTIC RUN: Encoder={encoder_type.upper()} ({pretrain_epochs} ep), Seed={seed}"
    )
    print(f"{'=' * 70}")
    set_seed(seed)

    tasks = get_split_mnist_tasks(
        data_dir="./data", batch_size=128, val_split=0.1, seed=seed
    )
    mnist_train = datasets.MNIST(
        "./data", train=True, download=False, transform=transforms.ToTensor()
    )
    unlabeled_loader = torch.utils.data.DataLoader(
        mnist_train, batch_size=256, shuffle=True
    )

    base_encoder = SharedEncoder(
        input_dim=784, hidden_dims=(256, 128), output_dim=128, arch="mlp"
    ).to(device)
    if encoder_type == "contrastive":
        print("  Running SimCLR contrastive pretraining...")
        base_encoder.pretrain_contrastive(
            unlabeled_loader, device=device, epochs=pretrain_epochs, lr=1e-3
        )
    else:
        print("  Running standard Autoencoder reconstruction pretraining...")
        base_encoder.pretrain_unsupervised(
            unlabeled_loader, device=device, epochs=pretrain_epochs
        )
    base_encoder.freeze()

    moe_model = build_moe(
        base_encoder,
        feature_dim=128,
        expert_hidden=64,
        num_classes=10,
        num_experts=1,
        router_type="dynamic",
        device=device,
    )

    prototype_mem = build_prototype_memory(
        128,
        distance_threshold=0.5,
        max_prototypes=60,
        store_raw=False,
    )
    trigger = QuantitativeTrigger(
        alpha=1.0, beta=0.4, gamma=0.6, delta=0.5, threshold_tau=0.5
    )
    builder = ExpertBuilder(
        min_acc_threshold=0.60,
        max_proto_drop=2.0,
        max_ece=0.35,
        distill_lambda=0.5,
    )
    trainer = ContinualTrainer(
        model=moe_model,
        prototype_memory=prototype_mem,
        trigger=trigger,
        builder=builder,
        lambda_r=0.5,
        lambda_e=2.5,
        lr=1e-3,
        max_experts=6,
        device=device,
    )

    progression_log = []

    for t_idx, task in enumerate(tasks):
        trainer.train_task(
            task_id=t_idx,
            train_loader=task.train_loader,
            val_loader=task.val_loader,
            epochs=epochs,
            enable_expansion=True,
        )
        stats = inspect_routing_distribution(moe_model, tasks, device)
        progression_log.append(
            {
                "after_task": t_idx,
                "num_experts": moe_model.num_experts,
                "tasks": stats,
            }
        )

        print(
            f"\n--- After Training Task {t_idx} (Active Experts: {moe_model.num_experts}) ---"
        )
        table_rows = []
        for s in stats[: t_idx + 1]:
            routes = ", ".join(
                [f"E{i}:{f:.0%}" for i, f in enumerate(s["top1_fractions"])]
            )
            exps = ", ".join([f"E{i}:{a:.0%}" for i, a in enumerate(s["expert_accs"])])
            table_rows.append(
                [
                    f"Task {s['task_id']} ({s['classes']})",
                    f"{s['moe_acc']:.1%}",
                    routes,
                    exps,
                ]
            )
        print(
            tabulate(
                table_rows,
                headers=[
                    "Task",
                    "MoE Acc",
                    "Top-1 Routing Choices",
                    "Individual Expert Accs",
                ],
                tablefmt="simple",
            )
        )

    evaluator = ContinualEvaluator(num_tasks=len(tasks), device=device)
    for t_idx, _task in enumerate(tasks):
        evaluator.evaluate_all_seen_tasks(moe_model, t_idx, tasks)

    final_acc = evaluator.compute_average_accuracy()
    final_forg = evaluator.compute_forgetting()
    print(f"\nFinal Result: Avg Acc = {final_acc:.2%}, Forgetting = {final_forg:.2%}")

    return {
        "encoder_type": encoder_type,
        "pretrain_epochs": pretrain_epochs,
        "final_acc": final_acc,
        "final_forgetting": final_forg,
        "progression": progression_log,
        "final_r_matrix": evaluator.R.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and args.device == "auto"
        else ("cpu" if args.device == "auto" else args.device)
    )

    res_ae = run_diagnostic_experiment(
        encoder_type="autoencoder",
        pretrain_epochs=1,
        epochs=args.epochs,
        device=device,
        seed=args.seed,
    )

    res_simclr = run_diagnostic_experiment(
        encoder_type="contrastive",
        pretrain_epochs=3,
        epochs=args.epochs,
        device=device,
        seed=args.seed,
    )

    summary_file = os.path.join(args.output_dir, "routing_asymmetry_debug.json")
    with open(summary_file, "w") as f:
        json.dump(
            {"autoencoder_1ep": res_ae, "contrastive_simclr_3ep": res_simclr},
            f,
            indent=2,
        )
    print(f"\nDiagnostics completed. Saved detailed log to {summary_file}")


if __name__ == "__main__":
    main()
