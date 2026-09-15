"""
Split-CIFAR-10 Benchmark with ConvNet Encoder & Unfrozen Encoder Regime.
Systematically tests the hypothesis:
1. Frozen Conv Encoder Baseline
2. Unfrozen Conv Encoder WITHOUT L_encoder_stab (lambda_enc = 0) -> Catastrophic representation drift
3. Unfrozen Conv Encoder WITH L_encoder_stab (lambda_enc = 1.0) -> Feature-space distillation prevents drift
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import copy
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from tabulate import tabulate

from pal_moe.data.split_cifar import get_split_cifar10_tasks
from pal_moe.models.encoder import SharedEncoder
from pal_moe.models.router import DynamicRouter
from pal_moe.models.expert import MLPExpert
from pal_moe.models.moe import DynamicMoE
from pal_moe.memory.prototype_memory import PrototypeMemory
from pal_moe.trigger.expert_trigger import QuantitativeTrigger
from pal_moe.builder.expert_builder import ExpertBuilder
from pal_moe.adaptation.ttt import ContinualTrainer
from pal_moe.evaluation.metrics import ContinualEvaluator


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_single_cifar_config(
    tasks,
    base_encoder: SharedEncoder,
    config_name: str,
    adapt_encoder: bool = False,
    lambda_enc: float = 0.0,
    lambda_r: float = 0.5,
    lambda_e: float = 2.5,
    epochs: int = 3,
    lr: float = 1e-3,
    encoder_lr: float = 2e-4,
    device: torch.device = torch.device("cpu"),
    seed: int = 42,
):
    print(f"\n---> Evaluating CIFAR-10 Config: {config_name} (seed={seed})")
    set_seed(seed)

    encoder = copy.deepcopy(base_encoder).to(device)
    if adapt_encoder:
        encoder.unfreeze()
    else:
        encoder.freeze()

    router = DynamicRouter(input_dim=128, num_experts=1, top_k=1, temperature=1.0).to(
        device
    )
    initial_experts = [
        MLPExpert(input_dim=128, hidden_dim=64, num_classes=10, expert_id=0).to(device)
    ]
    moe_model = DynamicMoE(
        encoder=encoder,
        router=router,
        experts=initial_experts,
        use_ema_encoder=False,
    ).to(device)

    prototype_mem = PrototypeMemory(
        feature_dim=128,
        distance_threshold=0.5,
        ema_alpha=0.9,
        max_prototypes=60,
        store_raw=True,  # Store raw images for drift stabilization and gate checks
    )
    trigger = QuantitativeTrigger(
        alpha=1.0, beta=0.4, gamma=0.6, delta=0.5, threshold_tau=0.5
    )
    builder = ExpertBuilder(
        min_acc_threshold=0.60,
        max_proto_drop=2.0,
        max_proto_acc_drop=0.10,
        max_ece=0.35,
        distill_lambda=0.5,
        enable_gate=True,
    )

    trainer = ContinualTrainer(
        model=moe_model,
        prototype_memory=prototype_mem,
        trigger=trigger,
        builder=builder,
        lambda_r=lambda_r,
        lambda_e=lambda_e,
        lambda_enc=lambda_enc,
        lr=lr,
        encoder_lr=encoder_lr if adapt_encoder else None,
        max_experts=6,
        device=device,
    )

    evaluator = ContinualEvaluator(num_tasks=len(tasks), device=device)
    total_experts_added = 0
    total_gate_rejections = 0

    for t_idx, task in enumerate(tasks):
        hist = trainer.train_task(
            task_id=t_idx,
            train_loader=task.train_loader,
            val_loader=task.val_loader,
            epochs=epochs,
            enable_expansion=True,
        )
        total_experts_added += hist.get("experts_added", 0)
        total_gate_rejections += hist.get("gate_rejections", 0)
        accs = evaluator.evaluate_all_seen_tasks(moe_model, t_idx, tasks)
        print(f"  [Task {t_idx}] Accuracies: {[f'{a:.1%}' for a in accs]}")

    avg_acc = evaluator.compute_average_accuracy()
    forgetting = evaluator.compute_forgetting()
    bwt = evaluator.compute_backward_transfer()
    router_kl = ContinualEvaluator.compute_router_stability(
        moe_model, prototype_mem, device
    )

    print(
        f"  Result -> Acc: {avg_acc:.2%}, Forgetting: {forgetting:.2%}, Router KL: {router_kl:.4f}, Experts: {moe_model.num_experts}"
    )

    return {
        "name": config_name,
        "acc": avg_acc,
        "forgetting": forgetting,
        "bwt": bwt,
        "router_stability_kl": router_kl,
        "final_experts": moe_model.num_experts,
        "experts_added": total_experts_added,
        "gate_rejections": total_gate_rejections,
        "acc_matrix": evaluator.R.tolist(),
    }


def run_cifar_benchmark(
    epochs: int = 3,
    device_str: str = "auto",
    output_dir: str = "./results",
    seed: int = 42,
    max_train_samples: int = 1500,
):
    os.makedirs(output_dir, exist_ok=True)
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"[CIFAR-10 Benchmark] Compute device: {device}")

    tasks = get_split_cifar10_tasks(
        data_dir="./data",
        batch_size=64,
        val_split=0.1,
        seed=seed,
        max_train_samples_per_task=max_train_samples,
    )

    print("Pretraining shared ConvNet base encoder contrastively on CIFAR-10...")
    set_seed(seed)
    from torchvision import datasets, transforms

    cifar_train = datasets.CIFAR10(
        "./data", train=True, download=False, transform=transforms.ToTensor()
    )
    # Subsample for fast pretraining on CPU
    sub_indices = torch.randperm(len(cifar_train))[:10000]
    sub_cifar = torch.utils.data.Subset(cifar_train, sub_indices)
    unlabeled_loader = torch.utils.data.DataLoader(
        sub_cifar, batch_size=128, shuffle=True
    )
    base_encoder = SharedEncoder(input_dim=3072, output_dim=128, arch="conv").to(device)
    base_encoder.pretrain_contrastive(
        unlabeled_loader, device=device, epochs=2, lr=2e-3
    )
    base_encoder.freeze()
    print("  Base ConvNet encoder successfully pretrained.")

    configs = [
        ("Frozen Conv Encoder", dict(adapt_encoder=False, lambda_enc=0.0)),
        ("Unfrozen Conv WITHOUT L_enc_stab", dict(adapt_encoder=True, lambda_enc=0.0)),
        ("Unfrozen Conv WITH L_enc_stab", dict(adapt_encoder=True, lambda_enc=1.5)),
    ]

    results = {}
    table_data = []

    for name, kwargs in configs:
        res = run_single_cifar_config(
            tasks,
            copy.deepcopy(base_encoder),
            name,
            epochs=epochs,
            device=device,
            seed=seed,
            **kwargs,
        )
        results[name] = res
        table_data.append(
            [
                name,
                f"{res['acc']:.2%}",
                f"{res['forgetting']:.2%}",
                f"{res['bwt']:.2%}",
                f"{res['router_stability_kl']:.4f}",
                str(res["final_experts"]),
            ]
        )

    headers = [
        "CIFAR-10 Configuration",
        "Avg Acc (↑)",
        "Forgetting (↓)",
        "BWT (↑)",
        "Router KL (↓)",
        "Experts",
    ]
    print("\n" + "=" * 80)
    print("CONTINUAL LEARNING ON SPLIT-CIFAR-10 (CONV ENCODER)")
    print("=" * 80)
    print(tabulate(table_data, headers=headers, tablefmt="github"))
    print("=" * 80)

    out_file = os.path.join(output_dir, "cifar10_unfrozen_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved CIFAR-10 results to {out_file}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_samples", type=int, default=1500)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_cifar_benchmark(
        epochs=args.epochs,
        device_str=args.device,
        output_dir=args.output_dir,
        seed=args.seed,
        max_train_samples=args.max_samples,
    )
