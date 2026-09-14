"""
Ablation Study Runner for PAL-MoE (Prototype-Anchored Lifelong Mixture of Experts).
Systematically verifies the necessity of each component in Section 10:
1. Full Proposed PAL-MoE
2. Ablation A: No Stability Loss (lambda_r = 0, lambda_e = 0)
3. Ablation B: No Expert Output Anchor (lambda_e = 0, lambda_r = 2.5)
4. Ablation C: Random Expert Init (No Function-Preserving Net2Net clone)
5. Ablation D: No Validation Gate (unconditional candidate admission)
6. Ablation E: Top-2 Routing vs Top-1 Routing
7. Ablation F: EMA Encoder vs Frozen Encoder
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import copy
import json
import argparse
from typing import Optional, List
import numpy as np
import torch
from torchvision import datasets, transforms
from tabulate import tabulate

from pal_moe.data.split_mnist import get_split_mnist_tasks
from pal_moe.models.encoder import SharedEncoder
from pal_moe.models.router import DynamicRouter
from pal_moe.models.expert import MLPExpert
from pal_moe.models.moe import DynamicMoE, PALMoE
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


def run_single_config(
    tasks,
    base_encoder: SharedEncoder,
    config_name: str,
    dataset: str = "mnist",
    lambda_r: float = 0.5,
    lambda_e: float = 2.5,
    use_function_preserving: bool = True,
    use_validation_gate: bool = True,
    use_ema_encoder: bool = False,
    adapt_encoder: bool = False,
    top_k: int = 1,
    epochs: int = 3,
    seed: int = 42,
    device: torch.device = torch.device("cpu"),
):
    print(f"\n---> Evaluating Config: {config_name} (seed={seed})")
    set_seed(seed)

    encoder = copy.deepcopy(base_encoder)
    if adapt_encoder:
        encoder.unfreeze()
    else:
        encoder.freeze()

    router = DynamicRouter(input_dim=128, num_experts=1, top_k=top_k, temperature=1.0).to(device)
    initial_experts = [
        MLPExpert(input_dim=128, hidden_dim=256, num_classes=10, expert_id=0).to(device)
    ]
    moe_model = DynamicMoE(
        encoder=encoder,
        router=router,
        experts=initial_experts,
        use_ema_encoder=use_ema_encoder,
        ema_decay=0.95,
    ).to(device)

    prototype_mem = PrototypeMemory(
        feature_dim=128,
        distance_threshold=0.5,
        ema_alpha=0.9,
        max_prototypes=60,
        store_raw=adapt_encoder,
    )
    trigger = QuantitativeTrigger(
        alpha=1.0, beta=0.4, gamma=0.6, delta=0.5, threshold_tau=0.5
    )
    if dataset == "cifar10":
        builder = ExpertBuilder(
            min_acc_threshold=0.0 if not use_validation_gate else 0.45,
            max_proto_drop=999.0,
            max_proto_acc_drop=999.0,
            max_ece=999.0,
            distill_lambda=0.5,
            enable_gate=use_validation_gate,
        )
        lambda_enc = 0.5
        encoder_lr = 1e-4 if adapt_encoder else None
    else:
        builder = ExpertBuilder(
            min_acc_threshold=0.0 if not use_validation_gate else 0.60,
            max_proto_drop=999.0,
            max_proto_acc_drop=999.0,
            max_ece=999.0,
            distill_lambda=0.5,
            enable_gate=use_validation_gate,
        )
        lambda_enc = 0.0
        encoder_lr = 1e-4 if adapt_encoder else None

    trainer = ContinualTrainer(
        model=moe_model,
        prototype_memory=prototype_mem,
        trigger=trigger,
        builder=builder,
        lambda_r=lambda_r,
        lambda_e=lambda_e,
        lambda_enc=lambda_enc,
        lr=1e-3,
        encoder_lr=encoder_lr,
        max_experts=6,
        device=device,
    )

    if not use_function_preserving:
        def random_create(parent_expert, new_expert_id, creation_task):
            return MLPExpert(input_dim=128, hidden_dim=256, num_classes=10, expert_id=new_expert_id).to(device)
        builder.create_candidate_from_parent = random_create

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

    avg_acc = evaluator.compute_average_accuracy()
    forgetting = evaluator.compute_forgetting()
    bwt = evaluator.compute_backward_transfer()
    router_kl = ContinualEvaluator.compute_router_stability(moe_model, prototype_mem, device)

    print(f"  Result -> Acc: {avg_acc:.2%}, Forgetting: {forgetting:.2%}, Router KL: {router_kl:.4f}, Experts: {moe_model.num_experts}, Rejections: {total_gate_rejections}")

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


def run_all_ablations(
    epochs: int = 3,
    seeds: list[int] = [42],
    selected_configs: Optional[list[str]] = None,
    device_str: str = "auto",
    output_dir: str = "./results",
    dataset: str = "mnist",
):
    os.makedirs(output_dir, exist_ok=True)
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"[Ablation] Using compute device: {device} | Seeds: {seeds} | Dataset: {dataset.upper()}")

    configs = [
        ("Full Proposed PAL-MoE", dict(lambda_r=0.5, lambda_e=2.5, use_function_preserving=True, use_validation_gate=True, use_ema_encoder=False, top_k=1)),
        ("No Stability Loss (lr=0, le=0)", dict(lambda_r=0.0, lambda_e=0.0, use_function_preserving=True, use_validation_gate=True, use_ema_encoder=False, top_k=1)),
        ("No Expert Anchor (lr=0.5, le=0)", dict(lambda_r=0.5, lambda_e=0.0, use_function_preserving=True, use_validation_gate=True, use_ema_encoder=False, top_k=1)),
        ("No Router Stability (lr=0, le=2.5)", dict(lambda_r=0.0, lambda_e=2.5, use_function_preserving=True, use_validation_gate=True, use_ema_encoder=False, top_k=1)),
        ("Random Expert Init (No Function-Preserving)", dict(lambda_r=0.5, lambda_e=2.5, use_function_preserving=False, use_validation_gate=True, use_ema_encoder=False, top_k=1)),
        ("No Validation Gate", dict(lambda_r=0.5, lambda_e=2.5, use_function_preserving=True, use_validation_gate=False, use_ema_encoder=False, top_k=1)),
        ("Top-2 Routing", dict(lambda_r=0.5, lambda_e=2.5, use_function_preserving=True, use_validation_gate=True, use_ema_encoder=False, top_k=2)),
        ("Online Encoder (No EMA)", dict(lambda_r=0.5, lambda_e=2.5, use_function_preserving=True, use_validation_gate=True, use_ema_encoder=False, adapt_encoder=True, top_k=1)),
        ("EMA Encoder (Adaptive)", dict(lambda_r=0.5, lambda_e=2.5, use_function_preserving=True, use_validation_gate=True, use_ema_encoder=True, adapt_encoder=True, top_k=1)),
    ]

    if selected_configs:
        configs = [c for c in configs if c[0] in selected_configs]

    all_results = {}
    table_data = []

    for name, kwargs in configs:
        seed_runs = []
        for s in seeds:
            set_seed(s)
            if dataset == "cifar10":
                from pal_moe.data.split_cifar import get_split_cifar10_tasks
                tasks = get_split_cifar10_tasks(data_dir="./data", batch_size=128, val_split=0.1, seed=s)
                cifar_train = datasets.CIFAR10("./data", train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]))
                unlabeled_loader = torch.utils.data.DataLoader(cifar_train, batch_size=256, shuffle=True)
                base_encoder = SharedEncoder(input_dim=3072, hidden_dims=None, output_dim=128, arch="conv").to(device)
            else:
                from pal_moe.data.split_mnist import get_split_mnist_tasks
                tasks = get_split_mnist_tasks(data_dir="./data", batch_size=128, val_split=0.1, seed=s)
                mnist_train = datasets.MNIST("./data", train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]))
                unlabeled_loader = torch.utils.data.DataLoader(mnist_train, batch_size=256, shuffle=True)
                base_encoder = SharedEncoder(input_dim=784, hidden_dims=(256, 128), output_dim=128, arch="mlp").to(device)
            
            if dataset == "cifar10":
                base_encoder.pretrain_contrastive(unlabeled_loader, device=device, epochs=50)
                # DO NOT freeze for CIFAR-10
            else:
                base_encoder.pretrain_unsupervised(unlabeled_loader, device=device, epochs=1)
                base_encoder.freeze()

            res = run_single_config(tasks, base_encoder, name, epochs=epochs, seed=s, device=device, dataset=dataset, **kwargs)
            seed_runs.append(res)

        accs = [r["acc"] for r in seed_runs]
        forg = [r["forgetting"] for r in seed_runs]
        bwts = [r["bwt"] for r in seed_runs]
        kls = [r["router_stability_kl"] for r in seed_runs if not np.isnan(r["router_stability_kl"])]
        exps = [r["final_experts"] for r in seed_runs]
        rejs = [r["gate_rejections"] for r in seed_runs]

        mean_acc, std_acc = float(np.mean(accs)), float(np.std(accs))
        mean_forg, std_forg = float(np.mean(forg)), float(np.std(forg))
        mean_bwt, std_bwt = float(np.mean(bwts)), float(np.std(bwts))
        mean_kl = float(np.mean(kls)) if kls else float("nan")
        std_kl = float(np.std(kls)) if len(kls) > 1 else 0.0

        if len(seeds) > 1:
            table_data.append([
                name,
                f"{mean_acc:.2%} ± {std_acc:.2%}",
                f"{mean_forg:.2%} ± {std_forg:.2%}",
                f"{mean_bwt:.2%} ± {std_bwt:.2%}",
                f"{mean_kl:.4f} ± {std_kl:.4f}" if not np.isnan(mean_kl) else "-",
                f"{np.mean(exps):.1f}",
                f"{np.mean(rejs):.1f}",
            ])
        else:
            table_data.append([
                name,
                f"{mean_acc:.2%}",
                f"{mean_forg:.2%}",
                f"{mean_bwt:.2%}",
                f"{mean_kl:.4f}" if not np.isnan(mean_kl) else "-",
                str(int(np.mean(exps))),
                str(int(np.mean(rejs))),
            ])

        all_results[name] = {
            "name": name,
            "acc": mean_acc,
            "acc_std": std_acc,
            "forgetting": mean_forg,
            "forgetting_std": std_forg,
            "bwt": mean_bwt,
            "bwt_std": std_bwt,
            "router_stability_kl": mean_kl,
            "router_stability_kl_std": std_kl,
            "final_experts": int(round(np.mean(exps))),
            "gate_rejections": int(round(np.mean(rejs))),
            "seed_runs": seed_runs,
            "acc_matrix": seed_runs[0]["acc_matrix"],
        }

    headers = ["Ablation Configuration", "Avg Acc (↑)", "Forgetting (↓)", "BWT (↑)", "Router KL (↓)", "Experts", "Rejections"]
    print("\n" + "=" * 80)
    print(f"ABLATION STUDY RESULTS (Split-MNIST, 5 Tasks, {len(seeds)} Seeds)")
    print("=" * 80)
    print(tabulate(table_data, headers=headers, tablefmt="github"))
    print("=" * 80)

    out_file = os.path.join(output_dir, "ablation_results.json")
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved ablation results to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42], help="List of seeds to evaluate (e.g. 42 43 44)")
    parser.add_argument("--configs", type=str, nargs="+", default=None, help="Specific configs to run")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar10"])
    args = parser.parse_args()

    run_all_ablations(
        epochs=args.epochs,
        seeds=args.seeds,
        selected_configs=args.configs,
        device_str=args.device,
        output_dir=args.output_dir,
        dataset=args.dataset,
    )
