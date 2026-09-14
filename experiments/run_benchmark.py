"""
Benchmark Runner for Continual Learning on Split-MNIST.
Compares:
1. Naive Sequential Fine-tuning
2. Elastic Weight Consolidation (EWC)
3. Experience Replay (Buffer Replay)
4. Standard MoE Fine-tuning (Fixed 4 experts, no prototype stability)
5. PAL-MoE (Proposed: Prototype-Anchored Lifelong Mixture of Experts)
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import copy
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
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
from pal_moe.adaptation.ttt import ContinualTrainer, TestTimeAdapter
from pal_moe.baselines.naive import NaiveFineTuning
from pal_moe.baselines.ewc import EWC
from pal_moe.baselines.replay import ReplayTrainer
from pal_moe.baselines.standard_moe import StandardMoE
from pal_moe.evaluation.metrics import ContinualEvaluator



def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_benchmark(epochs_per_task: int = 3, device_str: str = "auto", output_dir: str = "./results", dataset: str = "mnist"):
    os.makedirs(output_dir, exist_ok=True)

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"[Benchmark] Using compute device: {device}")


    if dataset == "cifar10":
        from pal_moe.data.split_cifar import get_split_cifar10_tasks
        tasks = get_split_cifar10_tasks(data_dir="./data", batch_size=128, val_split=0.1, seed=42)
        num_tasks = len(tasks)
        input_dim = 3072
        mnist_train = datasets.CIFAR10("./data", train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]))
        unlabeled_loader = torch.utils.data.DataLoader(mnist_train, batch_size=256, shuffle=True)
        base_encoder = SharedEncoder(input_dim=input_dim, hidden_dims=None, output_dim=128, arch="conv").to(device)
        base_encoder.pretrain_contrastive(unlabeled_loader, device=device, epochs=50)
        # unfreezing happens implicitly
    else:
        tasks = get_split_mnist_tasks(data_dir="./data", batch_size=128, val_split=0.1, seed=42)
        num_tasks = len(tasks)
        input_dim = 784
        mnist_train = datasets.MNIST("./data", train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]))
        unlabeled_loader = torch.utils.data.DataLoader(mnist_train, batch_size=256, shuffle=True)
        base_encoder = SharedEncoder(input_dim=input_dim, hidden_dims=(256, 128), output_dim=128, arch="mlp").to(device)
        base_encoder.pretrain_unsupervised(unlabeled_loader, device=device, epochs=1)
        base_encoder.freeze()

    print("  Shared Encoder successfully pretrained and frozen for all benchmark models.")

    results = {}

    # -------------------------------------------------------------
    # 1. Baseline: Naive Sequential Fine-tuning
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Running Proposed: PAL-MoE (Prototype-Anchored Lifelong Mixture of Experts)")
    print("=" * 60)
    set_seed(42)

    dyn_encoder = copy.deepcopy(base_encoder)
    dyn_router = DynamicRouter(input_dim=128, num_experts=1, top_k=1, temperature=1.0).to(device)
    initial_experts = [
        MLPExpert(input_dim=128, hidden_dim=256, num_classes=10, expert_id=0).to(device)
    ]
    moe_model = DynamicMoE(
        encoder=dyn_encoder,
        router=dyn_router,
        experts=initial_experts,
        use_ema_encoder=False,
    ).to(device)

    prototype_mem = PrototypeMemory(
        feature_dim=128,
        distance_threshold=0.5,
        ema_alpha=0.9,
        max_prototypes=250,
        store_raw=False,
    )
    trigger = QuantitativeTrigger(
        alpha=1.0,
        beta=0.4,
        gamma=0.6,
        delta=0.5,
        threshold_tau=0.5,
    )
    builder = ExpertBuilder(
        min_acc_threshold=0.45 if dataset == "cifar10" else 0.60,
        max_proto_drop=999.0 if dataset == "cifar10" else 2.0,
        max_ece=999.0,
        distill_lambda=0.5,
        max_proto_acc_drop=999.0,
    )
    trainer = ContinualTrainer(
        model=moe_model,
        prototype_memory=prototype_mem,
        trigger=trigger,
        builder=builder,
        lambda_r=0.5,
        lambda_e=2.5,
        lambda_enc=0.5 if dataset == "cifar10" else 0.0,
        encoder_lr=1e-4 if dataset == "cifar10" else None,
        lr=1e-3,
        max_experts=6,
        device=device,
    )
    evaluator_dynamic = ContinualEvaluator(num_tasks=num_tasks, device=device)

    for t_idx, task in enumerate(tasks):
        print(f"  Training Task {t_idx} (classes {task.classes})... Experts before: {moe_model.num_experts}")
        hist = trainer.train_task(
            task_id=t_idx,
            train_loader=task.train_loader,
            val_loader=task.val_loader,
            epochs=epochs_per_task,
            enable_expansion=True,
        )
        print(f"    Triggers: {hist['trigger_events']} | Experts added: {hist['experts_added']} | Gate rejects: {hist['gate_rejections']} | Total experts: {moe_model.num_experts}")
        accs = evaluator_dynamic.evaluate_all_seen_tasks(moe_model, t_idx, tasks)
        print(f"  Accuracies after Task {t_idx}: {[f'{a:.1%}' for a in accs]}")

    # Compute advanced stability and information metrics
    router_kl = ContinualEvaluator.compute_router_stability(moe_model, prototype_mem, device)
    mi_dyn, util_dyn = ContinualEvaluator.compute_expert_specialization_and_utilization(
        moe_model, tasks, device
    )
    footprint = prototype_mem.estimate_memory_footprint()
    print(f"  Prototype Memory Footprint: {footprint['num_prototypes']} prototypes, {footprint['total_elements']} floats ({footprint['size_kb']:.1f} KB)")

    results["PAL-MoE (Ours)"] = {
        "acc": evaluator_dynamic.compute_average_accuracy(),
        "forgetting": evaluator_dynamic.compute_forgetting(),
        "bwt": evaluator_dynamic.compute_backward_transfer(),
        "router_stability_kl": router_kl,
        "specialization_mi": mi_dyn,
        "utilization": util_dyn,
        "final_experts": moe_model.num_experts,
        "prototype_elements": footprint["total_elements"],
        "acc_matrix": evaluator_dynamic.R.tolist(),
    }

    # -------------------------------------------------------------
    # 8. Proposed Extension: PAL-MoE + Replay (Hybrid, P=250)
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Running Method 8: PAL-MoE + Replay (Hybrid, P=250 Exemplars)")
    print("=" * 60)
    set_seed(42)
    hyb_encoder = copy.deepcopy(base_encoder)
    hyb_router = DynamicRouter(input_dim=128, num_experts=1, top_k=1, temperature=1.0).to(device)
    initial_experts_hyb = [
        MLPExpert(input_dim=128, hidden_dim=256, num_classes=10, expert_id=0).to(device)
    ]
    moe_hyb = DynamicMoE(
        encoder=hyb_encoder,
        router=hyb_router,
        experts=initial_experts_hyb,
        use_ema_encoder=False,
    ).to(device)
    prototype_mem_hyb = PrototypeMemory(
        feature_dim=128,
        distance_threshold=0.5,
        ema_alpha=0.9,
        max_prototypes=250,
        store_raw=True,
    )
    trigger_hyb = QuantitativeTrigger(
        alpha=1.0, beta=0.4, gamma=0.6, delta=0.5, threshold_tau=0.5
    )
    builder_hyb = ExpertBuilder(
        min_acc_threshold=0.45 if dataset == "cifar10" else 0.60,
        max_proto_drop=999.0 if dataset == "cifar10" else 2.0,
        max_ece=999.0,
        distill_lambda=0.5,
        max_proto_acc_drop=999.0,
    )
    trainer_hyb = ContinualTrainer(
        model=moe_hyb,
        prototype_memory=prototype_mem_hyb,
        trigger=trigger_hyb,
        builder=builder_hyb,
        lambda_r=0.5,
        lambda_e=2.5,
        lambda_enc=0.5 if dataset == "cifar10" else 0.0,
        encoder_lr=1e-4 if dataset == "cifar10" else None,
        replay_exemplars=True,
        lambda_replay=1.0,
        lr=1e-3,
        max_experts=6,
        device=device,
    )
    evaluator_hyb = ContinualEvaluator(num_tasks=num_tasks, device=device)

    for t_idx, task in enumerate(tasks):
        print(f"  Training Task {t_idx} (classes {task.classes})... Experts before: {moe_hyb.num_experts}")
        hist = trainer_hyb.train_task(
            task_id=t_idx,
            train_loader=task.train_loader,
            val_loader=task.val_loader,
            epochs=epochs_per_task,
            enable_expansion=True,
        )
        print(f"    Triggers: {hist['trigger_events']} | Experts added: {hist['experts_added']} | Gate rejects: {hist['gate_rejections']} | Total experts: {moe_hyb.num_experts}")
        accs = evaluator_hyb.evaluate_all_seen_tasks(moe_hyb, t_idx, tasks)
        print(f"  Accuracies after Task {t_idx}: {[f'{a:.1%}' for a in accs]}")

    router_kl_hyb = ContinualEvaluator.compute_router_stability(moe_hyb, prototype_mem_hyb, device)
    mi_hyb, util_hyb = ContinualEvaluator.compute_expert_specialization_and_utilization(
        moe_hyb, tasks, device
    )
    results["PAL-MoE + Replay (Hybrid, P=250)"] = {
        "acc": evaluator_hyb.compute_average_accuracy(),
        "forgetting": evaluator_hyb.compute_forgetting(),
        "bwt": evaluator_hyb.compute_backward_transfer(),
        "router_stability_kl": router_kl_hyb,
        "specialization_mi": mi_hyb,
        "utilization": util_hyb,
        "final_experts": moe_hyb.num_experts,
        "prototype_elements": prototype_mem_hyb.estimate_memory_footprint()["total_elements"],
        "acc_matrix": evaluator_hyb.R.tolist(),
    }

    # -------------------------------------------------------------
    # Summary Table
    # -------------------------------------------------------------
    table_data = []
    headers = ["Method", "Avg Acc (↑)", "Forgetting (↓)", "BWT (↑)", "Router KL (↓)", "Spec. MI (↑)", "Util. Entropy", "Experts"]

    for name, m in results.items():
        table_data.append([
            name,
            f"{m['acc']:.2%}",
            f"{m['forgetting']:.2%}",
            f"{m['bwt']:.2%}",
            f"{m['router_stability_kl']:.4f}" if not np.isnan(m['router_stability_kl']) else "-",
            f"{m['specialization_mi']:.3f}" if not np.isnan(m['specialization_mi']) else "-",
            f"{m['utilization']:.3f}" if not np.isnan(m['utilization']) else "-",
            str(m['final_experts']),
        ])

    print("\n" + "=" * 80)
    print("FINAL CONTINUAL LEARNING BENCHMARK RESULTS (Split-MNIST, 5 Tasks)")
    print("=" * 80)
    print(tabulate(table_data, headers=headers, tablefmt="github"))
    print("=" * 80)

    # Save to json
    results_path = os.path.join(output_dir, "benchmark_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved benchmark results to {results_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3, help="Epochs per task")
    parser.add_argument("--device", type=str, default="auto", help="Compute device: cuda or cpu")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--dataset", type=str, default="mnist")
    args = parser.parse_args()

    run_benchmark(epochs_per_task=args.epochs, device_str=args.device, output_dir=args.output_dir, dataset=args.dataset)
