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


def run_benchmark(epochs_per_task: int = 3, device_str: str = "auto", output_dir: str = "./results"):
    os.makedirs(output_dir, exist_ok=True)

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"[Benchmark] Using compute device: {device}")

    # Load 5 Split-MNIST tasks
    tasks = get_split_mnist_tasks(data_dir="./data", batch_size=128, val_split=0.1, seed=42)
    num_tasks = len(tasks)

    # -------------------------------------------------------------
    # Step 0: Shared Domain Encoder Pretraining (Unsupervised, 1 epoch)
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 0: Pretraining Task-Agnostic Shared Encoder (Unsupervised)")
    print("=" * 60)
    set_seed(42)
    mnist_train = datasets.MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
    unlabeled_loader = torch.utils.data.DataLoader(mnist_train, batch_size=256, shuffle=True)
    base_encoder = SharedEncoder(input_dim=784, hidden_dims=(256, 128), output_dim=128, arch="mlp").to(device)
    base_encoder.pretrain_unsupervised(unlabeled_loader, device=device, epochs=1)
    base_encoder.freeze()
    print("  Shared Encoder successfully pretrained and frozen for all benchmark models.")

    results = {}

    # -------------------------------------------------------------
    # 1. Baseline: Naive Sequential Fine-tuning
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Running Baseline 1: Naive Sequential Fine-tuning")
    print("=" * 60)
    set_seed(42)
    naive_net = nn.Sequential(
        copy.deepcopy(base_encoder),
        MLPExpert(input_dim=128, hidden_dim=64, num_classes=10, expert_id=0),
    ).to(device)
    naive_trainer = NaiveFineTuning(naive_net, lr=1e-3, device=device)
    evaluator_naive = ContinualEvaluator(num_tasks=num_tasks, device=device)

    for t_idx, task in enumerate(tasks):
        print(f"  Training Task {t_idx} (classes {task.classes})...")
        naive_trainer.train_task(t_idx, task.train_loader, epochs=epochs_per_task)
        accs = evaluator_naive.evaluate_all_seen_tasks(naive_net, t_idx, tasks)
        print(f"  Accuracies after Task {t_idx}: {[f'{a:.1%}' for a in accs]}")

    results["Naive Fine-tuning"] = {
        "acc": evaluator_naive.compute_average_accuracy(),
        "forgetting": evaluator_naive.compute_forgetting(),
        "bwt": evaluator_naive.compute_backward_transfer(),
        "router_stability_kl": float("nan"),
        "specialization_mi": float("nan"),
        "utilization": float("nan"),
        "final_experts": 1,
        "acc_matrix": evaluator_naive.R.tolist(),
    }

    # -------------------------------------------------------------
    # 2. Baseline: Elastic Weight Consolidation (EWC)
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Running Baseline 2: Elastic Weight Consolidation (EWC)")
    print("=" * 60)
    set_seed(42)
    ewc_net = nn.Sequential(
        copy.deepcopy(base_encoder),
        MLPExpert(input_dim=128, hidden_dim=64, num_classes=10, expert_id=0),
    ).to(device)
    ewc_trainer = EWC(ewc_net, ewc_lambda=1000.0, lr=1e-3, device=device)
    evaluator_ewc = ContinualEvaluator(num_tasks=num_tasks, device=device)

    for t_idx, task in enumerate(tasks):
        print(f"  Training Task {t_idx} (classes {task.classes})...")
        ewc_trainer.train_task(t_idx, task.train_loader, epochs=epochs_per_task)
        accs = evaluator_ewc.evaluate_all_seen_tasks(ewc_net, t_idx, tasks)
        print(f"  Accuracies after Task {t_idx}: {[f'{a:.1%}' for a in accs]}")

    results["EWC"] = {
        "acc": evaluator_ewc.compute_average_accuracy(),
        "forgetting": evaluator_ewc.compute_forgetting(),
        "bwt": evaluator_ewc.compute_backward_transfer(),
        "router_stability_kl": float("nan"),
        "specialization_mi": float("nan"),
        "utilization": float("nan"),
        "final_experts": 1,
        "acc_matrix": evaluator_ewc.R.tolist(),
    }

    # -------------------------------------------------------------
    # 3. Baseline: Experience Replay (Budgeted P=60, matching prototype memory)
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Running Baseline 3: Experience Replay (Budgeted P=60)")
    print("=" * 60)
    set_seed(42)
    replay_net_budget = nn.Sequential(
        copy.deepcopy(base_encoder),
        MLPExpert(input_dim=128, hidden_dim=64, num_classes=10, expert_id=0),
    ).to(device)
    replay_trainer_budget = ReplayTrainer(replay_net_budget, buffer_size=60, lr=1e-3, device=device)
    evaluator_replay_budget = ContinualEvaluator(num_tasks=num_tasks, device=device)

    for t_idx, task in enumerate(tasks):
        print(f"  Training Task {t_idx} (classes {task.classes})...")
        replay_trainer_budget.train_task(t_idx, task.train_loader, epochs=epochs_per_task)
        accs = evaluator_replay_budget.evaluate_all_seen_tasks(replay_net_budget, t_idx, tasks)
        print(f"  Accuracies after Task {t_idx}: {[f'{a:.1%}' for a in accs]}")

    results["Replay (P=60)"] = {
        "acc": evaluator_replay_budget.compute_average_accuracy(),
        "forgetting": evaluator_replay_budget.compute_forgetting(),
        "bwt": evaluator_replay_budget.compute_backward_transfer(),
        "router_stability_kl": float("nan"),
        "specialization_mi": float("nan"),
        "utilization": float("nan"),
        "final_experts": 1,
        "acc_matrix": evaluator_replay_budget.R.tolist(),
    }

    # -------------------------------------------------------------
    # 4. Baseline: Experience Replay (Buffer=250)
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Running Baseline 4: Experience Replay (Buffer=250)")
    print("=" * 60)
    set_seed(42)
    replay_net = nn.Sequential(
        copy.deepcopy(base_encoder),
        MLPExpert(input_dim=128, hidden_dim=64, num_classes=10, expert_id=0),
    ).to(device)
    replay_trainer = ReplayTrainer(replay_net, buffer_size=250, lr=1e-3, device=device)
    evaluator_replay = ContinualEvaluator(num_tasks=num_tasks, device=device)

    for t_idx, task in enumerate(tasks):
        print(f"  Training Task {t_idx} (classes {task.classes})...")
        replay_trainer.train_task(t_idx, task.train_loader, epochs=epochs_per_task)
        accs = evaluator_replay.evaluate_all_seen_tasks(replay_net, t_idx, tasks)
        print(f"  Accuracies after Task {t_idx}: {[f'{a:.1%}' for a in accs]}")

    results["Experience Replay (Buffer=250)"] = {
        "acc": evaluator_replay.compute_average_accuracy(),
        "forgetting": evaluator_replay.compute_forgetting(),
        "bwt": evaluator_replay.compute_backward_transfer(),
        "router_stability_kl": float("nan"),
        "specialization_mi": float("nan"),
        "utilization": float("nan"),
        "final_experts": 1,
        "acc_matrix": evaluator_replay.R.tolist(),
    }

    # -------------------------------------------------------------
    # 4. Baseline: Standard MoE Fine-tuning (Fixed 4 Experts)
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Running Baseline 4: Standard MoE Fine-tuning (Fixed 4 Experts, No Stability)")
    print("=" * 60)
    set_seed(42)
    std_encoder = copy.deepcopy(base_encoder)
    std_router = DynamicRouter(input_dim=128, num_experts=4, top_k=1).to(device)
    std_experts = [
        MLPExpert(input_dim=128, hidden_dim=64, num_classes=10, expert_id=i).to(device)
        for i in range(4)
    ]
    std_moe = DynamicMoE(encoder=std_encoder, router=std_router, experts=std_experts, use_ema_encoder=False).to(device)
    opt_std = torch.optim.Adam(std_moe.parameters(), lr=1e-3)
    evaluator_std_moe = ContinualEvaluator(num_tasks=num_tasks, device=device)

    for t_idx, task in enumerate(tasks):
        print(f"  Training Task {t_idx} (classes {task.classes})...")
        std_moe.train()
        for epoch in range(epochs_per_task):
            for x, y in task.train_loader:
                x, y = x.to(device), y.to(device)
                opt_std.zero_grad()
                logits = std_moe(x)
                loss = nn.functional.cross_entropy(logits, y)
                loss.backward()
                opt_std.step()

        accs = evaluator_std_moe.evaluate_all_seen_tasks(std_moe, t_idx, tasks)
        print(f"  Accuracies after Task {t_idx}: {[f'{a:.1%}' for a in accs]}")

    mi_std, util_std = ContinualEvaluator.compute_expert_specialization_and_utilization(
        std_moe, tasks, device
    )

    results["Standard MoE"] = {
        "acc": evaluator_std_moe.compute_average_accuracy(),
        "forgetting": evaluator_std_moe.compute_forgetting(),
        "bwt": evaluator_std_moe.compute_backward_transfer(),
        "router_stability_kl": float("nan"),
        "specialization_mi": mi_std,
        "utilization": util_std,
        "final_experts": 4,
        "acc_matrix": evaluator_std_moe.R.tolist(),
    }

    # -------------------------------------------------------------
    # 5. Proposed: PAL-MoE
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Running Proposed: PAL-MoE (Prototype-Anchored Lifelong Mixture of Experts)")
    print("=" * 60)
    set_seed(42)

    dyn_encoder = copy.deepcopy(base_encoder)
    dyn_router = DynamicRouter(input_dim=128, num_experts=1, top_k=1, temperature=1.0).to(device)
    initial_experts = [
        MLPExpert(input_dim=128, hidden_dim=64, num_classes=10, expert_id=0).to(device)
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
        max_prototypes=60,
    )
    trigger = QuantitativeTrigger(
        alpha=1.0,
        beta=0.4,
        gamma=0.6,
        delta=0.5,
        threshold_tau=0.5,
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

    results["PAL-MoE (Ours)"] = {
        "acc": evaluator_dynamic.compute_average_accuracy(),
        "forgetting": evaluator_dynamic.compute_forgetting(),
        "bwt": evaluator_dynamic.compute_backward_transfer(),
        "router_stability_kl": router_kl,
        "specialization_mi": mi_dyn,
        "utilization": util_dyn,
        "final_experts": moe_model.num_experts,
        "acc_matrix": evaluator_dynamic.R.tolist(),
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
    args = parser.parse_args()

    run_benchmark(epochs_per_task=args.epochs, device_str=args.device, output_dir=args.output_dir)
