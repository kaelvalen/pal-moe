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
    lambda_r: float = 0.5,
    lambda_e: float = 2.5,
    use_function_preserving: bool = True,
    use_validation_gate: bool = True,
    use_ema_encoder: bool = False,
    adapt_encoder: bool = False,
    top_k: int = 1,
    epochs: int = 3,
    device: torch.device = torch.device("cpu"),
):
    print(f"\n---> Evaluating Config: {config_name}")
    set_seed(42)

    encoder = copy.deepcopy(base_encoder)
    if adapt_encoder:
        encoder.unfreeze()
    else:
        encoder.freeze()

    router = DynamicRouter(input_dim=128, num_experts=1, top_k=top_k, temperature=1.0).to(device)
    initial_experts = [
        MLPExpert(input_dim=128, hidden_dim=64, num_classes=10, expert_id=0).to(device)
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
    builder = ExpertBuilder(
        min_acc_threshold=0.0 if not use_validation_gate else 0.60,
        max_proto_drop=999.0 if not use_validation_gate else 2.0,
        max_proto_acc_drop=999.0 if not use_validation_gate else 0.05,
        max_ece=999.0 if not use_validation_gate else 0.35,
        distill_lambda=0.5,
        enable_gate=use_validation_gate,
    )

    trainer = ContinualTrainer(
        model=moe_model,
        prototype_memory=prototype_mem,
        trigger=trigger,
        builder=builder,
        lambda_r=lambda_r,
        lambda_e=lambda_e,
        lr=1e-3,
        encoder_lr=1e-4 if adapt_encoder else None,
        max_experts=6,
        device=device,
    )

    if not use_function_preserving:
        def random_create(parent_expert, new_expert_id, creation_task):
            return MLPExpert(input_dim=128, hidden_dim=64, num_classes=10, expert_id=new_expert_id).to(device)
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


def run_all_ablations(epochs: int = 3, device_str: str = "auto", output_dir: str = "./results"):
    os.makedirs(output_dir, exist_ok=True)
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"[Ablation] Using compute device: {device}")

    tasks = get_split_mnist_tasks(data_dir="./data", batch_size=128, val_split=0.1, seed=42)

    # Pretrain shared encoder
    print("Pretraining shared base encoder...")
    set_seed(42)
    mnist_train = datasets.MNIST("./data", train=True, download=True, transform=transforms.ToTensor())
    unlabeled_loader = torch.utils.data.DataLoader(mnist_train, batch_size=256, shuffle=True)
    base_encoder = SharedEncoder(input_dim=784, hidden_dims=(256, 128), output_dim=128, arch="mlp").to(device)
    base_encoder.pretrain_unsupervised(unlabeled_loader, device=device, epochs=1)
    base_encoder.freeze()

    configs = [
        ("Full Proposed PAL-MoE", dict(lambda_r=0.5, lambda_e=2.5, use_function_preserving=True, use_validation_gate=True, use_ema_encoder=False, top_k=1)),
        ("No Stability Loss (lr=0, le=0)", dict(lambda_r=0.0, lambda_e=0.0, use_function_preserving=True, use_validation_gate=True, use_ema_encoder=False, top_k=1)),
        ("No Expert Anchor (lr=0.5, le=0)", dict(lambda_r=0.5, lambda_e=0.0, use_function_preserving=True, use_validation_gate=True, use_ema_encoder=False, top_k=1)),
        ("Random Expert Init (No Function-Preserving)", dict(lambda_r=0.5, lambda_e=2.5, use_function_preserving=False, use_validation_gate=True, use_ema_encoder=False, top_k=1)),
        ("No Validation Gate", dict(lambda_r=0.5, lambda_e=2.5, use_function_preserving=True, use_validation_gate=False, use_ema_encoder=False, top_k=1)),
        ("Top-2 Routing", dict(lambda_r=0.5, lambda_e=2.5, use_function_preserving=True, use_validation_gate=True, use_ema_encoder=False, top_k=2)),
        ("Online Encoder (No EMA)", dict(lambda_r=0.5, lambda_e=2.5, use_function_preserving=True, use_validation_gate=True, use_ema_encoder=False, adapt_encoder=True, top_k=1)),
        ("EMA Encoder (Adaptive)", dict(lambda_r=0.5, lambda_e=2.5, use_function_preserving=True, use_validation_gate=True, use_ema_encoder=True, adapt_encoder=True, top_k=1)),
    ]

    all_results = {}
    table_data = []

    for name, kwargs in configs:
        res = run_single_config(tasks, base_encoder, name, epochs=epochs, device=device, **kwargs)
        all_results[name] = res
        table_data.append([
            name,
            f"{res['acc']:.2%}",
            f"{res['forgetting']:.2%}",
            f"{res['bwt']:.2%}",
            f"{res['router_stability_kl']:.4f}",
            str(res['final_experts']),
            str(res['gate_rejections']),
        ])

    headers = ["Ablation Configuration", "Avg Acc (↑)", "Forgetting (↓)", "BWT (↑)", "Router KL (↓)", "Experts", "Rejections"]
    print("\n" + "=" * 80)
    print("ABLATION STUDY RESULTS (Split-MNIST, 5 Tasks)")
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
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_dir", type=str, default="./results")
    args = parser.parse_args()

    run_all_ablations(epochs=args.epochs, device_str=args.device, output_dir=args.output_dir)
