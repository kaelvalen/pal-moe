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
from pal_moe.models.router import (
    DynamicRouter,
    DistanceRouter,
    DistanceRouter,
    DistanceRouter,
)
from pal_moe.models.expert import MLPExpert
from pal_moe.models.moe import DynamicMoE, PALMoE
from pal_moe.memory.prototype_memory import PrototypeMemory
from pal_moe.trigger.expert_trigger import QuantitativeTrigger
from pal_moe.builder.expert_builder import ExpertBuilder
from pal_moe.adaptation.ttt import ContinualTrainer, TestTimeAdapter
from pal_moe.baselines.naive import NaiveFineTuning
from pal_moe.baselines.der import DERPP, ERACE
from pal_moe.baselines.agem import AGEM
from pal_moe.baselines.icarl import ICaRL
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


def run_benchmark(
    epochs_per_task: int = 3,
    device_str: str = "auto",
    output_dir: str = "./results",
    dataset: str = "mnist",
):
    os.makedirs(output_dir, exist_ok=True)

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"[Benchmark] Using compute device: {device}")

    if args.max_proto_drop is None:
        args.max_proto_drop = 999.0 if dataset in ("cifar10", "cifar100") else 2.0
    if args.max_proto_acc_drop is None:
        args.max_proto_acc_drop = 999.0
    if args.pretrain_epochs is None:
        args.pretrain_epochs = 50 if dataset in ("cifar10", "cifar100") else 1
    num_classes = 100 if dataset == "cifar100" else 10

    feature_dim = args.feature_dim
    expert_hidden = args.expert_hidden
    conv_channels = tuple(int(c) for c in args.conv_channels.split(",") if c.strip())
    selected = (
        {m.strip() for m in args.methods.split(",") if m.strip()}
        if args.methods
        else None
    )

    def run(method_id: str) -> bool:
        """Whether a method block is enabled (empty --methods = run everything)."""
        return selected is None or method_id in selected

    method_keys = {
        "naive": "Naive Fine-tuning",
        "ewc": "EWC",
        "replay60": "Replay (P=60)",
        "replay360": "Replay (P=360)",
        "replay250": "Experience Replay (Buffer=250)",
        "derpp": "DER++ (P=250)",
        "erace": "ER-ACE (P=250)",
        "agem": "AGEM (P=250)",
        "icarl": "iCaRL (k=25)",
        "stdmoe": "Standard MoE",
        "palmoe": "PAL-MoE (Ours)",
        "hybrid": "PAL-MoE + Replay (Hybrid, P=250)",
    }

    if dataset == "cifar10":
        from pal_moe.data.split_cifar import get_split_cifar10_tasks

        tasks = get_split_cifar10_tasks(
            data_dir="./data",
            batch_size=128,
            val_split=0.1,
            seed=args.seed,
            num_workers=args.num_workers,
            pin_memory=args.num_workers > 0 and device.type == "cuda",
        )
        num_tasks = len(tasks)
        input_dim = 3072
        mnist_train = datasets.CIFAR10(
            "./data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
                    ),
                ]
            ),
        )
        unlabeled_loader = torch.utils.data.DataLoader(
            mnist_train,
            batch_size=256,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.num_workers > 0 and device.type == "cuda",
        )
        base_encoder = SharedEncoder(
            input_dim=input_dim,
            hidden_dims=None,
            output_dim=feature_dim,
            arch="conv",
            conv_channels=conv_channels,
        ).to(device)
        base_encoder.pretrain_contrastive(
            unlabeled_loader, device=device, epochs=args.pretrain_epochs
        )
        # unfreezing happens implicitly
    elif dataset == "cifar100":
        from pal_moe.data.split_cifar100 import get_split_cifar100_tasks

        tasks = get_split_cifar100_tasks(
            data_dir="./data",
            batch_size=128,
            val_split=0.1,
            seed=args.seed,
            num_workers=args.num_workers,
            pin_memory=args.num_workers > 0 and device.type == "cuda",
        )
        num_tasks = len(tasks)
        input_dim = 3072
        cifar100_train = datasets.CIFAR100(
            "./data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                    ),
                ]
            ),
        )
        unlabeled_loader = torch.utils.data.DataLoader(
            cifar100_train,
            batch_size=256,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.num_workers > 0 and device.type == "cuda",
        )
        base_encoder = SharedEncoder(
            input_dim=input_dim,
            hidden_dims=None,
            output_dim=feature_dim,
            arch="conv",
            conv_channels=conv_channels,
        ).to(device)
        base_encoder.pretrain_contrastive(
            unlabeled_loader, device=device, epochs=args.pretrain_epochs
        )
        # encoder stays unfrozen (adapts online)
    else:
        tasks = get_split_mnist_tasks(
            data_dir="./data", batch_size=128, val_split=0.1, seed=args.seed
        )
        num_tasks = len(tasks)
        input_dim = 784
        mnist_train = datasets.MNIST(
            "./data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        )
        unlabeled_loader = torch.utils.data.DataLoader(
            mnist_train, batch_size=256, shuffle=True
        )
        base_encoder = SharedEncoder(
            input_dim=input_dim,
            hidden_dims=(256, 128),
            output_dim=feature_dim,
            arch="mlp",
        ).to(device)
        base_encoder.pretrain_unsupervised(unlabeled_loader, device=device, epochs=1)
        base_encoder.freeze()

    print(
        "  Shared Encoder successfully pretrained and frozen for all benchmark models."
    )

    results = {}

    # -------------------------------------------------------------
    # 1. Baseline: Naive Sequential Fine-tuning
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Running Baseline 1: Naive Sequential Fine-tuning")
    print("=" * 60)
    set_seed(args.seed)
    naive_net = nn.Sequential(
        copy.deepcopy(base_encoder),
        MLPExpert(input_dim=feature_dim, hidden_dim=expert_hidden, num_classes=num_classes, expert_id=0),
    ).to(device)
    naive_trainer = NaiveFineTuning(naive_net, lr=1e-3, device=device)
    evaluator_naive = ContinualEvaluator(num_tasks=num_tasks, device=device)

    for t_idx, task in enumerate(tasks if run("naive") else []):
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
    set_seed(args.seed)
    ewc_net = nn.Sequential(
        copy.deepcopy(base_encoder),
        MLPExpert(input_dim=feature_dim, hidden_dim=expert_hidden, num_classes=num_classes, expert_id=0),
    ).to(device)
    ewc_trainer = EWC(ewc_net, ewc_lambda=1000.0, lr=1e-3, device=device)
    evaluator_ewc = ContinualEvaluator(num_tasks=num_tasks, device=device)

    for t_idx, task in enumerate(tasks if run("ewc") else []):
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
    set_seed(args.seed)
    replay_net_budget = nn.Sequential(
        copy.deepcopy(base_encoder),
        MLPExpert(input_dim=feature_dim, hidden_dim=expert_hidden, num_classes=num_classes, expert_id=0),
    ).to(device)
    replay_trainer_budget = ReplayTrainer(
        replay_net_budget, buffer_size=60, lr=1e-3, device=device
    )
    evaluator_replay_budget = ContinualEvaluator(num_tasks=num_tasks, device=device)

    for t_idx, task in enumerate(tasks if run("replay60") else []):
        print(f"  Training Task {t_idx} (classes {task.classes})...")
        replay_trainer_budget.train_task(
            t_idx, task.train_loader, epochs=epochs_per_task
        )
        accs = evaluator_replay_budget.evaluate_all_seen_tasks(
            replay_net_budget, t_idx, tasks
        )
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
    # 4. Baseline: Experience Replay (Budgeted P=360, theoretical raw memory)
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Running Baseline 4: Experience Replay (Budgeted P=360)")
    print("=" * 60)
    set_seed(args.seed)
    replay_net_360 = nn.Sequential(
        copy.deepcopy(base_encoder),
        MLPExpert(input_dim=feature_dim, hidden_dim=expert_hidden, num_classes=num_classes, expert_id=0),
    ).to(device)
    replay_trainer_360 = ReplayTrainer(
        replay_net_360, buffer_size=360, lr=1e-3, device=device
    )
    evaluator_replay_360 = ContinualEvaluator(num_tasks=num_tasks, device=device)

    for t_idx, task in enumerate(tasks if run("replay360") else []):
        print(f"  Training Task {t_idx} (classes {task.classes})...")
        replay_trainer_360.train_task(t_idx, task.train_loader, epochs=epochs_per_task)
        accs = evaluator_replay_360.evaluate_all_seen_tasks(
            replay_net_360, t_idx, tasks
        )
        print(f"  Accuracies after Task {t_idx}: {[f'{a:.1%}' for a in accs]}")

    results["Replay (P=360)"] = {
        "acc": evaluator_replay_360.compute_average_accuracy(),
        "forgetting": evaluator_replay_360.compute_forgetting(),
        "bwt": evaluator_replay_360.compute_backward_transfer(),
        "router_stability_kl": float("nan"),
        "specialization_mi": float("nan"),
        "utilization": float("nan"),
        "final_experts": 1,
        "acc_matrix": evaluator_replay_360.R.tolist(),
    }

    # -------------------------------------------------------------
    # 5. Baseline: Experience Replay (Buffer=250)
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Running Baseline 5: Experience Replay (Buffer=250)")
    print("=" * 60)
    set_seed(args.seed)
    replay_net = nn.Sequential(
        copy.deepcopy(base_encoder),
        MLPExpert(input_dim=feature_dim, hidden_dim=expert_hidden, num_classes=num_classes, expert_id=0),
    ).to(device)
    replay_trainer = ReplayTrainer(replay_net, buffer_size=250, lr=1e-3, device=device)
    evaluator_replay = ContinualEvaluator(num_tasks=num_tasks, device=device)

    for t_idx, task in enumerate(tasks if run("replay250") else []):
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
    # 6. Baseline: DER++ (Dark Experience Replay++, P=250)
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Running Baseline 6: DER++ (Dark Experience Replay++, P=250)")
    print("=" * 60)
    set_seed(args.seed)
    der_net = nn.Sequential(
        copy.deepcopy(base_encoder),
        MLPExpert(input_dim=feature_dim, hidden_dim=expert_hidden, num_classes=num_classes, expert_id=0),
    ).to(device)
    der_trainer = DERPP(der_net, buffer_size=250, lr=1e-3, device=device)
    evaluator_der = ContinualEvaluator(num_tasks=num_tasks, device=device)

    for t_idx, task in enumerate(tasks if run("derpp") else []):
        print(f"  Training Task {t_idx} (classes {task.classes})...")
        der_trainer.train_task(t_idx, task.train_loader, epochs=epochs_per_task)
        accs = evaluator_der.evaluate_all_seen_tasks(der_net, t_idx, tasks)
        print(f"  Accuracies after Task {t_idx}: {[f'{a:.1%}' for a in accs]}")

    results["DER++ (P=250)"] = {
        "acc": evaluator_der.compute_average_accuracy(),
        "forgetting": evaluator_der.compute_forgetting(),
        "bwt": evaluator_der.compute_backward_transfer(),
        "router_stability_kl": float("nan"),
        "specialization_mi": float("nan"),
        "utilization": float("nan"),
        "final_experts": 1,
        "acc_matrix": evaluator_der.R.tolist(),
    }

    # -------------------------------------------------------------
    # 7. Baseline: ER-ACE (Asymmetric Cross-Entropy Replay, P=250)
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Running Baseline 7: ER-ACE (Asymmetric Cross-Entropy Replay, P=250)")
    print("=" * 60)
    set_seed(args.seed)
    erace_net = nn.Sequential(
        copy.deepcopy(base_encoder),
        MLPExpert(input_dim=feature_dim, hidden_dim=expert_hidden, num_classes=num_classes, expert_id=0),
    ).to(device)
    erace_trainer = ERACE(erace_net, buffer_size=250, lr=1e-3, device=device)
    evaluator_erace = ContinualEvaluator(num_tasks=num_tasks, device=device)

    for t_idx, task in enumerate(tasks if run("erace") else []):
        print(f"  Training Task {t_idx} (classes {task.classes})...")
        erace_trainer.train_task(
            t_idx,
            task.train_loader,
            epochs=epochs_per_task,
            current_classes=list(task.classes),
        )
        accs = evaluator_erace.evaluate_all_seen_tasks(erace_net, t_idx, tasks)
        print(f"  Accuracies after Task {t_idx}: {[f'{a:.1%}' for a in accs]}")

    results["ER-ACE (P=250)"] = {
        "acc": evaluator_erace.compute_average_accuracy(),
        "forgetting": evaluator_erace.compute_forgetting(),
        "bwt": evaluator_erace.compute_backward_transfer(),
        "router_stability_kl": float("nan"),
        "specialization_mi": float("nan"),
        "utilization": float("nan"),
        "final_experts": 1,
        "acc_matrix": evaluator_erace.R.tolist(),
    }

    # -------------------------------------------------------------
    # 8. Baseline: AGEM (Average Gradient Episodic Memory, P=250)
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Running Baseline 8: AGEM (Average Gradient Episodic Memory, P=250)")
    print("=" * 60)
    set_seed(args.seed)
    agem_net = nn.Sequential(
        copy.deepcopy(base_encoder),
        MLPExpert(input_dim=feature_dim, hidden_dim=expert_hidden, num_classes=num_classes, expert_id=0),
    ).to(device)
    agem_trainer = AGEM(agem_net, buffer_size=250, lr=1e-3, device=device)
    evaluator_agem = ContinualEvaluator(num_tasks=num_tasks, device=device)

    for t_idx, task in enumerate(tasks if run("agem") else []):
        print(f"  Training Task {t_idx} (classes {task.classes})...")
        agem_trainer.train_task(t_idx, task.train_loader, epochs=epochs_per_task)
        accs = evaluator_agem.evaluate_all_seen_tasks(agem_net, t_idx, tasks)
        print(f"  Accuracies after Task {t_idx}: {[f'{a:.1%}' for a in accs]}")

    results["AGEM (P=250)"] = {
        "acc": evaluator_agem.compute_average_accuracy(),
        "forgetting": evaluator_agem.compute_forgetting(),
        "bwt": evaluator_agem.compute_backward_transfer(),
        "router_stability_kl": float("nan"),
        "specialization_mi": float("nan"),
        "utilization": float("nan"),
        "final_experts": 1,
        "acc_matrix": evaluator_agem.R.tolist(),
    }

    # -------------------------------------------------------------
    # 9. Baseline: iCaRL (Incremental Classifier & Representation Learning, k=25/class)
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print(
        "Running Baseline 9: iCaRL (Incremental Classifier & Representation Learning)"
    )
    print("=" * 60)
    set_seed(args.seed)
    icarl_net = nn.Sequential(
        copy.deepcopy(base_encoder),
        MLPExpert(input_dim=feature_dim, hidden_dim=expert_hidden, num_classes=num_classes, expert_id=0),
    ).to(device)
    icarl_trainer = ICaRL(
        icarl_net,
        exemplars_per_class=25,
        num_classes=num_classes,
        lr=1e-3,
        device=device,
    )
    evaluator_icarl = ContinualEvaluator(num_tasks=num_tasks, device=device)

    for t_idx, task in enumerate(tasks if run("icarl") else []):
        print(f"  Training Task {t_idx} (classes {task.classes})...")
        icarl_trainer.train_task(
            t_idx,
            task.train_loader,
            epochs=epochs_per_task,
            current_classes=list(task.classes),
        )
        accs = evaluator_icarl.evaluate_all_seen_tasks(
            icarl_trainer.eval_model(), t_idx, tasks
        )
        print(f"  Accuracies after Task {t_idx}: {[f'{a:.1%}' for a in accs]}")

    results["iCaRL (k=25)"] = {
        "acc": evaluator_icarl.compute_average_accuracy(),
        "forgetting": evaluator_icarl.compute_forgetting(),
        "bwt": evaluator_icarl.compute_backward_transfer(),
        "router_stability_kl": float("nan"),
        "specialization_mi": float("nan"),
        "utilization": float("nan"),
        "final_experts": 1,
        "acc_matrix": evaluator_icarl.R.tolist(),
    }

    # -------------------------------------------------------------
    # 10. Baseline: Standard MoE Fine-tuning (Fixed 4 Experts)
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Running Baseline 6: Standard MoE Fine-tuning (Fixed 4 Experts, Balanced)")
    print("=" * 60)
    set_seed(args.seed)
    std_encoder = copy.deepcopy(base_encoder)
    std_router = DynamicRouter(input_dim=feature_dim, num_experts=4, top_k=1).to(device)
    std_experts = [
        MLPExpert(
            input_dim=feature_dim, hidden_dim=expert_hidden, num_classes=num_classes, expert_id=i
        ).to(device)
        for i in range(4)
    ]
    std_moe = DynamicMoE(
        encoder=std_encoder,
        router=std_router,
        experts=std_experts,
        use_ema_encoder=False,
    ).to(device)
    opt_std = torch.optim.Adam(std_moe.parameters(), lr=1e-3)
    evaluator_std_moe = ContinualEvaluator(num_tasks=num_tasks, device=device)

    for t_idx, task in enumerate(tasks if run("stdmoe") else []):
        print(f"  Training Task {t_idx} (classes {task.classes})...")
        std_moe.train()
        for epoch in range(epochs_per_task):
            for x, y in task.train_loader:
                x, y = x.to(device), y.to(device)
                opt_std.zero_grad()
                feats = std_moe.encoder(x)
                logits = std_moe(x)
                loss_ce = nn.functional.cross_entropy(logits, y)

                # Switch Transformer auxiliary load-balancing loss: L_balance = N * sum_{i=1}^N f_i * P_i
                dense_probs = std_moe.router.get_full_distribution(feats)
                _, topk_idx, _ = std_moe.router(feats)
                f_i = torch.bincount(topk_idx.flatten(), minlength=4).float() / x.size(
                    0
                )
                P_i = dense_probs.mean(dim=0)
                loss_balance = 4.0 * torch.sum(f_i * P_i)

                loss = loss_ce + 0.1 * loss_balance
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
    # 7. Proposed: PAL-MoE
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Running Proposed: PAL-MoE (Prototype-Anchored Lifelong Mixture of Experts)")
    print("=" * 60)
    set_seed(args.seed)

    dyn_encoder = copy.deepcopy(base_encoder)
    if args.router_type == "distance":
        dyn_router = DistanceRouter(
            input_dim=feature_dim, num_experts=1, top_k=1, temperature=0.05
        ).to(device)
    else:
        dyn_router = DynamicRouter(
            input_dim=feature_dim, num_experts=1, top_k=1, temperature=1.0
        ).to(device)
    initial_experts = [
        MLPExpert(
            input_dim=feature_dim, hidden_dim=expert_hidden, num_classes=num_classes, expert_id=0
        ).to(device)
    ]
    moe_model = DynamicMoE(
        encoder=dyn_encoder,
        router=dyn_router,
        experts=initial_experts,
        use_ema_encoder=False,
    ).to(device)

    prototype_mem = PrototypeMemory(
        feature_dim=feature_dim,
        distance_threshold=0.5,
        ema_alpha=0.9,
        max_prototypes=args.proto_size,
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
        min_acc_threshold=0.45 if dataset in ("cifar10", "cifar100") else 0.60,
        max_proto_drop=args.max_proto_drop,
        max_proto_acc_drop=args.max_proto_acc_drop,
        max_ece=999.0,
        distill_lambda=0.5,
    )
    trainer = ContinualTrainer(
        model=moe_model,
        prototype_memory=prototype_mem,
        trigger=trigger,
        builder=builder,
        lambda_r=0.5,
        lambda_e=2.5,
        lambda_enc=0.5 if dataset in ("cifar10", "cifar100") else 0.0,
        encoder_lr=1e-4 if dataset in ("cifar10", "cifar100") else None,
        lr=1e-3,
        max_experts=6,
        joint_keep_routing_lock=args.joint_keep_routing_lock,
        joint_freeze_router=args.joint_freeze_router,
        lambda_ood=args.lambda_ood,
        device=device,
    )
    evaluator_dynamic = ContinualEvaluator(num_tasks=num_tasks, device=device)

    for t_idx, task in enumerate(tasks if run("palmoe") else []):
        print(
            f"  Training Task {t_idx} (classes {task.classes})... Experts before: {moe_model.num_experts}"
        )
        hist = trainer.train_task(
            task_id=t_idx,
            train_loader=task.train_loader,
            val_loader=task.val_loader,
            epochs=epochs_per_task,
            enable_expansion=True,
            enable_anchor=args.anchor,
        )
        print(
            f"    Triggers: {hist['trigger_events']} | Experts added: {hist['experts_added']} | Gate rejects: {hist['gate_rejections']} | Total experts: {moe_model.num_experts}"
        )
        accs = evaluator_dynamic.evaluate_all_seen_tasks(moe_model, t_idx, tasks)
        print(f"  Accuracies after Task {t_idx}: {[f'{a:.1%}' for a in accs]}")

    # Compute advanced stability and information metrics
    router_kl = ContinualEvaluator.compute_router_stability(
        moe_model, prototype_mem, device
    )
    mi_dyn, util_dyn = ContinualEvaluator.compute_expert_specialization_and_utilization(
        moe_model, tasks, device
    )
    footprint = prototype_mem.estimate_memory_footprint()
    print(
        f"  Prototype Memory Footprint: {footprint['num_prototypes']} prototypes, {footprint['total_elements']} floats ({footprint['size_kb']:.1f} KB)"
    )

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
    set_seed(args.seed)
    hyb_encoder = copy.deepcopy(base_encoder)
    if args.router_type == "distance":
        hyb_router = DistanceRouter(
            input_dim=feature_dim, num_experts=1, top_k=1, temperature=0.05
        ).to(device)
    else:
        hyb_router = DynamicRouter(
            input_dim=feature_dim, num_experts=1, top_k=1, temperature=1.0
        ).to(device)
    initial_experts_hyb = [
        MLPExpert(
            input_dim=feature_dim, hidden_dim=expert_hidden, num_classes=num_classes, expert_id=0
        ).to(device)
    ]
    moe_hyb = DynamicMoE(
        encoder=hyb_encoder,
        router=hyb_router,
        experts=initial_experts_hyb,
        use_ema_encoder=False,
    ).to(device)
    prototype_mem_hyb = PrototypeMemory(
        feature_dim=feature_dim,
        distance_threshold=0.5,
        ema_alpha=0.9,
        max_prototypes=args.proto_size,
        store_raw=True,
    )
    trigger_hyb = QuantitativeTrigger(
        alpha=1.0, beta=0.4, gamma=0.6, delta=0.5, threshold_tau=0.5
    )
    builder_hyb = ExpertBuilder(
        min_acc_threshold=0.45 if dataset in ("cifar10", "cifar100") else 0.60,
        max_proto_drop=args.max_proto_drop,
        max_proto_acc_drop=args.max_proto_acc_drop,
        max_ece=999.0,
        distill_lambda=0.5,
    )
    trainer_hyb = ContinualTrainer(
        model=moe_hyb,
        prototype_memory=prototype_mem_hyb,
        trigger=trigger_hyb,
        builder=builder_hyb,
        lambda_r=0.5,
        lambda_e=2.5,
        lambda_enc=0.5 if dataset in ("cifar10", "cifar100") else 0.0,
        encoder_lr=1e-4 if dataset in ("cifar10", "cifar100") else None,
        replay_exemplars=True,
        lambda_replay=1.0,
        lr=1e-3,
        max_experts=6,
        joint_keep_routing_lock=args.joint_keep_routing_lock,
        joint_freeze_router=args.joint_freeze_router,
        lambda_ood=args.lambda_ood,
        device=device,
    )
    evaluator_hyb = ContinualEvaluator(num_tasks=num_tasks, device=device)

    for t_idx, task in enumerate(tasks if run("hybrid") else []):
        print(
            f"  Training Task {t_idx} (classes {task.classes})... Experts before: {moe_hyb.num_experts}"
        )
        hist = trainer_hyb.train_task(
            task_id=t_idx,
            train_loader=task.train_loader,
            val_loader=task.val_loader,
            epochs=epochs_per_task,
            enable_expansion=True,
            enable_anchor=args.anchor,
        )
        print(
            f"    Triggers: {hist['trigger_events']} | Experts added: {hist['experts_added']} | Gate rejects: {hist['gate_rejections']} | Total experts: {moe_hyb.num_experts}"
        )
        accs = evaluator_hyb.evaluate_all_seen_tasks(moe_hyb, t_idx, tasks)
        print(f"  Accuracies after Task {t_idx}: {[f'{a:.1%}' for a in accs]}")

    router_kl_hyb = ContinualEvaluator.compute_router_stability(
        moe_hyb, prototype_mem_hyb, device
    )
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
        "prototype_elements": prototype_mem_hyb.estimate_memory_footprint()[
            "total_elements"
        ],
        "acc_matrix": evaluator_hyb.R.tolist(),
    }

    # Drop skipped methods so the table/JSON only contain what actually ran.
    if selected is not None:
        kept = {name for mid, name in method_keys.items() if run(mid)}
        results = {k: v for k, v in results.items() if k in kept}

    # -------------------------------------------------------------
    # Summary Table
    # -------------------------------------------------------------
    table_data = []
    headers = [
        "Method",
        "Avg Acc (↑)",
        "Forgetting (↓)",
        "BWT (↑)",
        "Router KL (↓)",
        "Spec. MI (↑)",
        "Util. Entropy",
        "Experts",
    ]

    for name, m in results.items():
        table_data.append(
            [
                name,
                f"{m['acc']:.2%}",
                f"{m['forgetting']:.2%}",
                f"{m['bwt']:.2%}",
                f"{m['router_stability_kl']:.4f}"
                if not np.isnan(m["router_stability_kl"])
                else "-",
                f"{m['specialization_mi']:.3f}"
                if not np.isnan(m["specialization_mi"])
                else "-",
                f"{m['utilization']:.3f}" if not np.isnan(m["utilization"]) else "-",
                str(m["final_experts"]),
            ]
        )

    print("\n" + "=" * 80)
    print(
        f"FINAL CONTINUAL LEARNING BENCHMARK RESULTS (Split-{dataset.upper()} , {num_tasks} Tasks)"
    )
    print("=" * 80)
    print(tabulate(table_data, headers=headers, tablefmt="github"))
    print("=" * 80)

    # Save to json
    results_path = os.path.join(output_dir, f"benchmark_results_seed{args.seed}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved benchmark results to {results_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3, help="Epochs per task")
    parser.add_argument(
        "--device", type=str, default="auto", help="Compute device: cuda or cpu"
    )
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument(
        "--dataset", type=str, default="mnist", choices=["mnist", "cifar10", "cifar100"]
    )
    parser.add_argument(
        "--pretrain_epochs",
        type=int,
        default=None,
        help="Encoder pretraining epochs (None = dataset default)",
    )
    parser.add_argument(
        "--feature_dim",
        type=int,
        default=128,
        help="Latent/feature dimension (encoder output, router and expert input)",
    )
    parser.add_argument(
        "--expert_hidden",
        type=int,
        default=256,
        help="Hidden width of every MLP expert",
    )
    parser.add_argument(
        "--conv_channels",
        type=str,
        default="32,64,128",
        help="Comma-separated conv encoder channels (default reproduces the original net)",
    )
    parser.add_argument(
        "--proto_size",
        type=int,
        default=250,
        help="PAL-MoE prototype memory budget (also sets the hybrid's latent store size)",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="",
        help=(
            "Comma-separated method ids to run (empty = all). Ids: naive, ewc, "
            "replay60, replay360, replay250, derpp, erace, agem, icarl, stdmoe, "
            "palmoe, hybrid"
        ),
    )
    parser.add_argument(
        "--anchor",
        action="store_true",
        help="Enable null-space routing anchoring (experimental; measured harmful on Split-MNIST)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help=(
            "DataLoader workers for CIFAR loaders (0 = single-process). CIFAR "
            "transforms are deterministic: speed knob only, results-neutral."
        ),
    )
    parser.add_argument(
        "--router_type", type=str, default="dynamic", choices=["dynamic", "distance"]
    )
    parser.add_argument(
        "--max_proto_drop",
        type=float,
        default=None,
        help="Prototype drift gate threshold (None = dataset default)",
    )
    parser.add_argument(
        "--joint_keep_routing_lock",
        action="store_true",
        help="Joint FT: unfreeze all experts but keep historical router rows locked",
    )
    parser.add_argument(
        "--joint_freeze_router",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Joint FT: freeze the ENTIRE router during calibration",
    )
    parser.add_argument(
        "--lambda_ood",
        type=float,
        default=0.1,
        help="OOD negative-boundary weight (entropy-max on old prototypes for newest expert)",
    )
    parser.add_argument(
        "--max_proto_acc_drop",
        type=float,
        default=None,
        help="Historical prototype accuracy drop tolerance (None = dataset default)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="JSON config file overriding CLI defaults",
    )
    args = parser.parse_args()

    if args.config:
        import json as _json

        with open(args.config) as _cf:
            _cfg = _json.load(_cf)
        for _k, _v in _cfg.items():
            if hasattr(args, _k):
                setattr(args, _k, _v)

    run_benchmark(
        epochs_per_task=args.epochs,
        device_str=args.device,
        output_dir=args.output_dir,
        dataset=args.dataset,
    )
