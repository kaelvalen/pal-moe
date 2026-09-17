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

import argparse
import copy
import datetime
import hashlib
import json
import time
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from tabulate import tabulate

from pal_moe.adaptation.ttt import ContinualTrainer
from pal_moe.baselines.agem import AGEM
from pal_moe.baselines.der import DERPP, ERACE
from pal_moe.baselines.ewc import EWC
from pal_moe.baselines.icarl import ICaRL
from pal_moe.baselines.naive import NaiveFineTuning
from pal_moe.baselines.replay import ReplayTrainer
from pal_moe.builder.expert_builder import ExpertBuilder
from pal_moe.config import ConfigError, apply_config
from pal_moe.data.split_mnist import get_split_mnist_tasks
from pal_moe.evaluation.diagnostics import (
    print_router_diagnostics,
    router_diagnostics,
)
from pal_moe.evaluation.metrics import ContinualEvaluator
from pal_moe.factory import (
    build_prototype_memory,
    build_router,
    build_single_head,
)
from pal_moe.models.encoder import SharedEncoder
from pal_moe.models.expert import MLPExpert
from pal_moe.models.moe import DynamicMoE
from pal_moe.models.router import DynamicRouter
from pal_moe.trigger.expert_trigger import QuantitativeTrigger


def _git_commit() -> str:
    """Short git hash of the working tree (for reproducibility metadata)."""
    try:
        import subprocess

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=root)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _banner(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def _record_baseline_result(
    model: nn.Module, evaluator: ContinualEvaluator, final_experts: int = 1
) -> dict:
    """Common result payload; `trainable_params` documents the capacity actually used."""
    return {
        "acc": evaluator.compute_average_accuracy(),
        "forgetting": evaluator.compute_forgetting(),
        "bwt": evaluator.compute_backward_transfer(),
        "router_stability_kl": float("nan"),
        "specialization_mi": float("nan"),
        "utilization": float("nan"),
        "final_experts": final_experts,
        "trainable_params": int(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        ),
        "fit_seconds": getattr(evaluator, "fit_seconds", None),
        "geometry": getattr(evaluator, "geometry", None),
        "acc_matrix": evaluator.R.tolist(),
    }


def _method_features(model: nn.Module, x: torch.Tensor) -> Optional[torch.Tensor]:
    """Best-effort representation accessor for the geometry report."""
    if hasattr(model, "get_routing_features"):
        return model.get_routing_features(x)
    if isinstance(model, nn.Sequential) and len(model) > 0:
        return model[0](x)
    return None


@torch.no_grad()
def _test_geometry(
    model: nn.Module, tasks: list, device: torch.device, max_batches_per_task: int = 4
) -> Optional[dict]:
    """Class-geometry report on a bounded slice of every task's test set."""
    from pal_moe.evaluation.geometry import geometry_report

    if model is None:
        return None
    feats, labels = [], []
    for task in tasks:
        for batch_idx, (x, y) in enumerate(task.test_loader):
            if batch_idx >= max_batches_per_task:
                break
            h = _method_features(model, x.to(device))
            if h is None:
                return None
            feats.append(h.detach().cpu())
            labels.append(y)
    if not feats:
        return None
    return geometry_report(torch.cat(feats, dim=0), torch.cat(labels, dim=0))


def _run_baseline_loop(
    trainer: Any,
    model: nn.Module,
    evaluator: ContinualEvaluator,
    tasks: list,
    epochs_per_task: int,
    per_task_kwargs: Optional[Callable] = None,
    eval_model: Optional[Callable] = None,
) -> None:
    """Shared per-task training/evaluation loop for the single-head baselines."""
    t0 = time.time()
    for t_idx, task in enumerate(tasks):
        print(f"  Training Task {t_idx} (classes {task.classes})...")
        kwargs = per_task_kwargs(task) if per_task_kwargs is not None else {}
        trainer.train_task(t_idx, task.train_loader, epochs=epochs_per_task, **kwargs)
        eval_net = eval_model() if eval_model is not None else model
        accs = evaluator.evaluate_all_seen_tasks(eval_net, t_idx, tasks)
        print(f"  Accuracies after Task {t_idx}: {[f'{a:.1%}' for a in accs]}")
    evaluator.fit_seconds = round(time.time() - t0, 1)
    evaluator.geometry = _test_geometry(
        model, tasks, getattr(trainer, "device", None) or torch.device("cpu")
    )


def _pretrain_cache_path(
    cache_dir: str,
    dataset: str,
    mode: str,
    seed: int,
    epochs: int,
    feature_dim: int,
    conv_channels: tuple,
) -> str:
    """Content-addressed path for a cached unsupervised pretraining result."""
    key = (
        f"v1|{dataset}|{mode}|seed={seed}|epochs={epochs}|fd={feature_dim}|"
        f"ch={conv_channels}|torch={torch.__version__}"
    )
    digest = hashlib.sha1(key.encode()).hexdigest()[:16]
    return os.path.join(cache_dir, f"pretrain_{dataset}_{mode}_{digest}.pt")


def _load_pretrained_encoder(encoder: nn.Module, path: str) -> bool:
    if not os.path.exists(path):
        return False
    state = torch.load(path, map_location="cpu", weights_only=True)
    encoder.load_state_dict(state, strict=True)
    return True


def _save_pretrained_encoder(encoder: nn.Module, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in encoder.state_dict().items()}, path)


def _pretrain_encoder(
    encoder: nn.Module,
    loader: Any,
    device: torch.device,
    *,
    dataset: str,
    mode: str,
    seed: int,
    epochs: int,
    feature_dim: int,
    conv_channels: tuple,
    use_cache: bool,
    cache_dir: str = "./data/pretrain_cache",
) -> None:
    """
    Pretrains the shared encoder, optionally reusing a cached state dict.

    A cache hit skips the RNG consumed by pretraining, so cached runs are seeded
    immediately before and after the (possibly skipped) pretraining to stay
    internally reproducible. They are distributionally equivalent to uncached
    runs, not bit-identical (same caveat as the feature cache).
    """
    if not use_cache:
        if mode == "simclr":
            encoder.pretrain_contrastive(loader, device=device, epochs=epochs)
        else:
            encoder.pretrain_unsupervised(loader, device=device, epochs=epochs)
        return

    set_seed(seed)
    path = _pretrain_cache_path(
        cache_dir, dataset, mode, seed, epochs, feature_dim, conv_channels
    )
    if _load_pretrained_encoder(encoder, path):
        print(f"  [pretrain-cache] loaded {path}")
    else:
        if mode == "simclr":
            encoder.pretrain_contrastive(loader, device=device, epochs=epochs)
        else:
            encoder.pretrain_unsupervised(loader, device=device, epochs=epochs)
        _save_pretrained_encoder(encoder, path)
        print(f"  [pretrain-cache] saved {path}")
    set_seed(seed)


def _run_standard_moe(
    base_encoder: nn.Module,
    tasks: list,
    device: torch.device,
    seed: int,
    epochs_per_task: int,
    feature_dim: int,
    expert_hidden: int,
    num_classes: int,
    num_tasks: int,
) -> dict:
    """Fixed 4-expert MoE with the Switch load-balancing loss (no stability)."""
    set_seed(seed)
    std_moe = DynamicMoE(
        encoder=copy.deepcopy(base_encoder),
        router=DynamicRouter(input_dim=feature_dim, num_experts=4, top_k=1).to(device),
        experts=[
            MLPExpert(
                input_dim=feature_dim,
                hidden_dim=expert_hidden,
                num_classes=num_classes,
                expert_id=i,
            ).to(device)
            for i in range(4)
        ],
        use_ema_encoder=False,
    ).to(device)
    opt_std = torch.optim.Adam(std_moe.parameters(), lr=1e-3)
    evaluator = ContinualEvaluator(num_tasks=num_tasks, device=device)

    for t_idx, task in enumerate(tasks):
        print(f"  Training Task {t_idx} (classes {task.classes})...")
        std_moe.train()
        for _ in range(epochs_per_task):
            for x, y in task.train_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                opt_std.zero_grad()
                feats = std_moe.encoder(x)
                logits = std_moe(x)
                loss_ce = nn.functional.cross_entropy(logits, y)

                # Switch Transformer auxiliary load-balancing loss: L_balance = N * sum_i f_i * P_i
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

        accs = evaluator.evaluate_all_seen_tasks(std_moe, t_idx, tasks)
        print(f"  Accuracies after Task {t_idx}: {[f'{a:.1%}' for a in accs]}")

    mi, util = ContinualEvaluator.compute_expert_specialization_and_utilization(
        std_moe, tasks, device
    )
    result = _record_baseline_result(std_moe, evaluator, final_experts=4)
    result["specialization_mi"] = mi
    result["utilization"] = util
    return result


def _run_palmoe_variant(
    base_encoder: nn.Module,
    tasks: list,
    device: torch.device,
    args: argparse.Namespace,
    *,
    dataset: str,
    feature_dim: int,
    expert_hidden: int,
    num_classes: int,
    proto_threshold: Optional[float],
    encoder_ft: bool,
    epochs_per_task: int,
    output_dir: str,
    checkpoint_subdir: str,
    replay: bool,
    max_proto_drop: float,
    max_proto_acc_drop: float,
) -> dict:
    """
    Runs one PAL-MoE variant (pure or hybrid). Configuration is identical to
    the published recipe except for the replay/latent-store switches, so the
    two variants cannot drift apart in this file.
    """
    set_seed(args.seed)
    model = DynamicMoE(
        encoder=copy.deepcopy(base_encoder),
        router=build_router(args.router_type, feature_dim, args.top_k, device=device),
        experts=[
            MLPExpert(
                input_dim=feature_dim,
                hidden_dim=expert_hidden,
                num_classes=num_classes,
                expert_id=0,
            ).to(device)
        ],
        use_ema_encoder=False,
    ).to(device)
    memory = build_prototype_memory(
        feature_dim,
        proto_threshold,
        args.proto_size,
        args.proto_per_class,
        (not args.feature_cache) if replay else False,
    )
    trigger = QuantitativeTrigger(
        alpha=1.0, beta=0.4, gamma=0.6, delta=0.5, threshold_tau=0.5
    )
    builder = ExpertBuilder(
        min_acc_threshold=0.45 if dataset in ("cifar10", "cifar100") else 0.60,
        max_proto_drop=max_proto_drop,
        max_proto_acc_drop=max_proto_acc_drop,
        max_ece=999.0,
        distill_lambda=0.5,
    )
    trainer = ContinualTrainer(
        model=model,
        prototype_memory=memory,
        trigger=trigger,
        builder=builder,
        lambda_r=args.lambda_r,
        lambda_e=args.lambda_e,
        lambda_enc=0.5 if encoder_ft else 0.0,
        encoder_lr=1e-4 if encoder_ft else None,
        replay_exemplars=replay,
        lambda_replay=1.0,
        lr=1e-3,
        max_experts=args.max_experts,
        joint_keep_routing_lock=args.joint_keep_routing_lock,
        joint_freeze_router=args.joint_freeze_router,
        lambda_ood=args.lambda_ood,
        router_anchor_steps=args.router_anchor_steps,
        router_anchor_lr=args.router_anchor_lr,
        joint_calib_epochs=args.joint_calib_epochs,
        proto_samples=args.proto_samples,
        refresh_anchors_after_calib=args.refresh_anchors_after_calib,
        keep_optimizer_state=args.keep_optimizer_state,
        stability_every=args.stability_every,
        ood_every=args.ood_every,
        device=device,
        checkpoint_dir=(
            os.path.join(output_dir, checkpoint_subdir)
            if args.save_checkpoints
            else None
        ),
    )
    evaluator = ContinualEvaluator(num_tasks=len(tasks), device=device)
    if args.proto_routing_alpha > 0:
        model.set_prototype_routing(
            memory,
            alpha=args.proto_routing_alpha,
            threshold=args.proto_routing_threshold,
        )

    t0 = time.time()
    for t_idx, task in enumerate(tasks):
        print(
            f"  Training Task {t_idx} (classes {task.classes})... "
            f"Experts before: {model.num_experts}"
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
            f"    Triggers: {hist['trigger_events']} | Experts added: "
            f"{hist['experts_added']} | Gate rejects: {hist['gate_rejections']} | "
            f"Total experts: {model.num_experts}"
        )
        accs = evaluator.evaluate_all_seen_tasks(model, t_idx, tasks)
        print(f"  Accuracies after Task {t_idx}: {[f'{a:.1%}' for a in accs]}")

    router_kl = ContinualEvaluator.compute_router_stability(model, memory, device)
    mi, util = ContinualEvaluator.compute_expert_specialization_and_utilization(
        model, tasks, device
    )
    footprint = memory.estimate_memory_footprint()
    print(
        f"  Prototype Memory Footprint: {footprint['num_prototypes']} prototypes, "
        f"{footprint['total_elements']} floats ({footprint['size_kb']:.1f} KB)"
    )

    evaluator.geometry = _test_geometry(model, tasks, device)
    result = _record_baseline_result(model, evaluator, final_experts=model.num_experts)
    result["router_stability_kl"] = router_kl
    result["specialization_mi"] = mi
    result["utilization"] = util
    result["prototype_elements"] = footprint["total_elements"]
    result["fit_seconds"] = round(time.time() - t0, 1)

    diagnostics = router_diagnostics(model, tasks, device, memory)
    print_router_diagnostics(diagnostics)
    result["router_diagnostics"] = diagnostics
    return result


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_benchmark(
    args: argparse.Namespace,
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

    # Resolve dataset-dependent defaults into locals instead of mutating `args`
    # (the caller may reuse the namespace, and mutation made the function hard
    # to call programmatically).
    max_proto_drop = (
        args.max_proto_drop
        if args.max_proto_drop is not None
        else (999.0 if dataset in ("cifar10", "cifar100") else 2.0)
    )
    max_proto_acc_drop = (
        args.max_proto_acc_drop if args.max_proto_acc_drop is not None else 999.0
    )
    pretrain_epochs = (
        args.pretrain_epochs
        if args.pretrain_epochs is not None
        else (50 if dataset in ("cifar10", "cifar100") else 1)
    )
    num_classes = 100 if dataset == "cifar100" else 10

    feature_dim = args.feature_dim
    expert_hidden = args.expert_hidden
    conv_channels = tuple(int(c) for c in args.conv_channels.split(",") if c.strip())
    # CIFAR fine-tunes the encoder online; --freeze_encoder keeps it fixed so the
    # prototype anchors cannot go stale (mirrors the MNIST setup).
    encoder_ft = dataset in ("cifar10", "cifar100") and not args.freeze_encoder
    proto_threshold = (
        None
        if str(args.proto_threshold).lower() in ("auto", "none")
        else float(args.proto_threshold)
    )
    _t_start = time.time()
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

    pin_memory = device.type == "cuda"
    if dataset == "cifar10":
        from pal_moe.data.split_cifar import get_split_cifar10_tasks

        tasks = get_split_cifar10_tasks(
            data_dir="./data",
            batch_size=128,
            val_split=0.1,
            seed=args.seed,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
        num_tasks = len(tasks)
        input_dim = 3072
        # Reuse the train split built by the task loader (identical transforms)
        # instead of instantiating a second CIFAR-10 dataset object.
        unlabeled_loader = torch.utils.data.DataLoader(
            tasks[0].train_loader.dataset.dataset,
            batch_size=256,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
        base_encoder = SharedEncoder(
            input_dim=input_dim,
            hidden_dims=None,
            output_dim=feature_dim,
            arch="conv",
            conv_channels=conv_channels,
        ).to(device)
        _pretrain_encoder(
            base_encoder,
            unlabeled_loader,
            device,
            dataset=dataset,
            mode="simclr",
            seed=args.seed,
            epochs=pretrain_epochs,
            feature_dim=feature_dim,
            conv_channels=conv_channels,
            use_cache=args.pretrain_cache,
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
            pin_memory=pin_memory,
        )
        num_tasks = len(tasks)
        input_dim = 3072
        unlabeled_loader = torch.utils.data.DataLoader(
            tasks[0].train_loader.dataset.dataset,
            batch_size=256,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
        base_encoder = SharedEncoder(
            input_dim=input_dim,
            hidden_dims=None,
            output_dim=feature_dim,
            arch="conv",
            conv_channels=conv_channels,
        ).to(device)
        _pretrain_encoder(
            base_encoder,
            unlabeled_loader,
            device,
            dataset=dataset,
            mode="simclr",
            seed=args.seed,
            epochs=pretrain_epochs,
            feature_dim=feature_dim,
            conv_channels=conv_channels,
            use_cache=args.pretrain_cache,
        )
        # encoder stays unfrozen (adapts online)
    else:
        tasks = get_split_mnist_tasks(
            data_dir="./data",
            batch_size=128,
            val_split=0.1,
            seed=args.seed,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
        num_tasks = len(tasks)
        input_dim = 784
        unlabeled_loader = torch.utils.data.DataLoader(
            tasks[0].train_loader.dataset.dataset,
            batch_size=256,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
        base_encoder = SharedEncoder(
            input_dim=input_dim,
            hidden_dims=(256, 128),
            output_dim=feature_dim,
            arch="mlp",
        ).to(device)
        _pretrain_encoder(
            base_encoder,
            unlabeled_loader,
            device,
            dataset=dataset,
            mode="ae",
            seed=args.seed,
            epochs=pretrain_epochs,
            feature_dim=feature_dim,
            conv_channels=conv_channels,
            use_cache=args.pretrain_cache,
        )
        base_encoder.freeze()

    if args.freeze_encoder:
        base_encoder.freeze()

    print(
        "  Shared Encoder successfully pretrained and frozen for all benchmark models."
    )

    if args.feature_cache:
        from pal_moe.data.feature_cache import build_feature_cache

        cache = build_feature_cache(
            base_encoder,
            tasks,
            device,
            dtype=torch.float16 if device.type == "cuda" else torch.float32,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
        tasks = cache.tasks
        base_encoder = cache.encoder
        print(
            "  [feature-cache] enabled: all methods run on cached frozen features "
            "(encoder removed from the training loop)"
        )

    results = {}

    # -------------------------------------------------------------
    # 1. Baseline: Naive Sequential Fine-tuning
    # -------------------------------------------------------------
    if run("naive"):
        _banner("Running Baseline 1: Naive Sequential Fine-tuning")
        set_seed(args.seed)
        naive_net = build_single_head(
            base_encoder, feature_dim, expert_hidden, num_classes, device
        )
        naive_trainer = NaiveFineTuning(naive_net, lr=1e-3, device=device)
        evaluator_naive = ContinualEvaluator(num_tasks=num_tasks, device=device)
        _run_baseline_loop(
            naive_trainer, naive_net, evaluator_naive, tasks, epochs_per_task
        )
        results["Naive Fine-tuning"] = _record_baseline_result(
            naive_net, evaluator_naive
        )

    # -------------------------------------------------------------
    # 2. Baseline: Elastic Weight Consolidation (EWC)
    # -------------------------------------------------------------
    if run("ewc"):
        _banner("Running Baseline 2: Elastic Weight Consolidation (EWC)")
        set_seed(args.seed)
        ewc_net = build_single_head(
            base_encoder, feature_dim, expert_hidden, num_classes, device
        )
        ewc_trainer = EWC(
            ewc_net, ewc_lambda=1000.0, lr=1e-3, device=device, online=args.ewc_online
        )
        evaluator_ewc = ContinualEvaluator(num_tasks=num_tasks, device=device)
        _run_baseline_loop(ewc_trainer, ewc_net, evaluator_ewc, tasks, epochs_per_task)
        results["EWC"] = _record_baseline_result(ewc_net, evaluator_ewc)

    # -------------------------------------------------------------
    # 3. Baseline: Experience Replay (Budgeted P=60)
    # -------------------------------------------------------------
    if run("replay60"):
        _banner("Running Baseline 3: Experience Replay (Budgeted P=60)")
        set_seed(args.seed)
        replay_net_budget = build_single_head(
            base_encoder, feature_dim, expert_hidden, num_classes, device
        )
        replay_trainer_budget = ReplayTrainer(
            replay_net_budget,
            buffer_size=60,
            lr=1e-3,
            device=device,
            sampling=args.buffer_sampling,
        )
        evaluator_replay_budget = ContinualEvaluator(num_tasks=num_tasks, device=device)
        _run_baseline_loop(
            replay_trainer_budget,
            replay_net_budget,
            evaluator_replay_budget,
            tasks,
            epochs_per_task,
        )
        results["Replay (P=60)"] = _record_baseline_result(
            replay_net_budget, evaluator_replay_budget
        )

    # -------------------------------------------------------------
    # 4. Baseline: Experience Replay (Budgeted P=360)
    # -------------------------------------------------------------
    if run("replay360"):
        _banner("Running Baseline 4: Experience Replay (Budgeted P=360)")
        set_seed(args.seed)
        replay_net_360 = build_single_head(
            base_encoder, feature_dim, expert_hidden, num_classes, device
        )
        replay_trainer_360 = ReplayTrainer(
            replay_net_360,
            buffer_size=360,
            lr=1e-3,
            device=device,
            sampling=args.buffer_sampling,
        )
        evaluator_replay_360 = ContinualEvaluator(num_tasks=num_tasks, device=device)
        _run_baseline_loop(
            replay_trainer_360,
            replay_net_360,
            evaluator_replay_360,
            tasks,
            epochs_per_task,
        )
        results["Replay (P=360)"] = _record_baseline_result(
            replay_net_360, evaluator_replay_360
        )

    # -------------------------------------------------------------
    # 5. Baseline: Experience Replay (Buffer=250)
    # -------------------------------------------------------------
    if run("replay250"):
        _banner("Running Baseline 5: Experience Replay (Buffer=250)")
        set_seed(args.seed)
        replay_net = build_single_head(
            base_encoder, feature_dim, expert_hidden, num_classes, device
        )
        replay_trainer = ReplayTrainer(
            replay_net,
            buffer_size=250,
            lr=1e-3,
            device=device,
            sampling=args.buffer_sampling,
        )
        evaluator_replay = ContinualEvaluator(num_tasks=num_tasks, device=device)
        _run_baseline_loop(
            replay_trainer, replay_net, evaluator_replay, tasks, epochs_per_task
        )
        results["Experience Replay (Buffer=250)"] = _record_baseline_result(
            replay_net, evaluator_replay
        )

    # -------------------------------------------------------------
    # 6. Baseline: DER++ (Dark Experience Replay++, P=250)
    # -------------------------------------------------------------
    if run("derpp"):
        _banner("Running Baseline 6: DER++ (Dark Experience Replay++, P=250)")
        set_seed(args.seed)
        der_net = build_single_head(
            base_encoder, feature_dim, expert_hidden, num_classes, device
        )
        der_trainer = DERPP(
            der_net,
            buffer_size=250,
            lr=1e-3,
            device=device,
            sampling=args.buffer_sampling,
        )
        evaluator_der = ContinualEvaluator(num_tasks=num_tasks, device=device)
        _run_baseline_loop(der_trainer, der_net, evaluator_der, tasks, epochs_per_task)
        results["DER++ (P=250)"] = _record_baseline_result(der_net, evaluator_der)

    # -------------------------------------------------------------
    # 7. Baseline: ER-ACE (Asymmetric Cross-Entropy Replay, P=250)
    # -------------------------------------------------------------
    if run("erace"):
        _banner("Running Baseline 7: ER-ACE (Asymmetric Cross-Entropy Replay, P=250)")
        set_seed(args.seed)
        erace_net = build_single_head(
            base_encoder, feature_dim, expert_hidden, num_classes, device
        )
        erace_trainer = ERACE(
            erace_net,
            buffer_size=250,
            lr=1e-3,
            device=device,
            sampling=args.buffer_sampling,
        )
        evaluator_erace = ContinualEvaluator(num_tasks=num_tasks, device=device)
        _run_baseline_loop(
            erace_trainer,
            erace_net,
            evaluator_erace,
            tasks,
            epochs_per_task,
            per_task_kwargs=lambda task: {"current_classes": list(task.classes)},
        )
        results["ER-ACE (P=250)"] = _record_baseline_result(erace_net, evaluator_erace)

    # -------------------------------------------------------------
    # 8. Baseline: AGEM (Average Gradient Episodic Memory, P=250)
    # -------------------------------------------------------------
    if run("agem"):
        _banner("Running Baseline 8: AGEM (Average Gradient Episodic Memory, P=250)")
        set_seed(args.seed)
        agem_net = build_single_head(
            base_encoder, feature_dim, expert_hidden, num_classes, device
        )
        agem_trainer = AGEM(
            agem_net,
            buffer_size=250,
            lr=1e-3,
            device=device,
            sampling=args.buffer_sampling,
        )
        evaluator_agem = ContinualEvaluator(num_tasks=num_tasks, device=device)
        _run_baseline_loop(
            agem_trainer, agem_net, evaluator_agem, tasks, epochs_per_task
        )
        results["AGEM (P=250)"] = _record_baseline_result(agem_net, evaluator_agem)

    # -------------------------------------------------------------
    # 9. Baseline: iCaRL (Incremental Classifier & Representation Learning)
    # -------------------------------------------------------------
    if run("icarl"):
        _banner(
            "Running Baseline 9: iCaRL (Incremental Classifier & Representation Learning)"
        )
        set_seed(args.seed)
        icarl_net = build_single_head(
            base_encoder, feature_dim, expert_hidden, num_classes, device
        )
        icarl_trainer = ICaRL(
            icarl_net,
            exemplars_per_class=25,
            num_classes=num_classes,
            lr=1e-3,
            device=device,
        )
        evaluator_icarl = ContinualEvaluator(num_tasks=num_tasks, device=device)
        _run_baseline_loop(
            icarl_trainer,
            icarl_net,
            evaluator_icarl,
            tasks,
            epochs_per_task,
            per_task_kwargs=lambda task: {"current_classes": list(task.classes)},
            eval_model=icarl_trainer.eval_model,
        )
        results["iCaRL (k=25)"] = _record_baseline_result(icarl_net, evaluator_icarl)

    # -------------------------------------------------------------
    # 10. Baseline: Standard MoE Fine-tuning (Fixed 4 Experts)
    # -------------------------------------------------------------
    if run("stdmoe"):
        _banner(
            "Running Baseline 6: Standard MoE Fine-tuning (Fixed 4 Experts, Balanced)"
        )
        results["Standard MoE"] = _run_standard_moe(
            base_encoder,
            tasks,
            device,
            args.seed,
            epochs_per_task,
            feature_dim,
            expert_hidden,
            num_classes,
            num_tasks,
        )

    # -------------------------------------------------------------
    # 11. Proposed: PAL-MoE (pure, zero raw replay)
    # -------------------------------------------------------------
    if run("palmoe"):
        _banner(
            "Running Proposed: PAL-MoE (Prototype-Anchored Lifelong Mixture of Experts)"
        )
        results["PAL-MoE (Ours)"] = _run_palmoe_variant(
            base_encoder,
            tasks,
            device,
            args,
            dataset=dataset,
            feature_dim=feature_dim,
            expert_hidden=expert_hidden,
            num_classes=num_classes,
            proto_threshold=proto_threshold,
            encoder_ft=encoder_ft,
            epochs_per_task=epochs_per_task,
            output_dir=output_dir,
            checkpoint_subdir="checkpoints_palmoe",
            replay=False,
            max_proto_drop=max_proto_drop,
            max_proto_acc_drop=max_proto_acc_drop,
        )

    # -------------------------------------------------------------
    # 12. Proposed Extension: PAL-MoE + Replay (Hybrid, P=250)
    # -------------------------------------------------------------
    if run("hybrid"):
        _banner("Running Method 8: PAL-MoE + Replay (Hybrid, P=250 Exemplars)")
        results["PAL-MoE + Replay (Hybrid, P=250)"] = _run_palmoe_variant(
            base_encoder,
            tasks,
            device,
            args,
            dataset=dataset,
            feature_dim=feature_dim,
            expert_hidden=expert_hidden,
            num_classes=num_classes,
            proto_threshold=proto_threshold,
            encoder_ft=encoder_ft,
            epochs_per_task=epochs_per_task,
            output_dir=output_dir,
            checkpoint_subdir="checkpoints_hybrid",
            replay=True,
            max_proto_drop=max_proto_drop,
            max_proto_acc_drop=max_proto_acc_drop,
        )

    # Drop skipped methods so the table/JSON only contain what actually ran.
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
                (
                    f"{m['router_stability_kl']:.4f}"
                    if not np.isnan(m["router_stability_kl"])
                    else "-"
                ),
                (
                    f"{m['specialization_mi']:.3f}"
                    if not np.isnan(m["specialization_mi"])
                    else "-"
                ),
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

    meta_path = os.path.join(output_dir, f"benchmark_meta_seed{args.seed}.json")
    run_meta = {
        "seed": args.seed,
        "dataset": dataset,
        "device": str(device),
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "duration_sec": round(time.time() - _t_start, 1),
        "git_commit": _git_commit(),
        "torch": torch.__version__,
        "python": sys.version.split()[0],
        "args": vars(args),
    }
    with open(meta_path, "w") as f:
        json.dump(run_meta, f, indent=2, default=str)
    print(f"Saved run metadata to {meta_path}")

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
        help="Encoder pretraining epochs (None = dataset default: 1 for MNIST)",
    )
    parser.add_argument(
        "--max_experts",
        type=int,
        default=6,
        help="Expert pool budget; expansion beyond it triggers prune/merge",
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
        "--freeze_encoder",
        action="store_true",
        help=(
            "Keep the pretrained encoder fixed (recommended for prototype-anchored "
            "CIFAR runs; MNIST is always frozen)"
        ),
    )
    parser.add_argument(
        "--proto_routing_alpha",
        type=float,
        default=0.0,
        help="Prototype-anchored inference routing weight (0 = off, 1 = pure prototype routing)",
    )
    parser.add_argument(
        "--proto_routing_threshold",
        type=float,
        default=None,
        help="Max prototype distance for anchoring (None = memory distance_threshold)",
    )
    parser.add_argument(
        "--buffer_sampling",
        type=str,
        default="recency",
        choices=["recency", "reservoir"],
        help=(
            "Rehearsal-buffer policy for the memory baselines: 'recency' is the "
            "published behaviour, 'reservoir' is uniform over the task stream"
        ),
    )
    parser.add_argument(
        "--ewc_online",
        action="store_true",
        default=False,
        help="EWC: keep a single gamma-decayed Fisher instead of one per task",
    )
    parser.add_argument(
        "--router_anchor_steps",
        type=int,
        default=0,
        help=(
            "End-of-task router distillation steps onto the explicit prototype "
            "owners (zero-replay; 0 = off)"
        ),
    )
    parser.add_argument(
        "--router_anchor_lr",
        type=float,
        default=1e-3,
        help="Learning rate for the router anchor distillation",
    )
    parser.add_argument(
        "--feature_cache",
        action="store_true",
        default=False,
        help=(
            "Frozen-encoder feature caching: precompute h(x) once and run the whole "
            "pipeline on cached features (requires --freeze_encoder; mathematically "
            "equivalent, removes all encoder work from the training loop)"
        ),
    )
    parser.add_argument(
        "--proto_samples",
        type=int,
        default=256,
        help="Training samples registered into prototype memory per task",
    )
    parser.add_argument(
        "--proto_threshold",
        type=str,
        default="0.5",
        help=(
            "Prototype merge distance threshold; 'auto' calibrates it from the "
            "median nearest-neighbour distance of the registration batch"
        ),
    )
    parser.add_argument(
        "--proto_per_class",
        type=int,
        default=None,
        help="Class-balanced prototype eviction group size (None = per-task eviction)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=1,
        help="Experts mixed per input (1 = hard top-1; >1 = soft mixture)",
    )
    parser.add_argument(
        "--joint_calib_epochs",
        type=int,
        default=5,
        help="End-of-task joint latent calibration epochs (0 = disable calibration)",
    )
    parser.add_argument(
        "--refresh_anchors_after_calib",
        action="store_true",
        default=False,
        help="Recompute prototype r_p/o_p anchors with the calibrated model",
    )
    parser.add_argument(
        "--keep_optimizer_state",
        action="store_true",
        default=False,
        help="Carry Adam moments across task/expansion optimizer rebuilds",
    )
    parser.add_argument(
        "--save_checkpoints",
        action="store_true",
        default=False,
        help=(
            "Write per-task checkpoint files (model + prototype memory) for the "
            "PAL-MoE variants; off by default, they add hundreds of MB per run"
        ),
    )
    parser.add_argument(
        "--pretrain_cache",
        action="store_true",
        default=False,
        help=(
            "Cache the unsupervised/contrastive encoder pretraining under "
            "./data/pretrain_cache and reuse it when dataset, seed, epochs and "
            "architecture match (distributionally equivalent, not bit-identical)"
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
        "--router_type",
        type=str,
        default="dynamic",
        choices=["dynamic", "distance", "attention"],
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
    parser.add_argument(
        "--lambda_r",
        type=float,
        default=0.5,
        help="Router stability loss weight (KL to stored routing targets)",
    )
    parser.add_argument(
        "--lambda_e",
        type=float,
        default=2.5,
        help="Expert stability loss weight (MSE to stored output anchors)",
    )
    parser.add_argument(
        "--stability_every",
        type=int,
        default=1,
        help=(
            "Apply the stability losses every k-th step with weight scaled by k "
            "(1 = every step, the published recipe)"
        ),
    )
    parser.add_argument(
        "--ood_every",
        type=int,
        default=1,
        help=(
            "Apply the OOD negative-boundary term every k-th step with weight "
            "scaled by k (1 = every step, the published recipe)"
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "JSON config file (validated; explicit CLI flags take precedence, "
            "config values take precedence over defaults)"
        ),
    )
    args = parser.parse_args()

    if args.config:
        try:
            apply_config(args, parser, sys.argv[1:], args.config)
        except ConfigError as _cfg_err:
            raise SystemExit(f"[config] {_cfg_err}") from None

    if args.feature_cache and not args.freeze_encoder:
        raise SystemExit(
            "--feature_cache requires --freeze_encoder: cached features are only "
            "valid while the encoder is frozen."
        )
    if (
        args.router_anchor_steps > 0
        and not args.freeze_encoder
        and args.dataset in ("cifar10", "cifar100")
    ):
        print(
            "[warn] --router_anchor_steps with a trainable CIFAR encoder distills "
            "stale prototype anchors; add --freeze_encoder for correct behaviour."
        )

    run_benchmark(
        args,
        epochs_per_task=args.epochs,
        device_str=args.device,
        output_dir=args.output_dir,
        dataset=args.dataset,
    )
