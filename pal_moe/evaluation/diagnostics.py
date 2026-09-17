"""
Router / expert diagnostics panel.

The routing lock incident (BENCHMARK design fact 15) showed that aggregate
accuracy hides what the router actually does. This module makes the routing
behaviour observable in every run:

- per-task top-1 share for every expert,
- global expert-usage entropy (1.0 = perfectly balanced),
- routing entropy on stored prototypes (sharp vs. uncertain routing),
- owner routing accuracy: how often the router sends a prototype to the expert
  that was trained for its task (explicit `owner_expert`),
- prototype-space geometry report (margin / silhouette / class separation).

All metrics are computed on a bounded number of batches so they can run after
every task without dominating the benchmark.
"""

from typing import Any, Optional

import torch

from .geometry import geometry_report

__all__ = ["router_diagnostics", "print_router_diagnostics"]


@torch.no_grad()
def router_diagnostics(
    model: Any,
    tasks: list[Any],
    device: torch.device,
    prototype_memory: Optional[Any] = None,
    max_batches_per_task: int = 8,
) -> dict[str, Any]:
    """Computes the routing/expert panel for the current model state."""
    model.eval()
    num_experts = model.num_experts
    num_tasks = len(tasks)

    counts = torch.zeros(num_tasks, num_experts, dtype=torch.float64)
    routing_entropies = []
    for t_idx, task in enumerate(tasks):
        loader = getattr(task, "val_loader", None) or task.test_loader
        for batch_idx, (x, _) in enumerate(loader):
            if batch_idx >= max_batches_per_task:
                break
            h = model.get_routing_features(x.to(device))
            weights, topk_idx, _ = model.router(h)
            flat = topk_idx.flatten()
            counts[t_idx] += torch.bincount(flat.cpu(), minlength=num_experts)
            routing_entropies.append(
                model.router.compute_entropy(weights).mean().item()
            )

    total = counts.sum().item()
    usage = counts.sum(dim=0)
    usage_frac = usage / max(total, 1.0)
    usage_entropy = float(-(usage_frac * torch.log(usage_frac + 1e-12)).sum().item())
    diagnostics: dict[str, Any] = {
        "num_experts": num_experts,
        "task_expert_counts": counts.tolist(),
        "usage_entropy": usage_entropy,
        "usage_entropy_normalized": float(
            usage_entropy
            / max(float(torch.log(torch.tensor(float(num_experts)))), 1e-9)
        ),
        "routing_entropy_mean": float(
            sum(routing_entropies) / max(len(routing_entropies), 1)
        ),
        "owners": {},
        "prototype_geometry": {},
    }

    if prototype_memory is not None and not prototype_memory.is_empty():
        v_mat = prototype_memory.get_prototype_matrix(device)
        owners = [
            p.owner_expert
            for p in prototype_memory.prototypes
            if p.owner_expert is not None
        ]
        if v_mat is not None:
            probs = model.router.get_full_distribution(v_mat)
            diagnostics["routing_entropy_on_prototypes"] = float(
                model.router.compute_entropy(probs).mean().item()
            )
            owner_v = torch.tensor(
                [p.owner_expert for p in prototype_memory.prototypes],
                device=device,
            )
            task_ids = prototype_memory.get_task_id_vector(device)
            owner_mask = owner_v >= 0
            if bool(owner_mask.any()):
                pred = probs[owner_mask].argmax(dim=-1)
                diagnostics["owners"]["owner_routing_accuracy"] = float(
                    (pred == owner_v[owner_mask]).float().mean().item()
                )
                diagnostics["owners"]["num_owned_prototypes"] = int(
                    owner_mask.sum().item()
                )
            diagnostics["owners"]["owner_histogram"] = {
                int(k): int(v)
                for k, v in zip(*torch.unique(owner_v, return_counts=True))
            }
            diagnostics["prototype_geometry"] = geometry_report(
                v_mat, task_ids, max_samples=2000
            )
        diagnostics["num_prototypes"] = len(prototype_memory.prototypes)
        diagnostics["owners"]["num_owner_experts"] = len(set(owners))

    return diagnostics


def print_router_diagnostics(diagnostics: dict[str, Any], prefix: str = "    ") -> None:
    """Compact human-readable summary of `router_diagnostics`."""
    print(
        f"{prefix}[router] experts={diagnostics['num_experts']} "
        f"usage_entropy={diagnostics['usage_entropy']:.3f} "
        f"(norm {diagnostics['usage_entropy_normalized']:.3f}) "
        f"routing_H={diagnostics['routing_entropy_mean']:.3f}"
    )
    owners = diagnostics.get("owners", {})
    if "owner_routing_accuracy" in owners:
        print(
            f"{prefix}[router] owner routing accuracy="
            f"{owners['owner_routing_accuracy']:.2%} "
            f"over {owners['num_owned_prototypes']} prototypes"
        )
    geom = diagnostics.get("prototype_geometry", {})
    if geom:
        print(
            f"{prefix}[geometry] margin={geom['margin_mean']:.3f} "
            f"(pos {geom['margin_positive_frac']:.0%}) "
            f"silhouette={geom['silhouette']:.3f} "
            f"sep_ratio={geom.get('class_mean_separation_ratio', float('nan')):.3f}"
        )
