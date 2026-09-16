"""
Checkpoint persistence for PAL-MoE.

Bundles the DynamicMoE state (encoder + router + experts), the PrototypeMemory
prototype anchors, and arbitrary metadata (e.g. task_id, metrics) into a single
torch.save file. Tensors are moved to CPU on save and back to `device` on load.
"""

import os
from typing import Any, Dict, Optional

import torch

from .memory.prototype_memory import Prototype, PrototypeMemory


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    prototype_memory: PrototypeMemory,
    meta: Optional[Dict[str, Any]] = None,
) -> str:
    """Saves model + prototype memory state to `path`. Returns the path."""
    mem_state = []
    for proto in prototype_memory.prototypes:
        mem_state.append(
            {
                "v_p": proto.v_p.detach().cpu(),
                "r_p": proto.r_p.detach().cpu(),
                "o_p": proto.o_p.detach().cpu(),
                "task_id": proto.task_id,
                "owner_expert": proto.owner_expert,
                "count": proto.count,
                "x_p": proto.x_p.detach().cpu() if proto.x_p is not None else None,
                "y_p": proto.y_p.detach().cpu() if proto.y_p is not None else None,
                "raw_x": proto.raw_x.detach().cpu()
                if proto.raw_x is not None
                else None,
            }
        )
    payload = {
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "prototypes": mem_state,
        "meta": meta or {},
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(payload, path)
    return path


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    prototype_memory: PrototypeMemory,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """Restores model + prototype memory from `path`. Returns stored meta."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state"], strict=False)
    model.to(device)

    prototype_memory.prototypes = []
    for p in payload["prototypes"]:
        prototype_memory.prototypes.append(
            Prototype(
                v_p=p["v_p"].to(device),
                r_p=p["r_p"].to(device),
                o_p=p["o_p"].to(device),
                task_id=p["task_id"],
                owner_expert=p.get("owner_expert"),
                count=p.get("count", 1),
                x_p=p["x_p"].to(device) if p["x_p"] is not None else None,
                y_p=p["y_p"].to(device) if p["y_p"] is not None else None,
                raw_x=p["raw_x"].to(device) if p.get("raw_x") is not None else None,
            )
        )
    return payload.get("meta", {})
