"""
The PAL-MoE measurement contract, in code (docs/MEASUREMENT_CONTRACT.md, S0).

Every runner emits a *run record*; every analysis consumes run records and
emits a *study record*. The contract exists so that no metric has to be added
after an experiment has run, and so that one table can hold NCM, a linear
probe, iCaRL and an MoE without special cases.

Design rules enforced here (see the doc for the rationale):

- R1 factors and metrics never mix; a budget is a factor, the achieved cost is
  a metric.
- R2 the class order is recorded explicitly, not only as a seed.
- R3 the protocol is explicit; aggregation across different protocols is
  refused.
- R4 a record declares the metric BLOCKS it fully satisfies. Blocks are
  derived from the data by `build_run_record`, so a declaration cannot drift
  from reality, and a legacy run that never measured its byte cost simply
  declares `["learning"]` instead of being rejected.
- R5 `null` means "not measured"; `0` means zero.
- R6 the existing v0 result JSONs stay readable via `upgrade_v0_records`.
- R7 paired comparisons are the default claim.
- R8 no required field may reference a model-specific structure.

Stdlib only on purpose: analysis scripts must be able to read records without
torch.
"""

from __future__ import annotations

import glob
import json
import os
import random
from collections.abc import Iterable
from typing import Any

__all__ = [
    "SCHEMA_VERSION",
    "BLOCKS",
    "BLOCK_FIELDS",
    "build_run_record",
    "validate_run_record",
    "load_run_records",
    "load_run_record",
    "upgrade_v0_result",
    "upgrade_v0_payload",
    "record_from_runner_result",
    "aggregate_runs",
    "paired_delta",
]

SCHEMA_VERSION = "1.0"

# A block is a group of metrics that a study either needs or does not need.
# Declaring a block is a claim that every field in it is non-null (R4).
BLOCKS = ("learning", "cost", "compute", "modular", "generalization", "robustness")

BLOCK_FIELDS: dict[str, tuple[str, ...]] = {
    "learning": (
        "metrics.learning.accuracy",
        "metrics.learning.forgetting",
        "metrics.learning.acc_matrix",
    ),
    "cost": (
        "metrics.cost.stored_bytes",
        "metrics.cost.total_params",
    ),
    "compute": (
        "metrics.cost.flops_forward",
        "metrics.cost.latency_ms",
        "metrics.cost.optimizer_steps",
    ),
    "modular": (
        "metrics.modular.expert_count",
        "metrics.modular.routing_entropy",
        "metrics.modular.utilization",
    ),
    "generalization": (
        "metrics.learning.fwt",
        "metrics.generalization.transfer_accuracy",
        "metrics.stability.representation_drift",
        "metrics.stability.routing_drift",
    ),
    "robustness": (
        "metrics.generalization.corruption_accuracy",
        "metrics.generalization.confounder_split_accuracy",
    ),
}

# Factors that must be PRESENT as keys for a record to be interpretable at all.
# Their values may be null when the run predates the field (R5/R6); a study that
# needs them (e.g. pairing) checks `reproducible`.
REQUIRED_FACTOR_KEYS = (
    "dataset",
    "protocol",
    "task_id_at_inference",
    "seed",
    "model_family",
    "backbone",
    "backbone_pretraining",
)

_FACTOR_DEFAULTS: dict[str, Any] = {
    "dataset": None,
    "protocol": None,
    "task_id_at_inference": None,
    "num_tasks": None,
    "classes_per_task": None,
    "class_order": None,
    "task_order_seed": None,
    "seed": None,
    "model_family": None,
    "backbone": None,
    "backbone_pretraining": None,
    "readout": None,
    "readout_estimator": None,
    "expert": None,
    # Protocol refinements (S5). `protocol` says which stream this is;
    # `class_masking` and `routing_mode` say which information the model was
    # given at INFERENCE, and they are independent: an oracle-routed class-IL
    # run is not a Task-IL run. Mixing them is the mistake the schema prevents
    # (docs/MEASUREMENT_CONTRACT.md R3).
    "class_masking": None,
    "routing_mode": None,
    "increment_type": None,
    "class_space": None,
    "data_fraction": 1.0,
    "budget": {
        "memory_bytes": None,
        "params": None,
        "compute_flops": None,
        "steps": None,
    },
}

_METRIC_DEFAULTS: dict[str, Any] = {
    "learning": {
        "accuracy": None,
        "forgetting": None,
        "bwt": None,
        "fwt": None,
        "acc_matrix": None,
        "acc_curve": None,
    },
    "generalization": {
        "transfer_accuracy": None,
        "unseen_task_accuracy": None,
        "unseen_domain_accuracy": None,
        "corruption_accuracy": None,
        "confounder_split_accuracy": None,
    },
    "cost": {
        "memory_bytes": None,
        "state_bytes": None,
        "stored_bytes": None,
        "total_params": None,
        "trainable_params": None,
        "active_params": None,
        "flops_forward": None,
        "latency_ms": None,
        "fit_seconds": None,
        "optimizer_steps": None,
    },
    "modular": {
        "expert_count": None,
        "experts_per_task": None,
        "reuse_rate": None,
        "routing_entropy": None,
        "utilization": None,
        "specialization_mi": None,
        "task_recall_at_k": None,
        "oracle_accuracy": None,
    },
    "stability": {
        "representation_drift": None,
        "routing_drift": None,
        "plasticity": None,
        "stability": None,
    },
}


# --------------------------------------------------------------------------
# accessors
# --------------------------------------------------------------------------


def _get(record: dict, path: str) -> Any:
    node: Any = record
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _clean(value: Any) -> Any:
    """NaN/Inf mean 'not measured' and become null (R5).

    The v0 files store `float('nan')` for diagnostics a baseline does not have;
    leaving NaN in place would break strict JSON and would silently count as a
    measured value in the block check.
    """
    if isinstance(value, float) and (
        value != value or value in (float("inf"), float("-inf"))
    ):
        return None
    return value


def _deep_defaults(defaults: dict, provided: dict | None) -> dict:
    """Fill missing keys from `defaults`, recursively. Provided keys win."""
    out: dict[str, Any] = {}
    for key, default in defaults.items():
        if isinstance(default, dict):
            sub = (provided or {}).get(key)
            out[key] = _deep_defaults(default, sub if isinstance(sub, dict) else None)
        else:
            out[key] = _clean((provided or {}).get(key, default))
    for key, value in (provided or {}).items():
        out.setdefault(key, _clean(value))
    return out


def satisfied_blocks(record: dict) -> list[str]:
    """Blocks whose every field is non-null."""
    return [
        block
        for block in BLOCKS
        if all(_get(record, path) is not None for path in BLOCK_FIELDS[block])
    ]


# --------------------------------------------------------------------------
# run records
# --------------------------------------------------------------------------


def build_run_record(
    factors: dict,
    metrics: dict | None = None,
    provenance: dict | None = None,
    blocks: Iterable[str] | None = None,
    run_id: str | None = None,
) -> dict:
    """Assemble a run record.

    `blocks` is derived from the data by default. Passing it explicitly is
    allowed and is checked by the validator, which is what makes a hand-written
    record fail loudly instead of over-claiming.
    """
    record = {
        "schema_version": SCHEMA_VERSION,
        "factors": _deep_defaults(_FACTOR_DEFAULTS, factors),
        "metrics": _deep_defaults(_METRIC_DEFAULTS, metrics),
        "provenance": dict(provenance or {}),
    }
    derived = satisfied_blocks(record)
    record["blocks"] = sorted(set(blocks) if blocks is not None else derived)
    record["reproducible"] = bool(
        record["factors"].get("seed") is not None
        and record["provenance"].get("git_commit") is not None
        and record["factors"].get("backbone") not in (None, "unknown")
    )
    record["run_id"] = run_id or _default_run_id(record)
    return record


def _default_run_id(record: dict) -> str:
    f = record["factors"]
    return "__".join(
        [
            str(f.get("dataset")),
            str(f.get("protocol")),
            str(f.get("backbone")),
            str(f.get("model_family")),
            f"seed{f.get('seed')}",
            f"order{f.get('task_order_seed')}",
        ]
    )


def validate_run_record(record: dict) -> list[str]:
    """Return a list of problems; empty means the record is conformant."""
    problems: list[str] = []
    if not isinstance(record, dict):
        return ["record is not an object"]

    for key in ("factors", "metrics", "provenance"):
        if not isinstance(record.get(key), dict):
            problems.append(f"{key} must be an object")
    if not isinstance(record.get("blocks"), list):
        problems.append("blocks must be a list")

    # R1: a budget is a factor, never a metric.
    for budget_key in ("budget", "budget_bytes", "budget_params"):
        if budget_key in (record.get("metrics") or {}):
            problems.append(f"metrics.{budget_key} is a factor, not a metric")

    # The factors that make a record interpretable must at least be present.
    for key in REQUIRED_FACTOR_KEYS:
        if key not in (record.get("factors") or {}):
            problems.append(f"factors.{key} is missing (may be null, not absent)")

    # R4: a declared block must actually be complete.
    declared = record.get("blocks") or []
    for block in declared:
        if block not in BLOCKS:
            problems.append(f"unknown block {block!r}")
            continue
        for path in BLOCK_FIELDS[block]:
            if _get(record, path) is None:
                problems.append(f"block {block!r} requires non-null {path}")
    return problems


def load_run_records(path: str) -> list[dict]:
    """Load every record in a file: one per method for v0, or the single v1 record.

    The v0 layout is `<dir>/benchmark_results_seed42.json` next to
    `<dir>/benchmark_meta_seed42.json`, and the dataset/seed/git fields live in
    the meta. The sibling meta is discovered automatically so that ingesting
    the 200+ existing result files needs no per-file arguments.
    """
    with open(path) as fh:
        payload = json.load(fh)
    if "schema_version" in payload and "factors" in payload:
        return [payload]
    return upgrade_v0_payload(payload, _sibling_meta(path))


def load_run_record(path: str) -> dict:
    """First record of a file. See `load_run_records`."""
    return load_run_records(path)[0]


def _sibling_meta(path: str) -> dict | None:
    candidates = sorted(
        glob.glob(os.path.join(os.path.dirname(path), "benchmark_meta_*.json"))
    )
    if not candidates:
        return None
    with open(candidates[0]) as fh:
        return json.load(fh)


def upgrade_v0_payload(payload: dict, meta: dict | None = None) -> list[dict]:
    """Upgrade a v0 results payload (one result or a dict of methods)."""
    if "acc" in payload or "forgetting" in payload:
        return [upgrade_v0_result(payload, meta)]
    records = []
    for method, result in payload.items():
        if not isinstance(result, dict):
            continue
        with_method = dict(result)
        with_method.setdefault("method", method)
        records.append(upgrade_v0_result(with_method, meta))
    return records


def record_from_runner_result(result: dict, meta: dict | None = None) -> dict:
    """Emit the contract record for a runner result (the native production path).

    `upgrade_v0_result` is the same mapping, named for the historical case;
    new runner code calls this one so the call site reads as what it is.
    """
    return upgrade_v0_result(result, meta)


def upgrade_v0_result(result: dict, meta: dict | None = None) -> dict:
    """Map a v0 benchmark result (and its meta) onto the contract.

    Unmeasured metrics stay `null` (R5); metadata that was never recorded
    becomes the string "unknown" so the record is still interpretable as a
    run, and `reproducible` is False. The derived `stored_bytes` is the
    documented sum of its components, not a fabricated number.
    """
    meta = meta or {}
    args = meta.get("args") if isinstance(meta.get("args"), dict) else {}
    args = args or {}
    method = str(result.get("method") or args.get("methods") or "unknown")

    frozen = args.get("freeze_encoder")
    weights = args.get("encoder_weights") or "none"
    if frozen is True:
        pretraining = f"{weights}_frozen" if weights != "none" else "frozen"
    elif frozen is False:
        pretraining = f"{weights}_finetuned" if weights != "none" else "finetuned"
    else:
        pretraining = "unknown"

    memory_bytes = _clean(result.get("memory_bytes"))
    state_bytes = _clean(result.get("state_bytes"))
    stored_bytes = _clean(result.get("stored_bytes"))
    if stored_bytes is None and memory_bytes is not None:
        stored_bytes = int(memory_bytes) + int(state_bytes or 0)

    acc_matrix = result.get("acc_matrix")
    diagnostics = result.get("router_diagnostics")
    modular = {
        "expert_count": _clean(result.get("final_experts")),
        "experts_per_task": _clean(result.get("experts_per_task")),
        "utilization": _clean(result.get("utilization")),
        "specialization_mi": _clean(result.get("specialization_mi")),
        "routing_entropy": _clean(
            diagnostics.get("routing_entropy_mean")
            if isinstance(diagnostics, dict)
            else None
        ),
    }
    factors = {
        "dataset": meta.get("dataset") or "unknown",
        "protocol": "class_il",
        "task_id_at_inference": False,
        "num_tasks": len(acc_matrix) if isinstance(acc_matrix, list) else None,
        "classes_per_task": _clean(args.get("classes_per_task")),
        "class_order": None,  # not recorded in v0
        "task_order_seed": None,
        "seed": _clean(meta.get("seed")),
        "model_family": method,
        "backbone": meta.get("backbone") or args.get("encoder_arch") or "unknown",
        "backbone_pretraining": pretraining,
        "readout": args.get("eval_head") or "moe",
        "expert": "mlp_head",
    }
    metrics = {
        "learning": {
            "accuracy": _clean(result.get("acc")),
            "forgetting": _clean(result.get("forgetting")),
            "bwt": _clean(result.get("bwt")),
            "acc_matrix": acc_matrix,
        },
        "cost": {
            "memory_bytes": memory_bytes,
            "state_bytes": state_bytes,
            "stored_bytes": stored_bytes,
            "total_params": _clean(result.get("total_params")),
            "trainable_params": _clean(result.get("trainable_params")),
            "active_params": _clean(result.get("active_params")),
            "fit_seconds": _clean(result.get("fit_seconds")),
        },
        "modular": modular,
    }
    return build_run_record(
        factors=factors,
        metrics=metrics,
        provenance={
            "git_commit": meta.get("git_commit") or "unknown",
            "timestamp": meta.get("timestamp"),
            "duration_sec": _clean(meta.get("duration_sec")),
            "torch": meta.get("torch"),
            "python": meta.get("python"),
            "device": meta.get("device"),
        },
    )


# --------------------------------------------------------------------------
# study records
# --------------------------------------------------------------------------


def aggregate_runs(
    records: Iterable[dict],
    path: str = "metrics.learning.accuracy",
    group_by: str | None = None,
    requires_blocks: Iterable[str] = (),
    n_boot: int = 2000,
    boot_seed: int = 0,
) -> dict:
    """Mean / std / bootstrap CI95 of `path`, optionally per group.

    R3: refuses to aggregate records with different `(protocol,
    task_id_at_inference)`. R4: refuses records that do not declare the blocks
    the study requires.
    """
    records = list(records)
    protocols = {
        (
            r["factors"].get("protocol"),
            r["factors"].get("task_id_at_inference"),
            r["factors"].get("class_masking"),
            r["factors"].get("routing_mode"),
        )
        for r in records
    }
    if len(protocols) > 1:
        raise ValueError(
            "refusing to aggregate across protocols (R3): "
            f"{sorted(protocols, key=str)}"
        )
    requires = set(requires_blocks)
    missing = [
        r.get("run_id")
        for r in records
        if not requires.issubset(set(r.get("blocks") or ()))
    ]
    if missing:
        raise ValueError(
            f"records missing required blocks {sorted(requires)} (R4): {missing[:3]}"
        )

    groups: dict[Any, list[float]] = {}
    for record in records:
        value = _get(record, path)
        if value is None:
            continue
        key = _get(record, group_by) if group_by else "all"
        groups.setdefault(key, []).append(float(value))

    return {
        key: {
            "mean": sum(values) / len(values),
            "std": _std(values),
            "ci95": _bootstrap_ci(values, n_boot=n_boot, seed=boot_seed),
            "n": len(values),
        }
        for key, values in groups.items()
    }


def paired_delta(
    records: Iterable[dict],
    baseline_family: str,
    path: str = "metrics.learning.accuracy",
    family_key: str = "factors.model_family",
) -> dict:
    """Paired delta of every family against `baseline_family` (R7).

    Pairs are matched on `(seed, class_order, backbone)`; a family with an
    unmatched cell is flagged `unpaired` and its difference is descriptive.
    """
    cells: dict[tuple, dict] = {}
    for record in records:
        f = record["factors"]
        key = (
            f.get("seed"),
            json.dumps(f.get("class_order"), sort_keys=True),
            f.get("backbone"),
        )
        cells.setdefault(key, {})[_get(record, family_key)] = _get(record, path)

    deltas: dict[str, list[float]] = {}
    unpaired: dict[str, int] = {}
    for cell in cells.values():
        base = cell.get(baseline_family)
        for family, value in cell.items():
            if family == baseline_family:
                continue
            if base is None or value is None:
                unpaired[family] = unpaired.get(family, 0) + 1
                continue
            deltas.setdefault(family, []).append(float(value) - float(base))

    return {
        family: {
            "mean_delta": sum(values) / len(values),
            "ci95": _bootstrap_ci(values, n_boot=2000, seed=0),
            "n": len(values),
            "wins": sum(1 for v in values if v > 0),
            "losses": sum(1 for v in values if v < 0),
            "unpaired": unpaired.get(family, 0) > 0,
        }
        for family, values in deltas.items()
    }


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5


def _bootstrap_ci(
    values: list[float], n_boot: int = 2000, seed: int = 0
) -> list[float]:
    """Percentile bootstrap CI95. Small n is the norm here, so no normal approx."""
    if len(values) < 2:
        return [values[0], values[0]] if values else [float("nan")] * 2
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[min(int(0.975 * n_boot), n_boot - 1)]
    return [lo, hi]
