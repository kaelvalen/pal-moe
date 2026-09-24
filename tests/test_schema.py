"""
Tests for the measurement contract (docs/MEASUREMENT_CONTRACT.md, S0).

The contract is only useful if violations fail loudly: a record that declares a
metric block without the block's fields, or that mixes a budget into the
metrics, must be rejected rather than silently averaged into a table.
"""

import json

import pytest

from pal_moe.evaluation.schema import (
    BLOCK_FIELDS,
    BLOCKS,
    aggregate_runs,
    build_run_record,
    load_run_records,
    paired_delta,
    satisfied_blocks,
    upgrade_v0_result,
    validate_run_record,
)


def _deep_merge(base: dict, extra: dict) -> dict:
    """Recursive merge so overriding one metric keeps the others."""
    out = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _record(**overrides):
    factors = {
        "dataset": "cifar100",
        "protocol": "class_il",
        "task_id_at_inference": False,
        "seed": 42,
        "model_family": "ncm",
        "backbone": "vit_b_16",
        "backbone_pretraining": "imagenet_frozen",
    }
    metrics = {
        "learning": {
            "accuracy": 0.7034,
            "forgetting": 0.0975,
            "bwt": -0.0975,
            "acc_matrix": [[0.7]],
        },
        "cost": {"stored_bytes": 307200, "total_params": 0},
    }
    provenance = {"git_commit": "abc1234"}
    factors = _deep_merge(factors, overrides.pop("factors", {}))
    metrics = _deep_merge(metrics, overrides.pop("metrics", {}))
    provenance = _deep_merge(provenance, overrides.pop("provenance", {}))
    return build_run_record(
        factors=factors,
        metrics=metrics,
        provenance=provenance,
        blocks=overrides.pop("blocks", None),
        **overrides,
    )


# ---------------------------------------------------------------- run records


def test_core_record_validates_and_declares_its_blocks():
    record = _record()
    assert validate_run_record(record) == []
    assert record["schema_version"] == "1.0"
    assert record["blocks"] == ["cost", "learning"]
    assert record["reproducible"] is True
    assert record["run_id"].startswith("cifar100__class_il__vit_b_16__ncm")


def test_blocks_are_derived_from_the_data():
    """R4: a declaration cannot drift from reality."""
    record = _record(
        metrics={
            "modular": {
                "expert_count": 20,
                "routing_entropy": 0.1,
                "utilization": 0.99,
            }
        }
    )
    assert record["blocks"] == ["cost", "learning", "modular"]
    assert validate_run_record(record) == []


def test_declaring_a_block_without_its_fields_is_an_error():
    record = _record(blocks=["learning", "modular"])
    problems = validate_run_record(record)
    assert any("metrics.modular.expert_count" in p for p in problems)
    assert any("metrics.modular.utilization" in p for p in problems)


def test_generalization_block_requires_order_transfer_and_drift():
    """R2/R4: the order is recorded explicitly, not only as a seed."""
    record = _record(blocks=["learning", "generalization"])
    problems = validate_run_record(record)
    assert any("metrics.learning.fwt" in p for p in problems)
    assert any("metrics.generalization.transfer_accuracy" in p for p in problems)
    assert any("metrics.stability.representation_drift" in p for p in problems)


def test_generalization_block_is_reachable_when_measured():
    record = _record(
        factors={"class_order": [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]},
        metrics={
            "learning": {"fwt": 0.011},
            "generalization": {"transfer_accuracy": 0.61},
            "stability": {"representation_drift": 0.0, "routing_drift": 0.0},
        },
    )
    assert "generalization" in record["blocks"]
    assert validate_run_record(record) == []


def test_a_legacy_run_can_declare_learning_only():
    """R4/R6: a run from before byte accounting is ingestible, not rejected."""
    legacy = _record(metrics={"cost": {"stored_bytes": None, "total_params": None}})
    assert satisfied_blocks(legacy) == ["learning"]
    assert validate_run_record(legacy) == []


def test_budget_is_a_factor_not_a_metric():
    """R1."""
    record = _record()
    record["metrics"]["budget"] = {"memory_bytes": 1024}
    problems = validate_run_record(record)
    assert any("is a factor, not a metric" in p for p in problems)


def test_missing_metrics_are_null_not_zero():
    """R5: a single-head baseline has no expert count, and that is not 0."""
    record = _record()
    assert record["metrics"]["modular"]["expert_count"] is None
    assert record["metrics"]["cost"]["memory_bytes"] is None
    assert record["factors"]["data_fraction"] == 1.0


def test_missing_seed_is_not_reproducible():
    record = _record(factors={"seed": None})
    assert record["reproducible"] is False
    assert validate_run_record(record) == []


def test_nan_is_treated_as_unmeasured():
    record = _record(metrics={"cost": {"stored_bytes": float("nan")}})
    assert record["metrics"]["cost"]["stored_bytes"] is None
    assert "cost" not in record["blocks"]


# ------------------------------------------------------------- study records


def test_aggregate_refuses_to_mix_protocols():
    """R3."""
    a = _record()
    b = _record(factors={"protocol": "task_il", "task_id_at_inference": True})
    with pytest.raises(ValueError, match="refusing to aggregate across protocols"):
        aggregate_runs([a, b])


def test_aggregate_refuses_records_missing_required_blocks():
    """R4 at the study level."""
    thin = _record(metrics={"cost": {"stored_bytes": None, "total_params": None}})
    with pytest.raises(ValueError, match="missing required blocks"):
        aggregate_runs([thin], requires_blocks=["learning", "cost"])


def test_aggregate_reports_mean_and_bootstrap_ci():
    runs = [
        _record(
            factors={"seed": seed},
            metrics={"learning": {"accuracy": 0.70 + 0.01 * i}},
        )
        for i, seed in enumerate([42, 1, 2])
    ]
    agg = aggregate_runs(runs, group_by="factors.model_family")
    assert agg["ncm"]["n"] == 3
    assert agg["ncm"]["mean"] == pytest.approx(0.71)
    lo, hi = agg["ncm"]["ci95"]
    assert lo <= agg["ncm"]["mean"] <= hi


def test_paired_delta_matches_cells_and_flags_unpaired():
    """R7: a claim needs a partner cell with the same seed and order."""
    base = _record(factors={"seed": 42, "model_family": "ncm"})
    better = _record(
        factors={"seed": 42, "model_family": "moe"},
        metrics={"learning": {"accuracy": 0.75}},
    )
    lonely = _record(factors={"seed": 1, "model_family": "moe"})
    out = paired_delta([base, better, lonely], baseline_family="ncm")
    assert out["moe"]["n"] == 1
    assert out["moe"]["mean_delta"] == pytest.approx(0.75 - 0.7034)
    assert out["moe"]["wins"] == 1
    assert out["moe"]["unpaired"] is True


# ------------------------------------------------------------ v0 ingestion


def _v0_pal_moe():
    result = {
        "acc": 0.593,
        "forgetting": 0.1864,
        "bwt": -0.1546,
        "final_experts": 20,
        "total_params": 13360101,
        "trainable_params": 682616,
        "active_params": 682617,
        "memory_bytes": 14231076,
        "state_bytes": 0,
        "stored_bytes": 14231076,
        "fit_seconds": 112.9,
        "utilization": 0.993,
        "specialization_mi": 1.887,
        "router_diagnostics": {"routing_entropy_mean": 0.097},
        "acc_matrix": [[0.5]],
    }
    meta = {
        "dataset": "cifar100",
        "seed": 42,
        "git_commit": "82b5684",
        "args": {
            "encoder_arch": "vit_b_16",
            "encoder_weights": "imagenet",
            "freeze_encoder": True,
            "classes_per_task": 5,
            "eval_head": "moe",
        },
    }
    return result, meta


def test_v0_result_upgrades_without_inventing_numbers():
    """R6: old results stay readable, and unmeasured fields stay null."""
    result, meta = _v0_pal_moe()
    record = upgrade_v0_result(result, meta)
    assert record["blocks"] == ["cost", "learning", "modular"]
    assert record["reproducible"] is True
    assert validate_run_record(record) == []
    assert record["factors"]["backbone_pretraining"] == "imagenet_frozen"
    assert record["metrics"]["cost"]["stored_bytes"] == 14231076
    # Never measured in v0, so null rather than a fabricated zero.
    assert record["factors"]["class_order"] is None
    assert record["metrics"]["learning"]["fwt"] is None
    assert record["metrics"]["generalization"]["transfer_accuracy"] is None


def test_v0_stored_bytes_is_derived_from_its_components():
    """`stored_bytes` is defined as memory + state, so deriving it is faithful."""
    result, meta = _v0_pal_moe()
    del result["stored_bytes"]
    record = upgrade_v0_result(result, meta)
    assert record["metrics"]["cost"]["stored_bytes"] == 14231076


def test_v0_multi_method_payload_yields_one_record_per_method(tmp_path):
    result, meta = _v0_pal_moe()
    # A baseline without the modular diagnostics declares only what it has.
    other = {
        "acc": 0.1,
        "forgetting": 0.5,
        "acc_matrix": [[0.1]],
        "total_params": 667237,
        "memory_bytes": 0,
        "state_bytes": 0,
        "stored_bytes": 0,
    }
    (tmp_path / "benchmark_results_seed42.json").write_text(
        json.dumps({"PAL-MoE (Ours)": result, "Naive Fine-tuning": other})
    )
    (tmp_path / "benchmark_meta_seed42.json").write_text(json.dumps(meta))
    records = load_run_records(str(tmp_path / "benchmark_results_seed42.json"))
    assert [r["factors"]["model_family"] for r in records] == [
        "PAL-MoE (Ours)",
        "Naive Fine-tuning",
    ]
    assert records[0]["blocks"] == ["cost", "learning", "modular"]
    assert records[1]["blocks"] == ["cost", "learning"]


def test_blocks_are_a_closed_set_with_fields():
    for block in BLOCKS:
        assert BLOCK_FIELDS[block]
