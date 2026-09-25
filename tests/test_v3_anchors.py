"""The v3 restructure's anchors: one test per anchor.

Fast tests (always run when the stored JSONs exist): the moved code is the same
object behind every old import path, and the moved statistics reproduce the stored
reports exactly.

Cell re-runs (GPU, ~40 s per cell): opt in with `PAL_MOE_ANCHORS=1`. Each re-runs
seed 42 of an anchor through the current code and compares against the stored cell.
The full grid (all seeds) is `experiments/v3_anchors.py`.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))

S11_JSON = ROOT / "results/s11/s11_confirmatory_study.json"
AC3_JSON = ROOT / "results/ac3/ac3_address_space_study.json"
ETID2_JSON = ROOT / "results/e_tid2/e_tid2_ridge_router.json"
CACHE = ROOT / "results/feature_cache/cifar100_vit_b16/feature_cache.pt"

needs = lambda p: pytest.mark.skipif(not p.exists(), reason=f"{p.name} not present (results/ is untracked)")  # noqa: E731
slow = pytest.mark.skipif(
    os.environ.get("PAL_MOE_ANCHORS") != "1" or not CACHE.exists(),
    reason="cell re-runs are opt-in: PAL_MOE_ANCHORS=1 and the ViT feature cache",
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -- shims -------------------------------------------------------------------------


def test_moved_names_are_the_same_objects():
    import e_tid2_ridge_router  # noqa: F401  (imports cleanly through the shims)
    import s2_ladder
    import s6b_difficulty
    import s11_confirmatory as s11

    import pal_moe.core.constructions as cons
    import pal_moe.core.features as feats
    import pal_moe.eval.stats as stats
    import pal_moe.experts.ladder as ladder

    assert s2_ladder.LadderModel is ladder.LadderModel
    assert s2_ladder.LEVELS_BY_NAME is ladder.LEVELS_BY_NAME
    assert s2_ladder.load_tasks is feats.load_tasks and s2_ladder.set_seed is feats.set_seed
    assert s6b_difficulty.build_construction is cons.build_construction
    assert s11.train_model is ladder.train_model and s11.evaluate is ladder.evaluate
    for name in ("paired_stats", "tost", "westfall_young", "holm", "signed_rank_statistic"):
        assert getattr(s11, name) is getattr(stats, name)


def test_legacy_shims_alias_not_copy():
    import pal_moe.evaluation.schema as old_schema
    import pal_moe.eval.schema as new_schema
    import pal_moe.factory as old_factory
    import pal_moe.legacy.factory as new_factory
    import pal_moe.legacy.models.moe as new_moe
    import pal_moe.memory.prototype_memory as old_pm
    import pal_moe.legacy.memory.prototype_memory as new_pm
    import pal_moe.models.moe as old_moe

    assert old_moe is new_moe and old_factory is new_factory
    assert old_pm is new_pm and old_schema is new_schema
    files = {}
    for name, mod in list(sys.modules.items()):
        if name.startswith("pal_moe") and getattr(mod, "__file__", None):
            files.setdefault(mod.__file__, set()).add(id(mod))
    assert not [f for f, ids in files.items() if len(ids) > 1], "a module was loaded twice"


# -- statistics moved from s11 ----------------------------------------------------


@needs(ETID2_JSON)
def test_anchor_etid2_report_recomputes_exactly():
    from pal_moe.eval.stats import paired_stats

    d = json.loads(ETID2_JSON.read_text())
    for regime, fam in d["report"].items():
        rows = [c for c in d["cells"] if c["regime"] == regime]
        got = paired_stats([r["ridge_routed"] - r["proto"] for r in rows], "P1")
        assert got == fam["P1_ridge_routed_minus_proto"]
        got = paired_stats([r["ridge_routed"] - r["ridge_alone"] for r in rows], "P2", sesoi=0.01)
        assert got == fam["P2_ridge_routed_minus_ridge_alone"]


@needs(S11_JSON)
def test_anchor_s11_hypotheses_recompute_exactly():
    import s11_confirmatory as s11

    d = json.loads(S11_JSON.read_text())
    assert s11.build_hypotheses(d["cells"], d["seeds"]) == d["hypotheses"]


@needs(S11_JSON)
def test_anchor_s11_e0_stored_means():
    d = json.loads(S11_JSON.read_text())
    for construct, mean in (("coherent", 73.29), ("dispersed", 70.66)):
        acc = [c["accuracy"] for c in d["cells"] if c["part"] == "B" and c["level"] == "L3_per_task"
               and c["rank"] == 8 and c["num_tasks"] == 20 and c["construct"] == construct]
        assert len(acc) == 6 and round(100 * sum(acc) / 6, 2) == mean


# -- cell re-runs (opt-in) -------------------------------------------------------


@slow
@needs(S11_JSON)
def test_anchor_s11_e0_cell_bitwise():
    import s11_confirmatory as s11

    stored = json.loads(S11_JSON.read_text())
    args = argparse.Namespace(epochs=10, lr=1e-3, batch_size=128, lambda_func=1.0, seed=None)
    for construct in ("coherent", "dispersed"):
        ref = next(c for c in stored["cells"] if c["part"] == "B" and c["level"] == "L3_per_task"
                   and c["rank"] == 8 and c["num_tasks"] == 20 and c["construct"] == construct
                   and c["seed"] == 42)
        cell = {k: ref[k] for k in ("part", "dataset", "construct", "level", "rank", "protos",
                                    "top_k", "num_tasks", "seed")}
        got = s11.run_cell(cell, args, torch.device(DEVICE), {})
        assert got["accuracy"] == ref["accuracy"]
        assert got["acc_matrix"] == ref["acc_matrix"]


@slow
@needs(AC3_JSON)
def test_anchor_ac3_fixed_proto_cell():
    import ac3_address_space as ac3
    from v3_anchors import _max_abs

    ref = next(c for c in json.loads(AC3_JSON.read_text())["cells"]
               if c["regime"] == "coherent" and c["seed"] == 42)
    got = ac3.run_cell("coherent", 42, argparse.Namespace(epochs=10, lr=1e-3, batch_size=128),
                       torch.device(DEVICE))
    assert _max_abs(got["fixed_proto"], ref["fixed_proto"]) <= 1e-6
    assert got["router_param_count"] == 0


@slow
@needs(ETID2_JSON)
def test_anchor_etid2_cell():
    from v3_anchors import ETID2_ARMS, ETID2_COV, check_etid2

    rows = check_etid2([42], None, torch.device(DEVICE))
    for r in rows:
        assert max(abs(r[f"delta_{k}"]) for k in ETID2_ARMS + ETID2_COV) <= 1e-6
        assert r["guards"]["G3_argmax_mismatch"] == 0
