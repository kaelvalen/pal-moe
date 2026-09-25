"""
S8 report: the four capability curves, the mechanism metrics, and the resource
axes, per routing regime.

Reads `results/s8/s8_budget_study.json`, refreshes its aggregate and contracts
with the current metric definitions (so a cell written before a metric existed
still contributes - `forgetting` and `R_iso_ncm` are recomputed from the stored
accuracy matrix), verifies that the operating point reproduces S6b exactly, and
prints the tables S8 is read from:

    capability   L2b   L3   L4   L3 - L0
    mechanism    routing tax   R_iso   R_iso_ncm   forgetting
    resources    memory_bytes   trainable_params   active_params

`R_iso` answers "how much of the isolation capacity is realised relative to the
current shared-sequential system"; `R_iso_ncm` answers "relative to the simple
readout reference". They are kept apart because `R_iso`'s denominator moves with
the budget while `L0` does not (docs/STAGE1_PLAN.md section 8.1).

Usage:
    python experiments/s8_report.py
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import s8_budget  # noqa: E402

S6B_STUDY = "results/s6b/s6b_difficulty_study.json"


def operating_point(cells: list[dict]) -> dict:
    return {
        (c["construct"], c["level"], c["seed"]): c["accuracy"]
        for c in cells
        if c["rank"] == s8_budget.OPERATING_RANK
        and c["protos"] == 1
        and c["top_k"] == 1
    }


def verify_against_s6b(cells: list[dict]) -> dict:
    """The operating point must reproduce S6b: the new knobs are no-ops there."""
    if not os.path.exists(S6B_STUDY):
        return {"checked": 0, "note": "S6b study not found"}
    s6b = {
        (c["construct"], c["level"], c["seed"]): c["accuracy"]
        for c in json.load(open(S6B_STUDY))["cells"]
    }
    here = operating_point(cells)
    matched = sorted(set(s6b) & set(here))
    deltas = [abs(s6b[k] - here[k]) for k in matched]
    return {
        "checked": len(matched),
        "max_abs_delta": max(deltas) if deltas else None,
        "missing_from_s8": sorted(set(s6b) - set(here)),
        "note": "same seeds, same data, same order; default resource knobs",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", default="results/s8/s8_budget_study.json")
    parser.add_argument("--out", default="results/s8/s8_budget_report.json")
    parser.add_argument("--refresh", action="store_true", default=True)
    parser.add_argument("--no-refresh", dest="refresh", action="store_false")
    args = parser.parse_args()

    payload = json.load(open(args.study))
    recipe = payload["recipe"]
    cells = payload["cells"]
    ranks = recipe["ranks"]
    prototypes = recipe["prototypes"]
    topks = recipe["topks"]

    aggregate = s8_budget.aggregate(cells, ranks, prototypes, topks)
    guard = verify_against_s6b(cells)

    if args.refresh:
        payload["aggregate"] = aggregate
        payload["contracts"] = {
            "__".join(str(part) for part in s8_budget._key(c)): s8_budget._contract(c)
            for c in cells
        }
        payload["s6b_reproduction_guard"] = guard
        with open(args.study, "w") as fh:
            json.dump(payload, fh, indent=1)
        print(f"[S8 report] refreshed {args.study} ({len(cells)} cells)")

    s8_budget._print(aggregate)

    print("\n" + "=" * 100)
    print("GUARD: the operating point must reproduce S6b (new knobs are no-ops there)")
    print("=" * 100)
    print(
        f"  matched {guard['checked']} cells, "
        f"max |delta| = "
        f"{guard['max_abs_delta'] * 100 if guard['max_abs_delta'] is not None else float('nan'):.4f} points"
    )
    if guard.get("missing_from_s8"):
        print(f"  not yet in S8: {len(guard['missing_from_s8'])} cells")

    with open(args.out, "w") as fh:
        json.dump(
            {
                "schema_version": "1.0",
                "study": "s8_resource_budget_report",
                "aggregate": aggregate,
                "s6b_reproduction_guard": guard,
                "recipe": recipe,
            },
            fh,
            indent=1,
        )
    print(f"\n[S8 report] wrote {args.out}")


if __name__ == "__main__":
    main()
