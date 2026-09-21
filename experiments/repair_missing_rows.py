"""
Merge missing method rows into existing benchmark result JSONs.

Use case: a method was added to an existing run after the fact (e.g. the
hybrid result-key fix), and re-running every method would waste GPU time.
This tool copies rows that are absent in the target per-seed JSONs from a
source directory with the same layout:

    target_dir/seed42/benchmark_results_seed42.json
    source_dir/seed42/benchmark_results_seed42.json

Existing rows are never overwritten unless --overwrite is passed. The source
row must come from the same protocol/seed (the runner is deterministic per
method and seed, so a single-method re-run is equivalent to the row the full
run would have produced).

After merging, rebuild the multi-seed aggregate with:

    python experiments/run_benchmark_multi.py --seeds "42 1 2" \
        --config <config> --output_dir <target_dir> --aggregate_only
"""

import argparse
import glob
import json
import os


def merge_file(target_path: str, source_path: str, overwrite: bool) -> list[str]:
    with open(target_path) as fh:
        target = json.load(fh)
    with open(source_path) as fh:
        source = json.load(fh)
    added = []
    for method, row in source.items():
        if method in target and not overwrite:
            continue
        target[method] = row
        added.append(method)
    if added:
        with open(target_path, "w") as fh:
            json.dump(target, fh, indent=2)
    return added


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_dir", type=str, required=True)
    parser.add_argument("--source_dir", type=str, required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace rows that already exist in the target",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Actually write; without this flag the merge is a dry run",
    )
    args = parser.parse_args()

    targets = sorted(
        glob.glob(
            os.path.join(args.target_dir, "**", "benchmark_results_seed*.json"),
            recursive=True,
        )
    )
    if not targets:
        raise SystemExit(f"no target result files under {args.target_dir}")

    total_added = 0
    for target_path in targets:
        filename = os.path.basename(target_path)
        source_matches = glob.glob(
            os.path.join(args.source_dir, "**", filename), recursive=True
        )
        if not source_matches:
            print(f"  [skip] no source for {filename}")
            continue
        if args.write:
            added = merge_file(target_path, source_matches[0], args.overwrite)
        else:
            with open(target_path) as fh:
                target = json.load(fh)
            with open(source_matches[0]) as fh:
                source = json.load(fh)
            added = [m for m in source if m not in target or args.overwrite]
        total_added += len(added)
        print(f"  {target_path}: added {added}")
    print(f"{'merged' if args.write else 'would merge'} {total_added} rows")


if __name__ == "__main__":
    main()
