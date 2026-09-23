"""
Paper report generator.

Scans the result directories produced by the paper experiment plan and writes
one Markdown report plus figures:

- equal-byte Pareto (accuracy/forgetting vs realised stored bytes);
- expert growth and routing retention under gated vs forced expansion;
- component-ablation ladder;
- capacity/parameter scaling;
- anchor-drift cells;
- every multi-seed table directory found under results/.

Usage:
    python experiments/paper_report.py --results_dir results \
        --output results/paper_report.md --figures_dir results/figures

The script is read-only and idempotent: directories that do not exist yet are
simply skipped, so it can be re-run while the queue is still producing results.
"""

import argparse
import glob
import json
import os
from collections import defaultdict
from typing import Optional

import numpy as np

try:  # optional: figures are nice-to-have, the report must still be written
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - matplotlib is a declared dependency
    plt = None


def _load_results(path: str) -> Optional[dict]:
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return None


def _iter_result_files(root: str):
    """Yield (directory, seed, results) for every benchmark JSON under root."""
    for path in sorted(
        glob.glob(
            os.path.join(root, "**", "benchmark_results_seed*.json"), recursive=True
        )
    ):
        seed = (
            os.path.basename(path)
            .replace("benchmark_results_seed", "")
            .replace(".json", "")
        )
        data = _load_results(path)
        if data:
            yield os.path.dirname(path), seed, data


def _agg(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(
        [v for v in values if v is not None and not np.isnan(v)], dtype=float
    )
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std())


def aggregate_dir(directory: str) -> dict[str, dict]:
    """Mean +/- std over seeds for every method in a result directory."""
    per_method: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    seeds = []
    for _, seed, data in _iter_result_files(directory):
        seeds.append(seed)
        for method, row in data.items():
            per_method[method]["acc"].append(row.get("acc"))
            per_method[method]["forgetting"].append(row.get("forgetting"))
            per_method[method]["stored_bytes"].append(row.get("stored_bytes"))
            per_method[method]["memory_bytes"].append(row.get("memory_bytes"))
            per_method[method]["final_experts"].append(row.get("final_experts"))
            per_method[method]["fit_seconds"].append(row.get("fit_seconds"))
            per_method[method]["routing_retention"].append(row.get("routing_retention"))
            per_method[method]["experts_per_task"].append(row.get("experts_per_task"))
    out = {}
    for method, cols in per_method.items():
        acc_m, acc_s = _agg(cols["acc"])
        f_m, f_s = _agg(cols["forgetting"])
        b_m, b_s = _agg(cols["stored_bytes"])
        d_m, d_s = _agg(cols["memory_bytes"])
        out[method] = {
            "acc_mean": acc_m,
            "acc_std": acc_s,
            "forgetting_mean": f_m,
            "forgetting_std": f_s,
            "stored_bytes_mean": b_m,
            "stored_bytes_std": b_s,
            "memory_bytes_mean": d_m,
            "memory_bytes_std": d_s,
            "final_experts": cols["final_experts"],
            "fit_seconds": cols["fit_seconds"],
            "routing_retention": cols["routing_retention"],
            "experts_per_task": cols["experts_per_task"],
            "num_seeds": len(cols["acc"]),
        }
    return out


def fmt_pct(value: float, std: float = float("nan")) -> str:
    if np.isnan(value):
        return "-"
    if np.isnan(std):
        return f"{value * 100:.2f}%"
    return f"{value * 100:.2f} ± {std * 100:.2f}%"


def fmt_bytes(value: float) -> str:
    if np.isnan(value):
        return "-"
    for unit, scale in (("MiB", 1024**2), ("KiB", 1024), ("B", 1)):
        if value >= scale or unit == "B":
            return f"{value / scale:.2f} {unit}"
    return f"{value:.0f} B"


def _method_table(rows: dict[str, dict]) -> list[str]:
    lines = [
        "| Method | Avg Acc | Forgetting | Data bytes | Total stored | Experts | Seeds |",
        "| :-- | :--: | :--: | :--: | :--: | :--: | :--: |",
    ]
    for method in sorted(rows):
        r = rows[method]
        experts = r["final_experts"][-1] if r["final_experts"] else "-"
        lines.append(
            f"| {method} | {fmt_pct(r['acc_mean'], r['acc_std'])} | "
            f"{fmt_pct(r['forgetting_mean'], r['forgetting_std'])} | "
            f"{fmt_bytes(r['memory_bytes_mean'])} | "
            f"{fmt_bytes(r['stored_bytes_mean'])} | {experts} | {r['num_seeds']} |"
        )
    return lines


def equal_byte_section(
    root: str, group: str, dataset_key: str, lines: list[str]
) -> Optional[list]:
    """Collect per-(method, budget) points across seeds for the Pareto figure."""
    pattern = os.path.join(root, group, dataset_key, "s*_b*_*")
    grouped: dict[tuple[str, int], dict] = defaultdict(
        lambda: {"acc": [], "f": [], "b": [], "seeds": []}
    )
    for directory in sorted(glob.glob(pattern)):
        base = os.path.basename(directory)
        parts = base.split("_")
        if len(parts) < 3:
            continue
        seed, budget, method = parts[0], parts[1], "_".join(parts[2:])
        rows = aggregate_dir(directory)
        for _name, r in rows.items():
            if np.isnan(r["acc_mean"]):
                continue
            key = (method, int(budget[1:]))
            grouped[key]["acc"].append(r["acc_mean"])
            grouped[key]["f"].append(r["forgetting_mean"])
            grouped[key]["b"].append(r["memory_bytes_mean"])
            grouped[key]["seeds"].append(seed)
    if not grouped:
        return None
    lines.append(f"### Equal-byte Pareto: {group}/{dataset_key}")
    lines.append("")
    lines.append(
        "| Method | Budget | Realised data bytes | Avg Acc | Forgetting | Seeds |"
    )
    lines.append("| :-- | --: | --: | :--: | :--: | --: |")
    for method, budget in sorted(grouped):
        cell = grouped[(method, budget)]
        acc_m, acc_s = _agg(cell["acc"])
        f_m, f_s = _agg(cell["f"])
        b_m, _ = _agg(cell["b"])
        lines.append(
            f"| {method} | {fmt_bytes(budget)} | {fmt_bytes(b_m)} | "
            f"{fmt_pct(acc_m, acc_s)} | {fmt_pct(f_m, f_s)} | {len(cell['seeds'])} |"
        )
    lines.append("")
    if plt is not None:
        by_method: dict[str, list] = defaultdict(list)
        for (method, _budget), cell in grouped.items():
            acc_m, acc_s = _agg(cell["acc"])
            b_m, _ = _agg(cell["b"])
            by_method[method].append((b_m, acc_m, acc_s))
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for method in sorted(by_method):
            pts = sorted(by_method[method])
            xs = [max(p[0], 1.0) for p in pts]
            ys = [p[1] for p in pts]
            es = [p[2] for p in pts]
            ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=method)
        ax.set_xscale("log")
        ax.set_xlabel("Stored data bytes")
        ax.set_ylabel("Avg accuracy")
        ax.set_title(f"Equal-byte Pareto: {group}/{dataset_key}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        figures_dir = os.path.join(root, "figures")
        os.makedirs(figures_dir, exist_ok=True)
        path = os.path.join(figures_dir, f"pareto_{group}_{dataset_key}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        lines.append(f"![pareto](figures/pareto_{group}_{dataset_key}.png)")
        lines.append("")
    return lines


def growth_section(root: str, lines: list[str]) -> None:
    base = os.path.join(root, "growth")
    if not os.path.isdir(base):
        return
    lines.append("### Expert growth and routing retention (gated vs forced)")
    lines.append("")
    lines.append(
        "| Protocol | Method | Experts per task (mean over seeds) | Final retention |"
    )
    lines.append("| :-- | :-- | :-- | :--: |")
    plot_data = {}
    for protocol in sorted(os.listdir(base)):
        proto_dir = os.path.join(base, protocol)
        rows = aggregate_dir(proto_dir)
        for method in sorted(rows):
            r = rows[method]
            curves = [c for c in r["experts_per_task"] if c]
            if not curves:
                continue
            max_len = max(len(c) for c in curves)
            mean_curve = []
            for i in range(max_len):
                vals = [c[i] for c in curves if len(c) > i]
                mean_curve.append(float(np.mean(vals)))
            retention = [x for x in r["routing_retention"] if x]
            last_ret = (
                retention[-1][-1] if retention and retention[-1] else float("nan")
            )
            lines.append(
                f"| {protocol} | {method} | {', '.join(str(round(v, 2)) for v in mean_curve)} | "
                f"{'-' if np.isnan(last_ret) else f'{last_ret:.3f}'} |"
            )
            plot_data[f"{protocol}:{method}"] = mean_curve
    lines.append("")
    if plt is not None and plot_data:
        fig, ax = plt.subplots(figsize=(7, 4))
        for label, curve in plot_data.items():
            ax.plot(range(1, len(curve) + 1), curve, marker="o", label=label)
        ax.set_xlabel("Task index (after training)")
        ax.set_ylabel("Number of experts")
        ax.set_title("Expert growth under gated vs forced expansion")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        figures_dir = os.path.join(root, "figures")
        os.makedirs(figures_dir, exist_ok=True)
        fig.savefig(os.path.join(figures_dir, "expert_growth.png"), dpi=150)
        plt.close(fig)
        lines.append("![growth](figures/expert_growth.png)")
        lines.append("")


def simple_group_section(root: str, group: str, title: str, lines: list[str]) -> None:
    base = os.path.join(root, group)
    if not os.path.isdir(base):
        return
    lines.append(f"### {title}")
    lines.append("")
    for cell in sorted(os.listdir(base)):
        rows = aggregate_dir(os.path.join(base, cell))
        if not rows:
            continue
        lines.append(f"**{cell}**")
        lines.append("")
        lines.extend(_method_table(rows))
        lines.append("")


def routing_matrix_section(root: str, lines: list[str]) -> None:
    """Mean row-normalised (task x expert) routing shares for the growth runs."""
    base = os.path.join(root, "growth")
    if not os.path.isdir(base):
        return
    lines.append("### Final routing matrices (task x expert, row-normalised)")
    lines.append("")
    for protocol in sorted(os.listdir(base)):
        proto_dir = os.path.join(base, protocol)
        matrices: dict[str, list] = defaultdict(list)
        for _, _, data in _iter_result_files(proto_dir):
            for method, row in data.items():
                diag = row.get("router_diagnostics") or {}
                counts = diag.get("task_expert_counts")
                if not counts:
                    continue
                arr = np.asarray(counts, dtype=float)
                row_sums = arr.sum(axis=1, keepdims=True)
                shares = np.divide(
                    arr, row_sums, out=np.zeros_like(arr), where=row_sums > 0
                )
                matrices[method].append(shares)
        for method in sorted(matrices):
            stack = np.stack(matrices[method], axis=0)
            mean = stack.mean(axis=0)
            n_experts = mean.shape[1]
            lines.append(f"**{protocol} - {method}**")
            lines.append("")
            lines.append(
                "| Task | " + " | ".join(f"E{j}" for j in range(n_experts)) + " |"
            )
            lines.append(
                "| --: | " + " | ".join("--:" for _ in range(n_experts)) + " |"
            )
            for t in range(mean.shape[0]):
                cells = " | ".join(f"{mean[t, j]:.2f}" for j in range(n_experts))
                lines.append(f"| {t} | {cells} |")
            lines.append("")


def latency_section(root: str, lines: list[str]) -> None:
    base = os.path.join(root, "latency")
    if not os.path.isdir(base):
        return
    lines.append("## Forward latency (per sample)")
    lines.append("")
    for path in sorted(glob.glob(os.path.join(base, "*.json"))):
        try:
            with open(path) as fh:
                payload = json.load(fh)
        except Exception:
            continue
        lines.append(
            f"**{os.path.basename(path)}** - {payload.get('encoder_arch')}, "
            f"feature_dim={payload.get('feature_dim')}, "
            f"input={payload.get('input_shape')}, device={payload.get('device')}"
        )
        lines.append("")
        lines.append("| Model | Experts | Total params | Batch | ms/sample |")
        lines.append("| :-- | --: | --: | --: | --: |")
        for row in payload.get("rows", []):
            lines.append(
                f"| {row['model']} | {row['experts']} | {row['total_params']:,} | "
                f"{row['batch_size']} | {row['latency_ms_per_sample']:.4f} |"
            )
        lines.append("")


def multi_seed_section(root: str, lines: list[str]) -> None:
    lines.append("## Multi-seed tables")
    lines.append("")
    for directory in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(directory) or os.path.basename(directory) == "archive":
            continue
        if not glob.glob(
            os.path.join(directory, "**", "benchmark_results_seed*.json"),
            recursive=True,
        ):
            continue
        rows = aggregate_dir(directory)
        if not rows:
            continue
        lines.append(f"**{os.path.relpath(directory, root)}**")
        lines.append("")
        lines.extend(_method_table(rows))
        lines.append("")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--output", type=str, default="results/paper_report.md")
    parser.add_argument("--figures_dir", type=str, default="results/figures")
    args = parser.parse_args()
    _ = args.figures_dir

    lines = [
        "# PAL-MoE paper report (auto-generated)",
        "",
        "Generated by `experiments/paper_report.py` from the result JSONs under "
        f"`{args.results_dir}/`. Means ± std over the seeds present in each directory.",
        "",
    ]
    lines.append("## Equal-byte Pareto")
    lines.append("")
    equal_byte_section(args.results_dir, "equalbyte", "c10r18", lines)
    equal_byte_section(args.results_dir, "equalbyte", "c100r18", lines)
    equal_byte_section(args.results_dir, "equalbyte_raw", "c10r18", lines)
    equal_byte_section(args.results_dir, "equalbyte_raw", "c100r18", lines)
    lines.append("## Expert growth / reuse")
    lines.append("")
    growth_section(args.results_dir, lines)
    routing_matrix_section(args.results_dir, lines)
    lines.append("## Component ablation (final recipe)")
    lines.append("")
    simple_group_section(
        args.results_dir,
        "ablation_final",
        "Component ablation, CIFAR-10 ResNet-18",
        lines,
    )
    lines.append("## Capacity and parameter matching")
    lines.append("")
    simple_group_section(
        args.results_dir,
        "capacity",
        "Capacity sweep and parameter-matched baselines",
        lines,
    )
    lines.append("## Anchor drift / refresh")
    lines.append("")
    simple_group_section(args.results_dir, "drift", "Anchor refresh cells", lines)
    lines.append("")
    latency_section(args.results_dir, lines)
    multi_seed_section(args.results_dir, lines)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
