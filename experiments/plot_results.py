"""
Generates publication-quality figures from benchmark and ablation JSON results.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt


def plot_benchmark_results(results_file="results/benchmark_results.json", output_path="results/benchmark_comparison.png"):
    if not os.path.exists(results_file):
        print(f"Results file {results_file} not found.")
        return

    with open(results_file, "r") as f:
        data = json.load(f)

    methods = list(data.keys())
    accs = [data[m]["acc"] * 100 for m in methods]
    forgetting = [data[m]["forgetting"] * 100 for m in methods]

    x = np.arange(len(methods))
    width = 0.35

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

    rects1 = ax.bar(x - width / 2, accs, width, label="Average Accuracy (%)", color="#2b5c8f")
    rects2 = ax.bar(x + width / 2, forgetting, width, label="Catastrophic Forgetting (%)", color="#d95f02")

    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_title("Continual Learning Benchmark on Split-MNIST (5 Tasks)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha="right", fontsize=9)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 105)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved benchmark figure to {output_path}")


def plot_ablation_results(results_file="results/ablation_results.json", output_path="results/ablation_comparison.png"):
    if not os.path.exists(results_file):
        print(f"Ablation file {results_file} not found.")
        return

    with open(results_file, "r") as f:
        data = json.load(f)

    configs = list(data.keys())
    accs = [data[c]["acc"] * 100 for c in configs]
    forgetting = [data[c]["forgetting"] * 100 for c in configs]

    x = np.arange(len(configs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

    rects1 = ax.bar(x - width / 2, accs, width, label="Avg Accuracy (%)", color="#1b9e77")
    rects2 = ax.bar(x + width / 2, forgetting, width, label="Forgetting (%)", color="#e7298a")

    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_title("Ablation Study of PAL-MoE Components", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=25, ha="right", fontsize=9)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 105)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved ablation figure to {output_path}")


if __name__ == "__main__":
    plot_benchmark_results()
    plot_ablation_results()
