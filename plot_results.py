import json
import matplotlib.pyplot as plt
import numpy as np

with open("results/benchmark_results.json", "r") as f:
    data = json.load(f)

# Extract key metrics
methods = [
    "Naive Fine-tuning",
    "Experience Replay (Buffer=250)",
    "PAL-MoE (Ours)",
    "PAL-MoE + Replay (Hybrid, P=250)",
]

accs = [data[m]["acc"] * 100 for m in methods]
forgets = [data[m]["forgetting"] * 100 for m in methods]

# 1. Bar Chart for Accuracy and Forgetting
fig, ax1 = plt.subplots(figsize=(10, 6))

x = np.arange(len(methods))
width = 0.35

rects1 = ax1.bar(
    x - width / 2, accs, width, label="Average Accuracy (%)", color="#2ca02c"
)
rects2 = ax1.bar(x + width / 2, forgets, width, label="Forgetting (%)", color="#d62728")

ax1.set_ylabel("Percentage (%)", fontsize=12)
ax1.set_title(
    "Continual Learning Performance (Split-MNIST, 5 Tasks)", fontsize=14, pad=20
)
ax1.set_xticks(x)
ax1.set_xticklabels(methods, rotation=15, ha="right", fontsize=11)
ax1.legend(loc="upper right")


# Add value labels
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax1.annotate(
            f"{height:.1f}%",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
        )


autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.savefig("results/benchmark_summary.png", dpi=300)
print("Saved bar chart to results/benchmark_summary.png")

# 2. Line Plot for Accuracy Evolution
fig2, ax2 = plt.subplots(figsize=(10, 6))

colors = {
    "Naive Fine-tuning": "#7f7f7f",
    "Experience Replay (Buffer=250)": "#1f77b4",
    "PAL-MoE (Ours)": "#ff7f0e",
    "PAL-MoE + Replay (Hybrid, P=250)": "#2ca02c",
}

for m in methods:
    matrix = data[m]["acc_matrix"]
    # Compute average accuracy of all seen tasks at each step
    evolution = []
    for step in range(5):
        seen_accs = [matrix[step][i] * 100 for i in range(step + 1)]
        evolution.append(np.mean(seen_accs))

    marker = "o" if "PAL-MoE" in m else "s"
    ax2.plot(
        range(1, 6),
        evolution,
        marker=marker,
        linewidth=2.5,
        markersize=8,
        label=m,
        color=colors[m],
    )

ax2.set_xlabel("Tasks Learned", fontsize=12)
ax2.set_ylabel("Average Accuracy on Seen Tasks (%)", fontsize=12)
ax2.set_title("Learning Trajectory Over Time", fontsize=14)
ax2.set_xticks(range(1, 6))
ax2.grid(True, linestyle="--", alpha=0.7)
ax2.legend(loc="lower left")

plt.tight_layout()
plt.savefig("results/benchmark_evolution.png", dpi=300)
print("Saved evolution plot to results/benchmark_evolution.png")
