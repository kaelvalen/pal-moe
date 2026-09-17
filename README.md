# PAL-MoE: Prototype-Anchored Lifelong Mixture of Experts

**PAL-MoE** is a dynamic Mixture of Experts (MoE) architecture for **class-incremental continual learning** that avoids catastrophic forgetting.

Instead of overwriting past knowledge, PAL-MoE spawns a new expert network for each task while a **prototype-anchored linear router** selects the expert for each input. The replay memory stores **128-dimensional latent vectors instead of raw images**, so its footprint is measured in kilobytes, and the pure (zero-raw-replay) mode requires no exemplar images at all.

---

## Method

1. **OOD negative-boundary loss.**
   While a new expert trains on a new task, its predictions on historical prototypes are pushed toward maximum entropy (`lambda_ood * H`). The new expert learns the boundary of its own task and stays agnostic about old tasks, so the router has no incentive to divert old-task inputs to it. In the controlled ablation this lifted pure zero-replay accuracy from 49.85% to 76.60% (see `BENCHMARK.md`, design fact 1); the current Split-MNIST recipe reaches 79.36 ± 1.16% (design fact 15).

2. **Latent replay and end-of-task joint calibration.**
   PAL-MoE stores lightweight 128-dimensional latent vectors (prototype centroids and exemplars) instead of raw pixels. At the end of each task, all experts are jointly calibrated on these latent exemplars, temporarily unfreezing all experts and then re-locking history. The full 5-task model occupies approximately 284 KB.

3. **Expert freezing and routing protection.**
   Older experts are locked after their task concludes; during a new task only the newest expert and routing row adapt. The lock is exact: the router sits in a zero weight-decay parameter group, so Adam's decoupled L2 term cannot shrink the locked historical rows (it used to, and that implicit decay was part of the older numbers — see `BENCHMARK.md`, design fact 15). Fully freezing the router during joint calibration is harmful (pure 49.85 to 43.16, hybrid 82.23 to 66.94), and null-space row orthogonalization at initialization is harmful with weakly separated latents (49.85 to 22.87). Routing protection comes from the stability losses, the validation gate, the OOD term and the zero-replay prototype-owner router distillation (`--router_anchor_steps`), all verified by controlled ablation.

4. **Contrastive pretraining and dynamic capacity growth.**
   CIFAR uses a 50-epoch SimCLR-pretrained conv encoder (fine-tuned adaptively); MNIST uses a 1-epoch autoencoder (frozen). When the quantitative trigger `S(x)` detects a domain shift, a new expert is spawned via function-preserving expansion, gated by a validation gate (new-task accuracy, prototype drift, historical prototype-accuracy drop).

---

## Benchmark Results

The headline table is produced by
`python experiments/run_benchmark_multi.py --seeds "42 1 2 3 4" --config configs/mnist_default.json --device cuda`
(single-seed 42 via `run_benchmark.py --config configs/mnist_default.json`; full methodology in [`BENCHMARK.md`](BENCHMARK.md)).
All methods share the same pretrained encoder and the same head width
(`hidden_dim=256` per head); PAL-MoE grows to one head per spawned expert, so
its total head parameter count scales with the number of experts and is
recorded as `trainable_params` in every result JSON.

### 1. Split-MNIST (5 tasks, mean ± std over 5 seeds: 42 1 2 3 4)

| Method | Avg Acc (↑) | Forgetting (↓) | BWT (↑) | Experts | Replay |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Standard MoE (Fixed 4 Experts) | 16.53 ± 2.65% | 95.56 ± 2.24% | -95.56% | 4 | raw buffer |
| Naive Fine-tuning | 19.28 ± 0.07% | 98.66 ± 0.13% | -98.66% | 1 | none |
| EWC | 19.36 ± 0.02% | 98.70 ± 0.07% | -98.70% | 1 | none |
| AGEM (P=250) | 31.38 ± 4.01% | 83.54 ± 4.94% | -83.54% | 1 | raw buffer |
| iCaRL (k=25) | 59.15 ± 2.13% | 10.68 ± 1.93% | -10.68% | 1 | exemplars |
| ER-ACE (P=250) | 60.14 ± 7.14% | 44.92 ± 8.95% | -44.92% | 1 | raw buffer |
| Experience Replay (P=60) | 67.97 ± 0.98% | 36.89 ± 1.12% | -36.89% | 1 | raw buffer |
| **PAL-MoE (pure, zero raw replay)** | **79.36 ± 1.16%** | **7.91 ± 1.24%** | **-7.62%** | **5** | **~284 KB latents, no images** |
| **PAL-MoE + Replay (Hybrid, P=250)** | **80.05 ± 1.08%** | **5.61 ± 1.12%** | **-5.24%** | **5** | latent + raw exemplars |
| Experience Replay (Buffer=250) | 82.50 ± 1.10% | 17.42 ± 1.54% | -17.42% | 1 | raw buffer |
| Experience Replay (P=360) | 83.80 ± 0.69% | 15.51 ± 1.09% | -15.51% | 1 | raw buffer |
| DER++ (P=250) | 86.93 ± 0.91% | 7.31 ± 1.06% | -7.31% | 1 | raw buffer + logits |

All research knobs added in this revision (shared generalist expert, generative
latent replay, NCM/bias read-outs, energy OOD/trigger, auto anchoring, adapter
experts, width growth, merging, uncertainty weighting, task-free metrics, ...)
are opt-in, tested and documented in the research-toolkit table of
[`BENCHMARK.md`](BENCHMARK.md).

Note: The pure variant reaches 79.36% without any raw exemplars — ahead of
iCaRL, ER-ACE, ER(P=60), AGEM, EWC, naive fine-tuning and Standard-MoE — and its
7.91% forgetting is the second lowest in the table after DER++ (7.31%) despite
storing only ~284 KB of latents. The hybrid variant stores the same latents plus
250 raw exemplars and has the lowest forgetting overall (5.61%) at that budget,
and it matches the accuracy of Experience Replay with the same 250-item budget
(80.05% vs 82.50%) while forgetting roughly three times less (5.61% vs 17.42%).
DER++ remains the accuracy leader at 86.93%. The gains come from prototype-owner
router distillation, a zero-replay mechanism that sharpens cross-task routing
(see `BENCHMARK.md`, design facts 11 and 15).

**Provenance note (2026-09):** this table was regenerated end-to-end on the
current code after the historical-routing lock fix (design fact 15). Earlier
revisions reported pure 78.06 ± 1.40% / hybrid 82.96 ± 0.53%; the fix removes an
implicit weight-decay artifact and the recovery recipe in
`configs/mnist_default.json` restores the pure result above that level, while
some baseline numbers moved by 1-3 points. Any table cited from before the fix
should be re-run.

### 2. Split-CIFAR-10 (5 tasks, wide CNN encoder, frozen)

All methods use the same geometry and schedule (single seed 42): conv encoder 64/128/256 with 256-dimensional latents (50 SimCLR epochs), experts with hidden size 512, 5 epochs per task, and a frozen encoder with `--feature_cache`. PAL-MoE additionally distills its router onto the prototype owners at each task end (`--router_anchor_steps 300`, zero raw replay).

> The PAL-MoE rows were re-run with the current code (exact routing lock and
> the recipe from design fact 15; `results/cifar10_lockfix`) and **improved**:
> pure 37.03% / 23.90% forgetting and hybrid 39.11% / 23.26%. Baseline rows are
> from the earlier run; their code paths were verified unchanged (buffer,
> herding and chunking edits are numerically equivalent and covered by tests).

| Method | Avg Acc (↑) | Forgetting (↓) | Raw exemplars |
| :--- | :---: | :---: | :---: |
| Naive Fine-tuning | 17.21% | 83.84% | none |
| EWC | 17.17% | 83.89% | none |
| Standard MoE (4 experts) | 17.23% | 82.10% | none |
| ER-ACE (P=250) | 25.27% | 70.75% | 250 |
| Experience Replay (P=250) | 25.37% | 73.13% | 250 |
| iCaRL (k=25) | 26.09% | 16.23% | 250 |
| AGEM (P=250) | 28.27% | 69.03% | 250 |
| DER++ (P=250) | 32.82% | 60.49% | 250 + logits |
| **PAL-MoE (pure)** | **37.03%** | **23.90%** | **0 (latent only)** |
| PAL-MoE + Replay (Hybrid) | **39.11%** | **23.26%** | 0 raw (latent replay; `--feature_cache` stores no images) |

Note: The pure variant stores no raw inputs (latent prototypes and task ids only) and reaches 37.03%, ahead of the strongest baseline DER++ (32.82%) by 4.2 accuracy points with 2.5 times less forgetting (23.90% vs 60.49%). The hybrid variant leads by another 2.1 points. With `--feature_cache` the "hybrid" replay samples latents only (no raw exemplars can be stored), which the table's last column now reflects. An earlier revision of this table showed an approximately 14-point gap; that measurement was affected by a baseline-side BatchNorm artifact (the "frozen" encoder drifted during the baselines' own training loops) and has been corrected here (see `BENCHMARK.md`, design fact 14). The remaining gap to the per-expert oracle (~67%) reflects expert and representation quality rather than routing.

Note on the training regime: the frozen-representation setting is part of the method's design on CIFAR-10; with an unfrozen encoder, the earlier pure/hybrid runs scored 17.75% / 25.54%. At the longer 15-epoch schedule the calibrated recipe reaches 37.35 ± 0.56% pure and 38.87 ± 0.47% hybrid across 5 seeds (forgetting 24.3 ± 0.5 and 23.2 ± 0.6; the feature cache was verified neutral on seed 42: 37.60% vs 36.72%). Split-CIFAR-100 (20 tasks of 5 classes) is implemented in `pal_moe/data/split_cifar100.py` and selected with `--dataset cifar100`. A scaled 20-task stress run (frozen encoder, 20 SimCLR epochs, 1 epoch/task, feature cache, boundary-free evaluation) completes end-to-end on the fixed code: pure 5.82% / 26.79% forgetting, hybrid 7.13% / **13.68%** forgetting, router distillation at **100% owner-routing accuracy**, with the hybrid's task-free stream score doubling the pure one. At this budget the 256-d latent margin is negative (−0.157), so representation quality — not routing — is the bottleneck; the full recipe remains the next step (`results/cifar100_20task/`, `--task_free_eval`).

---

## Project Structure

```text
pal-moe/
├── pal_moe/
│   ├── models/
│   │   ├── encoder.py          # SharedEncoder (AE / SimCLR pretraining)
│   │   ├── expert.py           # MLPExpert with Net2Net Expansion
│   │   ├── router.py           # DynamicRouter / DistanceRouter / AttentionRouter (top-k)
│   │   └── moe.py              # DynamicMoE container (latent forward, freeze/unfreeze)
│   ├── memory/
│   │   └── prototype_memory.py # Prototype anchors (v_p, r_p, o_p) + stability losses
│   ├── adaptation/
│   │   └── ttt.py              # ContinualTrainer (OOD loss, joint calibration, checkpointing)
│   ├── baselines/
│   │   ├── naive.py / ewc.py / replay.py
│   │   ├── der.py              # DER++ and ER-ACE
│   │   ├── agem.py             # A-GEM
│   │   └── icarl.py            # iCaRL (nearest-class-mean + distillation)
│   ├── data/                   # split_mnist / split_cifar / split_cifar100
│   ├── config.py               # Validated JSON config (explicit CLI > config > defaults)
│   ├── persistence.py          # checkpoint save/load (model + prototype memory)
│   ├── evaluation/             # ContinualEvaluator (avg acc, forgetting, BWT, router KL)
│   ├── trigger/                # QuantitativeTrigger (expert spawning)
│   └── builder/                # ExpertBuilder (candidate training + validation gate)
├── experiments/
│   ├── run_benchmark.py        # Full benchmark (12 methods), single seed
│   ├── run_benchmark_multi.py  # Multi-seed driver (mean ± std)
│   ├── run_ablation.py         # Controlled ablation (shared encoder per seed)
│   ├── run_pure_explore.py     # Fast pure/hybrid mechanism sweeps
│   ├── debug_routing_asymmetry.py  # Task-to-expert routing diagnostics
│   ├── diagnose_checkpoint.py  # Per-task checkpoint diagnostics
│   └── plot_results.py         # Figure generation (single + multi-seed JSON)
├── configs/                    # JSON configs (mnist_default, cifar10_default, ...)
├── BENCHMARK.md                # Methodology, protocol and measured design facts
└── tests/test_pal_moe.py       # PyTest suite (90 tests)
```

---

## How to Run

```bash
source .venv/bin/activate
export PYTHONPATH=.

# Full benchmark, single seed (MNIST, current recipe)
python experiments/run_benchmark.py --config configs/mnist_default.json --device cuda

# Multi-seed results (5 seeds, mean ± std)
python experiments/run_benchmark_multi.py --seeds "42 1 2 3 4" \
  --config configs/mnist_default.json --device cuda

# CIFAR-10 / CIFAR-100
python experiments/run_benchmark.py --dataset cifar10 --device cuda
python experiments/run_benchmark.py --dataset cifar100 --device cuda

# Config-driven run (same as above, explicit)
python experiments/run_benchmark.py --config configs/mnist_default.json --device cuda

# Big CIFAR-10 run (wide encoder, 15 epochs/task, frozen encoder; see BENCHMARK.md)
python experiments/run_benchmark.py --config configs/cifar10_big_frozen.json --device cuda

# Same recipe with frozen-feature caching (faster, mathematically equivalent)
python experiments/run_benchmark.py --config configs/cifar10_big_frozen.json \
  --feature_cache --device cuda

# Run only a subset of methods (fast iteration)
python experiments/run_benchmark.py --dataset cifar10 --methods palmoe,hybrid --device cuda

# Diagnose forgetting from per-task checkpoints (written with --save_checkpoints;
# not available for --feature_cache runs, the encoder is not stored there)
python experiments/run_benchmark.py --config configs/mnist_default.json \
  --save_checkpoints --device cuda
python experiments/diagnose_checkpoint.py \
  --checkpoint results/checkpoints_palmoe/task_4.pt --dataset mnist

# Controlled ablation (shared pretrained encoder per seed)
python experiments/run_ablation.py --seeds 42 1 2 --configs "OOD" --device cuda

# Tests
python -m pytest tests/
```

---

## Citation

If you use **PAL-MoE** in your research or benchmarks, please cite:

```bibtex
@software{pal_moe2026,
  author = {Hakbilen, Mehmet Arda},
  title = {PAL-MoE: Prototype-Anchored Lifelong Mixture of Experts},
  url = {https://github.com/kaelvalen/pal-moe},
  version = {0.1.0},
  year = {2026}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).
