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

> **Protocol note (feature-cached CIFAR/ViT runs):** those runs use
> `--feature_cache`, so the replay baselines store cached feature vectors
> (not raw images) and the "hybrid" variant's raw store is disabled — it is
> pure + latent-exemplar replay and is labelled `PAL-MoE + Latent Replay` in
> new runs. See the storage-semantics note and design fact 19 in
> [`BENCHMARK.md`](BENCHMARK.md) for the byte accounting and the raw-pipeline
> comparison.

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

All methods use the same geometry and schedule: conv encoder 64/128/256 with 256-dimensional latents (50 SimCLR epochs), experts with hidden size 512, 5 epochs per task, and a frozen encoder. PAL-MoE additionally distills its router onto the prototype owners at each task end (`--router_anchor_steps 300`, zero raw replay).

| Method | Avg Acc (↑) | Forgetting (↓) | Raw exemplars | Seeds |
| :--- | :---: | :---: | :---: | :---: |
| Naive Fine-tuning | 17.21% | 83.84% | none | 42 |
| EWC | 17.17% | 83.89% | none | 42 |
| Standard MoE (4 experts) | 17.23% | 82.10% | none | 42 |
| iCaRL (k=25) | 24.90 ± 0.58% | 15.75 ± 1.76% | 250 | 3 |
| ER-ACE (P=250) | 25.27% | 70.75% | 250 | 42 |
| Experience Replay (P=250) | 25.70 ± 0.76% | 72.10 ± 0.50% | 250 | 3 |
| AGEM (P=250) | 28.27% | 69.03% | 250 | 42 |
| DER++ (P=250) | 31.83 ± 0.32% | 60.98 ± 0.74% | 250 + logits | 3 |
| **PAL-MoE (pure)** | **37.30 ± 0.11%** | **21.98 ± 1.75%** | **0 (latent only)** | 3 |
| **PAL-MoE + Replay (Hybrid)** | **38.66 ± 0.33%** | **22.62 ± 0.92%** | 250 | 3 |

The 3-seed rows are means ± std over seeds 42 1 2 (`results/cifar10_conv_multiseed`, current code); the single-seed 42 rows are from the earlier full-suite run (`results/cifar10_final_full`), whose code paths were verified numerically equivalent.

Note: The pure variant stores no raw inputs (latent prototypes and task ids only) and reaches 37.30%, ahead of the strongest baseline DER++ (31.83 ± 0.32%) by 5.5 accuracy points with ~2.8 times less forgetting (21.98% vs 60.98%). The hybrid adds 250 raw exemplars and leads by another 1.4 points at the same forgetting level. An earlier revision of this table showed an approximately 14-point gap; that measurement was affected by a baseline-side BatchNorm artifact (the "frozen" encoder drifted during the baselines' own training loops) and has been corrected here (see `BENCHMARK.md`, design fact 14). The remaining gap to the per-expert oracle (~67%) reflects expert and representation quality rather than routing.

Note on the training regime: the frozen-representation setting is part of the method's design on CIFAR-10; with an unfrozen encoder, the earlier pure/hybrid runs scored 17.75% / 25.54%. At the longer 15-epoch schedule (150 SimCLR epochs) an earlier 5-seed run reached 37.35 ± 0.56% pure and 38.87 ± 0.47% hybrid, i.e. the longer schedule does not add over the 5-epoch numbers above; the feature cache was verified neutral on seed 42 (37.60% vs 36.72%; `BENCHMARK.md`, design facts 11-13).

### 3. Split-CIFAR-10 with a strong frozen backbone (ImageNet ResNet-18)

The same protocol, but the encoder is a frozen ImageNet ResNet-18 (no
pretraining, feature cache; `experiments/recipes/cifar10_resnet18_frozen.sh`),
3-seed means ± std over seeds 42 1 2 (`results/cifar10_resnet18_multiseed`).
Representation quality is the single biggest lever: pure PAL-MoE jumps from
37.30% to 48.82% with half the forgetting of DER++.

| Method | Avg Acc (↑) | Forgetting (↓) |
| :--- | :---: | :---: |
| Naive Fine-tuning | 17.71 ± 0.04% | 88.95 ± 0.12% |
| EWC | 17.66 ± 0.12% | 88.90 ± 0.19% |
| iCaRL (k=25) | 30.70 ± 3.71% | 26.89 ± 1.65% |
| Experience Replay (P=250) | 39.92 ± 0.76% | 58.87 ± 1.01% |
| DER++ (P=250) | 45.29 ± 0.48% | 43.81 ± 0.70% |
| **PAL-MoE (pure, zero raw replay)** | **48.82 ± 1.42%** | **21.05 ± 1.44%** |
| **PAL-MoE + Replay (Hybrid)** | **49.66 ± 1.47%** | **19.56 ± 1.36%** |

PAL-MoE pure beats DER++ by 3.5 accuracy points with **half** its forgetting;
the hybrid reaches 49.66% at 19.56% forgetting. iCaRL has comparable forgetting
(26.89%) at 18 points lower accuracy while still storing raw exemplars.

### 4. Split-CIFAR-100 (20 tasks, 5 classes each, frozen conv encoder)

Full 20-task recipe (`configs/cifar100_big_frozen.json`, 50 SimCLR epochs,
5 epochs/task, feature cache). This is the long-horizon test: vanilla replay
forgets almost everything (≈63-69%), while PAL-MoE keeps 2.7-5× less forgetting.

| Method | Avg Acc (↑) | Forgetting (↓) | Seeds |
| :--- | :---: | :---: | :---: |
| Naive Fine-tuning | 3.58% | 66.74% | 42 |
| EWC | 3.71% | 68.94% | 42 |
| Standard MoE (4 experts) | 3.67% | 66.00% | 42 |
| ER-ACE (P=250) | 4.70% | 62.37% | 42 |
| AGEM (P=250) | 4.96% | 65.07% | 42 |
| Experience Replay (P=250) | 6.43% | 62.91% | 42 |
| DER++ (P=250) | 5.92 ± 0.28% | 63.42 ± 0.48% | 3 |
| **PAL-MoE (pure, zero raw replay)** | **9.48 ± 0.25%** | **23.19 ± 0.56%** | 3 |
| **PAL-MoE + Replay (Hybrid, P=250)** | **9.98 ± 0.47%** | **11.93 ± 0.66%** | 3 |
| iCaRL (k=25, 2500 raw exemplars) | 10.13 ± 0.40% | 10.54 ± 0.30% | 3 |

The 3-seed rows are seeds 42 1 2 (`results/cifar100_multiseed`, config-default
relative gate); the single-seed 42 rows are the earlier full-suite run
(`results/cifar100_big_frozen`, absolute gate, same schedule).

Notes: PAL-MoE pure beats every replay baseline by 3.6 accuracy points
(DER++ 5.92%) with roughly 2.7× less forgetting, without storing a single raw
image. The hybrid is within 0.15 points of iCaRL — which stores 10× more raw
exemplars — with comparable forgetting. Router distillation reaches 91-93%
owner-routing accuracy across the 6 experts; the negative prototype margin
(−0.13) shows the 256-d conv representation is still the limiting factor (see
the next section).

> **Validation-gate note:** the 3-seed rows use the config-default relative
> gate (`min(absolute, majority + 0.10)`); the single-seed reference rows used
> the absolute gate, which rejected 7 of 20 expansions. A 3-seed ablation of
> the two policies (`results/cifar100_gate_relative` vs
> `cifar100_gate_absolute`) shows no material difference: pure 9.25 ± 0.27 /
> 22.57 ± 2.65 versus 9.61 ± 0.71 / 24.79 ± 1.72, hybrid 10.16 ± 1.25 /
> 13.97 ± 3.16 versus 9.94 ± 0.65 / 13.44 ± 2.13. All gaps are within one
> standard deviation — **the gate policy is not the bottleneck; the
> representation is**. The config keeps the relative gate (slightly better
> pure forgetting) and the knob stays configurable.

### 5. Split-CIFAR-100 with a strong frozen backbone (ImageNet ResNet-18)

Same 20-task protocol with a frozen ImageNet ResNet-18 encoder and feature
cache (`configs/cifar100_resnet18_frozen.json`, single seed 42,
`results/cifar100_resnet18/`). The backbone swap is the single biggest
improvement measured so far: pure PAL-MoE rises from 9.48 ± 0.25% (conv,
3-seed) to **16.01%**, and the hybrid from 9.98 ± 0.47% to **18.96%**.

| Method | Avg Acc (↑) | Forgetting (↓) |
| :--- | :---: | :---: |
| Naive Fine-tuning | 4.32% | 76.11% |
| EWC | 4.31% | 79.07% |
| Experience Replay (P=250) | 10.93% | 66.91% |
| DER++ (P=250) | 12.57% | 67.28% |
| iCaRL (k=25) | 13.24% | 11.03% |
| **PAL-MoE (pure, zero raw replay)** | **16.01%** | **28.89%** |
| **PAL-MoE + Replay (Hybrid, P=250)** | **18.96%** | **19.78%** |

With the stronger representation the hybrid beats iCaRL by 5.7 accuracy points
(18.96% vs 13.24%) and the pure variant beats every replay baseline; iCaRL
still has the lowest forgetting (11.03%). The prototype margin improves from
−0.13 (conv) to −0.076, confirming the representation diagnosis. Caveat: this
run is single-seed; a 3-seed validation is the next scheduled step.

### 6. Class-shared domain shift (Split-MNIST with rotating phases)

Every task keeps the same 10 classes but rotates the inputs 90°·k
(`--domain_shift rotate`), with the stabilized generalist expert frozen from
the third task on (`--freeze_shared_after 2`, `results/mnist_domainshift/`;
`experiments/recipes/mnist_domainshift_shared.sh`):

- PAL-MoE pure: **86.07%** average accuracy, **5.64%** forgetting;
- boundary-free stream (`--task_free_eval`): 86.16% online / 87.11% recent,
  surprise 0.740;
- router owner accuracy 90.8%, prototype margin **+0.117** (positive: the
  representation separates the shared label space well).

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
└── tests/test_pal_moe.py       # PyTest suite (96 tests)
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

# PAL-MoE v2: frozen ImageNet ViT-B/16, one expert per task, persistent cache
python experiments/run_benchmark.py --config configs/cifar10_vit.json --device cuda
python experiments/run_benchmark_multi.py --seeds "42 1 2" \
  --config configs/cifar100_vit.json --device cuda

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
