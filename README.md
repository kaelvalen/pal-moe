# PAL-MoE: Prototype-Anchored Lifelong Mixture of Experts

**PAL-MoE** is a Dynamic Mixture of Experts (MoE) architecture for **Class-Incremental Continual Learning** without catastrophic forgetting.

Unlike traditional networks that overwrite past knowledge, PAL-MoE dynamically spawns new Expert networks for new tasks while a **Prototype-Anchored Linear Router** routes data to the correct experts. It stores **128-d latent vectors instead of raw images**, so its replay memory is measured in kilobytes, and its pure (zero-raw-replay) mode needs no exemplar images at all.

---

## Key Architectural Innovations

1. **OOD Negative-Boundary Loss (empirically the decisive component)**
   While a new expert trains on a new task, its predictions on **historical prototypes** are pushed toward maximum entropy (`lambda_ood * H`). The new expert learns the boundary of *its own* task and stays agnostic about old tasks, so the router has no incentive to divert old-task inputs to it. This was measured to lift pure zero-replay accuracy on Split-MNIST from 49.85% → 76.60% (see `BENCHMARK.md`).

2. **Latent Replay & End-of-Task Joint Fine-Tuning**
   PAL-MoE stores lightweight 128-dimensional latent vectors (prototype centroids + exemplars) instead of raw pixels. At the end of each task, all experts are jointly calibrated on these latent exemplars (temporarily unfreezing all experts, then re-locking history). Memory footprint of the full 5-task model: ~284 KB.

3. **Mathematical Freezing & Protection (routing-level)**
   Older experts are locked after their task concludes; during a new task only the newest expert and routing row adapt. **Measured design facts:** fully freezing the router during joint calibration is harmful (tested: pure 49.85→43.16, hybrid 82.23→66.94); null-space row orthogonalization at init is also harmful with weakly-separated latents (49.85→22.87). Routing protection is instead enforced by the stability losses + the validation gate + the OOD term — all verified by controlled ablation.

4. **Contrastive Pretraining & Dynamic Capacity Growth**
   CIFAR uses a 50-epoch SimCLR pretrained conv encoder (left adaptive); MNIST uses a 1-epoch autoencoder (frozen). When the quantitative trigger `S(x)` detects a domain shift, a new Expert is spawned via function-preserving expansion, gated by a validation gate (new-task accuracy, prototype drift, historical prototype-accuracy drop).

---

## Benchmark Results

Reproducible protocol: the headline table is produced by
`python experiments/run_benchmark_multi.py --seeds "42 1 2 3 4" --epochs 3 --device cuda`
(single-seed 42 via `run_benchmark.py`; full methodology in [`BENCHMARK.md`](BENCHMARK.md)).
All methods share the same pretrained encoder and matched head capacity.

### 1. Split-MNIST (5 Tasks, mean ± std over 5 seeds: 42 1 2 3 4)

| Method | Avg Acc (↑) | Forgetting (↓) | BWT (↑) | Experts | Replay |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Standard MoE (Fixed 4 Experts) | 16.53 ± 2.65% | 95.56 ± 2.24% | -95.56% | 4 | raw buffer |
| Naive Fine-tuning | 19.28 ± 0.07% | 98.66 ± 0.13% | -98.66% | 1 | — |
| EWC | 19.36 ± 0.02% | 98.70 ± 0.07% | -98.70% | 1 | — |
| AGEM (P=250) | 34.14 ± 5.70% | 80.04 ± 7.17% | -80.04% | 1 | raw buffer |
| iCaRL (k=25) | 59.15 ± 2.13% | 10.68 ± 1.93% | -10.68% | 1 | exemplars |
| ER-ACE (P=250) | 62.19 ± 5.55% | 42.17 ± 6.97% | -42.17% | 1 | raw buffer |
| Experience Replay (P=60) | 65.13 ± 1.60% | 40.39 ± 2.13% | -40.39% | 1 | raw buffer |
| **PAL-MoE (Ours - Pure / Zero Raw Replay)** | **78.06 ± 1.40%** | **6.16 ± 0.69%** | **-6.16%** | **5** | **~284 KB latents, no images** |
| **PAL-MoE + Replay (Hybrid, P=250)** | **82.96 ± 0.53%** | **3.50 ± 0.76%** | **-3.50%** | **4** | latent + raw exemplars |
| Experience Replay (Buffer=250) | 81.26 ± 1.25% | 18.91 ± 1.77% | -18.91% | 1 | raw buffer |
| Experience Replay (P=360) | 83.54 ± 1.03% | 15.83 ± 1.73% | -15.83% | 1 | raw buffer |
| DER++ (P=250) | 87.08 ± 1.12% | 6.81 ± 1.14% | -6.81% | 1 | raw buffer + logits |

*(Honest reading: **the hybrid now beats classic Experience Replay at the same 250-item budget (82.96 vs 81.26) with a quarter of the forgetting (3.50% vs 18.91%)**, and the **pure variant reaches 78.06% with zero raw exemplars** — ahead of iCaRL, ER-ACE, ER(P=60), AGEM, EWC, Naive and Standard-MoE from a ~284 KB latent store, with forgetting on par with DER++. DER++ remains the accuracy leader (87.08%); that is reported as-is. The improvements come from prototype-owner router distillation — a zero-replay mechanism that sharpens cross-task routing, see BENCHMARK.md design fact 11.)*

### 2. Split-CIFAR-10 (5 Tasks, Wide CNN Encoder, Frozen)

*Current code, single seed (42), same geometry and schedule for every method: conv encoder 64/128/256 → 256-dim latents (150 SimCLR epochs), experts hidden 512, 5 epochs/task, `--freeze_encoder`, multi-seed pending. PAL-MoE additionally distills its router onto the prototype owners at each task end (`--router_anchor_steps 300`, zero raw replay).*

| Method | Avg Acc (↑) | Forgetting (↓) | Raw exemplars |
| :--- | :---: | :---: | :---: |
| Naive Fine-tuning | 17.24% | 83.41% | – |
| EWC | 17.31% | 83.59% | – |
| Standard MoE (4 experts) | 17.17% | 82.31% | – |
| ER-ACE (P=250) | 19.64% | 75.33% | 250 |
| Experience Replay (P=250) | 20.63% | 76.96% | 250 |
| DER++ (P=250) | 21.61% | 68.50% | 250 |
| AGEM (P=250) | 23.00% | 74.35% | 250 |
| iCaRL (k=25) | 8.55% | 15.90% | 250 |
| **PAL-MoE (pure)** | **37.52%** | **23.54%** | **0 (latent only)** |
| PAL-MoE + Replay (Hybrid, P=250) | 37.85% | 25.89% | 250 |

*(Honest reading: **the pure variant, which stores no raw inputs at all (latent prototypes + task ids only), beats every replay-based baseline by ~14.5 points on average accuracy while forgetting far less than ER/DER++/AGEM.** The hybrid adds little at this geometry — the latent anchors carry the readout. At the longer 15-epoch schedule the same recipe reaches **37.35 ± 0.56% pure / 38.87 ± 0.47% hybrid across 5 seeds** (forgetting 24.3 ± 0.5 / 23.2 ± 0.6, routing utilization entropy 0.998). Remaining gap is per-expert quality (oracle 66.9%), not routing.)*

*Note: the encoder fine-tuning regime is harder for the replay baselines on CIFAR-10 from scratch — with an unfrozen encoder the earlier pure/hybrid runs scored 17.75% / 25.54%, so the frozen-representation setting is part of the method's design, not a free lunch. Split-CIFAR-100 (20 tasks × 5 classes) support is implemented — `pal_moe/data/split_cifar100.py`, `--dataset cifar100` — pending a full benchmark run.*

---

## Project Structure

```text
pal-moe/
├── pal_moe/
│   ├── models/
│   │   ├── encoder.py          # SharedEncoder (AE / SimCLR pretraining)
│   │   ├── expert.py           # MLPExpert with Net2Net Expansion
│   │   ├── router.py           # DynamicRouter + DistanceRouter (top-k, null-space utils)
│   │   └── moe.py              # DynamicMoE container (latent forward, freeze/unfreeze)
│   ├── memory/
│   │   └── prototype_memory.py # Prototype anchors (v_p, r_p, o_p) + stability losses
│   ├── adaptation/
│   │   └── ttt.py              # ContinualTrainer (OOD loss, joint calibration, checkpointing)
│   ├── baselines/
│   │   ├── naive.py / ewc.py / replay.py / standard_moe.py
│   │   ├── der.py              # DER++ and ER-ACE
│   │   ├── agem.py             # A-GEM
│   │   └── icarl.py            # iCaRL (nearest-class-mean + distillation)
│   ├── data/                   # split_mnist / split_cifar / split_cifar100
│   ├── persistence.py          # checkpoint save/load (model + prototype memory)
│   ├── evaluation/             # ContinualEvaluator (avg acc, forgetting, BWT, router KL)
│   ├── trigger/                # QuantitativeTrigger (expert spawning)
│   └── builder/                # ExpertBuilder (candidate training + validation gate)
├── experiments/
│   ├── run_benchmark.py        # Full benchmark (12 methods) — single seed
│   ├── run_benchmark_multi.py  # Multi-seed driver (mean ± std)
│   ├── run_ablation.py         # Controlled ablation (shared encoder per seed)
│   ├── run_pure_explore.py     # Fast pure/hybrid mechanism sweeps
│   ├── debug_routing_asymmetry.py  # Task→expert routing diagnostics
│   └── plot_results.py         # Figure generation (single + multi-seed JSON)
├── configs/                    # JSON configs (mnist_default, cifar10_default, ...)
├── BENCHMARK.md                # Methodology, protocol and measured design facts
└── tests/test_pal_moe.py       # PyTest suite (29 tests)
```

---

## How to Run

```bash
source .venv/bin/activate
export PYTHONPATH=.

# Full benchmark - single seed (MNIST)
python experiments/run_benchmark.py --epochs 3 --device cuda

# Multi-seed headline numbers (5 seeds, mean ± std)
python experiments/run_benchmark_multi.py --seeds "42 1 2 3 4" --epochs 3 --device cuda

# CIFAR-10 / CIFAR-100
python experiments/run_benchmark.py --dataset cifar10 --device cuda
python experiments/run_benchmark.py --dataset cifar100 --device cuda

# Config-driven run
python experiments/run_benchmark.py --config configs/mnist_default.json --device cuda

# Big CIFAR-10 run (wide encoder, 15 epochs/task, frozen encoder — see BENCHMARK.md)
python experiments/run_benchmark.py --config configs/cifar10_big_frozen.json --device cuda

# Same recipe with frozen-feature caching (faster, mathematically equivalent)
python experiments/run_benchmark.py --config configs/cifar10_big_frozen.json \
  --feature_cache --device cuda

# Run only a subset of methods (fast iteration)
python experiments/run_benchmark.py --dataset cifar10 --methods palmoe,hybrid --device cuda

# Diagnose forgetting from the per-task checkpoints written by every run
python experiments/diagnose_checkpoint.py \
  --checkpoint results/cifar10_big_frozen/checkpoints_palmoe/task_4.pt

# Controlled ablation (shared pretrained encoder per seed)
python experiments/run_ablation.py --seeds 42 1 2 --configs "OOD" --device cuda

# Tests
python -m pytest tests/
```

---

## Academic Integrity & Citation

If you use **PAL-MoE** in your research or benchmarks, please cite:

```bibtex
@software{pal_moe2026,
  author = {Hakbilen, Mehmet Arda},
  title = {PAL-MoE: Prototype-Anchored Lifelong Mixture of Experts},
  url = {https://github.com/kaelvalen/pal-moe},
  version = {1.1.0},
  year = {2026}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).