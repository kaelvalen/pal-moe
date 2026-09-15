# PAL-MoE Benchmark Methodology

This document defines the exact protocol used to produce every number in
`README.md`. Reproducibility is the contract: **one table = one code version =
one command**.

## Protocol

### Datasets

| Benchmark | Splits | Classes/task | Input | Base encoder |
| :-- | :-- | :-- | :-- | :-- |
| Split-MNIST | 5 tasks, 2 classes each | 0-9 | 784-d MLP | 256→128 autoencoder pretrain (1 epoch), frozen |
| Split-CIFAR-10 | 5 tasks, 2 classes each | 0-9 | 3×32×32 | 50-epoch SimCLR conv encoder, adaptive |
| Split-CIFAR-100 | 20 tasks, 5 classes each | 0-99 | 3×32×32 | 50-epoch SimCLR conv encoder, adaptive (loader implemented; benchmark pending) |

All methods share the **same frozen base encoder** (pretrained once, then reused
for every method). This isolates the continual-learning mechanisms.

### Memory budget

- **PAL-MoE prototypes:** latent vectors in the 128-d encoder space
  (`max_prototypes=250`) + optional raw exemplars in hybrid mode. Full 5-task
  pure model ≈ **284 KB** (`PrototypeMemory.estimate_memory_footprint()`).
- **Replay baselines:** raw exemplar buffer of the same cardinality
  (`buffer_size=250`; P=60/P=360 variants show budget sensitivity).
- Memory is reported both as buffer size and bytes.

### Evaluation

- Class-incremental protocol: at each task `t`, accuracy is measured on every
  seen task; **no task-id oracle** at test time (top-1 routing decides).
- Metrics (`pal_moe/evaluation/metrics.py`):
  - `Avg Acc` = mean of `R[T, i]` over all tasks `i` (after the final task).
  - `Forgetting` = mean over tasks of `max_{t>=i} R[t, i] - R[T, i]`.
  - `BWT` = `R[T, i] - R[i, i]` averaged over tasks.
- Multi-seed results report `mean ± std` (seeds 42 1 2 3 4) via
  `experiments/run_benchmark_multi.py`.

### Hyperparameters (all models)

| Setting | Value |
| :-- | :-- |
| Optimizer / LR | Adam, `1e-3` (adaptive encoder: `1e-4`) |
| Epochs per task | 3 |
| Batch size | 128 |
| Expert head | `hidden_dim=256`, matched capacity across ALL methods |
| PAL-MoE: `lambda_r` / `lambda_e` / `lambda_ood` | 0.5 / 2.5 / 0.1 |
| PAL-MoE: joint calibration | 5 epochs on latent exemplars (all experts calibrate) |
| Validation gate | `min_acc_threshold=0.60` (MNIST), `max_proto_drop=2.0`, `max_proto_acc_drop=999.0` |
| Null-space routing anchoring | **off** (measured harmful, see below) |

### Hardware

NVIDIA GeForce RTX 5060 Laptop GPU, torch 2.14.0+cu130, CUDA 13.0, Python 3.12.
CPU runs supported (`--device cpu`); MNIST is deterministic w.r.t. seed
(`set_seed` + `cudnn.deterministic=True`).

## Reproducing the README tables

```bash
source .venv/bin/activate
export PYTHONPATH=.

# Single run (MNIST, seed 42)
python experiments/run_benchmark.py --epochs 3 --device cuda

# Multi-seed (5 seeds, mean ± std) — the README headline source
python experiments/run_benchmark_multi.py --seeds "42 1 2 3 4" --epochs 3 --device cuda

# Config-driven
python experiments/run_benchmark.py --config configs/mnist_default.json --device cuda

# Controlled ablation (one pretrained encoder shared per seed across configs)
python experiments/run_ablation.py --seeds 42 1 2 --device cuda

# CIFAR runs: overlap CPU decode/transform with GPU compute (results-neutral)
python experiments/run_benchmark.py --dataset cifar10 --device cuda --num_workers 8
python experiments/run_benchmark_multi.py --dataset cifar10 --device cuda --num_workers 8
```

Results: `results/benchmark_results_seed{s}.json` (single),
`results/benchmark_multi.json` (aggregated), `results/ablation_results.json`.

### Scaling up: capacity and budget knobs

Model size and training budget are configurable from the CLI (or a JSON config,
which overrides CLI defaults):

| Flag | Default | Meaning |
| :--- | :---: | :--- |
| `--feature_dim` | 128 | latent width: encoder output, router input, expert input |
| `--expert_hidden` | 256 | hidden width of every MLP expert |
| `--conv_channels` | `32,64,128` | CIFAR conv encoder channels (default reproduces the original net exactly) |
| `--proto_size` | 250 | PAL-MoE prototype-store budget (hybrid's latent exemplars) |
| `--methods` | all | comma-separated subset: `naive, ewc, replay60, replay360, replay250, derpp, erace, agem, icarl, stdmoe, palmoe, hybrid` |
| `--num_workers` | 0 | DataLoader workers (results-neutral, see design fact 8) |

`configs/cifar10_big.json` is the reference "big" run: conv `64,128,256`
(438k-param encoder) → 256-dim latents, expert hidden 512, P=1000, 15 epochs/task,
150 SimCLR pretrain epochs, both PAL-MoE variants only (a full-table run at this
geometry means re-running every baseline, which the same flags support):

```bash
python experiments/run_benchmark.py --config configs/cifar10_big.json --device cuda
```

Note: a config file wins over CLI flags, so use explicit flags (not `--config`)
when you need a different long schedule for a smoke test.

## Measured design facts (controlled experiments)

All figures below are Split-MNIST, seed 42, 3 epochs/task, current code
(`experiments/run_pure_explore.py` + `debug_routing_asymmetry.py`):

1. **OOD negative-boundary loss is the decisive pure-mode mechanism.**
   Max-entropy on historical prototypes for the newest expert lifts pure
   zero-replay accuracy from **49.85% → 76.60%** (forgetting 54.00 → 23.05)
   while keeping the hybrid at ≈82%. This is the README's "negative
   boundaries / OOD penalty" claim, implemented and validated.

2. **Recency funnel (root cause of the old pure-mode collapse).** With the
   OOD term absent, each new expert's routing row captured *every* task's
   routing after joint calibration (measured: task-0 inputs route 100% to
   expert-1 after task 1). Old experts starved and the newest expert, trained
   only on ≤256 latent exemplars, could not replace them → old-task accuracy
   collapsed to ~0%.

3. **Freezing the router during joint calibration is harmful** (pure
   49.85→43.16, hybrid 82.23→66.94). Routing protection comes from the
   stability losses, the OOD term and the validation gate — not a hard freeze.

4. **Null-space anchoring at init is harmful** with weakly-separated latents
   (pure 49.85→22.87): orthogonalizing the new row against all historical
   centroids left it a fragile near-random direction. Default **off**
   (`--anchor` to enable, experimental).

5. **Higher stability-loss weight helps pure but kills the hybrid** (pure
   76.60→76.08 at `lambda_r=10`; hybrid 82.35→76.51).

6. **Longer autoencoder pretraining hurts** (pure 76.60 @1ep → 60.20 @5ep →
   58.48 @10ep): a more "abstract" frozen latent space anchors prototypes
   worse. SimCLR (3ep, noise/scale augments) is also worse than AE-1ep for
   MNIST MLP latents (28.86% vs 43.86% in the routing diagnostic).

7. **Modern baselines on Split-MNIST are strong.** DER++ (P=250) reaches
   88.14% / 4.95% forgetting (seed 42), above plain ER (83.07) and the PAL-MoE
   hybrid (82.35). The pure variant (76.60, zero raw replay) beats AGEM, ER-ACE,
   iCaRL, ER(P=60), EWC, Naive and Standard-MoE from a ~284 KB latent store.

8. **The CIFAR data pipeline, not the GPU, caps throughput** with the default
   single-process loaders. Measured on this machine (RTX 5060, SM ≈ 40-60%
   during a run): a 2-class Split-CIFAR-10 task loader takes ~8.5 ms/batch at
   `num_workers=0`, dropping to ~4.6 ms (4) / ~4.2 ms (8); the 50k-image SimCLR
   loader goes ~16.4 ms → ~7.3 / ~5.3 ms. The CIFAR transforms are
   deterministic and a clean A/B confirms identical batch order and RNG
   consumption, so `--num_workers 8` (forwarded by `run_benchmark_multi.py`)
   is a pure speed knob that does not change results.

## Ablations

`experiments/run_ablation.py` sweeps loss components, expert-init strategy,
the validation gate, routing top-k, encoder adaptation and the OOD term, with
one pretrained encoder **shared per seed** across all configs (controlled
ablation). Config selection supports case-insensitive substring filters
(`--configs "OOD"`). Results land in `results/ablation_results.json` +
`results/ablation_comparison.png`.

## CI verification

`.github/workflows/ci.yml`:
- **test**: pytest (29 tests) on Python 3.10-3.12 + coverage.
- **benchmark-verify**: CPU smoke of the full 12-method benchmark (1 epoch)
  asserting it completes and that PAL-MoE hybrid ≥ 50% + pure ≥ 25%
  (sanity bounds, not SOTA checks).

Full multi-seed reproductions are run on the GPU machine before every README
update; a full CI reproduction job can be enabled with a GPU runner.