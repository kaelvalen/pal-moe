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
| `--freeze_encoder` | off | Keep the CIFAR encoder fixed after pretraining (stable prototype anchors; mirrors the MNIST setup) |
| `--proto_routing_alpha` | 0 | Prototype-anchored **inference** routing: `g = (1 - a·conf)·g_router + a·conf·r_p(nearest)` with the threshold-free confidence `conf = clamp(1 - d1/d2, 0, 1)` |
| `--proto_routing_threshold` | off | Optional absolute distance gate on top of the confidence weight |
| `--methods` | all | comma-separated subset: `naive, ewc, replay60, replay360, replay250, derpp, erace, agem, icarl, stdmoe, palmoe, hybrid` |
| `--num_workers` | 0 | DataLoader workers (results-neutral, see design fact 8) |
| `--feature_cache` | off | Precompute frozen-encoder features and run the whole pipeline on them (requires `--freeze_encoder`; mathematically equivalent, removes all encoder work from training) |
| `--proto_samples` | 256 | Training samples registered into prototype memory per task |
| `--proto_threshold` | 0.5 | Prototype merge distance; `auto` = median nearest-neighbour distance of the registration batch (scale-free) |
| `--proto_per_class` | off | Class-balanced eviction group size (None = per-task eviction) |
| `--top_k` | 1 | Experts mixed per input (1 = hard top-1, >1 = soft mixture) |
| `--joint_calib_epochs` | 5 | End-of-task joint latent calibration epochs (0 = disable) |
| `--refresh_anchors_after_calib` | off | Recompute prototype `r_p`/`o_p` anchors with the calibrated model |
| `--keep_optimizer_state` | off | Carry Adam moments across task/expansion optimizer rebuilds |

**Prototype-anchored inference routing** is the zero-replay answer to the recency
funnel: training is untouched, but at inference an input close to a stored
prototype inherits that prototype's historical routing distribution instead of
blindly trusting a router that has drifted. The confidence is deliberately
threshold-free (nearest-class-mean style: `1 - d1/d2`, where `d2` is the nearest
prototype of a *different* task) because an absolute distance is not comparable
across feature spaces — measured prototype coverage at the memory's clustering
threshold (0.5) is ~0% on Split-MNIST features, so a hard gate would anchor
nothing. `--proto_routing_alpha 1` lets confident anchors fully override the
router.

`configs/cifar10_big.json` is the reference "big" run: conv `64,128,256`
(438k-param encoder) → 256-dim latents, expert hidden 512, P=1000, 15 epochs/task,
150 SimCLR pretrain epochs, both PAL-MoE variants only (a full-table run at this
geometry means re-running every baseline, which the same flags support):

```bash
python experiments/run_benchmark.py --config configs/cifar10_big.json --device cuda
```

Note: a config file wins over CLI flags, so use explicit flags (not `--config`)
when you need a different long schedule for a smoke test.

### Diagnosing forgetting from checkpoints

Every PAL-MoE / hybrid run now writes per-task checkpoints
(`<output_dir>/checkpoints_palmoe/task_{t}.pt`,
`<output_dir>/checkpoints_hybrid/task_{t}.pt`). The diagnostic separates a
*router* failure (experts still know their tasks, the router never picks them —
the recency funnel) from a *destructive calibration* failure (joint fine-tuning
overwrote the experts):

```bash
python experiments/diagnose_checkpoint.py \
  --checkpoint results/cifar10_big/checkpoints_palmoe/task_4.pt
```

It prints the (task x expert) expert-accuracy matrix, the (task x expert)
router top-1 assignment shares, and the end-to-end vs. oracle gap. A large
oracle gap with a collapsed share matrix means the experts are intact and the
router is the bottleneck.

### Engineering: measured speedups and invariants

Every optimization below was verified to be **numerically equivalent** to the
previous implementation (stability losses agree to <1e-6; prototype
registration produces identical prototypes) before being kept:

| Fix | Measured effect |
| :--- | :--- |
| `num_workers` for CIFAR loaders (`--num_workers 8`) | task loader 8.5 → 4.2 ms/batch; SimCLR 16.4 → 5.3 ms/batch |
| Cached prototype/route/output matrices in `PrototypeMemory` (was re-stacking ~1000 CPU tensors and doing per-prototype `.item()` host syncs every batch) | `compute_stability_losses` **10.13 s → 0.13 s** per call (1000 prototypes, 6 experts, RTX 5060) |
| Batch registration via a single working matrix | O(P·D) once per batch instead of per sample |
| Vectorized expert utilization/MI metrics (was a per-sample Python loop) | removes O(N) host syncs per test pass |
| Reuse the split loader's dataset for the SimCLR loader | one CIFAR dataset instance instead of two |
| Frozen encoders stay in `eval()` during task phases | no silent BatchNorm drift for "frozen" representations |
| `--feature_cache` (frozen encoder): precompute h(x) per split, run training/calibration/distillation/registration/eval on the cache | removes all conv forward/backward from the loop (CIFAR-10 cache ≈ 25 MB fp16); measured **2.15× end-to-end** on the 1-epoch Split-MNIST CPU smoke (23.6 s vs 50.8 s, including the uncached pretraining phase, so the cached fraction is faster still); the effect grows with encoder size |
| Loss history accumulated on-device, converted once per task (was 4 `.item()` syncs per batch) | removes ~280 host syncs per task |
| `--keep_optimizer_state`: Adam moments carried across optimizer rebuilds (matched by parameter name) | preserves momentum across tasks/expansions (opt-in) |

### Router/dynamics knobs and the planned ablation grid

The mechanism stack (OOD negative-boundary loss, joint latent calibration and
router-anchor distillation) is currently all enabled in the big configs; their
individual contributions are **not yet measured** (design fact 11 only isolates
the distillation indirectly). The flags now support the controlled grid:

```bash
# Distillation on/off  x  OOD on/off  x  calibration on/off (pure, frozen, CIFAR)
for ood in 0.0 0.1; do for calib in 0 5; do for dist in 0 300; do
  python experiments/run_benchmark.py --dataset cifar10 --device cuda \
    --feature_cache --freeze_encoder --methods palmoe --epochs 5 --pretrain_epochs 50 \
    --lambda_ood $ood --joint_calib_epochs $calib --router_anchor_steps $dist \
    --proto_threshold auto --output_dir results/ablation/ood${ood}_calib${calib}_dist${dist}
done; done; done
```

Still open (deliberately deferred until a validation run is possible): folding
the twelve copy-pasted method blocks in `run_benchmark.py` into a registry
(two shared factories for the router and prototype memory are already
extracted).

## Measured design facts (controlled experiments)

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

9. **On CIFAR the recency funnel is representation drift, not a router
   weakness.** Controlled sweep (Split-CIFAR-10, conv 64/128/256 → 256-dim
   latents, 5 epochs/task, 50 SimCLR epochs, pure PAL-MoE, seed 42):

   | Variant | Avg Acc | Forgetting | Router MI | Util. entropy |
   | :--- | :---: | :---: | :---: | :---: |
   | baseline (encoder fine-tuned online) | 18.5% | 85.1% | 0.001 | 0.015 |
   | **frozen encoder** | **33.9%** | **32.7%** | **0.228** | **0.758** |
   | prototype-anchored inference routing | 19.8% | 83.0% | 0.001 | 0.014 |
   | frozen + prototype routing | 31.4% | 32.0% | 0.185 | 0.327 |

   Freezing the encoder removes the funnel (routing entropy 0.015 → 0.758) and
   more than doubles pure accuracy. Prototype-anchored inference routing cannot
   compensate for drift — in a drifted space the nearest prototype of a
   *different* task is as close as the correct one, so confidence ≈ 0 — and it
   slightly hurts once the encoder is stable (stale `r_p` overrides a correct
   router). Prototype coverage at the memory's clustering threshold (0.5) is
   ~0% (measured min distances p50 ≈ 6.5-7.2), which is why the anchoring
   confidence is threshold-free and `--proto_routing_threshold` is optional.
   CIFAR pure mode therefore needs `--freeze_encoder` (see
   `configs/cifar10_big_frozen.json`).

10. **Attention-style routing does not help — the linear gate is already
    sufficient.** Same schedule as fact 9 (pure, frozen encoder, 5 epochs,
    50 SimCLR epochs, seed 42):

    | Router | Avg Acc | Forgetting | MI | Util. entropy |
    | :--- | :---: | :---: | :---: | :---: |
    | linear gate (`dynamic`) | **33.9%** | **32.7%** | 0.228 | **0.759** |
    | cosine attention (`attention`) | 20.9% | 41.7% | 0.187 | 0.487 |
    | cosine attention + task-0 key alignment | 25.0% | 34.6% | 0.097 | 0.521 |

    A learnable query projection makes the attention score bilinear, i.e.
    expressiveness-equivalent to the linear gate; the L2-bounded logits and the
    missing per-expert bias then hurt early routing. `--router_type attention`
    remains available as an opt-in experiment, but the CIFAR bottlenecks were
    representation drift and cross-task readout — not router capacity.

11. **Router-anchor distillation fixes the cross-task readout from latent memory
    only.** At the end of every task the router is trained (300 steps, lr 1e-3)
    to predict the explicit prototype owners — a supervised signal derived
    entirely from stored latents and task ids, no raw exemplars. On the
    15-epoch frozen CIFAR-10 checkpoint this lifts the raw router from **21.3% →
    38.0%** (offline sweep: 100-300 steps are equivalent, 1000 steps overfit)
    and balances the per-task profile (0.46/0.24/0.31/0.42/0.48); adding k-NN
    prototype anchoring on top gives 37.9%, so distillation alone is sufficient.
    Enabled in both big frozen configs via `--router_anchor_steps 300`.

    **End-to-end confirmation (Split-CIFAR-10, big geometry, 15 epochs/task,
    150 SimCLR epochs, seed 42, `configs/cifar10_big_frozen.json`):**

    | Method | Avg Acc | Forgetting | Router MI | Util. entropy |
    | :--- | :---: | :---: | :---: | :---: |
    | PAL-MoE pure, encoder fine-tuned (first big run) | 17.8% | 85.4% | 0.000 | 0.014 |
    | PAL-MoE pure, frozen encoder | 20.8% | 37.3% | 0.194 | 0.287 |
    | **PAL-MoE pure, frozen + anchor distillation (5 seeds)** | **37.35 ± 0.56%** | **24.32 ± 0.45%** | **0.379** | **0.998** |
    | **PAL-MoE hybrid, frozen + anchor distillation (5 seeds)** | **38.87 ± 0.47%** | **23.21 ± 0.61%** | 0.376 | 0.998 |
    | PAL-MoE pure, k-NN anchoring instead of distillation | 33.5% | - | - | - |

    The checkpoint diagnostic on the distilled model shows task-to-expert
    routing is now exact (each task's inputs route to its own expert) and
    inference-time anchoring adds nothing (37.5% vs 37.4%); the remaining gap to
    the per-expert oracle (66.9%) is expert/representation quality, not routing.

    **Split-MNIST validation (5 seeds, 3 epochs/task):** pure PAL-MoE improves
    from 72.65 ± 3.25% / 28.10 ± 4.04% forgetting (the previous headline) to
    **78.06 ± 1.40% / 6.16 ± 0.69%**, and the hybrid from 81.09 ± 1.37% /
    13.28 ± 2.11% to **82.96 ± 0.53% / 3.50 ± 0.76%** (MI 0.998, utilization
    entropy 0.992). The mechanism is not CIFAR-specific.

    One subtlety found by the multi-seed runs: when the validation gate rejects
    expansion, the task has no dedicated expert. The prototype owner must then
    be the *newest* expert (the one the task phase actually trained), not the
    trigger's best parent — anchoring a rejected task to the parent gave the
    router distillation the wrong target and collapsed that task (MNIST hybrid
    81.1% → 71.1% before the fix, 83.0% after).

    **Same-budget comparison (5 epochs/task, 50 SimCLR epochs, frozen encoder,
    seed 42, single run, all methods):** pure PAL-MoE reaches **37.52% / 23.5%
    forgetting** and beats every baseline — AGEM 23.0%, DER++ 21.6%, ER 20.6%,
    ER-ACE 19.6%, EWC 17.3%, Standard-MoE 17.2%, Naive 17.2%, iCaRL 8.6% —
    while storing **zero raw exemplars**. The hybrid (P=250) is statistically
    indistinguishable (37.85%), i.e. the latent anchors carry the readout at
    this geometry. Multi-seed validation is the remaining step.

12. **The training recipe simplifies: distillation replaces joint
    calibration.** Controlled grid on Split-CIFAR-10 (pure, frozen encoder,
    feature cache, 5 epochs/task, 50 SimCLR epochs, seed 42, 8 cells):

    | OOD | joint calib | distill | Avg Acc | Forgetting | Util. entropy |
    | :---: | :---: | :---: | :---: | :---: | :---: |
    | 0.1 | off | 300 | **32.5%** | **24.1%** | **0.997** |
    | 0.0 | off | 300 | 31.8% | 26.4% | 0.994 |
    | 0.1 | 5 | 300 | 31.4% | 26.4% | 0.995 |
    | 0.1 | 5 | off | 30.5% | 44.9% | 0.626 |
    | 0.0 | 5 | 300 | 30.2% | 30.2% | 0.997 |
    | 0.1 | off | off | 22.9% | 66.4% | 0.542 |
    | 0.0 | 5 | off | 19.4% | 48.0% | 0.433 |
    | 0.0 | off | off | 18.4% | 68.6% | 0.427 |

    (a) **Distillation is the dominant mechanism**: without it the best cell
    reaches 30.5% / 44.9% forgetting vs 32.5% / 24.1% with it.
    (b) **Calibration and distillation are substitutes**: with distillation,
    calibration adds nothing (32.5 → 31.4) and slightly hurts; without it,
    calibration is critical (22.9 → 30.5). The default recipe therefore sets
    `--joint_calib_epochs 0`, which also removes the stale-anchor tension.
    (c) **OOD is a small, consistent positive** (+0.7-1.2 acc, 2-4 pp less
    forgetting).

    **Budget caveat (measured at the 15-epoch schedule, seed 42):** with the
    fixed threshold, calibration is *not* redundant at longer schedules —
    calib-on reaches **37.60%** vs **35.03%** for the calib-off/auto-threshold
    cell (and 34.62% for calib-on/auto). The long-schedule default therefore
    keeps calibration and the fixed threshold
    (`configs/cifar10_big_final.json`), while the 5-epoch comparison table uses
    the simplified recipe (`configs/cifar10_big_final_full.json`).

13. **Registration needs owner-aware merging; soft top-2 does not pay off.**
    With the scale-free `--proto_threshold auto`, prototypes from different
    tasks could merge and inherit a single owner, so the router distillation
    routed one task to the other's expert (auto: 30.4% / 39.7% forgetting vs the
    fixed-0.5 baseline 32.5% / 24.1%). Merging is now restricted to same-owner
    prototypes; with that fix `auto` + `--proto_per_class` is the best cell
    (**33.0% / 22.6%**). Class-balanced eviction alone at threshold 0.5 changes
    little (32.4% / 26.6%), and `--proto_samples 1024` hurts under the 1000-
    prototype quota (29.5% / 53.9%). `--top_k 2` raises accuracy slightly
    (33.4%) but more than doubles forgetting (36.4%), so top-1 remains the
    default. Final validated recipe: `configs/cifar10_big_final.json`.

    **Budget caveat:** the auto threshold wins at the 5-epoch budget but costs
    ~3 points at 15 epochs (calib-on: 37.60% fixed vs 34.62% auto), so the fixed
    `0.5` threshold remains the long-schedule default. The feature cache itself
    is neutral (seed-42 with cache 37.60% vs 36.72% without).

All figures below are Split-MNIST, seed 42, 3 epochs/task, current code
(`experiments/run_pure_explore.py` + `debug_routing_asymmetry.py`):

14. **A frozen encoder must stay in `eval()` everywhere — including the
    baselines.** The earlier same-budget CIFAR table (`results/cifar10_big_frozen_full`)
    trained each baseline with `model.train()`, which flipped the "frozen" conv
    encoder's BatchNorm layers back to training mode: their features drifted with
    every task and the replay baselines were badly understated (DER++ 21.6%,
    iCaRL 8.6%, ER 20.6%). `--feature_cache` removes the encoder from the
    baselines' loops entirely; the corrected table (`results/cifar10_final_full`)
    reads DER++ **32.8%**, AGEM 28.3%, iCaRL 26.1%, ER 25.4%, ER-ACE 25.3% —
    while PAL-MoE pure reaches **35.5% / 24.1% forgetting** and the hybrid
    **36.9% / 22.1%**. The honest margin over the best baseline is therefore
    ~2.7 accuracy points with ~2.5x less forgetting (not the ~14-point gap the
    flawed table suggested; that discrepancy is a baseline-side BN-drift
    artifact, reported here for the record).

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