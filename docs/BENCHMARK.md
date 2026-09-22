# PAL-MoE Benchmark Methodology

This document defines the protocol used to produce every number in
`../README.md`. Each table corresponds to one code version and one command.

## Protocol

### Datasets

| Benchmark | Splits | Classes/task | Input | Base encoder |
| :-- | :-- | :-- | :-- | :-- |
| Split-MNIST | 5 tasks, 2 classes each | 0-9 | 784-d MLP | 256-to-128 autoencoder pretrain (1 epoch), frozen |
| Split-CIFAR-10 | 5 tasks, 2 classes each | 0-9 | 3×32×32 | 50-epoch SimCLR conv encoder, frozen; optional frozen ImageNet ResNet-18 / ViT-B/16 |
| Split-CIFAR-100 | 20 tasks, 5 classes each | 0-99 | 3×32×32 | 50-epoch SimCLR conv encoder, frozen; optional frozen ImageNet ResNet-18 / ViT-B/16 |

All methods share the **same frozen base encoder** (pretrained once, then reused
for every method). This isolates the continual-learning mechanisms.

### Storage semantics under `--feature_cache` (fairness note)

With `--feature_cache` the task loaders yield cached frozen features, so every
replay baseline and iCaRL store **feature vectors** (`feature_dim*4 + 8` bytes
per item), not raw images, and the hybrid's raw store is disabled
(`store_raw = not feature_cache`). Under the feature cache the "hybrid" variant
is therefore *pure + latent-exemplar replay*; new runs name it
`PAL-MoE + Latent Replay`. Consequences:

- `memory_bytes` in each result JSON is the honest per-run byte count; item
  counts are not byte counts and must not be compared across protocols;
- the raw-vs-latent storage comparison needs the raw pipeline
  (`configs/cifar{10,100}_resnet18_frozen_raw.json`): raw ER stores 12,296 B
  per item while PAL pure stores ~2,184 B per prototype;
- hybrid rows in feature-cached tables published before 2026-09-22 did not
  store raw exemplars; treat them as pure + latent replay (see design fact 19).

### Memory budget

- **PAL-MoE prototypes:** latent vectors in the 128-d encoder space
  (`max_prototypes=250`) + optional raw exemplars in hybrid mode. Full 5-task
  pure model is approximately 284 KB (`PrototypeMemory.estimate_memory_footprint()`).
- **Replay baselines:** raw exemplar buffer of the same cardinality
  (`buffer_size=250`; P=60/P=360 variants show budget sensitivity). The default
  eviction policy is `--buffer_sampling recency` (published behaviour); the
  optional `reservoir` policy keeps a uniform sample over the task stream
  instead of evicting the oldest tasks. Experience Replay uses
  `0.5*CE(current) + 0.5*CE(replay)`; DER++ additionally distils stored logits
  (`alpha=beta=0.5`); ER-ACE uses the asymmetric term.
- Memory is reported both as buffer size and bytes. Every result JSON records
  `trainable_params` per method.

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
| Expert head | `hidden_dim=256` per head, identical for every method; PAL-MoE grows to N experts, so its total head parameter count scales with N and is recorded as `trainable_params` in every result JSON |
| PAL-MoE: `--lambda_r` / `--lambda_e` / `--lambda_ood` | 0.5 / 2.5 / 0.5 (`configs/mnist_default.json`; see design fact 15) |
| PAL-MoE: router-anchor distillation | `--router_anchor_steps 300` (zero-replay) |
| PAL-MoE: inference prototype anchoring | `--proto_routing_alpha 0.5` (zero raw replay) |
| PAL-MoE: joint calibration | 5 epochs on latent exemplars (all experts calibrate) |
| Validation gate | `min_acc_threshold=0.60` (MNIST), `max_proto_drop=2.0`, `max_proto_acc_drop=999.0` |
| Null-space routing anchoring | off (measured harmful; see design fact 4) |

### Hardware

NVIDIA GeForce RTX 5060 Laptop GPU, torch 2.14.0+cu130, CUDA 13.0, Python 3.12.
CPU runs supported (`--device cpu`); MNIST is deterministic w.r.t. seed
(`set_seed` + `cudnn.deterministic=True`).

## Reproducing the README tables

```bash
source .venv/bin/activate
export PYTHONPATH=.

# Single run (MNIST, seed 42, current recipe)
python experiments/run_benchmark.py --config configs/mnist_default.json --device cuda

# Multi-seed run (5 seeds, mean ± std): source of the README headline table
python experiments/run_benchmark_multi.py --seeds "42 1 2 3 4" \
  --config configs/mnist_default.json --device cuda

# Same knobs as plain flags (what the config contains)
python experiments/run_benchmark.py --epochs 3 --device cuda \
  --lambda_ood 0.5 --router_anchor_steps 300 --proto_routing_alpha 0.5

# Controlled ablation (one pretrained encoder shared per seed across configs)
python experiments/run_ablation.py --seeds 42 1 2 --device cuda

# CIFAR runs: overlap CPU decode/transform with GPU compute (results-neutral)
python experiments/run_benchmark.py --dataset cifar10 --device cuda --num_workers 8
python experiments/run_benchmark_multi.py --dataset cifar10 --device cuda --num_workers 8

# 3-seed CIFAR error bars: CIFAR-10 ResNet/conv + CIFAR-100 (~1.5-2 h)
bash experiments/recipes/multiseed_cifar.sh

# CIFAR-100 + frozen ImageNet ResNet-18, 20 tasks (~15-25 min)
bash experiments/recipes/cifar100_resnet18_frozen.sh

# PAL-MoE v2: frozen ImageNet ViT-B/16, one expert per task, persistent cache
bash experiments/recipes/vit_cifar_quick.sh          # CIFAR-10, single seed
bash experiments/recipes/vit_cifar_multiseed.sh      # 3-seed CIFAR-10 + CIFAR-100
bash experiments/recipes/memory_pareto.sh            # bytes-vs-accuracy sweep
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
| `--feature_cache_dir` | off | Persist/load that cache (`feature_cache.pt`); seed-invariant encoders (identity head, e.g. ViT at `feature_dim == width`) share it across seeds |
| `--expand_every_task` | off | One expert per task: implies `--trigger always` and bypasses the validation gate up to `--max_experts` (long-horizon protocol) |
| `--encoder_arch` | dataset default | `mlp`/`conv`/`resnet18/34/50`/`vit_b_16`/`vit_b_32`/`vit_l_16`; ViTs resize to 224 and re-normalise to ImageNet statistics |
| `--proto_samples` | 256 | Training samples registered into prototype memory per task |
| `--proto_threshold` | 0.5 | Prototype merge distance; `auto` = median nearest-neighbour distance of the registration batch (scale-free) |
| `--proto_per_class` | off | Class-balanced eviction group size (None = per-task eviction) |
| `--top_k` | 1 | Experts mixed per input (1 = hard top-1, >1 = soft mixture) |
| `--joint_calib_epochs` | 5 | End-of-task joint latent calibration epochs (0 = disable) |
| `--refresh_anchors_after_calib` | off | Recompute prototype `r_p`/`o_p` anchors with the calibrated model |
| `--keep_optimizer_state` | off | Carry Adam moments across task/expansion optimizer rebuilds |

**Prototype-anchored inference routing** addresses the recency funnel without
replay: training is unchanged, but at inference an input close to a stored
prototype inherits that prototype's historical routing distribution instead of
relying on a router that may have drifted. The confidence is threshold-free
(nearest-class-mean style: `1 - d1/d2`, where `d2` is the nearest prototype of a
*different* task) because absolute distances are not comparable across feature
spaces: measured prototype coverage at the memory's clustering threshold (0.5)
is about 0% on Split-MNIST features, so a hard gate would anchor nothing.
`--proto_routing_alpha 1` lets confident anchors fully override the router.

`configs/cifar10_big.json` is the reference "big" run: conv `64,128,256`
(438k-param encoder) with 256-dim latents, expert hidden 512, P=1000,
15 epochs/task, 150 SimCLR pretrain epochs, and only the two PAL-MoE variants (a
full-table run at this geometry requires re-running every baseline, which the
same flags support):

```bash
python experiments/run_benchmark.py --config configs/cifar10_big.json --device cuda
```

Note: explicit CLI flags win over the config file (and config values win over
argparse defaults; see `pal_moe/config.py`), so any knob can be overridden on a
`--config` run.

### Diagnosing forgetting from checkpoints

Every PAL-MoE / hybrid run writes per-task checkpoints
(`<output_dir>/checkpoints_palmoe/task_{t}.pt`,
`<output_dir>/checkpoints_hybrid/task_{t}.pt`). The diagnostic separates a
*router* failure (experts still know their tasks, the router never picks them:
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
| `num_workers` for CIFAR loaders (`--num_workers 8`) | task loader 8.5 to 4.2 ms/batch; SimCLR 16.4 to 5.3 ms/batch |
| Cached prototype/route/output matrices in `PrototypeMemory` (was re-stacking ~1000 CPU tensors and doing per-prototype `.item()` host syncs every batch) | `compute_stability_losses` **10.13 s to 0.13 s** per call (1000 prototypes, 6 experts, RTX 5060) |
| Batch registration via a single working matrix | O(P·D) once per batch instead of per sample |
| Vectorized expert utilization/MI metrics (was a per-sample Python loop) | removes O(N) host syncs per test pass |
| Reuse the split loader's dataset for the SimCLR loader | one CIFAR dataset instance instead of two |
| Frozen encoders stay in `eval()` during task phases | no silent BatchNorm drift for "frozen" representations |
| `--feature_cache` (frozen encoder): precompute h(x) per split, run training/calibration/distillation/registration/eval on the cache | removes all conv forward/backward from the loop (CIFAR-10 cache approx. 25 MB fp16); measured **2.15× end-to-end** on the 1-epoch Split-MNIST CPU smoke (23.6 s vs 50.8 s, including the uncached pretraining phase, so the cached fraction is faster still); the effect grows with encoder size |
| Loss history accumulated on-device, converted once per task (was 4 `.item()` syncs per batch) | removes ~280 host syncs per task |
| `--keep_optimizer_state`: Adam moments carried across optimizer rebuilds (matched by parameter name) | preserves momentum across tasks/expansions (opt-in) |
| Exemplar batch cached until the memory mutates (hybrid latent replay re-fetched and re-transferred it every step) | 0.63 ms to 0.0002 ms per call |
| Stability-loss per-expert row indices cached (depended only on the cached routing matrix) | removes repeated mask/nonzero work per step |
| `refresh_representations` encodes all raw exemplars in one batched forward | one GPU call per task instead of one per prototype |
| DER++ buffer logits computed in 512-sample chunks | removes a full-split forward (multi-GB activations, OOM on raw CIFAR) |
| iCaRL herding vectorized (`argmin ||s+f_i||^2` via one matvec per pick) | 458 ms to 4 ms per class (k=25, N=5000, CPU) |
| Loss accumulation in all baselines (was `.item()` on every optimisation step) | one host sync per task |
| Conditional model construction in the runner (only selected methods are built) | `--methods palmoe` no longer deep-copies 11 unused baselines onto the GPU |
| `--pretrain_cache`: SimCLR/AE encoder weights cached under `./data/pretrain_cache` | skips 50-150 pretraining epochs on repeated runs |
| `--save_checkpoints` is opt-in | no more hundreds of MB of per-task checkpoints per run |
| Training mode restored after the trigger/gate inference paths (`_enter_training_mode`) | fixes a latent eval-mode leak: trainable-encoder BatchNorm stats and prototype-anchored routing no longer silently activate in training |
| Router parameters moved to a zero weight-decay Adam group | historical-routing rows stay exactly locked (Adam's L2 term used to shrink them every step; see design fact 15) |
| `--stability_every` / `--ood_every` (weight scaled by k) | optional amortisation of the two most expensive per-step terms (+35% / +10% measured at k=1) |
| `--buffer_sampling reservoir`, `--ewc_online` | optional fairness/memory alternatives for the baselines (defaults preserve published behaviour) |

### Research toolkit (new knobs, all opt-in)

This revision adds the brainstorm toolkit: every mechanism defaults to the
published behaviour and is covered by the test suite. The intent is "stronger,
more stable, more accurate" without forking the code.

| Area | Flag / API | Effect |
| :--- | :--- | :--- |
| Representation geometry | result `geometry` | nearest-other margin, silhouette, class-mean separation per run |
| Router panel | result `router_diagnostics` | usage entropy, routing entropy, owner-routing accuracy, prototype geometry |
| Metric router | `--router_learn_temperature` | learnable router temperature (all three routers) |
| Owner-contrastive routing | `--router_anchor_margin M` | owner logit must beat the best competitor by M during distillation |
| Lock control | `--router_weight_decay` | 0 = exact lock, 1e-5 = legacy implicit decay (design fact 15) |
| Energy OOD | `--ood_mode energy --ood_margin` | logsumexp hinge instead of entropy maximisation |
| Energy trigger | `--trigger energy --energy_threshold` | unsupervised novelty z-score trigger |
| Auto anchoring | `--proto_routing_auto` | fits the prototype-anchoring alpha per task on stored latents |
| Expert calibration | `--expert_temperature_calib` | per-expert temperature scaling on routed validation samples |
| Generalist expert | `--shared_expert` | always-on expert mixed by a learned gate, never frozen |
| Read-out heads | `--eval_head {ncm,bias}` | nearest-class-mean or bias-corrected logits at evaluation |
| Generative replay | `--generative_replay N --generative_replay_mode {gaussian,vae}` | synthetic latent exemplars, zero raw data |
| Coreset memory | `--proto_selection {kcenter,uncertainty} --proto_candidate_pool` | farthest-point / boundary exemplar selection |
| Eviction policy | `--proto_eviction {task,balanced,reservoir}` | balanced protects the newest task |
| ANN bridge | `PrototypeMemory.build_ann_index()` / `query_ann()` | optional FAISS for production-scale stores |
| LwF | `--lambda_lwf --lwf_temperature` | snapshot distillation on the current batch |
| EMA distillation | `--ema_encoder --lambda_ema` | trainable encoder anchored to its EMA copy |
| Uncertainty weighting | `--loss_weighting uncertainty` | learned homoscedastic weights for task/router/expert/ood |
| Adapter experts | `--freeze_expansion_base` | frozen clones, only the residual adapter trains |
| Width growth | `--expansion_action widen --widen_by` | function-preserving growth instead of new experts |
| Expert merging | `pal_moe.merge`, `experiments/merge_experts.py` | soup / TIES / task arithmetic into one serving head |
| Generic streams | `pal_moe.data.split_folder`, `pal_moe.data.domain_shift` | ImageFolder splits and domain-shifting phases |
| Task-free metrics | `pal_moe.evaluation.task_free.StreamingEvaluator` | online/recent accuracy, surprise, per-domain |
| External encoders | `--encoder_checkpoint` | plug exported foundation-backbone weights into `SharedEncoder` |
| Strong backbones | `--encoder_arch {resnet18,resnet34,resnet50,vit_b_16,vit_b_32,vit_l_16} --encoder_weights imagenet` | frozen ImageNet features (1-channel adaptation for MNIST built in; ViTs resize to 224 and re-normalise to ImageNet statistics), no pretraining needed |
| Persistent feature cache | `--feature_cache_dir DIR` | save/load the frozen-feature cache; seed-invariant encoders (identity head, e.g. ViT at `feature_dim == width`) share one cache across seeds |
| One expert per task | `--expand_every_task` (implies `--trigger always`) | bypass the validation gate so every task gets a dedicated expert up to `--max_experts` (long-horizon protocol; removes expert sharing) |
| Domain-shift streams | `--domain_shift {permute,rotate}` | class-shared phase stream wired into the runner for stability stress tests |
| Ready recipes | `experiments/recipes/*.sh` | CIFAR-100 full, CIFAR-10 ResNet-18, ViT-B/16 quick + multiseed, memory Pareto, read-out ablation, MNIST domain-shift + shared expert, one-expert-per-task |
| Relative validation gate | `--gate_mode relative --gate_margin` | effective threshold `min(absolute, majority + margin)`: relaxes the bar for weak-majority tasks (5-way CIFAR-100) without changing 2-way behaviour |
| Compute reporting | `total_params`, `trainable_params`, `active_params`, `fit_seconds`, `geometry` | full size, gradient-receiving size and per-sample forward cost per run |

First single-seed measurements of the new knobs (Split-MNIST, seed 42, 3
epochs, `configs/mnist_default.json`; base = 79.69% / 6.85% forgetting):

| Variant | Avg Acc | Forgetting | Verdict |
| :-- | :--: | :--: | :--- |
| `--router_anchor_margin 1.0` | 80.77% | 6.51% | looked best on seed 42, but a controlled 3-seed check (42 1 2: 79.40 ± 1.29 / 7.10 ± 0.30) is statistically indistinguishable from the base recipe (79.54 ± 1.30 / 6.97 ± 0.27): not adopted |
| `--generative_replay 64` | 79.29 ± 1.20% | 8.02 ± 1.25% | 5-seed validation: statistically identical to the base recipe (79.36 ± 1.16 / 7.91 ± 1.24); neutral on MNIST, worth retesting when the exemplar budget is tighter |
| `--eval_head ncm` | 76.89% | 10.11% | worse here (top-1 routing already recovers the classes) |
| `--shared_expert` | 78.06% / 7.84 | −1.9 vs base | first measurement without anchoring was catastrophic (45.16% / 63.99) because the always-on pathway drifted into the newest task. With the prototype anchor now added it is stable but still trails the base recipe (CPU base 80.14% / 7.09; `--freeze_shared_after 1`: 78.26% / 7.73). Kept as an opt-in for domain-shift streams where a shared pathway is expected to pay off. |
| shared + margin + ncm + generative | 77.64% | 9.07% | combination does not rescue the shared-expert drift |

These are single-seed measurements to guide the next validation round, not
published claims.

**20-task Split-CIFAR-100, full recipe** (`configs/cifar100_big_frozen.json`,
3 seeds 42 1 2, `results/cifar100_multiseed/`): 50 SimCLR epochs, 5
epochs/task. PAL-MoE pure 9.48 ± 0.25% / 23.19 ± 0.56% forgetting; hybrid
9.98 ± 0.47% / 11.93 ± 0.66%; DER++ 5.92 ± 0.28% / 63.42 ± 0.48%; iCaRL
10.13 ± 0.40% / 10.54 ± 0.30%. Pure beats every replay baseline by 3.6 points
with ~2.7× less forgetting and zero raw storage; the hybrid is within 0.15
points of iCaRL (which stores 10× more raw exemplars). The earlier single-seed
run (absolute gate, which rejected 7 of 20 expansions) is in
`results/cifar100_big_frozen/`; the 3-seed gate ablation shows the policy is a
wash (design fact 16) and the bottleneck is the representation, not the gate.
Router distillation reaches 91-93% owner-routing accuracy over 6 experts; the
negative prototype margin (−0.13) again points at the representation as the
limiting factor. Parameters are reported three ways now: total (1.65M across 6
experts), trainable (275k after freezing history) and active per sample (275k,
one expert).

**Strong-backbone CIFAR-10** (`configs/cifar10_resnet18_frozen.json`,
ImageNet ResNet-18, frozen, feature cache, 3 seeds 42 1 2,
`results/cifar10_resnet18_multiseed/`): pure **48.82 ± 1.42%** / 21.05 ± 1.44
forgetting and hybrid **49.66 ± 1.47%** / 19.56 ± 1.36, versus DER++ 45.29 ±
0.48% / 43.81 ± 0.70% and ER 39.92 ± 0.76% / 58.87 ± 1.01%. The 20-task
CIFAR-100 equivalent (single seed 42, `results/cifar100_resnet18/`) lifts pure
from 9.48 ± 0.25 to 16.01 and hybrid from 9.98 ± 0.47 to 18.96 — the strongest
single lever observed in this project (brainstorm 1.1, design fact 17).

**Class-shared domain shift** (`--domain_shift rotate`, MNIST, shared expert
frozen after the first task): pure 86.07% / 5.64% forgetting, boundary-free
stream 86.16% online, prototype margin +0.117 — the stabilized generalist no
longer hurts when the label space is shared.

Deliberately staged (not implemented here): GPM/Adam-NSCL gradient projection
(needs stored raw activations), hierarchical MoE-of-MoE routing, boundary-free
*task-free training* (training still uses task ids; the streaming evaluator and
energy trigger provide the measurement/novelty half), diffusion-based
generative replay (the latent VAE is the cheaper stand-in), and foundation
backbones whose architecture differs from `SharedEncoder` (they need an export
adapter into that contract).

### Router/dynamics knobs and the ablation grid

The mechanism stack (OOD negative-boundary loss, joint latent calibration and
router-anchor distillation) is enabled as a whole in the big configs. The
individual contributions were measured with the controlled grid below; the
results are summarized in design fact 12 and stored under `results/ablation/`:

```bash
# Distillation on/off  x  OOD on/off  x  calibration on/off (pure, frozen, CIFAR)
for ood in 0.0 0.1; do for calib in 0 5; do for dist in 0 300; do
  python experiments/run_benchmark.py --dataset cifar10 --device cuda \
    --feature_cache --freeze_encoder --methods palmoe --epochs 5 --pretrain_epochs 50 \
    --lambda_ood $ood --joint_calib_epochs $calib --router_anchor_steps $dist \
    --proto_threshold auto --output_dir results/ablation/ood${ood}_calib${calib}_dist${dist}
done; done; done
```

The twelve method blocks in `run_benchmark.py` share `pal_moe.factory` and the
`_run_baseline_loop`/`_run_palmoe_variant` helpers, so the PAL-MoE variants and
the single-head baselines cannot drift apart.

## Measured design facts (controlled experiments)

1. **The OOD negative-boundary loss is a key pure-mode mechanism.**
   Max-entropy on historical prototypes for the newest expert lifts pure
   zero-replay accuracy from **49.85% to 76.60%** (forgetting 54.00 to 23.05)
   while keeping the hybrid at about 82%. This validates the OOD penalty
   described in the README.

2. **Recency funnel (root cause of the earlier pure-mode collapse).** With the
   OOD term absent, each new expert's routing row captured *every* task's
   routing after joint calibration (measured: task-0 inputs route 100% to
   expert-1 after task 1). Old experts starved and the newest expert, trained
   only on ≤256 latent exemplars, could not replace them, so old-task accuracy
   collapsed to about 0%.

3. **Freezing the router during joint calibration is harmful** (pure 49.85 to
   43.16, hybrid 82.23 to 66.94). Routing protection comes from the stability
   losses, the OOD term and the validation gate, not from a hard freeze.

4. **Null-space anchoring at initialization is harmful** with weakly separated
   latents (pure 49.85 to 22.87): orthogonalizing the new row against all
   historical centroids left it a fragile near-random direction. Default off
   (`--anchor` enables it; experimental).

5. **Higher stability-loss weight helps the pure variant but hurts the hybrid**
   (pure 76.60 to 76.08 at `lambda_r=10`; hybrid 82.35 to 76.51).

6. **Longer autoencoder pretraining hurts** (pure 76.60 at 1 epoch, 60.20 at 5,
   58.48 at 10): a more "abstract" frozen latent space anchors prototypes
   worse. SimCLR (3 epochs, noise/scale augments) is also worse than AE-1ep for
   MNIST MLP latents (28.86% vs 43.86% in the routing diagnostic).

7. **Modern baselines on Split-MNIST are strong.** DER++ (P=250) reaches
   88.14% / 4.95% forgetting (seed 42), above plain ER (83.07) and the PAL-MoE
   hybrid (82.35). The pure variant (76.60, zero raw replay) beats AGEM, ER-ACE,
   iCaRL, ER(P=60), EWC, Naive and Standard-MoE from a ~284 KB latent store.

8. **The CIFAR data pipeline, not the GPU, caps throughput** with the default
   single-process loaders. Measured on this machine (RTX 5060, SM utilization
   40-60% during a run): a 2-class Split-CIFAR-10 task loader takes ~8.5 ms/batch
   at `num_workers=0`, dropping to ~4.6 ms (4 workers) / ~4.2 ms (8 workers);
   the 50k-image SimCLR loader goes from ~16.4 ms to ~7.3 / ~5.3 ms. The CIFAR
   transforms are deterministic and a clean A/B confirms identical batch order
   and RNG consumption, so `--num_workers 8` (forwarded by
   `run_benchmark_multi.py`) is a pure speed knob that does not change results.

9. **On CIFAR the recency funnel is representation drift, not a router
   weakness.** Controlled sweep (Split-CIFAR-10, conv 64/128/256 with 256-dim
   latents, 5 epochs/task, 50 SimCLR epochs, pure PAL-MoE, seed 42):

   | Variant | Avg Acc | Forgetting | Router MI | Util. entropy |
   | :--- | :---: | :---: | :---: | :---: |
   | baseline (encoder fine-tuned online) | 18.5% | 85.1% | 0.001 | 0.015 |
   | **frozen encoder** | **33.9%** | **32.7%** | **0.228** | **0.758** |
   | prototype-anchored inference routing | 19.8% | 83.0% | 0.001 | 0.014 |
   | frozen + prototype routing | 31.4% | 32.0% | 0.185 | 0.327 |

   Freezing the encoder removes the funnel (routing entropy 0.015 to 0.758) and
   more than doubles pure accuracy. Prototype-anchored inference routing cannot
   compensate for drift: in a drifted space the nearest prototype of a
   *different* task is as close as the correct one, so confidence is about 0,
   and it slightly hurts once the encoder is stable (stale `r_p` overrides a
   correct router). Prototype coverage at the memory's clustering threshold
   (0.5) is about 0% (measured min distances p50 of 6.5-7.2), which is why the
   anchoring confidence is threshold-free and `--proto_routing_threshold` is
   optional.
   CIFAR pure mode therefore needs `--freeze_encoder` (see
   `configs/cifar10_big_frozen.json`).

10. **Attention-style routing does not help; the linear gate is already
    sufficient.** Same schedule as fact 9 (pure, frozen encoder, 5 epochs,
    50 SimCLR epochs, seed 42):

    | Router | Avg Acc | Forgetting | MI | Util. entropy |
    | :--- | :---: | :---: | :---: | :---: |
    | linear gate (`dynamic`) | **33.9%** | **32.7%** | 0.228 | **0.759** |
    | cosine attention (`attention`) | 20.9% | 41.7% | 0.187 | 0.487 |
    | cosine attention + task-0 key alignment | 25.0% | 34.6% | 0.097 | 0.521 |

    A learnable query projection makes the attention score bilinear, so the two
    are equivalent in expressiveness; the L2-bounded logits and the
    missing per-expert bias then hurt early routing. `--router_type attention`
    remains available as an opt-in experiment, but the CIFAR bottlenecks were
    representation drift and cross-task readout, not router capacity.

11. **Router-anchor distillation improves the cross-task readout using latent
    memory only.** At the end of every task the router is trained (300 steps,
    lr 1e-3) to predict the explicit prototype owners, a supervised signal
    derived entirely from stored latents and task ids, with no raw exemplars. On
    the 15-epoch frozen CIFAR-10 checkpoint this lifts the raw router from
    **21.3% to 38.0%** (offline sweep: 100-300 steps are equivalent, 1000 steps
    overfit) and balances the per-task profile (0.46/0.24/0.31/0.42/0.48);
    adding k-NN prototype anchoring on top gives 37.9%, so distillation alone is
    sufficient. Enabled in both big frozen configs via
    `--router_anchor_steps 300`.

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
    trigger's best parent. Anchoring a rejected task to the parent gave the
    router distillation the wrong target and collapsed that task (MNIST hybrid
    81.1% to 71.1% before the fix, 83.0% after).

    **Same-budget comparison (5 epochs/task, 50 SimCLR epochs, frozen encoder,
    3 seeds 42 1 2, `results/cifar10_conv_multiseed`):** pure PAL-MoE reaches
    **37.30 ± 0.11% / 21.98 ± 1.75% forgetting** and outperforms every baseline:
    DER++ 31.83 ± 0.32%, AGEM 28.3% (seed 42), ER 25.70 ± 0.76%, ER-ACE 25.3%
    (seed 42), iCaRL 24.90 ± 0.58%, EWC 17.3% (seed 42), Standard-MoE 17.2%
    (seed 42), Naive 17.2% (seed 42). It stores **zero raw exemplars**. The
    hybrid (P=250) adds 1.4 points (38.66 ± 0.33% / 22.62 ± 0.92%); the latent
    anchors already carry the readout at this geometry.

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
    calibration adds nothing (32.5 to 31.4) and slightly hurts; without
    distillation, calibration is critical (22.9 to 30.5). The default recipe
    therefore sets `--joint_calib_epochs 0`, which also removes the
    stale-anchor tension.
    (c) **OOD is a small, consistent positive** (+0.7-1.2 accuracy, 2-4 pp less
    forgetting).

    **Budget caveat (measured at the 15-epoch schedule, seed 42):** with the
    fixed threshold, calibration is *not* redundant at longer schedules:
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

The following fact documents the correction of the CIFAR-10 comparison table.

14. **A frozen encoder must stay in `eval()` everywhere, including in the
    baselines.** The earlier same-budget CIFAR table
    (`results/cifar10_big_frozen_full`) trained each baseline with
    `model.train()`, which flipped the "frozen" conv encoder's BatchNorm layers
    back to training mode: their features drifted with every task and the replay
    baselines were badly understated (DER++ 21.6%, iCaRL 8.6%, ER 20.6%).
    `--feature_cache` removes the encoder from the baselines' loops entirely;
    the corrected table (`results/cifar10_final_full`) reads DER++ **32.8%**,
    AGEM 28.3%, iCaRL 26.1%, ER 25.4%, ER-ACE 25.3%, while PAL-MoE pure reaches
    **35.5% / 24.1% forgetting** and the hybrid **36.9% / 22.1%**. The corrected
    margin over the best baseline is therefore ~2.7 accuracy points with ~2.5x
    less forgetting; the earlier ~14-point gap was a baseline-side BatchNorm
    drift artifact. The PAL-MoE rows were later re-run with the exact routing
    lock (design fact 15); the 3-seed error-bar run puts pure at
    **37.30 ± 0.11% / 21.98 ± 1.75% forgetting** and hybrid at
    **38.66 ± 0.33% / 22.62 ± 0.92%** (`results/cifar10_conv_multiseed`),
    reproducing the seed-42 rows (`results/cifar10_lockfix`) within noise.

15. **The historical-routing lock must exclude optimizer weight decay.** Router
    rows of frozen experts are protected by backward hooks that zero their task
    gradients, but `Adam(weight_decay=1e-5)` adds its decoupled L2 term inside
    `step()`; because Adam normalises by the second moment, a locked row whose
    only gradient is `wd * theta` receives updates of order `lr` per step and
    slowly collapses toward zero. That implicit decay was part of the tables
    published before this commit. Moving the router parameters into a
    zero-decay group makes the lock exact, which by itself changes the pure
    recipe (MNIST, seed 42, 3 epochs: **76.60% -> 65.60%**): without the decay
    the newest expert's routing row can no longer out-compete the historical
    rows on its own task, so tasks 2-4 lose their routing.

    The recovery uses mechanisms that are already part of the method and stay
    zero-raw-replay: prototype-owner router distillation
    (`--router_anchor_steps 300`), a stronger OOD negative-boundary weight
    (`--lambda_ood 0.5`) and inference-time prototype anchoring
    (`--proto_routing_alpha 0.5`). This is the recipe in
    `configs/mnist_default.json`; 5-seed results (42 1 2 3 4, 3 epochs,
    reproduced with the config command above):

    | Variant | Avg Acc | Forgetting | Experts |
    | :-- | :--: | :--: | :--: |
    | PAL-MoE pure | **79.36 ± 1.16%** | 7.91 ± 1.24% | 5 |
    | PAL-MoE + Replay (hybrid) | 80.05 ± 1.08% | 5.61 ± 1.12% | 5 |

    The hybrid prefers the weaker OOD weight used before the fix: with
    `--lambda_ood 0.1` and everything else unchanged it reaches 81.92% /
    3.66% forgetting on seed 42 (pure: 78.78% / 8.78%), because its raw replay
    already supplies the boundary signal.

    **Provenance:** the Split-MNIST and Split-CIFAR-10 tables have been re-run
    with the fixed lock (results in this document); other tables produced
    before the fix should be re-run before citing them. Split-CIFAR-100 runs a
    scaled 20-task protocol on the fixed code (`results/cifar100_20task`).

16. **CIFAR 3-seed error bars close the validation loop.** All CIFAR headline
    rows now average seeds 42 1 2 (`experiments/recipes/multiseed_cifar.sh`):

    | Benchmark / method | Avg Acc | Forgetting | Seeds |
    | :-- | :--: | :--: | :--: |
    | CIFAR-10 conv, pure | 37.30 ± 0.11 | 21.98 ± 1.75 | 42 1 2 |
    | CIFAR-10 conv, hybrid | 38.66 ± 0.33 | 22.62 ± 0.92 | 42 1 2 |
    | CIFAR-10 ResNet-18, pure | 48.82 ± 1.42 | 21.05 ± 1.44 | 42 1 2 |
    | CIFAR-10 ResNet-18, hybrid | 49.66 ± 1.47 | 19.56 ± 1.36 | 42 1 2 |
    | CIFAR-100 conv, pure | 9.48 ± 0.25 | 23.19 ± 0.56 | 42 1 2 |
    | CIFAR-100 conv, hybrid | 9.98 ± 0.47 | 11.93 ± 0.66 | 42 1 2 |

    Run-to-run drift on the same config and seeds (same code, different process)
    was measured at up to ~1 accuracy point and ~3.5 forgetting points on one
    CIFAR-100 seed (the gate-ablation PAL rows versus this multiseed run);
    same-code CIFAR-10 repeats matched exactly. Comparisons inside one driver
    session (baselines and PAL-MoE in the same run, as in all headline tables)
    are unaffected, and repeated configs must stay in one session to be
    comparable. The CIFAR-100 ResNet-18 table is still single-seed.

17. **A stronger frozen backbone is the largest single lever on Split-CIFAR-100.**
    Swapping the frozen 50-epoch SimCLR conv encoder for a frozen ImageNet
    ResNet-18 (no pretraining, feature cache) lifts the 20-task result from
    9.48 ± 0.25 (conv, 3 seeds) to **16.01** pure and from 9.98 ± 0.47 to
    **18.96** hybrid (single seed 42, `results/cifar100_resnet18/`,
    `configs/cifar100_resnet18_frozen.json`). The baselines improve too
    (DER++ 12.57, iCaRL 13.24 with the lowest forgetting at 11.03), so the
    ranking story is unchanged while the absolute level roughly doubles. The
    prototype margin improves from −0.13 (conv) to −0.076, confirming that the
    conv representation — not the router or the validation gate — was the
    long-horizon bottleneck. The hybrid now beats iCaRL by 5.7 points; iCaRL
    keeps the lowest forgetting. 3-seed validation of this backbone is the next
    run.

18. **Frozen ViT-B/16 promotion (3 seeds, one expert per task).** The ViT
    results were previously single-seed and undocumented. Re-run in one session
    on `configs/cifar10_vit.json` (frozen ImageNet ViT-B/16, feature cache
    shared across seeds because the projection is the identity, 15 epochs/task,
    `--expand_every_task --max_experts 5`), seeds 42 1 2
    (`results/cifar10_vit_multiseed`, git `bfd71d0`):

    | Method | Avg Acc | Forgetting | Data bytes |
    | :-- | :--: | :--: | --: |
    | Naive Fine-tuning | 24.17 ± 0.64% | 93.86 ± 0.78% | 0 |
    | EWC | 23.72 ± 2.75% | 94.42 ± 3.44% | 0 |
    | DER++ (P=250) | 73.28 ± 0.31% | 32.10 ± 0.34% | 0.78 MB |
    | Experience Replay (P=250) | 82.89 ± 0.10% | 20.29 ± 0.17% | 0.77 MB |
    | iCaRL (k=25) | 84.18 ± 0.00% | 10.38 ± 0.00% | 0.80 MB |
    | **PAL-MoE (pure)** | **91.74 ± 0.28%** | **5.61 ± 0.12%** | **6.37 MB** |
    | PAL-MoE + Latent Replay (P=1000) | 91.74 ± 0.33% | 5.55 ± 0.11% | 6.37 MB |

    PAL-MoE pure beats the best baseline (iCaRL) by 7.6 accuracy points with
    about half its forgetting, storing no raw inputs. The new byte column also
    shows the honest caveat: at this geometry PAL stores ~8x the data bytes of
    the replay baselines, so this table is item-budget matched, not byte
    matched. The equal-byte Pareto (EXPERIMENT_PLAN.md, E4) is the fair
    comparison. iCaRL's zero variance is expected: the frozen ViT features are
    seed-independent and herding is deterministic. The latent-replay row is
    identical to pure, as expected under the feature cache (design fact 19:
    the hybrid's raw store is disabled and only latent exemplars are added).

    The CIFAR-100 counterpart (`configs/cifar100_vit.json`, 20 tasks, 10
    epochs, `results/cifar100_vit_multiseed`) is honest about a baseline that
    wins: iCaRL **64.94 ± 0.00% / 12.25%** at 8.0 MB beats PAL-MoE pure
    **59.34 ± 0.32% / 18.17%** at 14.2 MB (latent replay 59.03 ± 0.47 /
    18.85). On this benchmark the method is not the accuracy leader; the
    equal-byte and long-horizon forgetting results (facts 19-20, plan §5) are
    where the trade-off lives.

19. **Feature-cached runs store features, not raw inputs — and the hybrid's raw
    store is disabled there.** `--feature_cache` replaces the loaders with
    cached encoder outputs, so (a) every replay baseline and iCaRL store
    `feature_dim*4 + 8` bytes per item (256-d ResNet: 1,032 B; 768-d ViT:
    3,080 B) instead of raw 32x32 images (12,296 B), and (b)
    `store_raw = not feature_cache`, so the hybrid variant trains with latent
    exemplars only. Measured confirmation: on CIFAR-100 ResNet-18 the replay
    baselines' `memory_bytes` at P=250 is 258 KB (ER) — 250 stored features —
    not the 3.07 MB a raw-image buffer would need; and the ViT CIFAR-10
    promotion table's latent-replay row is a pure + latent-replay variant, not
    raw replay. This was found while preparing the equal-byte sweep (E4): the
    first sweep assumed raw-image item sizes and underfilled the baselines by
    ~12x. The corrected protocol uses feature item sizes for feature-cached
    runs and the raw pipeline (`configs/*_frozen_raw.json`) for the
    raw-vs-latent comparison. New PAL runs record `stores_raw` so the JSON is
    self-documenting.

## Ablations

`experiments/run_ablation.py` sweeps loss components, expert-init strategy,
the validation gate, routing top-k, encoder adaptation and the OOD term, with
one pretrained encoder **shared per seed** across all configs (controlled
ablation). Config selection supports case-insensitive substring filters
(`--configs "OOD"`). Results land in `results/ablation_results.json` +
`results/ablation_comparison.png`.

## CI verification

`.github/workflows/ci.yml`:
- **lint**: `ruff check` + `black --check` (versions pinned).
- **test**: pytest (96 tests) on Python 3.10-3.12, with coverage.
- **benchmark-verify**: CPU smoke of the full 12-method benchmark (1 epoch)
  asserting it completes and that PAL-MoE hybrid ≥ 50% + pure ≥ 25%
  (sanity bounds, not state-of-the-art checks).

Full multi-seed reproductions are run on the GPU machine before every README
update; a full CI reproduction job can be enabled with a GPU runner.

## Paper-track plan

[`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md) audits the results in this document
and `../README.md` against the paper's claims (what is strong, what is
apples-to-oranges, what is missing) and freezes the outstanding runs: the
equal-byte memory protocol, the component ablation at the final recipe, expert
growth/reuse curves and the long-horizon benchmarks.
