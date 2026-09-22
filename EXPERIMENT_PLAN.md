# PAL-MoE — Paper Experiment Plan

> Planning document, not a result record. It audits the results in `README.md` /
> `BENCHMARK.md` against the paper's claims, lists what is strong, what is
> apples-to-oranges and what is missing, then freezes a 13-experiment plan with
> exact commands, budgets and kill criteria.
>
> Companion docs: `README.md` (headline results), `BENCHMARK.md` (protocol and
> measured design facts). Every number quoted here is from the repo; nothing in
> this file may be cited as a result until it points at a `results/` artifact.

## Status (code ready 2026-09-21; runs deferred to 2026-09-22)

| Item | Status | Artifact |
| :-- | :-- | :-- |
| M1 byte accounting + budget overrides | **Done** | `memory_bytes`/`state_bytes`/`stored_bytes` in every result; `--buffer_size`, `--icarl_k`; generic `replay`/`latent_replay`/`mir` ids |
| M2 per-task instrumentation | **Done** | `--track_routing` records `experts_per_task`, expansion/gate counts and routing retention `RR_t` |
| M3 folder streams | **Done** | `--dataset folder --data_dir ... --classes_per_task --image_size`; Tiny-ImageNet flattened via `experiments/prepare_tiny_imagenet.py` (100k images ready under `data/`) |
| M4 latency | **Done** | `experiments/measure_latency.py` |
| E1a CIFAR-10 ViT 3-seed | **Done** (commit `fb69e63`) | pure **91.74 ± 0.28 / 5.61**; iCaRL 84.18 / 10.38; ER 82.89 / 20.29 (`results/cifar10_vit_multiseed`) |
| E1b CIFAR-100 ViT 3-seed | **Done, latent-replay row queued** | pure **59.34 ± 0.32 / 18.17**; iCaRL 64.94 / 12.25 (8.0 MB vs PAL 14.2 MB); DER++ 50.97 / 47.30 (`results/cifar100_vit_multiseed`) |
| E2 CIFAR-100 ResNet-18 3-seed | **Done** (commit `82b5684`) | pure **15.65 ± 0.39 / 31.69**; iCaRL 13.97 / 10.96; DER++ 13.12 / 66.91 (`results/cifar100_resnet18_multiseed`) |
| E4 equal-byte Pareto | **Feature-cache sweep done; raw sweep queued** | `results/equalbyte/` (real seeds 42 1 2 at 1/4 MiB, seed 42 at 256K/16M); `results/equalbyte_raw/` in wave 2; item sizes per design fact 19 |
| E5 component ablation | **Queued (wave 1c, real seeds)** | `results/ablation_final/` |
| E6 routing retention | **Running** (piggyback on the gated runs) | `routing_retention` fields |
| E7 growth/reuse | **Running** (wave 1c, `--max_experts 20` for both protocols) | `results/growth/` |
| E8 anchor drift + trainable-encoder cell | **Queued** | `results/drift/` |
| E9 capacity/param matching | **Queued** | `results/capacity/` |
| E10 Tiny-ImageNet | **Queued** (dataset prepared) | `results/tinyimagenet_multiseed` |
| E11 domain-incremental | **Pilot queued** (MNIST rotate 5 seeds); CORe50 blocked on a domain-stream loader | `results/mnist_domainshift_multiseed` |
| E12 modern baselines | **Partial**: MIR implemented and queued; prompt-based methods deferred (scope decision) | `pal_moe/baselines/mir.py` |
| E13 report/release | **Partial**: `experiments/paper_report.py` writes `results/paper_report.md` + figures; final tables after the run | `results/paper_report.md` |

### How to run (one command, detached)

```bash
cd /home/kael/pal-moe
nohup bash experiments/recipes/paper_all.sh > results/paper_run.log 2>&1 &
tail -f results/paper_run.log          # follow
grep "::" results/paper_wave1b.log results/paper_wave2.log   # step markers
```

`paper_all.sh` runs `paper_wave1b.sh` then `paper_wave2.sh` back to back:
corrected equal-byte sweeps (feature-cache and raw), E7/E5/E8/E9/E3-fast, the
E1a/E1b latent-replay repair, MIR, Tiny-ImageNet 20×10, the slow conv
regenerations, latency, the 5-seed domain-shift pilot and the appendix cells.
Expected ~2.5–3 h + ~5–6 h on the RTX 5060. Failures log `FAILED ...` and do
not stop the queue.

Afterwards:

```bash
python experiments/paper_report.py     # tables + Pareto/growth/matrix figures
```

then fold the numbers into `README.md` / `BENCHMARK.md` (design facts 18–19
already cover the ViT promotion and the feature-cache storage semantics).

**Protocol correction applied before the run day (design fact 19):** under
`--feature_cache` the replay baselines and iCaRL store cached features, and the
hybrid's raw store is disabled (it is pure + latent replay). The equal-byte
sweeps now use the correct item sizes; the true raw-vs-latent comparison runs
in the raw pipeline (`configs/*_frozen_raw.json`).

**Run-day corrections (2026-09-22):** the first wave wrote seed-1/seed-2
directories without passing `--seed` (cells were seed-42 repeats); all per-seed
helpers now pass it and the equal-byte cells were re-run (`d4e0f7a`). E7's
gated protocol hit the very slow prune/merge path at `--max_experts 6`
(296 s vs 7494 s for identical work); both protocols now run with
`--max_experts 20`, so the only difference is the allocation policy. Wave 1c
resumes from the fixed code and is followed by wave 2.

**Run-day findings so far (2026-09-22):**

- **Equal-byte (feature-cache, CIFAR-10, 3 seeds):** ER leads accuracy at every
  budget (49.86 at 1 MiB, 54.95 at 4 MiB); PAL pure is 46.32 / 50.54 with the
  lowest forgetting among non-iCaRL methods (21.37 / 19.15); iCaRL has low
  forgetting and much lower accuracy. CIFAR-100 at 1 MiB: DER++ 16.97, ER
  13.64, PAL 12.15 / 34.21 forgetting, iCaRL 10.72 / 9.93.
- **H5 refuted (E7):** gated allocation equals forced expansion (20/20 experts,
  14.25 ± 0.22 vs 14.34 ± 0.41 accuracy; owner-routing 94.5%). No expert reuse
  on CIFAR-100.
- **Router generalization gap (E7/E6):** prototype owner accuracy is 94.5% but
  test-time routing is diffuse (top-expert share 0.15–0.24); RR_t 0.78–0.85.

## 0. Framing

**Research question.** Under a fixed memory budget, *when* should a
class-incremental learner allocate new expert capacity instead of reusing
existing experts, and *how* should routing be stabilized so old knowledge
survives the allocation?

**Mechanisms.** (i) dynamic expert allocation (trigger + validation gate +
function-preserving expansion), (ii) latent replay / prototype memory
(`v_p`, `r_p`, `o_p`, latent exemplars), (iii) prototype-anchored routing
(owner distillation, inference anchoring, negative-boundary regularization).

**Hypotheses.**

| ID | Hypothesis | Current status |
| :-- | :-- | :-- |
| H1 | Dynamic specialization reduces task interference vs a shared single network. | Partially supported (no parameter-matched control, no gated-vs-forced run on a strong backbone). |
| H2 | At equal **bytes**, latent replay is competitive with raw replay. | Untested: all tables are item-budget matched, not byte matched. |
| H3 | Prototype anchors improve routing stability/accuracy. | Supported on MNIST/CIFAR-10 (facts 11, 15) but not isolated at the final recipe, and only at the end state. |
| H4 | The negative-boundary (OOD) objective reduces misassignment of new data to old experts. | Supported (facts 1, 12): +0.7–1.2 acc, 2–4 pp less forgetting. |
| H5 | The allocation policy reuses experts for related tasks instead of one expert per task. | **Refuted on CIFAR-100 (2026-09-22):** gated and forced both allocate one expert per task (20/20, no gate rejections) and are statistically identical (14.25 ± 0.22 vs 14.34 ± 0.41 accuracy). Reframe as fixed-budget capacity expansion. |
| H6 | Old-task routing is retained while new tasks get new capacity. | **Measured (2026-09-22):** routing retention RR_t stays 0.78–0.85 across 20 tasks; prototype owner accuracy 94.5%, but test-time routing is diffuse (top-expert share only 0.15–0.24) — a router generalization gap to report. |

**Novelty statement to defend (draft).** Not "a MoE for continual learning" —
that space is occupied. The defensible combination is: *memory-constrained
class-incremental learning with dynamically allocated experts, compact latent
replay, and prototype-anchored routing*, evaluated under an equal-byte
protocol. The paper must therefore prove the memory claim byte-for-byte and
show what the allocation policy actually decides.

---

## 1. Result audit

### 1.1 Evidence inventory and verdicts

| # | Result (evidence) | Verdict | Notes |
| :-- | :-- | :-- | :-- |
| R1 | **CIFAR-10 ResNet-18, 3 seeds** (`results/cifar10_resnet18_multiseed`): pure 48.82 ± 1.42 / 21.05 ± 1.44 forgetting vs DER++ 45.29 ± 0.48 / 43.81, ER 39.92 ± 0.76 / 58.87, iCaRL 30.70 ± 3.71 / 26.89. | **Strong** | Same encoder, same session, same schedule; zero raw storage. Cleanest headline. |
| R2 | **CIFAR-100 conv 20-task, 3 seeds** (`results/cifar100_multiseed`): pure 9.48 ± 0.25 / 23.19 ± 0.56 vs DER++ 5.92 ± 0.28 / 63.42 ± 0.48; hybrid 9.98 ± 0.47 / 11.93 ± 0.66 vs iCaRL 10.13 ± 0.40 / 10.54 ± 0.30 (iCaRL stores 2500 raw). | **Strong (long-horizon forgetting)** | The ~40 pp forgetting gap is far outside seed noise. Absolute accuracy is low; representation is the documented bottleneck (fact 17). |
| R3 | **ViT-B/16 CIFAR-10** (`results/cifar10_vit_multiseed`, seed 42 + 1; seed 2 missing): pure 92.03 / 5.54 vs iCaRL 84.18 / 10.38, ER 82.75 / 20.50, DER++ 72.96 / 32.46. | **Strong but undocumented and protocol-skewed** | Highest absolute numbers in the project, never written into `README.md`/`BENCHMARK.md`. Uses `--expand_every_task` (forced expansion, gate bypassed), 15 epochs, `proto_size=1000`; single seed. Must be promoted only after E1. |
| R4 | **MNIST 5-seed** (`results/benchmark_multi.json`): pure 79.36 ± 1.16 / 7.91 ± 1.24; hybrid 80.05 ± 1.08 / 5.61 ± 1.12; DER++ 86.93 ± 0.91 / 7.31. | Medium | Toy dataset; keep as a sanity/ablation bed, not as a headline. |
| R5 | **CIFAR-100 ResNet-18 single seed** (`results/cifar100_resnet18`): pure 16.01 / 28.89; hybrid 18.96 / 19.78 vs DER++ 12.57 / 67.28, iCaRL 13.24 / 11.03. | Medium (single seed) | Backbone swap is the largest lever measured (fact 17). Recipe for 3 seeds exists (`cifar100_resnet18_multiseed.sh`), not yet run. |
| R6 | **Domain-shift MNIST** (`results/mnist_domainshift`): pure 86.07 / 5.64; task-free online 86.16; margin +0.117. | Medium (single seed, pilot) | Supports the shared-expert/domain-incremental story; needs a real domain benchmark (E11). |
| R7 | **Controlled mechanism ablations** (facts 1, 11, 12, 15): OOD 49.85→76.60; distillation 21.3→38.0 raw router; calibration/distillation substitutes; exact routing lock. | Strong internal validity | Seed-42 / small budgets. Must be repeated as one matrix at the final recipe (E5). |
| R8 | **Gate policy ablation** (facts 16, `results/cifar100_gate_*`): relative vs absolute gate is a wash. | Strong negative result | Honest, but it weakens H5: the allocation policy currently does not change outcomes on conv CIFAR-100. |

### 1.2 Apples-to-oranges list (fix or flag before publishing)

| ID | Issue | Where | Fix |
| :-- | :-- | :-- | :-- |
| AO1 | **Item budget ≠ byte budget.** PAL `proto_size=1000` vs replay `buffer_size=250`; DER++ also stores logits; iCaRL CIFAR-100 stores 2500 raw exemplars (10×). Measured bytes: PAL pure 2.08 MiB (CIFAR-10 ResNet-18), 3.98 MiB (CIFAR-100 ResNet-18), 6.07 MiB (CIFAR-10 ViT); ER P=250 ≈ 2.93 MiB of raw CIFAR. The sign of the mismatch changes per dataset/encoder. | all tables | E4 equal-byte protocol; report bytes in every table (M1). |
| AO2 | **Parameter budget.** PAL total params scale with experts (1.65M vs 274k on CIFAR-100; ViT run 2.99M vs 598k) while active per-sample params are matched. Feature-cache runs exclude the cached backbone from `total_params`, so cross-run param counts are not comparable. | README param tables, result JSONs | Report total/trainable/active with a stated definition; add a parameter-matched baseline (E9). |
| AO3 | **Encoder changes across tables** (conv vs ResNet-18 vs ViT) are a representation lever, not a method comparison. | README §3/§5 | Keep encoders fixed within a table; present backbone swaps as a controlled lever only. |
| AO4 | **Single-seed rows inside multi-seed tables** (CIFAR-10 conv/ResNet baselines, CIFAR-100 baselines, domain shift). | README §2/§3/§4/§6 | Fill seeds or mark explicitly as single-seed. |
| AO5 | **Cross-process drift**: same config+seed can move ~1 acc / ~3.5 forgetting point across processes (fact 16). Tables produced before the routing-lock fix (fact 15) must not be cited. | BENCHMARK provenance | One session per table; regenerate all headline tables after the last code change (E3). |
| AO6 | **Forced expansion vs gated allocation** are different method variants (ViT configs use `--expand_every_task`; conv configs use the gate). | ViT vs conv tables | Name the protocol in every table; measure both (E7). |
| AO7 | **"Zero replay" is zero *raw* replay**, not memory-free: latent exemplars, prototypes, routing rows and output anchors are stored. | README wording | Use "zero raw replay" everywhere; state what is stored. |
| AO8 | **Forgetting is clamped at zero** (pessimistic convention) while BWT is signed. | `pal_moe/evaluation/metrics.py` | State the convention in the paper; report BWT alongside. |
| AO9 | **Task-free claims rest on one MNIST run**; the trigger uses labels (supervised novelty). | README §6 | Scope the claim; run a real domain stream (E11). |
| AO10 | **Buffer sampling policy** (`recency` default vs `reservoir`) is not explored for all baselines. | BENCHMARK protocol | Report both in the appendix or justify the default. |

### 1.3 Missing evidence (what the paper still needs)

| ID | Missing | Needed by |
| :-- | :-- | :-- |
| M1 | Byte accounting per method in every result JSON; `--buffer_size`/`--icarl_k` overrides so budgets can be swept. | H2, AO1, E3, E4 |
| M2 | Per-task instrumentation: experts per task, expansion/gate events, routing snapshots on a fixed probe set (for routing retention RR_t and routing matrices over time). | H5, H6, E6, E7 |
| M3 | Folder-stream wiring (`--dataset folder --data_dir`) for Tiny-ImageNet / ImageNet-100 / CUB / CORe50. Library exists (`pal_moe.data.split_folder`), runner does not expose it. | E7, E10, E11 |
| M4 | Latency / throughput per sample and FLOPs estimate (only `fit_seconds` exists). | E9, cost table |
| M5 | Equal-byte Pareto curves. | E4 |
| M6 | Component ablation at the **final** recipe (the fact-12 grid is a 5-epoch conv side study). | E5 |
| M7 | Oracle-routing decomposition and routing retention on a strong backbone. | E6 |
| M8 | Expert growth/reuse curves under the gate. | E7 |
| M9 | Latent/anchor drift study (`refresh_anchors_after_calib`, stale anchors). | E8 |
| M10 | Harder benchmarks and modern baselines (MIR/GSS/ASER; L2P/DualPrompt/CODA-Prompt; recent MoE-CL references). | E10–E12 |

---

## 2. Protocol rules (freeze before running anything)

1. **One encoder per table**, shared by every method, stated in the caption.
   Representation is a control variable (AO3).
2. **One session per table** (AO5): all methods and all seeds of a table are
   produced by one driver run after the final code commit; `benchmark_meta_*.json`
   records the git hash (already automatic).
3. **Seeds:** minimum 3 (42 1 2) for every published row; 5 (42 1 2 3 4) for
   the MNIST sanity table and the final headline tables.
4. **Budgets:** every table reports stored **bytes** and
   total/trainable/active params next to accuracy. Item counts are secondary.
5. **Metrics:** Avg Acc, Forgetting, BWT, owner-routing accuracy, routing
   retention RR_t, expert count, bytes, latency, `fit_seconds`.
6. **No test-set tuning**; validation splits only (the gate already uses them).
7. **Naming:** "zero raw replay" (AO7); "negative-boundary regularization"
   instead of "OOD detection".
8. **Configs are versioned** in `configs/`; every published cell names its
   config file and CLI overrides. No ad-hoc flag soups in the paper.

---

## 3. Prerequisites (measurement plumbing, no new science)

| ID | Work | Where | Done when |
| :-- | :-- | :-- | :-- |
| M1 | `memory_bytes` per result entry: replay buffer tensors + DER++ logits + iCaRL herding state + prototype elements + model params. Add `--buffer_size` / `--icarl_k` overrides (currently hardcoded 60/250/360 and k=25 in `run_benchmark.py`). | `experiments/run_benchmark.py`, `pal_moe/memory/prototype_memory.py` | every result JSON has `memory_bytes`; a byte sweep is a flag change. |
| M2 | Per-task instrumentation: `experts_per_task`, `expansions_per_task`, `gate_rejections`, and router top-1 snapshots on a fixed probe set at each task boundary. The trainer already returns `hist` (`experts_added`, `gate_rejections`) but the runner only prints it. | `experiments/run_benchmark.py` (`_run_palmoe_variant`), `pal_moe/adaptation/ttt.py` | E6/E7 can be computed from a normal run. |
| M3 | Folder-stream wiring: `--dataset folder --data_dir DIR` using `pal_moe.data.split_folder.get_split_folder_tasks`, with an encoder-appropriate transform. | `experiments/run_benchmark.py` dataset builders | Tiny-ImageNet / CORe50 run with one command. |
| M4 | Latency script: per-sample forward latency (batch 1 and 128) + FLOPs estimate, using the three parameter counts. | new `experiments/measure_latency.py` | cost table row per method. |

---

## 4. Experiment plan

Status: **READY** = runnable with current flags; **NEEDS Mx** = blocked on the
prerequisite above. Cost estimates are for the RTX 5060 laptop GPU and use
measured durations from the result meta files.

### E1 — Promote the ViT-B/16 results (H1, H3)

**Question.** Do the strongest absolute numbers survive multi-seed, and do they
survive documentation/protocol review?
**Setup.** `configs/cifar10_vit.json` and `configs/cifar100_vit.json` (frozen
ImageNet ViT-B/16, feature cache, one expert per task). Complete seed 2 and
re-run 42/1 in one session.

```bash
python experiments/run_benchmark_multi.py --seeds "42 1 2" --device cuda \
  --config configs/cifar10_vit.json --output_dir results/cifar10_vit_multiseed

python experiments/run_benchmark_multi.py --seeds "42 1 2" --device cuda \
  --config configs/cifar100_vit.json --output_dir results/cifar100_vit_multiseed
```

**Deliverable.** Two 3-seed tables + a `BENCHMARK.md` design fact; label the
protocol "one expert per task" explicitly (AO6).
**Cost.** CIFAR-10 ≈ 15 min (cache exists); CIFAR-100 ≈ 75 min plus a one-time
≈ 15–25 min feature extraction. **Status: READY.**

### E2 — CIFAR-100 ResNet-18 3-seed (H1)

```bash
bash experiments/recipes/cifar100_resnet18_multiseed.sh
```

**Deliverable.** Converts the "largest single lever" claim (fact 17) into a
3-seed row. **Cost.** ~35–45 min. **Status: READY.**

### E3 — Headline table regeneration with bytes and params (AO1, AO2, AO5)

**Setup.** After M1, regenerate MNIST (5 seeds), CIFAR-10 conv (3), CIFAR-10
ResNet-18 (3), CIFAR-100 conv (3), CIFAR-100 ResNet-18 (3) in one session per
dataset; every table gains `bytes` and `total/trainable/active params` columns.
**Deliverable.** The paper's Table 1 + a provenance paragraph (git hash, one
session, seed set). **Cost.** ~3 h total. **Status: NEEDS M1.**

### E4 — Equal-byte Pareto (H2) — flagship figure

**Question.** At a fixed byte budget, which method wins?
**Setup.** After M1, sweep byte budgets (log-spaced: 256 KB, 1 MB, 4 MB, 16 MB;
CIFAR-100: 256 KB, 1 MB, 4 MB) on CIFAR-10 ResNet-18 and CIFAR-100 ResNet-18,
methods `palmoe, hybrid, replay*, derpp, icarl`, each method's item budget set
to fit the byte budget exactly (prototype size for PAL, buffer size for replay,
per-class k for iCaRL). 3 seeds.

```bash
# template (after --buffer_size / --icarl_k land); one cell per method x budget
python experiments/run_benchmark.py --config configs/cifar10_resnet18_frozen.json \
  --methods replay250,derpp,icarl,palmoe,hybrid \
  --buffer_size 85 --icarl_k 8 --proto_size 480 \
  --output_dir results/equalbyte/c10r18_1MB --device cuda
```

(≈ 85 raw images ≈ 1 MiB; 480 PAL prototypes ≈ 1.04 MiB; DER++ adds its stored
logits. Each cell's exact item count is derived from the byte target, not
guessed: the M1 accounting reports the realised bytes.)

**Deliverable.** Accuracy/forgetting vs bytes Pareto (main figure), with the
crossover budgets stated. **Cost.** ≈ 2 h (CIFAR-10) + ≈ 2 h (CIFAR-100).
**Status: NEEDS M1.**

### E5 — Component ablation at the final recipe (H1, H3, H4)

**Setup.** 7-variant ladder × {CIFAR-10 ResNet-18, CIFAR-100 ResNet-18} × 3
seeds, one session per dataset:

| # | Variant | Recipe |
| :-- | :-- | :-- |
| 1 | Single NN | `--methods naive` |
| 2 | ER (raw) | `--methods replay250` |
| 3 | Latent replay, single head | new `latent_replay` method (M1; single head + stored latents) |
| 4 | Static MoE | `--methods palmoe --expand_every_task --lambda_r 0 --lambda_e 0 --lambda_ood 0 --router_anchor_steps 0 --joint_calib_epochs 0` |
| 5 | + prototype anchors | variant 4 + `--lambda_r 0.5 --lambda_e 2.5 --router_anchor_steps 300` |
| 6 | + gated allocation | variant 5 without `--expand_every_task` |
| 7 | + negative boundary (full) | published config (`--lambda_ood 0.1`, config defaults) |

The recipes are flag sketches: freeze each variant as a config file before
running (for the "static" variants, also shrink the prototype store so the
memory budget matches the baselines).

**Deliverable.** "Which component pays for itself" table; validates or corrects
facts 1/11/12 at the final recipe. **Cost.** ~2–3 h. **Status: NEEDS M1
(`latent_replay`), otherwise READY.**

### E6 — Routing decomposition and retention (H6)

**Setup.** After M2, for every E3/E7 run record at each task boundary on a fixed
probe set (all test splits seen so far, subsampled): top-1 expert, owner-routing
accuracy, end-to-end accuracy, expert-oracle accuracy, and
RR_t = P(route unchanged since task t−1). Report the (task × expert) routing
matrix at the end and its growth.
**Deliverable.** A figure separating router failure from expert/representation
failure; the RR_t curve is the routing stability/plasticity evidence.
**Cost.** Piggybacks on E3/E7 (~30–60 min extra). **Status: NEEDS M2.**

### E7 — Expert growth and reuse under the gate (H5)

**Question.** Does the allocation policy ever reuse an expert, or does every
task get one?
**Setup.** After M2 (and M3 for Tiny-ImageNet), gated vs forced expansion on
CIFAR-100 ResNet-18 and Tiny-ImageNet, 3 seeds. Report experts per task, gate
rejections and reasons, routing matrices, a reuse index (fraction of tasks whose
majority expert is shared with an earlier task), accuracy and params.
**Deliverable.** Growth curve figure; honest answer to H5.
**Cost.** ≈ 40 min (CIFAR-100) + ≈ 4–8 h (Tiny-ImageNet, after M3).
**Status: NEEDS M2 (+ M3 for Tiny-ImageNet).**

### E8 — Latent/anchor drift study (H2, stale-anchor risk)

**Setup.** CIFAR-100 ResNet-18, 3 seeds: `--refresh_anchors_after_calib` on/off ×
`--proto_routing_alpha {0, 0.5}`; plus one trainable-encoder cell
(no `--freeze_encoder`) on CIFAR-10 to measure true latent drift (frozen encoders
cannot drift; what goes stale there is `r_p`/`o_p` after calibration).
Metrics: prototype margin, router KL on `v_p`, expert-output drift on `v_p`,
accuracy/forgetting.
**Deliverable.** Anchor-refresh policy recommendation; the "latent replay stays
valid" argument the latent-replay literature demands.
**Cost.** ≈ 2 h. **Status: READY (metric exposure needs M2 or a small script).**

### E9 — Capacity/cost scaling and a parameter-matched baseline (H1, AO2)

**Setup.** CIFAR-100 ResNet-18, 3 seeds: `--max_experts {1,2,4,6,unlimited}` at
fixed bytes; plus a single-head baseline whose `--expert_hidden` is widened to
match PAL's **total** params (a sensitivity control, run as a separate
invocation so other methods are unaffected); latency/FLOPs from M4.
**Deliverable.** Accuracy/forgetting vs params/latency; kills the naive "more
experts = better" reading (cf. the MoE-in-CL theory result).
**Cost.** ≈ 2 h + latency script. **Status: READY (latency needs M4).**

### E10 — Tiny-ImageNet / ImageNet-100 (H1, H2, generalization)

**Setup.** After M3, 20-task class-IL (10 classes/task) with frozen ResNet-18
and/or ViT, feature cache; methods `palmoe, hybrid, replay*, derpp, icarl`;
3 seeds. Optionally a 100-task (2 classes/task) long-horizon variant to match
modern online-CL protocols.
**Deliverable.** The "serious benchmark" table that replaces Split-CIFAR-only
evidence. **Cost.** ≈ 4–8 h (Tiny-ImageNet, cache included); ImageNet-100 adds
download + cache (≈ 1–2 days). **Status: NEEDS M3.**

### E11 — Domain-incremental and task-free evaluation (H4, shared expert)

**Setup.** CORe50 (11 class-shared domains) via M3, shared expert
(`--shared_expert --freeze_shared_after 2`), task-free metrics
(`--task_free_eval`); compare PAL pure/hybrid vs ER/DER++ at equal bytes.
MNIST rotations (`--domain_shift rotate`) remains the pilot.
**Deliverable.** Online/recent accuracy + surprise curves; evidence that the
generalist expert pays off when the label space is shared.
**Cost.** ≈ 4–6 h (dataset + cache + runs). **Status: NEEDS M3.**

### E12 — Modern baselines and positioning (novelty defense)

**Setup.** (a) Implement MIR and optionally GSS in-codebase (small, replay
family, keeps the same-protocol rule). (b) Decide whether prompt-based methods
(L2P/DualPrompt/CODA-Prompt) are in scope: either implement on the frozen ViT or
run them in Mammoth as a **separate, clearly-caveated** table. (c) Verify and
discuss the 2024–2026 MoE-CL line (theory of MoE in CL; adaptive expert
expansion; dynamic MoE adapters; task-shared/specific experts) — this is the
related-work section's backbone, not necessarily an experiment.
**Deliverable.** "Why this is not just another MoE-CL paper" paragraph with
numbers; scope statement for rehearsal-free methods.
**Cost.** 1–2 days if prompt methods are included; ~half a day otherwise.
**Status: NEEDS implementation / scope decision.**

### E13 — Statistical protocol and artifact release

5-seed final tables for the 2–3 headline benchmarks; pre-registered configs;
same-session baselines; raw JSONs + figure scripts committed; README/BENCHMARK
rewrite with byte and param columns; artifact hash/DOI. **Status: process.**

---

## 5. Paper artifact map

| Artifact | Source |
| :-- | :-- |
| Table 1 — main class-IL results | E3 (+ E1 for the ViT row) |
| Figure 1 — equal-byte Pareto | E4 |
| Table 2 — component ablation | E5 |
| Figure 2 — expert growth + routing matrix | E7 |
| Figure 3 — routing retention RR_t | E6 |
| Table 3 — capacity/cost scaling | E9 |
| Table 4 — long-horizon Tiny-ImageNet | E10 |
| Table 5 — domain-incremental / task-free | E11 |
| Appendix — drift study, buffer-policy ablation, external baselines | E8, AO10, E12 |

## 6. Execution order and kill criteria

**Order:** M1 → E1 → E2 → E4 → E7 → E5 → E6 → E3 → E8 → E9 → E10 → E11 → E12
→ E13. (E1/E2 need no code and immediately strengthen the weakest documented
claims.)

**Decision points.**

- **After E4:** if baselines dominate PAL at equal bytes on both CIFAR-10 and
  CIFAR-100, narrow the claim to long-horizon forgetting and the hybrid mode,
  and report the crossover budgets instead of a blanket win.
- **After E7:** if the gate never reuses experts and forced expansion matches
  gated, drop H5 as a headline; reframe the contribution as fixed-budget
  capacity expansion + prototype-anchored routing. If reuse appears on
  Tiny-ImageNet/CORe50, feature it as the central result.
- **After E8:** if anchor refresh is clearly better, make it the default and
  re-run E3 once.
- **After E5:** if the prototype/routing components do not individually matter
  at the final recipe, state that the contribution is the combination, not any
  single component (this is already the honest novelty framing).
- **After E10:** if the gap to baselines shrinks with scale, say so and shift
  the emphasis to the memory/Pareto axis.

## 7. Related-work checklist (verify references before citing)

Names/years only; the links collected during the literature scan contain
title/URL mismatches, so every entry must be checked against the original paper
before it enters the bibliography.

| Line | Works to verify and cite | Role |
| :-- | :-- | :-- |
| CL foundations | EWC, SI, LwF, GEM, A-GEM, ER, DER/DER++, ER-ACE, MIR, GSS, ASER | replay/regularization baselines |
| Prototype/representation | iCaRL, CoPE, SCR, RanPAC, latent replay | prototype routing, latent memory |
| Parameter isolation | PNN, HAT, PackNet, Expert Gate | capacity expansion, expert selection |
| Rehearsal-free | L2P, DualPrompt, CODA-Prompt | scope decision (E12) |
| MoE in CL | theory of MoE in CL (2024), adaptive/Incremental MoE (2025), MoE-Adapters++ (2025), task-shared/specific MoE (2026), MOSE (CVPR 2024) | direct competitors, novelty boundary |
| Evaluation | Mammoth, equal-byte non-inferiority protocol (2026), modern CL survey (2026) | protocol and fairness |

## 8. Open decisions

1. **Headline protocol:** forced one-expert-per-task (clean, ViT-strong) vs
   gated allocation (tests the actual research question). Recommendation:
   report both, with the gated run as the mechanism study and the forced run as
   the fixed-budget comparison.
2. **Prompt-based baselines:** in scope (E12) or scope the paper to
   rehearsal-based CL?
3. **Domain benchmark:** CORe50 (small, class-shared) vs a DomainNet subset.
4. **Venue/format:** workshop vs conference decides whether E10–E12 are
   mandatory or appendix material.
5. **Compute:** all estimates assume the single laptop GPU. If a larger GPU is
   available, E10–E12 stop being the bottleneck.
