# Research Map - PAL-MoE components and their literature lineage

Which research line each part of the codebase comes from, what is
implemented vs adapted vs absent, and where the evidence lives. Companion to
`BENCHMARK.md` (protocol/design facts) and `EXPERIMENT_PLAN.md` (paper plan).
Bibliographic details must be verified against the original papers before
they enter a bibliography (see the checklist in `EXPERIMENT_PLAN.md` section 7).

## 1. Continual-learning foundations

| Literature line | Representative work | In this repo | Evidence |
| :-- | :-- | :-- | :-- |
| Regularization | EWC (Kirkpatrick et al., 2017) | `pal_moe/baselines/ewc.py` (multi-task + online Fisher) | README MNIST/CIFAR tables |
| Regularization | SI (Zenke et al., 2017) | not implemented | - |
| Distillation | LwF (Li & Hoiem, 2017) | `--lambda_lwf` in `adaptation/ttt.py` (opt-in) | design facts 11-13 |
| Gradient projection | GEM / A-GEM (Lopez-Paz 2017; Chaudhry 2019) | `pal_moe/baselines/agem.py` | README tables |
| Replay | ER (Chaudhry 2019 / Rolnick 2019) | `pal_moe/baselines/replay.py` + `baselines/buffer.py` | README tables |
| Replay + distillation | DER / DER++ (Buzzega et al., 2020) | `pal_moe/baselines/der.py` (`DERPP`) | README tables |
| Asymmetric replay | ER-ACE (Caccia et al., 2022) | `pal_moe/baselines/der.py` (`ERACE`) | README tables |
| Interference-based selection | MIR (Aljundi et al., 2019) | `pal_moe/baselines/mir.py` | wave-2 runs |
| Prototypes / NCM | iCaRL (Rebuffi et al., 2017) | `pal_moe/baselines/icarl.py`; prototype anchors `v_p`, `o_p` in `memory/prototype_memory.py` | README tables, E8 |
| Latent replay | Latent Replay (Pellegrini et al., 2020) | `pal_moe/baselines/latent_replay.py`; latent exemplars `x_p`; design fact 19 | E5, E4 |
| Coreset selection | k-center / uncertainty | `--proto_selection`, `--proto_candidate_pool` | toolbox tests |

## 2. Capacity expansion and parameter isolation

| Literature line | Representative work | In this repo | Evidence |
| :-- | :-- | :-- | :-- |
| Progressive networks | PNN (Rusu et al., 2016) | function-preserving clone in `pal_moe/models/expert.py` (`clone_function_preserving`, `widen`) | tests; `--expansion_action widen` |
| Expert selection | Expert Gate (Aljundi et al., 2017) | `pal_moe/trigger/` + `pal_moe/builder/expert_builder.py` (validation gate) | E7 (gated vs forced) |
| Parameter isolation | PackNet (Mallya & Lazebnik, 2018), HAT (Serra et al., 2018) | expert freezing + exact routing lock (`freeze_historical_experts`, zero-weight-decay router group) | design fact 15 |
| Merging | model soup / TIES / task arithmetic | `pal_moe/merge.py`, `experiments/merge_experts.py` | toolbox tests |

## 3. Mixture-of-Experts in continual learning

| Literature line | Representative work | In this repo | Evidence |
| :-- | :-- | :-- | :-- |
| Fixed sparse MoE | Switch-style load balancing | `experiments/run_benchmark.py::_run_standard_moe` ("Standard MoE") | README baselines |
| MoE theory in CL | theory of MoE in continual learning (2024) | motivation for the allocation/reuse question (H5); no theory implementation | E7 result (no reuse) |
| Adaptive expert expansion | adaptive / Incremental MoE (2025), MoE-Adapters++ (2025) | dynamic allocation via trigger + gate; direct comparison is future work | E7; `EXPERIMENT_PLAN.md §7` |
| Routing stability | router anchoring / distillation | `--router_anchor_steps`, `--router_anchor_margin`, prototype-owner distillation | design facts 11, 15; E5 |

## 4. Evaluation and protocol

| Literature line | Representative work | In this repo | Evidence |
| :-- | :-- | :-- | :-- |
| Benchmark framework | Mammoth (Boschini et al., 2022) | protocol inspiration; all methods re-implemented in one codebase | `experiments/run_benchmark.py` |
| Equal-byte fairness | equal-byte non-inferiority protocol (2026) | `memory_bytes`/`state_bytes`/`stored_bytes` in every result; `--buffer_size`/`--icarl_k`; E4 sweep | `results/equalbyte*` |
| Online / task-free CL | MOSE (Yan et al., CVPR 2024) | `pal_moe/evaluation/task_free.py`, `--task_free_eval`, `--trigger energy` | domain-shift pilot |
| Foundation backbones | frozen ImageNet ResNet-18 / ViT-B/16 | `pal_moe/models/encoder.py`, `--encoder_weights imagenet`, feature cache | ViT promotion (fact 18) |

## 5. What is novel here (and what is not)

**Not claimed as novel:** prototype/NCM memory, replay, distillation, gating,
function-preserving expansion, routing locks - all individually established.

**Claimed (to be defended by the experiments):**

1. the combination of *dynamically expanded experts + compact latent replay +
   prototype-anchored routing* under a **class-incremental, fixed-memory**
   protocol;
2. **function-space anchoring** of routing and expert behaviour through
   `v_p/r_p/o_p`, including the exact routing lock correction (fact 15);
3. an **equal-byte evaluation protocol** and the resulting trade-off
   (accuracy vs forgetting per stored byte).

**Explicitly refuted by the current evidence (do not claim):**

- "dynamic allocation reuses experts" (H5): gated allocation equals forced
  expansion on CIFAR-100 (20/20 experts);
- "the OOD negative-boundary loss is the key mechanism" in the current recipe:
  it is neutral once router distillation is present (E5, revising fact 1);
- "zero replay" as memory-free: it is *zero raw replay*.

## 6. Where new code should go

| If you add... | Put it in... | Wire it via |
| :-- | :-- | :-- |
| a new baseline trainer | `pal_moe/baselines/<name>.py` | `run_benchmark.py` method id + `method_keys` |
| a new dataset/stream | `pal_moe/data/` | `--dataset` branch in `run_benchmark.py` |
| a new router/expert/encoder | `pal_moe/models/` | `pal_moe/factory.py` |
| a new stability/regularization term | `pal_moe/adaptation/ttt.py` | a `--lambda_*` flag + config |
| a new metric | `pal_moe/evaluation/` | `_record_baseline_result` / result JSON |
| a new experiment batch | `experiments/recipes/*.sh` | commit a results dir + inventory entry |
