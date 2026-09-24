# PAL-MoE Measurement Contract (S0)

Status: proposed, 2026-09-23. Companion to `BENCHMARK.md` (protocol and design
facts), `EXPERIMENT_PLAN.md` (v1 paper plan) and `PALMOE_V2_SPEC.md` (v2 design
freeze). This document defines the *measurement* layer that every later stage
(S1-S11) must emit against. It changes no model code and no existing result.

The point of S0 is stated in one line:

> Every new experiment must emit a **model-independent record** whose fields
> were decided before the experiment ran, so that no metric has to be added
> after the fact.

---

## 1. Two levels, not one

A single JSON per run cannot answer "is the mechanism order-sensitive?" or "is
the accuracy difference significant?". Those are properties of a *set* of runs.
So the contract has two record types:

| level | answers | produced by |
| :-- | :-- | :-- |
| **run record** | what was run, and what was measured in that run | one runner invocation |
| **study record** | aggregation, variance, paired comparison over runs | an analysis step over run records |

`order_variance`, `ci95` and `paired` deltas belong to the study level. Putting
them in a run record (as an earlier sketch did) makes them ill-defined.

---

## 2. Run record

### 2.1 Shape

```json
{
  "schema_version": "1.0",
  "blocks": ["cost", "learning", "modular"],
  "reproducible": true,
  "run_id": "cifar100__class_il__vit_b_16__ncm__seed42__order0",

  "factors":    { "...": "what was run (inputs)" },
  "metrics":    { "...": "what was measured (outputs)" },
  "provenance": { "...": "reproducibility" }
}
```

### 2.2 Rules

**R1 - factors and metrics never mix.** Factors are inputs (dataset, protocol,
backbone, budgets, seed). Metrics are outputs (accuracy, achieved bytes,
expert count). A budget is a factor; the cost actually used is a metric. The
validator rejects a record where a budget key appears under `metrics`.

**R2 - the order is recorded explicitly, not only as a seed.** `class_order`
is the full per-task class sequence. A seed is not enough: it does not survive
a change to the splitter, and it cannot be read by a human looking at a table.
`task_order_seed` is kept alongside as the *generator* of that order.

**R3 - the protocol is explicit and comparable only to itself.** Every record
carries `protocol` (`class_il`, `task_il`, `domain_il`, `task_free`) and
`task_id_at_inference` (bool). The study aggregator refuses to aggregate
records with different `(protocol, task_id_at_inference)`: mixing them is the
single easiest way to publish an apples-to-oranges table.

**R4 - conformance is declared as metric BLOCKS, not as a level.** A record
declares which blocks it fully satisfies:

| block | fields | who can emit it |
| :-- | :-- | :-- |
| `learning` | accuracy, forgetting, `acc_matrix` | every run (bwt is a derived view of the matrix) |
| `cost` | `stored_bytes`, `total_params` | every run that recorded its byte/parameter cost (the Pareto essentials) |
| `compute` | `flops_forward`, `latency_ms`, `optimizer_steps` | requires the S8 plumbing; kept separate from `cost` so a run with storage accounting is not excluded from a Pareto study |
| `modular` | `expert_count`, `routing_entropy`, `utilization` | PAL-MoE and Standard-MoE |
| `generalization` | `fwt`, `transfer_accuracy`, `representation_drift`, `routing_drift` | requires the S6/S7 plumbing |
| `robustness` | `corruption_accuracy`, `confounder_split_accuracy` | requires S9 |

`build_run_record` **derives** the list from the data, so a declaration cannot
drift from reality, and the validator rejects a hand-written record that claims
a block whose fields are null. A study declares `requires_blocks` and the
aggregator refuses records that do not satisfy them.

Why blocks and not levels: a level forces a total order, and it made 81 legacy
runs (which predate byte accounting) unrepresentable. A run that never measured
its cost declares `["learning"]` and stays usable in an accuracy study; it is
simply excluded from a Pareto study. That is the honest behaviour, and it is
what let all 524 existing result records ingest cleanly (see 8.1).

**R4b - reproducibility is a separate flag.** `reproducible` is true iff the
seed, the git commit and the backbone are known. A record can be usable
(blocks satisfied) without being reproducible; a paired study must require
both.

**R5 - `null` is not `0`.** A metric that was not measured is `null`; a metric
that is genuinely zero (e.g. `final_experts == 1` for a single-head baseline)
is `0`. Every consumer must handle `null` explicitly.

**R6 - backward compatibility.** The 40+ existing `benchmark_results_*.json`
files stay readable. A record without `schema_version` is treated as `0.x` and
mapped through `pal_moe.evaluation.schema.upgrade_v0_result`, with unmeasured
fields set to `null`. Nothing already published is invalidated.

**R7 - paired comparison is the default claim.** A claim of the form "A beats
B" must be supported by paired cells (same `seed` x same `class_order`). An
unpaired mean difference is reported as `unpaired: true` and is not a claim.

**R8 - model independence.** No required field may reference a model-specific
structure. `final_experts` is `L1` and optional; a single-head baseline simply
leaves it `null` at `L0`. This is what makes one table hold NCM, a linear
probe, iCaRL and an MoE without special cases.

---

## 3. Factors

| field | type | meaning |
| :-- | :-- | :-- |
| `dataset` | str | `mnist`, `fashion_mnist`, `cifar10`, `cifar100`, `tiny_imagenet`, `folder`, ... |
| `protocol` | enum | `class_il`, `task_il`, `domain_il`, `task_free` |
| `task_id_at_inference` | bool | is the task id given to the model at test time |
| `num_tasks` | int | number of tasks/phases/windows |
| `classes_per_task` | int or null | null for domain/task-free streams |
| `class_order` | list[list[int]] | explicit per-task class ids |
| `task_order_seed` | int or null | generator of `class_order` |
| `seed` | int | parameter-init / data-order seed (separate from the above) |
| `model_family` | str | `ncm`, `linear_probe`, `ridge`, `single_head`, `static_moe`, `dynamic_moe`, ... |
| `backbone` | str | `mlp`, `conv`, `resnet18`, `vit_b_16`, `clip_vit_b32`, ... |
| `backbone_pretraining` | str | `random`, `imagenet_frozen`, `imagenet_finetuned`, `simclr_frozen`, `checkpoint:<path>` |
| `readout` | str | `ncm`, `cosine_head`, `linear`, `bias_corrected` |
| `expert` | str | `none`, `mlp_head`, `residual_adapter_r8`, ... |
| `budget.memory_bytes` | int or null | allowed stored bytes (null = unlimited) |
| `budget.params` | int or null | allowed trainable parameters |
| `budget.compute_flops` | int or null | allowed training FLOPs |
| `budget.steps` | int or null | allowed optimizer steps |
| `data_fraction` | float | fraction of each task's training data used (1.0 default) |

`backbone` + `backbone_pretraining` is deliberately two fields: `vit_b_16` with
`random` and with `imagenet_frozen` are different experimental conditions and
the whole point of S3 is to separate them.

---

## 4. Metrics

### 4.1 learning (all levels)

| field | definition | source |
| :-- | :-- | :-- |
| `accuracy` | mean over tasks of `R[T, i]` (final row) | exists |
| `forgetting` | repo convention: mean of `max(0, max_{t<T} R[t,i] - R[T,i])` | exists |
| `bwt` | backward transfer, same sign convention as the repo | exists |
| `fwt` | **forward transfer**, measured at the representation level: a closed-form probe fitted on the incoming task using the model's current representation, minus the same probe on the raw frozen features. Positive means the adaptation so far helps a task it has never seen | S2 (`experiments/s2_ladder.py::forward_transfer`) |
| `acc_matrix` | `R[t, i]`, the full lower triangle | exists |
| `acc_curve` | `accuracy(t)` over the run (derived from `acc_matrix`, exposed for plotting) | cheap |

`fwt` definition is pinned because it is easy to get wrong, and the obvious
version *is* wrong: "evaluate the model on task `k+1` before training on it"
returns structurally 0 in a growing-head class-incremental setting, because the
task's classes have no head rows yet and are masked out. S2 measured exactly
`-mean(1/5t) = -3.73%` after chance correction, identical for every level -
a metric that cannot distinguish anything. The pinned definition therefore
measures the *representation* instead: a closed-form ridge probe on the
incoming task using the model's current representation, minus the same probe on
the raw frozen features. That is comparable across methods, needs no head rows
for unseen classes, and has a clean sign (positive = forward transfer).

### 4.2 generalization (L2)

| field | definition |
| :-- | :-- |
| `transfer_accuracy` | linear probe trained on the **frozen** representation at the end of the run, on a dataset never seen during continual training. Fixed probe protocol: same optimizer, same steps, same split for every method compared |
| `unseen_task_accuracy` | accuracy on a held-out task `T_{k+1}` that was never trained on, task id not given |
| `unseen_domain_accuracy` | as above for a domain never seen in training |
| `corruption_accuracy` | accuracy on the final model under a fixed corruption set (S9) |
| `confounder_split_accuracy` | accuracy on the split where the train-time spurious correlation is broken (S9) |

### 4.3 cost (all levels)

| field | definition |
| :-- | :-- |
| `memory_bytes` | stored *data* memory (buffer, exemplars, prototypes) |
| `state_bytes` | stored non-model state (Fisher, distillation snapshots) |
| `stored_bytes` | `memory_bytes + state_bytes` |
| `total_params` | full model size |
| `trainable_params` | parameters still receiving gradients at the end |
| `active_params` | per-sample forward cost |
| `flops_forward` | FLOPs per sample at inference |
| `latency_ms` | measured inference latency per sample (median over N) |
| `fit_seconds` | wall clock of the training phase |
| `optimizer_steps` | total optimizer steps (the compute-parity field) |

`flops_forward` and `latency_ms` are separate on purpose: FLOPs are the fair
cross-architecture comparison, latency is the deployment number, and the two
can disagree by an order of magnitude on a laptop GPU.

### 4.4 modular (L1)

| field | definition |
| :-- | :-- |
| `expert_count` | final number of experts |
| `experts_per_task` | list, one entry per task |
| `reuse_rate` | fraction of tasks that reused an existing expert |
| `routing_entropy` | mean entropy of the routing distribution |
| `utilization` | utilization entropy (repo definition) |
| `specialization_mi` | task-expert mutual information (repo definition) |
| `task_recall_at_k` | `{k: recall}` of the router's candidate set |
| `oracle_accuracy` | accuracy with the true task id (task-IL upper bound) |

`oracle_accuracy` is the single most informative number when routing is in
question: it separates "the experts are bad" from "the selection is bad". E0
showed the two differ by 27 points on CIFAR-100/ViT.

### 4.5 stability (L2)

| field | definition |
| :-- | :-- |
| `representation_drift` | mean cosine distance between the representation of a fixed probe set at consecutive task boundaries |
| `routing_drift` | total-variation distance between routing distributions on a fixed probe set at consecutive task boundaries |
| `plasticity` | accuracy gained on the newest task relative to the previous task's model |
| `stability` | negative of the accuracy lost on old tasks at a boundary |

### 4.6 provenance (all levels)

`git_commit`, `timestamp`, `duration_sec`, `torch`, `python`, `device`,
`command` (the exact argv). The v1 runner already records all of these except
`command`.

---

## 5. Study record

```json
{
  "schema_version": "1.0",
  "study": "backbone_generalization",
  "protocol": "class_il",
  "task_id_at_inference": false,
  "min_conformance": "L1",
  "runs": ["run_id", "..."],
  "cells": {"seed": [42, 1, 2], "backbone": ["conv", "resnet18", "vit_b_16"]},
  "aggregates": {
    "accuracy": {"mean": 0.0, "std": 0.0, "ci95": [0.0, 0.0], "n": 3}
  },
  "paired": {
    "baseline": "ncm",
    "mean_delta": 0.0,
    "ci95": [0.0, 0.0],
    "wins": 0,
    "losses": 0,
    "unpaired": false
  }
}
```

Rules: `ci95` is a bootstrap interval over the runs in the cell (not a normal
approximation; n is small). `paired` requires that every run in the comparison
has a partner with the same `(seed, class_order)`. If not, `unpaired: true` and
the difference is descriptive only (R7).

---

## 6. Axis to stage map

Every axis from the S0 brief, with the stage that delivers it and its current
status in the repo. "factor" means a new experiment grid is needed; "metric"
means new plumbing.

| axis | kind | stage | status today |
| :-- | :-- | :-- | :-- |
| learning: accuracy, forgetting, bwt | metric | - | available |
| learning: fwt | metric | S6 | new (definition pinned in 4.1) |
| learning: acc curve `A(t)` | metric | S2 | derivable from `acc_matrix` |
| backbone generalization | factor | S3 | runner supports 8 archs; needs the grid |
| data generalization | factor | S4 | available (`--dataset`, `--dataset folder`) |
| task-order generalization | factor | S6 | **runner support missing** |
| class-order generalization | factor | S6 | **runner support missing** |
| domain generalization (seen / unseen) | factor | S6 | `--domain_shift` exists; unseen-domain split new |
| unseen-class / future-class | metric | S6 | new |
| representation transfer | metric | S7 | **new** (checkpoint -> frozen probe) |
| linear probe ladder | factor | S2 | partial: NCM only as `--eval_head`; ridge/logreg new |
| data-regime `{10,25,50,100}%` | factor | S8 | new (`--data_fraction`) |
| memory budget `{0..inf}` | factor | S8 | partial (`--proto_size`, `--buffer_size`, `--icarl_k`) |
| parameter budget `{0..1M}` | factor | S8 | partial (a param-matched recipe exists) |
| compute budget | factor + metric | S8 | `fit_seconds` exists; `flops`, `steps` new |
| robustness / corruption | factor | S9 | new |
| spurious correlation | factor | S9 | new (ConCon-style) |
| stability / plasticity per transition | metric | S2 | cheap, from `acc_matrix` + probe snapshots |
| representation / routing drift | metric | S2 | machinery exists (E0), needs wiring |
| expert specialization | metric | S2 | partial (MI, utilization) |
| expert transfer `P(E_i \| T_new)` | metric | S6 | new |
| scalability `T = 2..100` | factor | S10 | needs long streams |
| statistical robustness | method | S11 | multi-seed exists for some cells; task-order not |

---

## 7. Success criterion for Stage 1

Not "PAL-MoE beats iCaRL". The deliverable is a map of the form:

| regime | which capacity is actually needed |
| :-- | :-- |
| frozen pretrained representation | `NCM ~= MoE` (E0 measured this) |
| random representation | `?` (S3) |
| domain shift | `?` (S6) |
| task-free stream | `?` (S6) |
| tight memory | `?` (S8) |
| high shift | `?` (S9) |

with each cell supported by a paired comparison at a declared conformance
level, and with the negative cells reported as prominently as the positive
ones. A Stage 1 in which MoE loses everywhere and that is *measured
systematically* is a successful Stage 1.

---

## 8. Adoption plan

1. `pal_moe/evaluation/schema.py` (this commit) - the contract in code:
   builders, validators, and a v0 upgrade path for the existing result files.
2. S1 (interface refactor) wires `build_run_record` into the runner as an
   **additive** field in the result JSON: the existing keys stay byte-identical
   so the published tables do not move.
3. S2 onward, each stage populates the fields it owns and raises the declared
   block set.
4. The analysis scripts read only the contract, never the raw result keys.

### 8.1 Ingest check (measured, 2026-09-23)

Every `results/**/benchmark_results_*.json` in the repo was upgraded through
`load_run_records` and validated:

| blocks declared | records |
| :-- | --: |
| `cost + learning` | 248 |
| `cost + learning + modular` | 150 |
| `learning` only (pre-byte-accounting) | 81 |
| `learning + modular` | 45 |
| **total conformant** | **524** |
| rejected | 0 |

The 81 `learning`-only records are the older ablation sweeps that predate the
byte accounting. Under a level scheme they were unrepresentable; under blocks
they are simply excluded from Pareto studies and remain usable everywhere else.
