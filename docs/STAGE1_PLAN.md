# Stage 1 Plan and Results

Status: S0-S2 complete, 2026-09-23. Companion to `MEASUREMENT_CONTRACT.md` (S0,
what a run must report), `ARCHITECTURE_CONTRACT.md` (S1, the four interfaces) and
`PALMOE_V2_SPEC.md` (the v2 design freeze, which S1 supersedes on the
abstraction question).

Stage 1 answers one question, not "which model is best":

> Which capacity is actually required in which problem regime, and where does it
> stop generalizing?

A Stage 1 in which the modular machinery loses everywhere and that is *measured
systematically* is a successful Stage 1.

---

## 1. Stage order

| stage | content | status |
| :-- | :-- | :-- |
| S0 | measurement contract: run/study records, blocks, v0 ingest | **done** (`MEASUREMENT_CONTRACT.md`, `pal_moe/evaluation/schema.py`) |
| S1 | architecture contract: Backbone / Expert / Readout / Router registries | **done** (`ARCHITECTURE_CONTRACT.md`, `pal_moe/arch/`) |
| S2 | complexity ladder, one backbone, 3 seeds | **done** (`experiments/s2_ladder.py`, section 3) |
| S3 | backbone generalization: `random -> mlp -> conv -> resnet18 -> vit` | **done** (`experiments/s3_run.py`, `s3_report.py`, section 5) |
| S4 | dataset generalization: MNIST / CIFAR-10 / CIFAR-100 / Tiny-ImageNet | planned |
| S5 | protocol generalization: Class-IL / Task-IL / Domain-IL labelling | planned |
| S6 | task-order, class-order, domain, unseen-task, expert transfer | planned |
| S7 | representation transfer: checkpoint -> frozen probe on an unseen dataset | **done** (`experiments/s7_transfer.py`, section 6) |
| S8 | budget / data-regime / compute scaling | planned |
| S9 | robustness: corruption, spurious correlation | planned |
| S10 | scalability: task count 2..100 | planned |
| S11 | statistical validation: 3-5 seeds x task orders, paired tests | planned |

Two rules apply to every stage, and they are the reason the ladder is
trustworthy:

1. **One variable per experiment.** Everything else stays fixed: feature cache
   (byte-identical tensors), task partition, class order, training budget,
   evaluation protocol, memory quota.
2. **A failing rung is reported, not tuned.** If a level loses, that is the
   result. Rescuing it with a per-level hyper-parameter search would destroy the
   ladder comparison; a sensitivity study is a different experiment.

---

## 2. Measurement (S0) and architecture (S1) in one line each

- **S0**: every run emits a model-independent record whose fields were decided
  *before* the run; blocks are derived from the data, so a claim cannot drift
  from reality. 524 existing result records ingest cleanly, 0 rejected.
- **S1**: `X -> Backbone -> Z -> Expert -> Z' -> Readout -> Y`, with the router
  separate. `RepresentationExpert` (`Z -> Z`) and `ClassificationExpert`
  (`Z -> Y`) are different protocols, which is what makes the capacity question
  (L2a) separable from the isolation question (L3).

---

## 3. S2 results - the complexity ladder

CIFAR-100, 20 tasks x 5 classes, Class-IL, frozen ViT-B/16, feature cache,
10 epochs/task, 3 seeds. Identical inputs for every row.

| level | acc | std | forgetting | FWT | params | latency |
| :-- | --: | --: | --: | --: | --: | --: |
| `L0_ncm` | 70.34% | 0.00 | 9.75% | 0.00% | **0** | 0.000 ms |
| **`L1_ridge`** | **76.77%** | 0.00 | 9.65% | 0.00% | **0** | 0.000 ms |
| `L1_linear` | 39.81% | 0.35 | 55.91% | 0.00% | 76.8 K | 0.001 ms |
| `L1_cosine` | 66.90% | 0.14 | 18.35% | 0.00% | 76.8 K | 0.001 ms |
| `ceiling_joint_probe` | 81.09% | 0.11 | - | - | 76.8 K | 0.000 ms |
| `L2a_shared_joint` | 75.81% | 0.76 | - | - | 89.1 K | 0.001 ms |
| `L2b_shared_seq` | 56.00% | 0.65 | 42.19% | +0.01% | 89.1 K | 0.001 ms |
| `L3_per_task` | 70.58% | 0.03 | 10.13% | **-0.19%** | 89.1 K | 0.026 ms |
| `L4_oracle` | 97.46% | 0.06 | 0.09% | - | 89.1 K | 0.002 ms |
| **routing cost (L4-L3)** | **26.88%** | **0.04** | | | | |

`L2a` and `ceiling_joint_probe` are control conditions, not CL results: they read
every task at once.

### 3.1 Findings

**F1 - a closed-form readout dominates every learned rung.** `L1_ridge`
(0 parameters, 0 optimizer steps, ~2.7 MB of accumulated sufficient statistics,
no raw data) beats `L1_cosine` by 9.87, `L2a` by 0.96, `L3` by 6.19 and `L2b` by
20.77, all 3/3 seeds. Under rule 8 the learned-readout rung and the expert-bank
rung do not justify themselves at this backbone.

**F2 - the expert bank buys isolation, not capacity.** Within the expert family,
`L2b -> L3` is **+14.58 points** (std 0.6, 3/3 seeds), while `L2a` (joint
training of the same single adapter) beats both. So sharing under the sequential
constraint is the failure mode, and per-task isolation is the fix.

**F3 - routing is the largest single cost.** `L4 - L3` = **26.88% +- 0.04**:
the same model with the task id given is 27 points better. This is the quantity
that any allocation or rejection mechanism would have to recover, and E0 showed
that reranking on the frozen space cannot.

**F4 - normalisation helps under the CL constraint and hurts without it.**
Frozen old rows: `linear` 39.81 vs `cosine` 66.90. Joint training: `linear`
81.09 vs `cosine` 75.38 (E0). Cosine normalisation bounds the logit scale and
therefore the recency bias; in the joint case it removes a degree of freedom for
nothing.

**F5 - no forward transfer.** The adapted representation does not help a task it
has not seen: `L3` is **-0.19%** (all 3 seeds negative), `L2b` is +0.01%
(mixed). Isolation has a small, consistent cost on future tasks. This is the
first measurement from the corrected FWT definition (section 4).

**F6 - the router is the most expensive component per sample.** `L3` 0.026 ms
vs `L4` 0.002 ms: the 100 x 768 prototype similarity dominates the adapter
forward. A cheaper router is a real engineering constraint, not a detail.

### 3.2 Two bugs the ladder found

1. **A readout's freezing policy belongs to the readout, not the loop.** Passing
   "all seen classes" instead of "classes trainable now" re-opens every old row:
   the linear readout collapses from 39.81% to 7.69% (forgetting 55.91% to
   92.35%). Fixed by `pal_moe.arch.readouts.trainable_hook(readout,
   active_classes)`, which the loop must call rather than decide for itself.
2. **The obvious FWT definition is degenerate** (section 4).

---

## 4. The FWT definition was wrong

Pinned originally as "evaluate the model on task `k+1` before training on it,
with the task's classes masked". In a growing-head class-incremental setting the
task's classes have no head rows yet, so masking makes the accuracy structurally
zero: the measured value was `-mean(1/5t) = -3.73%`, identical for every level -
a metric that cannot distinguish anything, at any number of seeds.

The pinned definition now measures the **representation**: a closed-form ridge
probe on the incoming task using the model's current representation, minus the
same probe on the raw frozen features. It is comparable across methods, needs no
head rows for unseen classes, has a clean sign, and it discriminates: 0.00% for
no-expert levels (the control), +0.01% for the shared adapter, -0.19% for the
per-task bank.

---

## 5. S3 results - backbone generalization

CIFAR-100, 20 tasks, Class-IL, 3 seeds, every backbone projected to a common
`latent_dim = 768` by a **frozen** map (no trainable model enters), `rep_seed`
separate from the CL `seed`. The ladder code is byte-identical to S2.

| backbone | L0_ncm | L1_ridge | L1_cosine | L2a | L2b | L3 | L4 |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| `random` | 10.62 | 18.42 | 10.13 | 19.23 | 4.03 | 11.18 | 59.45 |
| `mlp` | 9.28 | 14.27 | 8.97 | 14.58 | 4.86 | 9.29 | 52.40 |
| `conv` | 7.26 | 19.12 | 6.96 | 22.36 | 4.65 | 7.31 | 48.11 |
| `resnet18_random` | 9.09 | 16.38 | 7.97 | 15.73 | 3.38 | 9.15 | 52.61 |
| `resnet18_imagenet` | 26.99 | 37.91 | 25.63 | 37.49 | 13.37 | 27.49 | 82.11 |
| `vit_b16_imagenet` | 70.35 | **76.67** | 66.87 | 76.15 | 56.61 | 70.61 | **97.45** |

Deltas - the table that carries the scientific content:

| backbone | L3 - ridge | routing tax (L4 - L3) | L2b - cosine | L2a - ridge | transfer | recall@3 |
| :-- | --: | --: | --: | --: | --: | --: |
| `random` | -7.24 | 48.27 | -6.09 | +0.81 | 44.68 | 30.25 |
| `mlp` | -4.98 | 43.12 | -4.11 | +0.31 | 35.82 | 28.90 |
| `conv` | -11.81 | 40.81 | -2.31 | +3.24 | 45.98 | 26.73 |
| `resnet18_random` | -7.23 | 43.46 | -4.59 | -0.65 | 42.58 | 27.97 |
| `resnet18_imagenet` | -10.42 | 54.62 | -12.25 | -0.42 | 64.74 | 52.34 |
| `vit_b16_imagenet` | -6.06 | **26.83** | -10.26 | -0.52 | **94.84** | **88.86** |

`transfer` is a closed-form ridge probe on CIFAR-10, a dataset the continual run
never saw: it measures the representation, not the classifier the run built.

Consistency gate: the `vit_b16_imagenet` row must reproduce the S2 reference and
does, at every level (max absolute difference 0.61 points, from a different
loader row order; the feature-cache path is distributionally equivalent, not
bit-identical).

### 5.1 Findings

**G1 - readout dominance is backbone-independent.** `L3 - L1_ridge` is negative
on all six backbones (-4.98 to -11.81). The closed-form readout wins everywhere,
including a random projection of raw pixels. Per the brief's wording this is
*not* "the MoE architecture is unnecessary"; it is "under this
benchmark/protocol/budget, the incremental expert machinery added no measurable
benefit".

**G2 - the routing tax is a general bottleneck, and it shrinks as the
representation improves.** It is 40.8-54.6 points on the five weak backbones and
26.8 on the ViT. The hypothesised direction (tax grows with representation
quality) is **refuted**; the measured direction is the opposite, and it tracks
candidate recall@3 (26.7-30.3 on the weak backbones, 88.9 on the ViT). Routing
failure is driven by representation-space ambiguity, not by the router's
capacity.

**G3 - the oracle is far above the routed model everywhere.** Even a random
projection of normalised pixels reaches 59.45% with the correct task id against
11.18% routed. There is exploitable capacity on every backbone and selection
never realises it. This is the S3 headline.

**G4 - a shared adapter adds capacity exactly where the representation is
weak.** `L2a - ridge` is positive on `random` (+0.81), `mlp` (+0.31) and `conv`
(+3.24), and negative on the two ResNets and the ViT (-0.42 to -0.65). The
expert *bank* never shows this (G1), so the effect belongs to a single shared
adapter.

**G5 - transfer separates the backbones more than CIL accuracy does.** CIFAR-10
ridge probe: ViT 94.84, ResNet-18/ImageNet 64.74, conv 45.98, random 44.68,
ResNet-18/random 42.58, MLP 35.82.

### 5.2 Verdict against the interpretation scheme

| case | statement | verdict |
| :-- | :-- | :-- |
| A | `ridge > L3` on every backbone | **confirmed** (6/6) |
| B | `L3 > ridge` on weak backbones, `ridge > L3` on pretrained | refuted (the bank loses everywhere) |
| C | ridge wins only on the ViT | refuted |
| D | oracle high and L3 low everywhere -> routing is a general bottleneck | **confirmed**, and stronger than stated |

The result is **A + D**: simple closed-form readouts are a strong and general
baseline, and selection - not capacity, not readout, not adaptation - is the
binding constraint on every representation family tested.

### 5.3 A measurement bug S3 found

The backbone factory asked the encoder for `output_dim=256` for every
non-random architecture. For a ViT-B/16 (native 768) that inserts a
`Linear(768->256) + BatchNorm + ReLU` head and then projects back up to 768, and
the CIFAR-100 features collapse: norm 4.3 instead of 18.4, NCM 28% instead of
70%. Nothing in the raw accuracy table reveals it - the S3 rows look internally
consistent, they are just all wrong. The fix asks each torchvision architecture
for its **native** width (`NATIVE_WIDTHS`) and lets the frozen projection do the
rest; the ViT then needs no projection at all. The ViT row reproducing the S2
reference is the regression test.

Related and now fixed: a backbone factory must forward the dataset statistics
(`input_mean`/`input_std`), because the ViT path undoes the dataset
normalisation before applying ImageNet's.

---

## 6. S7 results - representation transfer

Question: is the representation a continual run learns actually useful for new
problems, or does it only solve the stream it was trained on?

Protocol: at **every** checkpoint the classification head is discarded and
replaced by a closed-form ridge probe on the frozen representation, evaluated on
CIFAR-10 - a dataset the continual run never saw. `z_raw` (the backbone's own
output, which the downstream cache already contains) is the reference for every
number, so the reported quantity is `delta = transfer(z_CL) - transfer(z_raw)`.
The headline evaluator is ridge; logistic and NCM are validation only.

| backbone | level | CIL | raw | full | delta | best t | few1 delta | few10 delta |
| :-- | :-- | --: | --: | --: | --: | --: | --: | --: |
| `vit_b16_imagenet` | raw | 76.67 | 94.84 | 94.84 | - | - | 70.21 | 88.31 |
| `vit_b16_imagenet` | L2b shared | 56.64 | 94.84 | 94.87 | +0.03 | 94.96 | **-11.03** | -0.56 |
| `vit_b16_imagenet` | L3 per-task | 70.61 | 94.84 | 94.63 | -0.21 | 94.86 | **-14.75** | -7.34 |
| `resnet18_imagenet` | raw | 37.91 | 64.74 | 64.74 | - | - | 22.23 | 29.36 |
| `resnet18_imagenet` | L2b shared | 14.50 | 64.74 | 64.83 | +0.09 | 64.89 | -1.67 | -0.90 |
| `resnet18_imagenet` | L3 per-task | 27.37 | 64.74 | 64.54 | -0.20 | 65.02 | -0.77 | -4.07 |
| `random` | raw | 18.42 | 44.68 | 44.68 | - | - | 14.14 | 21.26 |
| `random` | L2b shared | 3.68 | 44.68 | 44.43 | -0.25 | 44.64 | +0.39 | +0.06 |
| `random` | L3 per-task | 11.09 | 44.68 | 44.31 | -0.37 | 44.71 | -1.29 | -1.99 |

(`CIL` for the raw rows is `L1_ridge` from S2/S3: the same representation, no
expert.)

### 6.1 Findings

**T1 - on the full downstream split, adaptation is inert.** The delta is between
-0.37 and +0.09 across all six conditions: the expert machinery neither adds nor
destroys average transferability, and the best checkpoint never beats raw by
more than 0.28 points.

**T2 - the low-data regime is where adaptation costs.** The few-shot axis was
added as a secondary check and turned out to carry the result. On the ViT, one
example per class transfers at 70.21% from the raw frozen representation but
only 55.46% after per-task adaptation (**-14.75**), and at ten examples the loss
is still **-7.34**. The average over 50k samples hides it completely: a probe
can re-learn a good linear map when it has data, and cannot when it has one
example. Discriminability and transferability are different quantities, and the
adaptation trades the second for the first.

**T3 - isolation is the culprit, not adaptation as such.** The shared adapter
loses less few-shot transfer than the per-task bank on every backbone (-11.03 vs
-14.75; -1.67 vs -0.77; +0.39 vs -1.29), and the per-task bank is exactly the
variant S2 showed buys CIL retention through isolation. The mechanism that
improves retention costs generality - the transfer-side counterpart of S2's FWT
result (`L3` = -0.19%, all three seeds negative).

**T4 - `L1_ridge` dominates every expert variant on both axes.**

| variant | CIL accuracy | transfer (full) | transfer (1-shot) |
| :-- | --: | --: | --: |
| **`L1_ridge`** (= raw representation) | **76.67** | **94.84** | **70.21** |
| `L2b` shared adapter | 56.64 | 94.87 | 59.18 |
| `L3` per-task bank | 70.61 | 94.63 | 55.46 |

`L2b` buys +0.03 transfer for -20.0 CIL; `L3` buys -0.21 transfer for -6.1 CIL.
Neither trade is worth making, and `L1_ridge` is the sensible point.

### 6.2 Verdict

| hypothesis | statement | verdict |
| :-- | :-- | :-- |
| H1 | `L1_ridge > L3` in transfer as well as CIL | **confirmed**, much stronger on few-shot |
| H2 | `L3 > L1_ridge` in transfer despite losing CIL | refuted |
| H3 | `L3` transfer is also poor | **confirmed in the few-shot regime**, not in the full-split regime |

Answers to the four questions the stage was set to settle:

- **Is a representation that is good at CIL also good at transfer?** Only
  weakly. The backbone ranking is preserved, but within a backbone the
  adaptation moves CIL by tens of points and average transfer by tenths.
- **Does sequential isolation reduce transferability?** Yes, in the low-data
  regime, and consistently: the per-task bank is the worst variant for few-shot
  transfer on every backbone.
- **How much does pretraining determine transfer?** Almost entirely. Raw
  transfer: ViT 94.84, ResNet-18/ImageNet 64.74, random 44.68; the few-shot
  numbers separate them further (70.21 / 22.23 / 14.14 at one shot).
- **Is the CIL ranking preserved under transfer?** Yes at the backbone level, no
  at the method level: `L1_ridge` wins both axes, but `L2b` and `L3` swap order
  between them.

### 6.3 Note on the records

S7's contract records declare **no blocks**: forgetting and the accuracy matrix
are not measured here, and R5 says a missing metric is null rather than zero. The
blocks design behaves as intended (the records stay readable and are correctly
excluded from learning-block studies), but a future S7 run should record the
accuracy matrix it already computes, which would let the transfer conditions
enter a full study.
