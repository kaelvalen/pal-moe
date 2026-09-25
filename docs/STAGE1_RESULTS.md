# Stage 1 Results

Status: S0-S4 and S7 complete, 2026-09-25. This is the single place to read what
was measured and what it means. Companions:

- [`MEASUREMENT_CONTRACT.md`](MEASUREMENT_CONTRACT.md) - what a run must report (S0)
- [`ARCHITECTURE_CONTRACT.md`](ARCHITECTURE_CONTRACT.md) - the four interfaces and registries (S1)
- [`STAGE1_PLAN.md`](STAGE1_PLAN.md) - the stage order, the rules, and the per-stage write-ups
- [`PALMOE_V2_SPEC.md`](PALMOE_V2_SPEC.md) - the pre-measurement v2 design freeze, kept for the record
- [`BENCHMARK.md`](BENCHMARK.md) - the v1 protocol, design facts and tables

Reproduce everything with one command (skips finished stages, logs per stage):

```bash
.venv/bin/python experiments/run_all.py --list      # what is done, what is left
.venv/bin/python experiments/run_all.py --dry-run   # the exact commands
nohup .venv/bin/python -u experiments/run_all.py > /tmp/run_all.log 2>&1 &
```

---

## 1. Why the programme was reset

The v1 result that triggered everything: on CIFAR-100 with a frozen ViT-B/16,
v1 uses **1.3x the stored bytes and 20x the total parameters of iCaRL for 5.6
points less accuracy**, and it was already known that its router is near-exact
(design fact 11: the remaining gap to the per-expert oracle is expert and
representation quality, not routing). So the question was no longer "how do we
forget less" but:

> Which capacity is actually required in which regime, and where does it stop
> generalizing?

Two rules govern every experiment below. One variable changes at a time
(identical feature cache, task partition, class order, budget, protocol); and a
failing rung is **reported, not tuned** - rescuing a level with its own
hyper-parameter search would destroy the comparison.

---

## 2. E0 - the ceiling, before any new architecture

CIFAR-100, 20 tasks, frozen ViT-B/16, 10 epochs, seed 42.

| configuration | accuracy | forgetting | params | stored |
| :-- | --: | --: | --: | --: |
| v1 PAL-MoE (reproduced at HEAD) | 59.30 | 18.64 | 13.4 M | 14.23 MB |
| iCaRL (reproduced at HEAD) | 64.94 | 12.25 | 0.67 M | 10.66 MB |
| **NCM, training-free** | **70.34** | **9.75** | **0** | **307 KB** |
| joint probe (not CL) | 75.38 | - | 76.8 K | 307 KB |
| trained cosine head | 66.75 | 18.51 | 76.8 K | 347 KB |
| trained head + function anchor | 66.45 | 21.22 | 76.8 K | 347 KB |
| adapter r8 + head | 70.56 | 10.16 | 322 K | 347 KB |
| adapter r32 / r64 + head | 70.62 / 70.60 | 10.14 / 10.17 | 1.06 M / 2.04 M | 347 KB |
| adapter + NCM readout | 70.39 | 10.67 | 246 K | 347 KB |
| **adapter + oracle routing** | **97.64** | - | 322 K | 347 KB |

Router candidate recall: @1 70.98, @2 83.23, **@3 88.61**, @5 93.50.

Findings: a training-free prototype readout beats the entire published v1
system and iCaRL; adapter capacity saturates at rank 8 and adds ~0.2 points
over NCM; and giving the same model the task id is worth **27 points**, so the
binding constraint is selection, not capacity (F1-F6 in `STAGE1_PLAN.md` 3).

---

## 3. S2 - the complexity ladder

CIFAR-100, frozen ViT-B/16, 3 seeds, identical inputs for every row.

| level | accuracy | std | forgetting | FWT | params |
| :-- | --: | --: | --: | --: | --: |
| `L0_ncm` | 70.34 | 0.00 | 9.75 | 0.00 | 0 |
| **`L1_ridge`** | **76.77** | 0.00 | 9.65 | 0.00 | **0** |
| `L1_linear` | 39.81 | 0.35 | 55.91 | 0.00 | 76.8 K |
| `L1_cosine` | 66.90 | 0.14 | 18.35 | 0.00 | 76.8 K |
| `ceiling_joint_probe` | 81.09 | 0.11 | - | - | 76.8 K |
| `L2a_shared_joint` | 75.81 | 0.76 | - | - | 89.1 K |
| `L2b_shared_seq` | 56.00 | 0.65 | 42.19 | +0.01 | 89.1 K |
| `L3_per_task` | 70.58 | 0.03 | 10.13 | **-0.19** | 89.1 K |
| `L4_oracle` | 97.46 | 0.06 | 0.09 | - | 89.1 K |
| routing cost (L4-L3) | **26.88** | **0.04** | | | |

A closed-form ridge readout with **zero parameters and zero optimizer steps**
beats every learned rung. Within the expert family the shared-to-per-task step
is **+14.58** (isolation), and the oracle is 27 points above the routed model.
Forward transfer is zero or slightly negative: isolation has a small consistent
cost on the next task.

---

## 4. S3 - backbone generalization

Six backbones, all projected to a common `latent_dim = 768` by a frozen
seeded map (no trainable model enters), `rep_seed` separate from the CL seed.

| backbone | L0_ncm | L1_ridge | L1_cos | L2a | L2b | L3 | L4 |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| `random` | 10.62 | 18.42 | 10.13 | 19.23 | 4.03 | 11.18 | 59.45 |
| `mlp` | 9.28 | 14.27 | 8.97 | 14.58 | 4.86 | 9.29 | 52.40 |
| `conv` | 7.26 | 19.12 | 6.96 | 22.36 | 4.65 | 7.31 | 48.11 |
| `resnet18_random` | 9.09 | 16.38 | 7.97 | 15.73 | 3.38 | 9.15 | 52.61 |
| `resnet18_imagenet` | 26.99 | 37.91 | 25.63 | 37.49 | 13.37 | 27.49 | 82.11 |
| `vit_b16_imagenet` | 70.35 | **76.67** | 66.87 | 76.15 | 56.61 | 70.61 | **97.45** |

| backbone | L3 - ridge | routing tax | L2b - cosine | L2a - ridge | transfer | recall@3 |
| :-- | --: | --: | --: | --: | --: | --: |
| `random` | -7.24 | 48.27 | -6.09 | +0.81 | 44.68 | 30.25 |
| `mlp` | -4.98 | 43.12 | -4.11 | +0.31 | 35.82 | 28.90 |
| `conv` | -11.81 | 40.81 | -2.31 | +3.24 | 45.98 | 26.73 |
| `resnet18_random` | -7.23 | 43.46 | -4.59 | -0.65 | 42.58 | 27.97 |
| `resnet18_imagenet` | -10.42 | 54.62 | -12.25 | -0.42 | 64.74 | 52.34 |
| `vit_b16_imagenet` | -6.06 | **26.83** | -10.26 | -0.52 | **94.84** | **88.86** |

Readout dominance holds on all six. The routing tax is a general bottleneck and
it **shrinks** as the representation improves, tracking candidate recall rather
than the router's capacity. Even a random projection of normalised pixels
reaches 59.45% with the task id against 11.18% routed. A shared adapter adds
capacity exactly where the representation is weak (`L2a - ridge` positive on
random/mlp/conv) and nothing on the strong ones.

The `vit_b16_imagenet` row reproduces the S2 reference at every level (max
difference 0.61 points, from a different loader row order).

---

## 5. S7 - representation transfer

Every checkpoint, the classifier is discarded and replaced by a closed-form
ridge probe on CIFAR-10, a dataset the continual run never saw. `z_raw` is the
reference, so the reported quantity is `transfer(z_CL) - transfer(z_raw)`.

| backbone | level | CIL | raw | full | delta | best t | 1-shot delta | 10-shot delta |
| :-- | :-- | --: | --: | --: | --: | --: | --: | --: |
| `vit_b16_imagenet` | raw | 76.67 | 94.84 | 94.84 | - | - | 70.21 | 88.31 |
| `vit_b16_imagenet` | L2b | 56.64 | 94.84 | 94.87 | +0.03 | 94.96 | **-11.03** | -0.56 |
| `vit_b16_imagenet` | L3 | 70.61 | 94.84 | 94.63 | -0.21 | 94.86 | **-14.75** | -7.34 |
| `resnet18_imagenet` | L2b | 14.50 | 64.74 | 64.83 | +0.09 | 64.89 | -1.67 | -0.90 |
| `resnet18_imagenet` | L3 | 27.37 | 64.74 | 64.54 | -0.20 | 65.02 | -0.77 | -4.07 |
| `random` | L2b | 3.68 | 44.68 | 44.43 | -0.25 | 44.64 | +0.39 | +0.06 |
| `random` | L3 | 11.09 | 44.68 | 44.31 | -0.37 | 44.71 | -1.29 | -1.99 |

Adaptation is **inert on the full split** (-0.37 to +0.09, best checkpoint
+0.28) and **costly in the low-data regime**: 14.75 points at one example per
class on the ViT, 7.34 at ten. The per-task bank loses more few-shot transfer
than the shared adapter on every backbone, so isolation is the culprit, not
adaptation as such. `L1_ridge` (which is the raw representation) dominates both
expert variants on both axes.

---

## 6. S4 - dataset generalization

Frozen ViT-B/16, Class-IL, 3 seeds, only the dataset changes; transfer is
always CIFAR-10, so **that row is in-domain and the other three are
cross-dataset**.

| dataset | T | NCM | Ridge | Shared-J | Shared-S | PerTask | Oracle | routing tax | isolation | 1-shot delta |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| MNIST | 5 | 84.99 | **97.76** | 96.49 | 83.86 | 85.56 | 99.70 | 14.14 | +1.69 | -3.73 |
| CIFAR-10 | 5 | 91.17 | **94.81** | 90.41 | 83.76 | 91.61 | 99.14 | 7.53 | +7.85 | **+17.23** |
| CIFAR-100 | 20 | 70.34 | **76.77** | 75.81 | 56.00 | 70.58 | 97.46 | 26.88 | +14.58 | -10.19 |
| Tiny-ImageNet | 20 | 79.95 | **84.03** | 82.57 | 64.93 | 80.38 | 94.10 | 13.72 | +15.45 | -5.46 |

| dataset | L3 - ridge | isolation (L3 - L2b) | routing tax | L2a - ridge |
| :-- | --: | --: | --: | --: |
| MNIST | -12.21 | +1.69 | +14.14 | -1.27 |
| CIFAR-10 | -3.20 | +7.85 | +7.53 | -4.40 |
| CIFAR-100 | -6.19 | +14.58 | +26.88 | -0.96 |
| Tiny-ImageNet | -3.65 | +15.45 | +13.72 | -1.46 |

Four results:

1. **Readout dominance holds on 4/4**, largest on the easiest dataset.
2. **Isolation benefit grows with the stream length**: +1.69/+7.85 at 5 tasks,
   +14.58/+15.45 at 20.
3. **The routing tax tracks difficulty, not length**: Tiny-ImageNet's 20 tasks
   cost 13.72 against CIFAR-100's 20 tasks at 26.88.
4. **The few-shot delta flips sign with the domain**: +17.23 in-domain,
   -3.73 to -10.19 cross-dataset.

And the sharpest statement of what the expert bank is for: `L2a - ridge` is
**-0.96 to -4.40**, i.e. one adapter trained on all tasks at once is within a
few points of the closed-form readout everywhere, while the same adapter under
the sequential constraint loses 12-20. The cost lives in the constraint, not in
the capacity.

---

## 7. S5 - the protocol axis

`oracle routing` and `Task-IL` are not the same thing: the first hands the model
the task id *for routing* while it still chooses among every class it has seen,
the second additionally restricts the class search space to the task's own
classes. S5 measures the 2x2 factorial, so the gain decomposes into two
mechanisms instead of one number:

    class_masking=False, routing=learned   Class-IL (the reference)
    class_masking=False, routing=oracle    task id for routing only
    class_masking=True,  routing=learned   class space restricted only
    class_masking=True,  routing=oracle    Task-IL (the standard protocol)

Training is protocol-independent, so each (level, seed) trains once and is
evaluated four times. Frozen ViT-B/16, 3 seeds.

**CIFAR-100**

| level | Class-IL | +oracle | mask | Task-IL | routing effect | masking effect | Task-IL - Class-IL |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| `L0_ncm` | 70.34 | 70.34 | 95.73 | 95.73 | 0.00 | **25.39** | 25.39 |
| `L1_ridge` | 76.77 | 76.77 | 96.79 | 96.79 | 0.00 | **20.02** | 20.02 |
| `L2a_shared_joint` | 75.81 | 75.81 | 96.97 | 96.97 | 0.00 | **21.17** | 21.17 |
| `L2b_shared_seq` | 56.00 | 56.00 | 92.85 | 92.85 | 0.00 | **36.85** | 36.85 |
| `L3_per_task` | 70.58 | 97.46 | 95.78 | 97.55 | **26.88** | **25.20** | 26.96 |
| `L4_oracle` | 97.46 | 97.46 | 97.55 | 97.55 | 0.00 | 0.09 | 0.09 |

**CIFAR-10**

| level | Class-IL | +oracle | mask | Task-IL | routing effect | masking effect | Task-IL - Class-IL |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| `L0_ncm` | 91.17 | 91.17 | 98.33 | 98.33 | 0.00 | 7.16 | 7.16 |
| `L1_ridge` | 94.81 | 94.81 | 99.07 | 99.07 | 0.00 | 4.26 | 4.26 |
| `L2a_shared_joint` | 90.41 | 90.41 | 97.79 | 97.79 | 0.00 | 7.38 | 7.38 |
| `L2b_shared_seq` | 83.76 | 83.76 | 97.92 | 97.92 | 0.00 | 14.16 | 14.16 |
| `L3_per_task` | 91.61 | 99.14 | 98.53 | 99.20 | **7.53** | 6.92 | 7.59 |
| `L4_oracle` | 99.14 | 99.14 | 99.20 | 99.20 | 0.00 | 0.06 | 0.06 |

### 7.1 Findings

**P1 - the setup validates itself.** Ten of the twelve rows must show a routing
effect of exactly zero, because L0/L1 have no router, L2a/L2b apply the same
adapter to everything, and L4 *is* the oracle. They do, to the digit. A mistake
in the factorial or in the oracle path would have shown up as a spurious gain.

**P2 - the protocol gap is large and strongly level-dependent.** Task-IL minus
Class-IL is +20.02 for ridge, +21.17 for the joint shared adapter, and
**+36.85 for the sequential shared adapter** on CIFAR-100. L2b's Class-IL
collapse (56.00) is therefore mostly a *class-space* failure, not an adaptation
failure: the same adapter reaches 92.85 under Task-IL. Under Task-IL the
expert bank's advantage over the shared adapter shrinks from +14.58 to +4.70.

**P3 - the two mechanisms carry the same information.** For L3 the routing
effect is 26.88 and the masking effect 25.20, but the combined gain is only
26.96, not ~52. They are almost fully redundant: telling the model which task
it is answering and restricting the classes it may answer *are the same
reminder*. This is the mechanism behind the E0 reranking failures - the
router's confusion and the cross-task class confusion are one quantity, so no
reranker reading the same frozen space has an independent signal to exploit.

**P4 - once the task id is known, the class restriction is redundant.** L4's
masking effect is 0.09 on CIFAR-100 and 0.06 on CIFAR-10. Routing to the right
expert already implies the right class set, because the experts were trained
per task.

**P5 - the router is a class-level decision, not a task-level one.**
`MI(Expert;Class)` is 0.619 against `MI(Expert;Task)` 0.545 on CIFAR-100
(0.796 vs 0.792 on CIFAR-10). The prototype router picks the nearest class mean
and the class-to-expert map is many-to-one, so "task specialization" is
imposed *after* a class decision. That is the structural reason P3 holds.

**P6 - transfer is protocol-independent, as it must be.** Every level reports
the same transfer under both protocols (94.60 vs 94.60 for L3 on CIFAR-100),
because the protocol changes only the evaluation, not the trained
representation. A difference there would have meant a leak between the axes.

### 7.2 What this changes

The Class-IL numbers in sections 3-6 are not a property of the *models*: the
same trained model is 20-37 points better when the protocol supplies task
information. Any claim of the form "method X reaches 76.77 on CIFAR-100" is a
Class-IL claim, and the Task-IL number for the same run is 96.79. The
measurement contract now records `protocol`, `class_masking` and `routing_mode`
as separate factors, and the aggregator refuses to mix them (R3), so the two
cannot be conflated in a table again.

---

## 8. S5b - Domain-IL: is the routing tax a Class-IL artifact?

Domain-IL keeps the label space fixed and moves the input distribution, so
`class_space` stays shared and class masking is a no-op **by construction**.
That removes the explanation S5 pointed at (P2/P3: the class-space restriction
and the routing decision carry the same information) and leaves the question
that decides how to read every earlier section: is isolation useful across
shifted distributions, or was its benefit the class-space problem of Class-IL?

Domains are rotations of MNIST (0/90/180/270 in the stream, 45 held out), each
carrying **all 10 classes**, through the canonical frozen ViT-B/16. Three seeds.

| level | learned | oracle | routing effect | unseen (learned) | unseen (oracle) | reuse | MI(E;Domain) | MI(E;Class) |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: |
| `L0_ncm` | 72.74 | 72.74 | 0.00 | 58.19 | 58.19 | - | - | - |
| `L1_ridge` | **91.58** | 91.58 | 0.00 | 61.88 | 61.88 | - | - | - |
| `L2a_shared_joint` | 89.02 | 89.02 | 0.00 | **65.54** | 65.54 | - | - | - |
| `L2b_shared_seq` | 82.15 | 82.15 | 0.00 | 46.10 | 46.10 | - | - | - |
| `L3_per_task` | 84.20 | 95.89 | **+11.69** | 56.44 | 73.37 | 1.00 | **0.3600** | 0.0024 |
| `L4_oracle` | 95.89 | 95.89 | 0.00 | 73.37 | 73.37 | - | - | - |

The structural check holds again: `L3` under oracle routing equals `L4` exactly
(95.54 / 96.05 / 96.06), so the oracle path and the factorial are sound.

### 8.1 Findings

**D1 - the routing tax is not a class-space artifact, but more than half of the
Class-IL tax was.** Under a genuine distribution shift with a fixed label space
the routing effect is **+11.69**, against **+26.88** in Class-IL. So a real
"identify the distribution" problem exists independently of the class-space
problem, and S5's redundancy (P3) explains the larger Class-IL number rather
than the whole of it.

**D2 - the router tracks whichever axis separates the tasks.** `MI(E;Domain)`
is **0.3600** and `MI(E;Class)` **0.0024**, the exact inverse of Class-IL
(`MI(E;Class)` 0.619 > `MI(E;Task)` 0.545). The class-first structure S5
measured is therefore a property of the *problem*, not a bias of the router:
when classes are shared there is nothing to gain from class geometry and the
prototypes organize by domain instead.

**D3 - isolation's available value is larger in Class-IL, and in both protocols
the learned router realises only a minority of it.** The correct comparison uses
the same routing on both sides, and it splits into *available* (oracle-routed)
and *realised* (learned):

| protocol | isolation, realised (`L3 - L2b`) | isolation, available (`L4 - L2b`) | realised / available |
| :-- | --: | --: | --: |
| Class-IL (CIFAR-100) | +14.58 | **+41.46** | 35% |
| Domain-IL (rotated MNIST) | **+2.05** | **+13.74** | 15% |

An earlier draft of this section compared Domain-IL's *available* +13.74 with
Class-IL's *realised* +14.58 - a category error. The corrected numbers say
something stronger: the bank's available benefit is three times larger under
Class-IL (+41.46) because there each expert owns a disjoint label set and is a
5-way classifier against the shared adapter's 100-way, while under Domain-IL
every expert covers all ten classes and only the domain specialisation can help.
And the learned router captures only a **minority of the available benefit in
both protocols**, 35% and 15% respectively - worst where the label space gives
it no help. Isolation is real; realising it is the bottleneck.

**D4 - readout dominance holds, and the oracle now beats it.** `L1_ridge`
(0 parameters) is 91.58 against L3's 84.20 and L2b's 82.15, exactly the S2-S4
ordering; only the oracle-routed bank (95.89) is above it.

**D5 - for a genuinely unseen distribution, joint training beats sequential by
19.4 points.** Unseen-domain accuracy: `L2a` 65.54 against `L2b` 46.10. The
sequential shared adapter overfits the four rotations it saw, while the joint
one keeps enough generality to face a fifth. `L3` with the learned router
(56.44) is worse than both, and the `L4` row (73.37) is the 0-degree expert
applied to the 45-degree domain - the closest match available, not a fair
oracle, because the held-out cache carries `task_id = 0` by construction.

**D6 - forgetting is 0 by construction and is the wrong metric here.** Every
domain adds training data for the *same* ten classes, so old-domain accuracy
never degrades. Under Domain-IL the informative measurements are the routing
tax and unseen-domain accuracy, and `forgetting` should not be quoted.

**D7 - no domain-to-domain forward transfer** (`L2b` +0.017, `L3` -0.024),
consistent with S2 and S7.

### 8.2 Bugs this stage found

Two, both of which made the whole stage meaningless rather than merely noisy:

1. **Prototype keys collided across domains.** Keying the router by class id
   meant every domain overwrote the previous one's prototypes: all ten classes
   ended up mapped to the last domain's expert. The tell was `reuse = 0.25` and
   `MI = 0.0000` with `L3` at 36.41%. Fixed by keying prototypes by
   `(task, class)`; `L3` then reached 84.20%.
2. **The rotation was applied to the training split only.** `train_x =
   rotated(...)` but `test_x = stack(test)`, so all five domains shared one
   unrotated test set and "domain accuracy" and "unseen-domain accuracy" were
   the same measurement. The tell was `torch.equal(test_0, test_45) == True`
   and `unseen == acc` to the digit. Fixed, with two guards that would have
   caught it immediately: an assertion that a non-zero rotation changes its
   test split, and a pairwise-distinctness check across the domain caches.

---

## 9. S6 - order sensitivity

Both order axes are free: a class order is a re-partition of an existing
feature cache (which classes share a task, features untouched) and a task order
is a permutation of the resulting groups. So the stream's *geometry* can be
varied without any new extraction, which makes order the single variable.

Nine configurations (3 class partitions x 3 visit orders), canonical frozen
ViT-B/16, CIFAR-100, 10 epochs.

| level | accuracy | std across orders | unseen transfer | expert share (unseen) | reading |
| :-- | --: | --: | --: | --: | :-- |
| `L0_ncm` | 70.76 | 0.64 | 95.47 | - | - |
| `L1_ridge` | 77.30 | 0.43 | 95.47 | - | - |
| `L2b_shared_seq` | 57.02 | 1.55 | 95.51 | - | - |
| `L3_per_task` | 71.17 | 0.65 | 95.00 | **0.205** | **spread** |
| `L4_oracle` | 96.58 | 0.43 | 95.47 | - | - |

Per-configuration decomposition (nine rows, abbreviated to the range):

| quantity | range over the nine orders | easy vs hard groups |
| :-- | --: | --: |
| routing tax (`L4 - L3`) | 25.19 - 25.92 | 25.22 vs 25.80 |
| isolation realised (`L3 - L2b`) | 12.36 - 16.88 | 14.14 vs 14.17 |
| isolation available (`L4 - L2b`) | 37.56 - 42.01 | 39.36 vs 39.97 |

### 9.1 Findings

**O1 - the routing tax is order-robust.** It spans 25.19-25.92 over nine
configurations, and the median split into easy and hard orders separates
25.22 from 25.80 - a 0.58-point spread on a 25.5-point effect. Correlations:
`tax` vs `order_confusion` **-0.053**, `isolation_realised` vs difficulty
-0.095, `isolation_available` vs difficulty +0.076. Every earlier stage measured
on one canonical order, and S6 says that was safe: the tax is a property of the
representation and the protocol, not of the stream's arrangement.

**O2 - order is not a meaningful variance source at this scale.** The standard
deviation across the nine configurations is 0.43-1.55 points per level, well
below every effect the earlier stages report. The largest is `L2b` at 1.55,
which is the one level whose accuracy is genuinely order-dependent (it has no
per-task isolation, so it absorbs the class ordering directly).

**O3 - the difficulty axis was not actually sampled, so the difficulty
hypothesis is untested.** The easy/hard split separates difficulty 0.289 from
0.299 - a one-point range - and `order_confusion` only spans 0.653-0.722. A
random permutation of CIFAR-100 classes produces tasks of nearly equal
difficulty, so there was nothing to correlate against. The honest conclusion is
therefore narrow: **the tax does not depend on the order permutations sampled
here**, not "the tax is independent of difficulty". Testing the difficulty
hypothesis needs a *designed* partition with a real contrast - for instance
grouping classes by CIFAR-100 superclass (coherent tasks, high cross-task
similarity) versus spreading one class per superclass across tasks
(incoherent tasks, low similarity) - which the cached features support at no
extraction cost (S6b).

**O4 - for an unseen task the router does not decide; it spreads.** The top
expert takes only **20.5%** of an unseen task's samples and the expert
distribution has entropy 2.556 against a maximum of ln(20) = 2.996, i.e. 85% of
uniform. Meanwhile the representation transfers to that task at **95.00%**. Of
the two failure modes this stage was built to separate - "decides, but wrongly"
(high concentration, low entropy, low transfer) versus "cannot decide" (low
concentration, high entropy) - the measurement is unambiguously the second:
**the prototypes are all from seen classes, and a new class has no clear winner
among them.** This is the specialisation-extrapolation failure, and it is the
same root as E0's reranking result (F5): the router has no basis on which to
decide, so nothing downstream of it can recover the decision.

**O5 - consistency with the canonical order.** S4's CIFAR-100 numbers (L2b
56.00, L3 70.58, L4 97.46, tax 26.88) sit inside this order distribution
(57.02 / 71.17 / 96.58 / 25.5), so the canonical order the other stages used is
not an outlier.

### 9.2 What this does not settle

Whether the routing tax grows with *task difficulty* remains open, because the
sampled orders had no difficulty spread (O3). The designed-partition experiment
(S6b) is the version that can answer it, and it is cheap: the class ids in the
cache can be re-grouped by superclass without touching the features.

---

## 10. Consolidated findings

**F1. The readout was the first bottleneck, and a training-free prototype
readout solves it.** NCM on frozen features beats v1 and iCaRL on CIFAR-100
with zero parameters and 307 KB.

**F2. Adapter capacity saturates at rank 8** (r8 70.56, r32 70.62, r64 70.60)
and adds ~0.2 points over NCM in the per-task setting.

**F3. Selection, not capacity, is the binding constraint.** Oracle routing is
+27 points on CIFAR-100/ViT, +14 on MNIST, +7.5 on CIFAR-10, and it is the
largest single term in every dataset tested.

**F4. The routing tax shrinks as the representation improves** and tracks
candidate recall@3 (88.9 on ViT, 26.7-30.3 on the weak backbones). Routing
failure is driven by representation-space ambiguity, not by router capacity.

**F5. No reranking on the frozen space recovers the gap.** Uniform mixtures,
expert filtering, router-prior reranking and acceptance reranking (independent
heads, joint softmax, shared and expert-conditioned input spaces) all fail to
beat hard top-1 routing. The measured reason: when the router's top-1 is wrong,
the rejector prefers the router's pick 64-67% of the time, because both read
the same `z_s` and their errors are correlated.

**F6. The expert bank buys isolation, not capacity.** `L2b -> L3` is +14.58 on
CIFAR-100 and +15.45 on Tiny-ImageNet, while `L2a` (joint) beats both and sits
within 1-4 points of the closed-form readout. Isolation matters more on longer
streams.

**F7. Isolation has a generalization cost.** Few-shot transfer drops by up to
14.75 points on the ViT; the per-task bank is the worst variant on every
backbone; forward transfer is slightly negative (3/3 seeds).

**F8. The direction of the few-shot effect depends on the domain.** In-domain
+17.23, cross-dataset -3.73 to -10.19. The same mechanism helps inside the
stream's distribution and hurts outside it.

**F9. The full-split transfer probe is uninformative where the representation
is saturated** (94.8-95.1 for every cell with ViT/CIFAR-10). The low-data axis
is what discriminates.

---

## 11. Pre-registered hypotheses and their verdicts

| hypothesis | statement | verdict |
| :-- | :-- | :-- |
| H5 (v1) | the allocation policy reuses experts for related tasks | **refuted** (E7: 20/20 experts, gated == forced) |
| A (S3) | `ridge > L3` on every backbone | **confirmed** (6/6) |
| B (S3) | `L3 > ridge` on weak backbones, ridge wins on pretrained | refuted |
| C (S3) | ridge wins only on the ViT | refuted |
| D (S3) | oracle high and L3 low everywhere -> routing is general | **confirmed**, stronger than stated |
| H1 (S7) | `L1_ridge > L3` in transfer as well as CIL | **confirmed**, much stronger on few-shot |
| H2 (S7) | `L3 > L1_ridge` in transfer despite losing CIL | refuted |
| H3 (S7) | `L3` transfer is also poor | **confirmed in the few-shot regime only** |
| G1 (S4) | `ridge > L3` on most datasets | **confirmed** (4/4) |
| G2 (S4) | isolation benefit positive and dataset-dependent | **confirmed**, and it grows with stream length |
| G3 (S4) | routing tax varies with the dataset | **confirmed**, tracks difficulty not length |
| G4 (S4) | transfer correlates with `L3 - ridge` | not testable: the full-split probe is saturated |
| G5 (S4) | the few-shot penalty is systematically worse for the per-task bank | **confirmed** cross-dataset, **reverses in-domain** |
| G6 (S4) | memory/params/latency scale with T | recorded per cell; the dedicated sweep is S4b |

---

## 12. Measurement bugs this programme found

Five, each of which would have produced a confident wrong number:

1. **The forward-transfer definition was degenerate.** "Evaluate on the next
   task before training it" returns structurally 0 in a growing-head
   class-incremental setting; the measured value was exactly
   `-mean(1/5t) = -3.73%`, identical for every level. Re-pinned to a
   representation-level probe.
2. **A readout's freezing policy belongs to the readout.** Passing "all seen
   classes" instead of "trainable now" collapses a linear readout from 39.81%
   to 7.69% (forgetting 55.91% to 92.35%).
3. **The backbone factory asked for the wrong width.** Requesting
   `output_dim=256` from a ViT-B/16 inserted a 768->256 BatchNorm+ReLU head
   plus an up-projection back to 768: feature norm 4.3 instead of 18.4, NCM 28%
   instead of 70%. The raw accuracy table looked internally consistent; every
   S3 row was wrong. A regression check now guards it
   (`experiments/s3_check_vit_features.py`).
4. **Dataset statistics must reach the encoder.** Without `input_mean`/
   `input_std` the ViT path double-normalises.
5. **A copied loop loses the original's branching.** S4's loop was re-derived
   from `s2_ladder` and dropped the `joint` branch, making `L2a` numerically
   identical to `L2b` on every dataset - visible in the log, invisible in the
   aggregate.
6. **A prototype key that is not unique silently collapses the router.**
   Domain-IL gives every task the same labels, so keying prototypes by class id
   made each domain overwrite the previous one's and the router collapsed to a
   single expert (`reuse = 0.25`, `MI = 0.0000`).
7. **A transform applied to train but not to test.** The domain rotation
   reached the training split only, so every domain shared one unrotated test
   set; `unseen == acc` to the digit was the tell. Both guards are now in
   place: non-zero transforms must change the test split, and domain caches
   must have pairwise-distinct test features.

The first four are measurement-layer bugs; the fifth is the reason the plan now
requires calling the existing entry point instead of re-deriving a loop.

---

## 13. What this evidence does NOT say

- It does **not** say "mixture of experts is unnecessary". It says that under
  this benchmark, protocol, budget and backbone, the incremental expert
  machinery added no measurable benefit over a closed-form readout on frozen
  features. A different budget (much smaller memory), a different protocol
  (task-free/online), or a plastic encoder could change that, and those are the
  unmeasured axes.
- The `L1_ridge` comparison is on accuracy, not on bytes at equal budget: ridge
  keeps `(d+1)^2 + (d+1)C` sufficient statistics (about 2.7 MB at d=768,
  C=100) against NCM's 307 KB and the expert bank's 347 KB. The Pareto claim
  must be made on `stored_bytes`, which the contract records for every cell.
- S7's transfer numbers are one seed over three backbones; S4's are three seeds
  over four datasets but only one backbone. The backbone and dataset axes were
  deliberately not crossed (that is S3 x S4, not measured).
- Robustness to image corruptions is **not measured** (S4-r), nor are
  task-order, class-order, unseen-task, unseen-domain or scalability axes.
- Every result is on frozen features. The plastic-encoder branch - the one
  thing that could plausibly change F3 - is untouched.

---

## 14. Open items and the stage order

| stage | content | blocker |
| :-- | :-- | :-- |
| S4-r | corruption robustness, routing stability `TV(p(x), p(x~))` | needs corrupted caches (~2 h of ViT extraction) |
| S4b | task-count sweep `T in {2,5,10,20}` | splitters have no sub-task support |
| S5 | Task-IL / Class-IL protocol axis | **done** (section 7) |
| S5b | Domain-IL rotated MNIST, unseen domain | **done** (section 8) |
| S6 | order sensitivity (class and task order) | **done** (section 9); the difficulty hypothesis needs S6b |
| S6 | task order, class order, unseen task, expert transfer | splitters have no order parameters |
| S8 | memory / data-regime / compute budgets | no `--data_fraction`, no generic budget flag |
| S9-S11 | robustness, scalability, statistics | unstarted |

The recommended next step is **S5**: it needs no new extraction, runs on the
existing caches, and closes the protocol-labelling gap (rule 4) that every
number above depends on being explicit about.

---

## 15. Artefacts

| path | contents |
| :-- | :-- |
| `results/e0/` | ceiling, adapter headroom, router recall, mixture/reranking/rejection negatives |
| `results/s2/` | complexity ladder, 3 seeds |
| `results/s3/` | six backbones, 3 seeds, per-condition studies, deltas, transfer, ViT consistency gate |
| `results/s7/` | per-checkpoint transfer curves, few-shot suite, evaluator validation |
| `results/s4/` | four datasets x six levels x three seeds, transfer per cell, S0 contracts |
| `results/logs/` | per-stage logs from `experiments/run_all.py` |
| feature caches | gitignored (`*.pt`); `experiments/s3_run.py` and `s4_datasets.py` rebuild them |

Every cell carries an S0 contract record; the S4 study stores its recipe
(epochs/lr/rank/levels) so a resume cannot mix incompatible rows.
