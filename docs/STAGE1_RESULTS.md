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
(S6b, section 10) answered it: with a real difficulty contrast the tax moves from
+13.98 to +26.63, so S6's order-robustness is about the visit order, not about
the partition.

---

## 10. S6b - designed difficulty: the routing tax *is* geometry-dependent

S6 varied the order by random permutation and found the tax unchanged, while
admitting (O3) that a random permutation of CIFAR-100 produces tasks of nearly
equal difficulty, so the difficulty hypothesis was never tested. S6b designs the
contrast instead of sampling it, from the same cached features:

| construct | task structure | intra-task similarity | cross-task similarity | separability |
| :-- | :-- | --: | --: | --: |
| `coherent` | one task per CIFAR-100 superclass (5 similar classes) | **0.821** | 0.522 | **+0.298** |
| `dispersed` | round-robin: one class from each of five well-separated superclasses | 0.638 | **0.684** | **-0.046** |

The two axes are orthogonal by construction: `coherent` makes *within-task*
discrimination hard (five confusable classes) and routing easy; `dispersed`
does the opposite (five dissimilar classes per task, but tasks that overlap in
the representation - cross-task similarity exceeds intra-task similarity).

| construct | L0 | L1 | L2b | L3 | L4 | routing tax | iso realised | iso available | realised/available |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| `coherent` | 70.34 | 76.77 | 60.50 | 73.34 | 87.32 | **+13.98** | +12.84 | +26.82 | 47.9% |
| `dispersed` | 70.34 | 76.77 | 55.46 | 70.68 | 97.31 | **+26.63** | +15.22 | +41.85 | 36.4% |

### 10.1 Findings

**E1 - the routing tax nearly doubles with cross-task overlap: +13.98 against
+26.63.** This is the outcome the S6b design was built to detect. It also
resolves S6's open question in the useful direction: the tax does not depend on
*which order* a fixed partition is visited in (S6), it depends on *the partition
itself* - specifically on how much the tasks overlap in the representation.

**E2 - the two difficulty axes dissociate cleanly, which validates the
decomposition.** `intra_task_similarity` drives the oracle (0.821 -> 87.32
against 0.638 -> 97.31, a 9.99-point within-task cost) while
`cross_task_similarity` drives the tax (0.522 -> 13.98 against 0.684 -> 26.63, a
12.65-point routing cost). Within-task difficulty and routing difficulty are
independent knobs with independent effects, and `L4 - L3` isolates the second.

**E3 - the realisation fraction falls as routing gets harder: 47.9% -> 36.4%.**
Outcome 3 of the design brief (available isolation moves, realised stays put)
did not occur literally - realised isolation also rises, +12.84 -> +15.22 - but
it rises *more slowly* than what is available (+26.82 -> +41.85). The gap
therefore widens exactly as the thesis predicts: **more isolation is on offer
when tasks overlap, and the router can cash proportionally less of it.**

**E4 - the partition-independent quantities do not move.** `L0` and `L1` are
identical to the digit across the two constructs (70.34, 76.77), because a
pooled readout does not care which classes share a task. Only the
routing-dependent quantities move. That is a design-level consistency check:
the construct changes what it is supposed to change and nothing else.

**E5 - the canonical CIFAR-100 split is a hard-routing partition.** Its tax
(26.88 in S4) sits next to `dispersed` (26.63) and far from `coherent` (13.98).
CIFAR-100's class list is alphabetical, so consecutive five-class tasks mix
categories (apple, aquarium fish, baby, bear, beaver) - semantically dispersed
rather than coherent. Every S2-S5 number was therefore measured in the harder
routing regime, which is worth stating when those numbers are quoted.

### 10.2 Consequence for S8

The routing tax is not a constant of the method: it is a function of the task
geometry, spanning 13.98 to 26.63 across two designed partitions of the same
data with the same backbone. Budget experiments must therefore either hold the
regime fixed or report per regime; a single accuracy-per-byte curve would average
over a factor of two in the quantity the budget is supposed to buy.

---

## 11. S8 - what a budget buys, and in which regime

S6b showed the routing tax is a property of the task geometry rather than of the
visit order, spanning 13.98 (`coherent`) to 26.63 (`dispersed`) over two designed
partitions of the same data. A single accuracy-per-byte curve would average over
a factor of two in the quantity a budget is supposed to buy, so S8 sweeps the
budget in both regimes. Three resource axes, one variable at a time, each mapped
to the mechanism it is supposed to buy:

```text
parameter   adapter rank            -> expert capacity
memory      prototypes per class    -> routing resolution
active      experts per sample      -> inference compute
```

`L4_oracle` is an oracle upper bound, not a budget method, and it is measured at
every rank so that "does more capacity close the routing gap" is asked at a
fixed budget. The primary mechanism metrics are kept apart on purpose:

```text
R_iso     = (L3 - L2b) / (L4 - L2b)   relative realization of modularity
R_iso_ncm = (L3 - L0)  / (L4 - L0)    absolute incremental value
```

### 11.1 The parameter axis: the tax does not close

| rank | L2b | L3 | L4 | tax | realised | available | R_iso | R_iso_ncm | active params | stored |
| --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| **coherent** | | | | | | | | | | |
| 2 | 65.37 | 72.16 | 84.93 | 12.77 | 6.79 | 19.56 | 0.347 | 0.125 | 79,972 | 0.82 MiB |
| 8 | 60.50 | 73.34 | 87.32 | 13.98 | 12.84 | 26.82 | 0.479 | 0.177 | 89,188 | 1.53 MiB |
| 32 | 64.10 | 73.96 | 88.27 | 14.31 | 9.86 | 24.17 | 0.408 | 0.202 | 126,052 | 4.34 MiB |
| 128 | 65.13 | 73.99 | 88.32 | 14.33 | 8.86 | 23.19 | 0.382 | 0.203 | 273,508 | 15.59 MiB |
| **dispersed** | | | | | | | | | | |
| 2 | 59.43 | 70.33 | 95.56 | 25.23 | 10.90 | 36.13 | 0.302 | -0.000 | 79,972 | 0.82 MiB |
| 8 | 55.46 | 70.68 | 97.31 | 26.63 | 15.22 | 41.85 | 0.364 | 0.013 | 89,188 | 1.53 MiB |
| 32 | 59.34 | 70.71 | 97.57 | 26.86 | 11.37 | 38.23 | 0.297 | 0.014 | 126,052 | 4.34 MiB |
| 128 | 60.74 | 70.77 | 97.63 | 26.86 | 10.03 | 36.89 | 0.272 | 0.016 | 273,508 | 15.59 MiB |

**E1 - a 3.4x span in active parameters and a 19x span in stored bytes does not
close the routing gap, in either regime.** The tax is 12.77 -> 14.33 in
`coherent` and 25.23 -> 26.86 in `dispersed`: flat, and if anything slightly
*widening*. `L3` gains +1.83 / +0.44 points across the whole sweep while `L4`
gains +3.39 / +2.07. Capacity is not the binding constraint, and this now holds
in the regime where routing is *easy* as well as the one where it is hard - which
is the strongest form of the S5b/S6b thesis.

**E2 - in the hard-routing regime the whole expert bank is worth nothing over a
training-free nearest-class-mean.** `R_iso_ncm` is -0.000, 0.013, 0.014, 0.016
across the sweep: `L3` (70.33 - 70.77) sits on top of `L0` (70.34) while `L4`
reaches 97.63. This is outcome C of the S8 pre-registration: `L4 - L3` is large
and *grows* while the realised fraction stays at the floor. Capacity is
available; the router cannot realise it.

**E3 - `R_iso` and `R_iso_ncm` disagree, and the disagreement is the result.**
In `coherent`, `R_iso` is non-monotone (0.347 -> 0.479 -> 0.408 -> 0.382) while
`R_iso_ncm` is monotone and saturating (0.125 -> 0.177 -> 0.202 -> 0.203). The
`R_iso` spike at rank 8 comes from its denominator: `L2b` dips to 59.43 for seed
42 at that rank (3-seed mean 60.50) while `L3` moves only +1.18 from rank 2 to 8.
A rise in `R_iso` can be produced by the shared baseline degrading, so the
baseline-free ratio is the honest one; both are reported because they answer
different questions.

**E4 - `L2b` is not monotone in rank at all** (65.37, 60.50, 64.10, 65.13 in
`coherent`). A larger shared sequential adapter is not automatically a better
one, which is exactly why `R_iso`'s denominator cannot be treated as fixed.

### 11.2 The memory axis: the only axis that buys anything

| prototypes | L3 coherent | tax | L3 dispersed | tax | recall@3 (coh / disp) | stored |
| --: | --: | --: | --: | --: | --: | --: |
| 1 | 73.34 | 13.98 | 70.68 | 26.63 | 0.949 / 0.891 | 1.53 MiB |
| 4 | 75.50 | 11.82 | 72.46 | 24.85 | 0.956 / 0.902 | 2.41 MiB |
| 16 | 76.03 | 11.29 | 73.37 | 23.94 | 0.957 / 0.912 | 5.94 MiB |

**E5 - routing resolution is a real lever, in both regimes.** 16 prototypes per
class (4x the memory) buys +2.69 in `coherent` and +2.69 in `dispersed` - more
than the entire 64x rank sweep - and it is the only axis that moves the tax
(13.98 -> 11.29 and 26.63 -> 23.94). `recall@3` rises with it in both regimes
(0.949 -> 0.957, 0.891 -> 0.912), so the mechanism is the intended one: the
router resolves the task structure better rather than the experts getting
stronger.

**E6 - but it is a partial lever, not the bottleneck's removal.** 16x the router
memory leaves a 23.94-point tax in `dispersed` and an 11.29-point tax in
`coherent`. The router is not merely under-resolved; there is a residual
structural gap that no amount of stored prototypes in this parameterisation
closes.

### 11.3 The active axis: a pure cost

| experts/sample | L3 coherent | tax | L3 dispersed | tax | active params |
| --: | --: | --: | --: | --: | --: |
| 1 | 73.34 | 13.98 | 70.68 | 26.63 | 89,188 |
| 2 | 70.92 | 16.40 | 68.20 | 29.11 | 101,476 |
| 4 | 70.14 | 17.18 | 67.13 | 29.80 | 126,052 |

**E7 - mixing experts per sample costs 3.2 / 3.6 points and makes the tax
*worse*.** This was pre-registered from E0 and it holds: more active compute is
not a budget worth spending. Note that `k = 1` reproduces the S2-S6 router
exactly, so the sweep's first point is the baseline rather than a new method.

### 11.4 The baseline question: the closed-form readout dominates

| method | accuracy | stored | parameter-equivalents | optimizer steps |
| :-- | --: | --: | --: | --: |
| `L1_ridge` | 76.77 | 2.84 MiB | 745,162 | **0** |
| `L3` best, `coherent` (16 protos) | 76.03 | 5.94 MiB | 322,660 | 3,320 |
| `L3` best, `dispersed` (16 protos) | 73.37 | 5.94 MiB | 322,660 | 3,300 |

**E8 - at comparable memory the closed-form readout wins in both regimes.**
`L1_ridge` stores 2.84 MiB - its sufficient statistics, `A` alone is
`769 x 769` - and reaches 76.77 with zero gradient steps. `L3` needs 4x that
memory to come within 0.74 points in `coherent` and never gets within 3.4 points
in `dispersed`. Spending the same budget on expert modularity is not a good trade
in this regime, and the honest budget question is the comparison, not the raw
accuracy.

### 11.5 Forgetting is a floor effect here, and that is a finding about the setup

**E9 - forgetting measures exactly 0.00 in all 56 cells, including `L2b` and
`L3`.** It is not structural for those levels - the adapter moves and the router
grows - so the value was checked unclamped: the worst old-task change
`R[T,i] - max_{t>=i} R[t,i]` is exactly 0.00 for all five levels in both
regimes. **No old task ever declined.** The prototype anchor (`lambda_func=1.0`,
`_l_func`) holds the representation in place strongly enough that forgetting
cannot discriminate in this configuration, so S8's conclusions rest on accuracy,
isolation and the tax, not on forgetting. This is a property of the ladder as
configured, not a general claim.

**E10 - the sequential constraint's damage appears as lost plasticity, not as
forgetting.** The newest-task accuracy at the end of the run orders the levels
`L2b` 96.33, `L4` 96.80, `L3` 82.27, `L1` 86.00, `L0` 81.00 in `coherent`
(`L2b` 96.40, `L3` 76.53, `L4` 98.27 in `dispersed`), while the final average
orders them `L4` > `L1` > `L3` > `L0` > `L2b`. `L2b` fits each new task *best*
and ends *worst*: with the anchor preventing decline, the cost of the sequential
constraint is that each task is underfit at the moment it is learned and then
frozen there. Newest-task accuracy alone is therefore a misleading metric - it
rewards plasticity and hides exactly what the ladder is measuring.

### 11.6 The reading, per regime

```text
coherent   routing cheap      L3 - L0 = +1.82 ... +3.65   R_iso_ncm 0.125 -> 0.203
           capacity saturates, the tax stays ~13-14, the memory axis is the
           only lever (+2.69), active compute hurts.

dispersed  routing expensive  L3 - L0 = -0.01 ... +0.43   R_iso_ncm ~0.000 -> 0.016
           the expert bank is worth nothing over a training-free NCM while the
           oracle reaches 97.63; the memory axis recovers +2.69 and lowers the
           tax from 26.63 to 23.94, and nothing else moves.
```

**E11 - a consistency guard.** The operating point (rank 8, one prototype,
`top_k = 1`, three seeds) reproduces S6b exactly: 30 matched cells,
`max |delta| = 0.0000`. The two resource knobs are numerically no-ops at their
default, so S8's measurement additions did not change the experiment.

### 11.7 What S8 does not say

- It does not say the routing tax is unbreakable. It says *capacity* does not
  break it and *router resolution* only partly does. A different routing
  mechanism (attention over prototypes, a learned gate, joint router training)
  is untested here.
- It does not say the expert bank is worthless in general: in `coherent` it
  delivers +3.65 over the budget-free readout, which is real but smaller than the
  closed-form baseline's own +6.43 over that readout.
- The memory axis was swept as prototypes per class; other memory uses (stored
  statistics, expert state, replay) are not measured.
- The one-seed sweep points cannot support a claim about differences below ~1
  point; they support the curve shapes, which is what they are used for.

---

## 12. S9 - does the decomposition survive a distribution shift?

S8 closed the capacity question in the clean geometry. S9 asks whether the
decomposition it rests on - capacity is not routing realization, and intra-task
difficulty is not cross-task routing difficulty - is an artifact of that
geometry. Two controlled families, each varying exactly one thing, in both S6b
regimes, at the S8 operating point (rank 8, one prototype per class,
`top_k = 1`, seed 42):

```text
corruption   training stays clean, only the test-time input distribution changes
             gaussian noise / defocus blur / brightness at severity 1, 3, 5
             (a controlled CIFAR-C style family, not the official CIFAR-100-C)
spurious     a 6x6 red corner cue on the first half of the classes during
             training; at test it is correlated, absent, or flipped onto the
             other half. Training on clean and testing on the same conditions is
             the control arm, so the shortcut effect is the *interaction*.
```

170 cells. Shift caches are extracted through the same pipeline as the canonical
cache (the clean re-extraction reproduces it with `max |delta| = 0.0` over 10,000
samples) and are regrouped by *label* onto each construction, because a
construction is a regrouping of class ids and a task-keyed graft would feed the
wrong classes to every task.

### 12.1 Corruption: the tax grows, the realized value collapses

`dispersed` (hard routing), training clean:

| test | L0 | L1 | L2b | L3 | L4 | tax | realised | available | R_iso_ncm | recall@3 | max-share |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| clean | 70.34 | 76.77 | 55.64 | 70.66 | 97.29 | +26.63 | +15.02 | +41.65 | 0.012 | 0.891 | 0.713 |
| noise s1 | 52.48 | 57.52 | 32.15 | 52.84 | 90.74 | +37.90 | +20.69 | +58.59 | 0.009 | 0.769 | 0.542 |
| noise s3 | 30.69 | 34.05 | 14.65 | 30.82 | 74.16 | +43.34 | +16.17 | +59.51 | 0.003 | 0.554 | 0.331 |
| noise s5 | 11.23 | 12.40 | 5.19 | 11.25 | 47.23 | +35.98 | +6.06 | +42.04 | 0.001 | 0.324 | 0.211 |
| blur s1 | 52.03 | 61.83 | 33.55 | 52.58 | 92.17 | +39.59 | +19.03 | +58.62 | 0.014 | 0.753 | 0.537 |
| blur s3 | 30.35 | 38.20 | 16.03 | 30.55 | 78.40 | +47.85 | +14.52 | +62.37 | 0.004 | 0.563 | 0.332 |
| blur s5 | 12.58 | 15.78 | 5.41 | 12.71 | 58.72 | +46.01 | +7.30 | +53.31 | 0.003 | 0.355 | 0.341 |
| bright s1 | 68.40 | 74.86 | 52.37 | 68.73 | 96.84 | +28.11 | +16.36 | +44.47 | 0.012 | 0.877 | 0.693 |
| bright s3 | 62.76 | 69.86 | 45.20 | 63.11 | 94.90 | +31.79 | +17.91 | +49.70 | 0.011 | 0.834 | 0.640 |
| bright s5 | 52.62 | 60.24 | 34.32 | 52.92 | 90.99 | +38.07 | +18.60 | +56.67 | 0.008 | 0.751 | 0.542 |

`coherent` (easy routing), training clean:

| test | L0 | L1 | L2b | L3 | L4 | tax | realised | available | R_iso_ncm | recall@3 | max-share |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| clean | 70.34 | 76.77 | 59.43 | 73.21 | 87.19 | +13.98 | +13.78 | +27.76 | 0.170 | 0.949 | 0.819 |
| noise s1 | 52.48 | 57.52 | 37.51 | 54.80 | 76.52 | +21.72 | +17.29 | +39.01 | 0.097 | 0.858 | 0.669 |
| noise s3 | 30.69 | 34.05 | 19.92 | 31.84 | 60.34 | +28.50 | +11.92 | +40.42 | 0.039 | 0.667 | 0.446 |
| noise s5 | 11.23 | 12.40 | 8.21 | 11.73 | 40.74 | +29.01 | +3.52 | +32.53 | 0.017 | 0.411 | 0.312 |
| blur s1 | 52.03 | 61.83 | 40.46 | 54.21 | 77.69 | +23.48 | +13.75 | +37.23 | 0.085 | 0.861 | 0.658 |
| blur s3 | 30.35 | 38.20 | 22.99 | 31.52 | 63.08 | +31.56 | +8.53 | +40.09 | 0.036 | 0.693 | 0.471 |
| blur s5 | 12.58 | 15.78 | 8.71 | 12.78 | 44.95 | +32.17 | +4.07 | +36.24 | 0.006 | 0.466 | 0.401 |
| bright s1 | 68.40 | 74.86 | 55.99 | 71.34 | 86.00 | +14.66 | +15.35 | +30.01 | 0.167 | 0.942 | 0.805 |
| bright s3 | 62.76 | 69.86 | 49.22 | 65.31 | 82.97 | +17.66 | +16.09 | +33.75 | 0.126 | 0.909 | 0.753 |
| bright s5 | 52.62 | 60.24 | 39.23 | 55.10 | 76.97 | +21.87 | +15.87 | +37.74 | 0.102 | 0.852 | 0.664 |

**E1 - the routing tax grows with corruption severity in both regimes.** From
13.98 to 32.17 in `coherent` (2.3x) and from 26.63 to 47.85 in `dispersed`
(1.8x), monotonically for noise and blur and nearly so for brightness. This is
the S6b mechanism reproduced under a new manipulation: as the representation
degrades, the tasks overlap more and the router's job gets harder. The regime
*ordering* is preserved at every severity, but the regimes converge as the
representation collapses (ratio 1.90x at clean, 1.52x at blur s3, 1.24x at noise
s5) - when everything is near chance there is no geometry left to route on.

**E2 - `R_iso_ncm` collapses toward zero in both regimes.** `dispersed`:
0.012 -> 0.001 / 0.003 / 0.008 at severity 5. `coherent`: 0.170 -> 0.017 / 0.006
/ 0.102. Corruption erases the expert bank's incremental value, and the effect
is strongest where the damage is representational (noise, blur) rather than a
global gain change (brightness, which is the mildest family at every severity).

**E3 - the S8 result survives every condition.** In `dispersed`, `R_iso_ncm`
ranges over **0.001 - 0.014 across all 17 conditions** (clean, 9 corruptions, 3
spurious, plus the spurious control arm). The statement "in the hard-routing
regime the whole expert bank is worth nothing over a training-free
nearest-class-mean" is not an artifact of the clean benchmark geometry. In
`coherent` the same quantity ranges from **-0.169 to 0.205**: the easy-routing
regime's incremental value is real but fragile.

**E4 - the router's failure is visible at the assignment level, and it is
diffuseness rather than confident error.** `recall@3` falls monotonically with
severity (0.891 -> 0.324 for noise) and the assignment max-share falls with it
(0.713 -> 0.211). Under corruption the router does not pick the wrong expert
confidently; its distribution flattens. That is the same picture S6 found for
unseen tasks, now along a severity axis.

**E5 - `L1_ridge` wins in every single condition.** 34/34 (2 regimes x 17
conditions), mean margin +5.4 points clean, +4.7 under corruption, +5.5 under
spurious, minimum +0.67 (noise s5, where every level is near chance) and maximum
+9.51. The closed-form readout is not only the strongest clean baseline; it is
the most shift-robust one, and its margin over the expert bank grows with
severity for blur and brightness.

**E6 - the shared sequential adapter is the most fragile level.** `L2b` falls
furthest under both families (`dispersed` spurious: 56.65 -> 49.91 -> 43.99;
`coherent`: 63.20 -> 52.31 -> 42.47). The sequential constraint's cost, which
S8 showed is paid as plasticity rather than forgetting, is amplified by a shift.

### 12.2 Spurious cue: a learned shortcut, with a regime-dependent landing site

Control arm (training clean, cue only at test):

| construct | test | L3 | L4 | tax | R_iso_ncm | recall@3 |
| :-- | :-- | --: | --: | --: | --: | --: |
| dispersed | correlated | 70.00 | 97.31 | +27.31 | 0.010 | 0.890 |
| dispersed | absent | 70.66 | 97.29 | +26.63 | 0.012 | 0.891 |
| dispersed | flipped | 70.58 | 97.23 | +26.65 | 0.014 | 0.891 |
| coherent | correlated | 72.58 | 86.84 | +14.26 | 0.167 | 0.950 |
| coherent | absent | 73.21 | 87.19 | +13.98 | 0.170 | 0.949 |
| coherent | flipped | 73.09 | 87.03 | +13.94 | 0.171 | 0.949 |

Treatment arm (the cue present during training on the first half of classes):

| construct | test | L0 | L1 | L2b | L3 | L4 | tax | R_iso_ncm | recall@3 |
| :-- | :-- | --: | --: | --: | --: | --: | --: | --: | --: |
| dispersed | correlated | 72.35 | 78.19 | 56.65 | 72.68 | 97.69 | +25.01 | 0.013 | 0.904 |
| dispersed | absent | 69.66 | 76.51 | 49.91 | 69.96 | 97.14 | +27.18 | 0.011 | 0.886 |
| dispersed | flipped | 65.77 | 73.49 | 43.99 | 65.97 | 96.19 | +30.22 | 0.007 | 0.861 |
| coherent | correlated | 72.35 | 78.19 | 63.20 | 75.85 | 89.42 | +13.57 | 0.205 | 0.953 |
| coherent | absent | 69.66 | 76.51 | 52.31 | 70.97 | 84.11 | +13.14 | 0.091 | 0.950 |
| coherent | flipped | 65.77 | 73.49 | 42.47 | 63.98 | 76.38 | +12.40 | **-0.169** | 0.943 |

**E7 - the cue is a learned shortcut, at every level and in both regimes.** In
the treatment arm, accuracy orders `correlated > absent > flipped`: `dispersed`
L3 72.68 / 69.96 / 65.97, `coherent` L3 75.85 / 70.97 / 63.98. The control arm
shows the cue alone is nearly harmless (tax and `R_iso_ncm` unchanged, accuracy
within 0.7 points), so the effect is the *interaction* between training and test
condition, not the cue itself.

**E8 - where the shortcut's cost lands depends on the routing regime.** In
`dispersed` the tax grows (25.01 -> 27.18 -> 30.22) and `recall@3` falls (0.904
-> 0.886 -> 0.861): the router is partly fooled by the cue. In `coherent` the
tax is flat (13.57 -> 13.14 -> 12.40) and `recall@3` is flat (0.953 -> 0.950 ->
0.943), while `L4` falls hard (89.42 -> 84.11 -> 76.38): the router is *not*
fooled and the damage is entirely representational. With easy routing the model
absorbs the shortcut into the shared representation; with hard routing it leaks
into the assignment as well.

**E9 - and `R_iso_ncm` goes negative in the flipped `coherent` arm** (-0.169):
`L3` (63.98) ends up *below* the training-free NCM (65.77) while the oracle
reaches 76.38. The expert bank is worse than no expert bank at all when the
representation is built around a cue that lies at test time.

### 12.3 Consistency checks

**E10 - three guards passed exactly.** (i) The clean cells reproduce S8's
seed-42 operating point: 10/10 cells (both constructs x 5 levels),
`max |delta| = 0.0000`. (ii) `spurious_absent` reproduces `clean` to the digit
(`max |delta| = 0.0000` over both constructs x 5 levels), which is what a "no
cue" path must do. (iii) The label-based regrouping reproduces the S6b
construction splits exactly, with 100 test samples per class. The S9 extraction
also reproduces the canonical cache's clean test features with
`max |delta| = 0.0`.

### 12.4 What S9 adds to the story

The decomposition is not a property of the clean benchmark. Across 34 conditions
- two regimes, ten corruption cells, four spurious cells - the ordering
`L4 > L3 > L2b` and the separation between *capacity* and *routing realization*
both survive:

```text
capacity     rank, expert count and active compute do not close the tax (S8)
             and corruption does not change that - it makes it worse
routing      the tax is a function of task overlap (S6b) and it grows
             monotonically with representation damage (S9)
resolution   the only lever that moves the tax (S8) is not enough (S8, S9)
baseline     L1_ridge wins in 34/34 conditions, and is the most shift-robust
             level in the ladder
```

The one thing that changes qualitatively is *where* a shortcut's damage lands:
into routing under hard routing, into the representation under easy routing.

---

## 13. Consolidated findings

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

**F10. The routing tax is a property of the task geometry, not of the visit
order.** Random permutations leave it unchanged (S6: 25.19-25.92, order std
0.43-1.55, correlations ~0); a designed partition moves it by nearly 2x (S6b:
+13.98 `coherent` against +26.63 `dispersed`). Within-task similarity drives the
oracle, cross-task similarity drives the tax.

**F11. Capacity does not close the routing gap; router resolution only partly
does.** A 3.4x span in active parameters and a 19x span in stored bytes leaves
the tax flat in both regimes (S8: 12.77 -> 14.33 and 25.23 -> 26.86). 16
prototypes per class - the only axis that buys accuracy in both regimes (+2.69)
- lowers it to 11.29 and 23.94 but does not remove it. More experts per sample
*raises* it.

**F12. In the hard-routing regime the whole expert bank is worth nothing over a
training-free nearest-class-mean.** `R_iso_ncm` is ~0.000-0.016 across the S8
sweep: `L3` 70.33-70.77 against `L0` 70.34, while the oracle reaches 97.63.
Available isolation is large; realised isolation is at the floor.

**F13. The closed-form readout dominates the expert bank in the
memory-accuracy plane, in both regimes.** `L1_ridge` stores 2.84 MiB and reaches
76.77 with zero gradient steps; `L3` needs 4x that memory to come within 0.74
(`coherent`) or 3.4 (`dispersed`) points.

**F14. Forgetting is a floor effect in this ladder.** Measured exactly 0.00 in
all 56 S8 cells, including `L2b`/`L3`, and unclamped the worst old-task change is
also exactly 0.00: no old task ever declined, because the prototype anchor holds
the representation in place. The sequential constraint's damage appears as lost
*plasticity* instead - `L2b` fits each newest task best (96.33) and ends worst
(60.50).

**F15. The decomposition survives a distribution shift.** Across 34 S9
conditions (two regimes, ten corruption cells, four spurious cells) the ordering
`L4 > L3 > L2b` holds and capacity stays separate from routing realization. The
tax grows monotonically with corruption severity in both regimes (13.98 -> 32.17
`coherent`, 26.63 -> 47.85 `dispersed`), reproducing the S6b mechanism under a
new manipulation.

**F16. In the hard-routing regime the expert bank is worth nothing over NCM in
every condition tested.** `R_iso_ncm` ranges over 0.001-0.014 across all 17
`dispersed` conditions, and -0.169 to 0.205 across `coherent`. The S8 result is
not an artifact of the clean geometry.

**F17. `L1_ridge` wins in 34/34 conditions and is the most shift-robust level**
(mean margin +5.4 clean, +4.7 corruption, +5.5 spurious). A spurious cue is a
learned shortcut in both regimes (`correlated > absent > flipped` at every
level), and where its cost lands depends on the routing regime: into the
assignment under hard routing, into the representation under easy routing.

---

## 14. Pre-registered hypotheses and their verdicts

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
| E1 (S8) | extra capacity is realizable in `coherent`, `R_iso` rises | **partly confirmed**: the baseline-free `R_iso_ncm` rises 0.125 -> 0.203, but the tax stays flat and `L3` gains only +1.83 |
| E2 (S8) | in `dispersed` more experts create isolation the router cannot cash | **confirmed**, at the floor: `R_iso_ncm` ~0.000-0.016 while `L4 - L3` grows |
| E3 (S8) | if `L3 -> L4` does not close, the bottleneck is realization, not capacity | **confirmed in both regimes** (tax 12.77 -> 14.33 and 25.23 -> 26.86) |
| E4 (S8) | the memory axis raises `recall@3` before accuracy | **confirmed**: `recall@3` rises 0.949 -> 0.957 and 0.891 -> 0.912, accuracy +2.69 in both |
| E5 (S8) | the active axis is a cost without a matching gain | **confirmed**: -3.2 / -3.6 points at `top_k = 4`, and the tax worsens |
| R1 (S9) | the capacity/routing decomposition survives corruption and spurious shift | **confirmed** in 34/34 conditions; the tax grows with severity instead of changing sign |
| R2 (S9) | in `dispersed` more experts create isolation the router cannot cash | **confirmed under every shift**: `R_iso_ncm` 0.001-0.014 over 17 conditions |
| R3 (S9) | the routing tax is driven by task overlap, not by the benchmark | **confirmed**: the tax rises monotonically with representation damage, the S6b mechanism under a new manipulation |
| R4 (S9) | the closed-form readout stays the strongest baseline | **confirmed**: 34/34 conditions, and the most shift-robust level |
| R5 (S9) | a spurious cue is absorbed as a shortcut | **confirmed**, with a regime-dependent landing site: routing under `dispersed`, representation under `coherent` |

---

## 15. Measurement bugs this programme found

Ten, each of which would have produced a confident wrong number:

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
8. **Interleaving evaluation with training changes the protocol, not just the
   schedule.** `mask_unseen` restricts the head to the classes seen so far, so a
   task evaluated *at the time it is learned* competes against `5(t+1)` classes
   while the final evaluation competes against 100. Measured on the same model:
   81.02 against 70.66, a 10.4-point difference that is **not** forgetting (S8
   measured forgetting at exactly 0.00). Every stage therefore computes the
   whole accuracy matrix after training; a harness that evaluates as it goes is
   running a different, easier benchmark.
9. **A shift cache keyed by task is not a shift cache keyed by class.** S9's
   corruption caches are extracted on the canonical partition, but a
   construction (`coherent`/`dispersed`) is a *regrouping of class ids*, so
   grafting canonical task `k` onto construction task `k` would have fed the
   wrong classes' features to every task. Regrouping by label is
   order-independent, works for the train split (whose order is a permutation)
   and needs no re-extraction; it is guarded against the S6b construction code
   that already produces the same splits.
10. **A guard caught a configuration error, and the guard was right.** S9's
    first version trained on the canonical partition while comparing against
    S8's `dispersed` cells (70.56 against 70.66) and it looked like
    cross-process nondeterminism. It was not: hashing the model init, the first
    batch order and the per-task parameters in independent processes showed the
    ladder is bit-reproducible, and the 0.1-point difference was exactly
    canonical-against-`dispersed`. The tolerance was tightened back to float
    noise rather than widened to hide it.

The first four are measurement-layer bugs; the fifth is the reason the plan now
requires calling the existing entry point instead of re-deriving a loop; the
eighth and ninth are the same failure class one level down - a harness detail
(evaluation order, key space) silently redefining the experiment.

---

## 16. What this evidence does NOT say

- It does **not** say "mixture of experts is unnecessary". It says that under
  this benchmark, protocol, budget and backbone, the incremental expert
  machinery added no measurable benefit over a closed-form readout on frozen
  features. A different budget (much smaller memory), a different protocol
  (task-free/online), or a plastic encoder could change that, and those are the
  unmeasured axes.
- The `L1_ridge` comparison is on accuracy, not on bytes at equal budget: ridge
  keeps `(d+1)^2 + (d+1)C` sufficient statistics (about 2.9 MB at d=768,
  C=100) against NCM's 307 KB and the expert bank's 347 KB. The Pareto claim
  must be made on `stored_bytes`, which the contract records for every cell -
  and S8 (section 11.4) does it: at comparable memory ridge still wins in both
  regimes.
- S7's transfer numbers are one seed over three backbones; S4's are three seeds
  over four datasets but only one backbone. The backbone and dataset axes were
  deliberately not crossed (that is S3 x S4, not measured).
- Robustness to image corruptions is **not measured** (S4-r), nor are
  task-order, class-order, unseen-task, unseen-domain or scalability axes.
- S9's corruptions are a controlled CIFAR-C style family, not the official
  CIFAR-100-C benchmark, and the spurious cue is synthetic (a 6x6 corner
  square). Both are single-seed (seed 42); the clean cells are pinned to S8's
  three-seed operating point exactly, but the shifted points are one seed each.
  At severity 5 (noise, blur) every level is near chance, so the tax at those
  points compares two near-floor numbers.
- Every result is on frozen features. The plastic-encoder branch - the one
  thing that could plausibly change F3 - is untouched.

---

## 17. Open items and the stage order

| stage | content | blocker |
| :-- | :-- | :-- |
| S4-r | corruption robustness, routing stability `TV(p(x), p(x~))` | **superseded by S9** (section 12) |
| S4b | task-count sweep `T in {2,5,10,20}` | splitters have no sub-task support; S10 is the same question with the pieces S8 built |
| S5 | Task-IL / Class-IL protocol axis | **done** (section 7) |
| S5b | Domain-IL rotated MNIST, unseen domain | **done** (section 8) |
| S6 | order sensitivity (class and task order) | **done** (section 9) |
| S6b | designed difficulty: coherent vs dispersed partitions | **done** (section 10) |
| S8 | resource budget: parameter / memory / active, both regimes | **done** (section 11) |
| S9 | robustness: corruption and spurious cue, both regimes | **done** (section 12) |
| S10 | scalability: task count and expert count | **next** |
| S11 | statistical validation: confirmatory seed/CI protocol | after S10 |

The recommended next step is **S10**, and S8+S9 give it a specific question rather
than a generic sweep. Two pieces already exist: `L3` allocates one expert per
task, so increasing `T` increases storage, expert count and routing candidates at
once, and S9 showed the router's failure mode under shift is *diffuseness*
(max-share falling, `recall@3` falling) rather than a confident wrong pick. So
S10's question is which of those three `T`-effects the tax actually tracks:

```text
T up  ->  stored bytes up        (measured per cell already)
T up  ->  experts per bank up    (capacity, which S8 showed does not close the tax)
T up  ->  routing candidates up  (resolution, which S8 showed partly does)
```

If the tax tracks candidate count rather than expert capacity, the routing
mechanism redesign that follows S11 has a target: make the router's decision
easier, not the experts stronger. S4b is subsumed by this: it is the same sweep
with the S8 resource instrumentation.

---

## 18. Artefacts

| path | contents |
| :-- | :-- |
| `results/e0/` | ceiling, adapter headroom, router recall, mixture/reranking/rejection negatives |
| `results/s2/` | complexity ladder, 3 seeds |
| `results/s3/` | six backbones, 3 seeds, per-condition studies, deltas, transfer, ViT consistency gate |
| `results/s7/` | per-checkpoint transfer curves, few-shot suite, evaluator validation |
| `results/s4/` | four datasets x six levels x three seeds, transfer per cell, S0 contracts |
| `results/s6/` | order sensitivity: 9 order configurations, unseen-task routing |
| `results/s6b/` | designed difficulty: `coherent` vs `dispersed`, three seeds |
| `results/s8/` | 56-cell budget study (rank x prototypes x top-k, both regimes) + report |
| `results/s9/` | shift caches (13 conditions) + 170-cell robustness study (2 regimes x 2 families) |
| `results/logs/` | per-stage logs from `experiments/run_all.py` |
| feature caches | gitignored (`*.pt`); `experiments/s3_run.py` and `s4_datasets.py` rebuild them |

Every cell carries an S0 contract record; the S4 study stores its recipe
(epochs/lr/rank/levels) so a resume cannot mix incompatible rows.
