# PAL-MoE v2 - Technical Specification

Status: design freeze for implementation. Written 2026-09-23 against commit
`9cf8c7a`. Companion to `BENCHMARK.md` (v1 protocol and design facts),
`RESEARCH_MAP.md` (literature lineage) and `CODE_REVIEW.md` (refactor
backlog).

This document is a *plan*. It does not change any v1 behaviour. The paper run
queue is active, so every v2 artefact is additive: a new package
(`pal_moe/v2/`), a new method id (`palmoe_v2`), new configs, new tests. v1
modules are imported, never edited, by the v2 path.

---

## 0. Summary of the change

v1 answers *"how do we not forget when a new task arrives?"* with one expert
per task, a shared softmax router, and a prototype store that anchors both.

v2 answers a different question:

> Given a fixed parameter and memory budget, how much of the incoming stream
> is already explained by the existing modular capacity, and when is buying
> new capacity actually worth it?

Concretely, v2 replaces:

| v1 | v2 |
| :-- | :-- |
| expert = classifier head `h -> C` | expert = residual adapter `z -> z + A_i(z)` |
| per-expert classifier | one global cosine classifier |
| softmax over a growing logit vector | nested residual gates (bounded initial mass) |
| hard `owner_expert` + owner CE distillation | no owner; soft responsibility is *derived* from the gates |
| heuristic trigger `S(x) > tau` | candidate filter + counterfactual reuse-vs-expand probe |
| null-space initialization | not needed: routing retention is bounded by construction |
| parameter-averaging merge | function-space merge with accept/reject |
| stored `o_p [N, C]`, `x_p`, `raw_x`, `r_p` | stored `v_p`, `p_anchor`, coverage stats |
| 10+ loss terms | 3 core terms (+1 opt-in ablation) |

The novelty claim is not any single component (all have prior art, see
`RESEARCH_MAP.md`). It is the combination: **expansion-safe residual routing +
derived soft responsibility + counterfactual allocation + function-space
lifecycle under an explicit byte budget**, evaluated on a budget-conditioned
Pareto front rather than on expert count.

---

## 1. Where v1 actually stands (measured, not assumed)

Everything in this section is read from `docs/BENCHMARK.md` and `results/`.
These numbers define v2's success criteria.

### 1.1 The routing problem is solved; the representation problem is not

Fact 11, end-to-end CIFAR-10, frozen encoder, seed 42:

- router-anchor distillation makes task-to-expert routing exact
  (task->expert routing correct, MI 0.379, utilization entropy 0.998);
- inference-time prototype anchoring adds nothing on top (37.5% vs 37.4%);
- **the remaining gap to the per-expert oracle (66.9%) is expert and
  representation quality, not routing.**

v2's premise follows directly: if routing is already near-perfect, further
routing work has no headroom. The capacity must move from the classification
head into the representation.

### 1.2 The strong-backbone Pareto point is currently losing

CIFAR-100, 20 tasks x 5 classes, frozen ViT-B/16, feature cache
(`results/cifar100_vit_multiseed`, 3 seeds):

| Method | Avg Acc | Forgetting | Experts | memory_bytes | stored_bytes | total_params |
| :-- | --: | --: | --: | --: | --: | --: |
| iCaRL (k=25) | **64.94%** | **12.25%** | 1 | 7.99 MB | 10.66 MB | 0.67 M |
| PAL-MoE v1 (`--expand_every_task`) | 59.34% | 18.64% | 20 | 14.23 MB | 14.23 MB | 13.36 M |
| DER++ (P=250) | 50.97% | 47.17% | 1 | 0.87 MB | 0.87 MB | 0.67 M |
| ER (P=250) | 34.06% | 67.68% | 1 | 0.77 MB | 0.77 MB | 0.67 M |

v1 uses **1.3x the stored bytes and 20x the total parameters of iCaRL for 5.6
points less accuracy** (1.8x on `memory_bytes` alone). That is the Pareto
point v2 must fix. A method that adds allocation machinery on top of v1
without moving this row is not worth publishing.

### 1.3 Where the bytes go

`estimate_memory_footprint` (`memory/prototype_memory.py:1042`) counts
`v_p + r_p + o_p + x_p + y_p + raw_x` as 4 bytes/element. At the ViT geometry
(d=768, C=100, N=20, k=5 exemplars, P=536 stored):

| field | shape | bytes/proto | share |
| :-- | :-- | --: | --: |
| `x_p` exemplar features | `[5, 768]` | 15,360 | 58% |
| `o_p` expert output anchors | `[20, 100]` | 8,000 | 30% |
| `v_p` prototype | `[768]` | 3,072 | 12% |
| `r_p`, `y_p` | `[20]`, `[5]` | 100 | <1% |
| **total** | | **26,532** | **14.23 MB** |

So the two fields v2 deletes (`x_p` by default, `o_p` structurally) are 88% of
the store. The remaining `v_p + p_anchor` is 3.2 KB/prototype: **1.76 MB at
P=536, an 8x reduction**, with the same prototype quota.

### 1.4 The v1 results that constrain the design

These are not opinions; they are measured facts from `BENCHMARK.md`.

- **Fact 4**: null-space anchoring at initialization is harmful (MNIST pure
  49.85 -> 22.87). Confirmed by inspection: at d=256 with P>=256 prototypes
  the basis spans `R^d`, `_project_to_null_space`
  (`models/router.py:13`) silently returns the unprojected direction, so the
  mechanism is both harmful and, past the rank limit, a no-op.
- **Fact 9**: with a trainable encoder the recency funnel is representation
  drift, not router capacity; freezing the encoder removes it (18.5 -> 33.9
  accuracy, routing entropy 0.015 -> 0.758).
- **Fact 12**: distillation is the dominant stability mechanism; joint
  calibration is redundant at short schedules (32.5 / 24.1 with distillation
  vs 22.9 / 66.4 without); the OOD term is a small consistent positive.
- **Fact 15**: the exact routing lock breaks the newest expert's ability to
  win its own task (MNIST pure 76.60 -> 65.60), which v1 recovers with owner
  distillation + a stronger OOD weight + inference anchoring. This is a
  symptom of the shared softmax denominator, not of the lock.
- **H5 refuted (E7)**: gated allocation equals forced expansion on CIFAR-100
  (20/20 experts, 14.25 +/- 0.22 vs 14.34 +/- 0.41). The trigger never reuses.
- **`AlwaysTrigger` docstring**: a 6-expert cap on 20-task CIFAR-100 cost
  accuracy, which is why `--expand_every_task` exists.

Consequences for v2:

1. The routing mechanism must be replaced, not retuned: under a shared
   softmax, expansion *necessarily* perturbs history, and locking it
   *necessarily* blocks the new expert. v2 removes the shared denominator.
2. v2's claim cannot be "fewer experts". It must be a Pareto claim
   (accuracy/forgetting at a fixed byte and parameter budget). Expert count is
   an instrument, not the result.
3. v2 must run on frozen ImageNet backbones with the feature cache. The
   counterfactual probe is only affordable because the encoder is out of the
   loop, and the 20-task CIFAR-100 conv row (15.65%) is too close to noise to
   measure anything.

---

## 2. Architecture

### 2.1 Two spaces, one direction of information flow

```
                    x
                    |
                    v
        Frozen backbone  phi_0        (feature cache; no drift, no EMA)
                    |
                    v
        stable routing space  z_s in R^d
                    |
        +-----------+-----------+
        |                       |
        v                       v
  coverage stats            gate stack        <-- routing reads the STABLE space
  (mu_i, log sigma^2_i)     p(e | z_s)
        |                       |
        v                       v
  allocation controller     expert adapters
                            z_mix = z_s + sum_i p_i * A_i(z_s)
                                        |
                                        v
                              global cosine classifier   <-- prediction reads the PLASTIC space
                                        |
                                        v
                                     logits
```

Rule: **routing reads `z_s` only; prediction reads `z_mix` only.** A gate never
sees its own adapter's output, so expansion cannot make routing
self-referential. `z_s` is produced by a frozen encoder, so both the gates and
the coverage statistics are drift-free by construction. This removes, as
unnecessary machinery: `EMAEncoder`, `refresh_representations`,
`refresh_anchors`, the encoder stability loss, and the trainable-encoder
warning in `_distill_router_anchors`.

Plasticity enters only through `A_i`. That is the architectural statement of
the stability/plasticity split (infrastructure, not novelty - DEMM 2026 uses
the same slow/fast idea; see `RESEARCH_MAP.md` section 3).

### 2.2 Expert = residual adapter

```python
class ResidualAdapter(nn.Module):
    """A_i(z) = U_i @ gelu(V_i @ z),  U_i zero-init -> A_i == 0 exactly."""
    V: nn.Parameter  # [r, d]  Kaiming-uniform, bias=False
    U: nn.Parameter  # [d, r]  zeros,           bias=False
```

Reference geometry `d=768, r=32`:

| | params | note |
| :-- | --: | :-- |
| `ResidualAdapter(r=32)` | 49,152 | 2*r*d |
| v1 `MLPExpert(hidden=512, C=100)`, base pathway | 445,028 | (d*h + h) + (h*C + C) |
| v1 `MLPExpert` incl. its residual adapter | 667,236 | matches the measured `trainable_params` of the v1 single-head baselines |
| ratio vs the full v1 expert | **13.6x smaller** | |

- `A_i == 0` at init, so `E_i(z) = z + A_i(z) = z` exactly. Adding an expert
  is function-preserving *without cloning a parent*:
  `clone_function_preserving` and the whole parent-copy path are deleted from
  the v2 path.
- `freeze_base` semantics disappear: an expert is *only* the adapter.
- Rank is a config knob (`--v2_rank`, default 32; sweep 8/16/32/64).
- Optional `--v2_adapter_norm` adds a LayerNorm before `V`. Default off.

### 2.3 Expansion-safe residual router

There is no router trunk. Routing is a stack of scalar gates, one per expert
created after the first:

```
a_k(z) = sigmoid( w_k . lift(z) + b_k )          # a_k in (0, 1), proper sigmoid
```

`lift` is an optional frozen random ReLU projection (section 2.3.2). `w_k, b_k`
are fit by a logistic probe and then calibrated so that the **top-1 routing on
every stored prototype is preserved** (section 2.3.1). They stay trainable
during ADAPT. The routing distribution is defined recursively:

```
p_0(0 | z) = 1

p_k(e | z) = (1 - a_k(z)) * p_{k-1}(e | z)     for e < k
p_k(k | z) = a_k(z)
```

Closed form for N experts:

```
p(e | z) = [ prod_{j=e+1..N-1} (1 - a_j(z)) ] * a_e(z)        e >= 1
p(0 | z) = prod_{j=1..N-1} (1 - a_j(z))
```

Why a plain sigmoid with a bias shift, and not a multiplicative `gamma`
initialized to zero: a product `gamma * s(z)` gives exact invariance at
`gamma = 0` but zeroes the gradient of `w, b` (a saddle), so `w, b` would never
learn from the task loss. The calibrated bias gives the same effect while
keeping every parameter trainable, and the retention claim becomes a measured
quantity (`eps0`) rather than a structural identity.

#### 2.3.1 Gate calibration: dense mass is not the invariant that matters

The safety criterion is **not** `max_p a_k(z_p) <= eps0` alone. Inference runs
`top_k=1`, so what must be preserved is the *argmax*, not the proportional mass.

Let `p` be the distribution before expansion and `m(z) = max_e p_e(z)` its
top-1 mass. After expansion the old top-1 carries `(1-a)m` and the new expert
carries `a`, so the old argmax survives iff

```
a < m / (1 + m)                    (top-1 preservation)
a <= (m - delta) / (1 + m - delta) (top-1 preservation with margin delta)
```

The calibration therefore solves

```
t* = min_p ( m_p - delta ) / ( 1 + m_p - delta )     over the protected anchors
t  = min( eps0, t* )
b_k = logit(t) - max_p ( w_k . lift(z_p) )           (exact, closed form)
```

Two consequences worth stating explicitly:

- the bias shift is `Delta b = logit(t) - logit(p_max)`, which is **negative
  when `p_max > t` and positive when `p_max < t`**; the spec's earlier "shift
  down" wording was wrong in general;
- with `eps0 = 0.02` the top-1 condition is *slack* whenever
  `m > eps0/(1-eps0) ~= 0.0204`, i.e. almost always. It becomes the binding
  constraint only on anchors where the old routing is already diffuse. The
  implementation must still take `min(eps0, t*)` so that raising `eps0` (a
  config knob) can never silently break top-1.

*Protected anchors*: anchors with `m_p <= delta` have no meaningful top-1 (the
old distribution is uniform there); they are excluded from the `min` and
reported separately. Without that exclusion a single diffuse anchor would
force `t* = 0` and block expansion.

The E6.4 test is therefore `argmax p_before(z_p) == argmax p_after(z_p)` on the
protected anchors, not the mass bound.

**Properties (all testable, section 10 E6):**

1. *Normalization*: `sum_e p(e|z) = 1` by induction.
2. *Expansion safety*: at the moment an expert is added, `argmax p(z_p)` is
   unchanged on every protected anchor and the new gate claims at most
   `eps0 = 2%` of the mass there. v1's "routing lock" could not do this at all:
   `softmax([l_1..l_N, l_new])` changes the denominator for every input, and
   fact 15 shows the consequences of trying to lock it by gradients.
3. *Proportional retention*: when `a_new > 0`, the relative proportions among
   older experts are preserved exactly; only a global `(1 - a_new)` factor is
   applied. "New expert takes mass" and "old routing is untouched" are no
   longer in conflict.
4. *No renormalization artifact*: nothing needs to be re-locked after
   expansion, so `lock_historical_routing`, the backward hooks and the
   zero-weight-decay router group are deleted from the v2 path.
5. *Degenerate check*: with `b_k <= -30` the routing distribution is unchanged
   to `atol=1e-6`, which is the unit-test form of the invariance claim.

The newest gate is outermost, so a new expert can take mass from every older
one. That is also the recency-funnel risk (fact 2). It is controlled by three
independent mechanisms, not by a lock: the candidate filter (section 4), the
gate's negative boundary `L_gate` (section 6), and the function anchor
`L_func`, which holds even if the gate does fire.

#### 2.3.2 Top-k

`p` is dense; top-k takes the k largest and renormalizes, exactly as v1 does.
Default `top_k=1`: fact 13 measured that top-2 raises accuracy slightly but
more than doubles forgetting.

#### 2.3.3 Frozen feature lift (optional, default on)

A gate linear in `z_s` can only carve halfspace regions. Following RanPAC
(frozen random projection + linear readout), the default gate input is

```
lift(z) = concat( z, relu(R z) )        R: [d_lift, d], d_lift = 256
```

with `R` drawn once from a fixed seed, frozen, and *not stored* (regenerated
from `--v2_lift_seed`; 0 stored bytes). `--v2_lift_dim 0` gives a plain linear
gate. This is a component, not a claim.

### 2.4 Global cosine classifier

```python
z_mix = z_s + sum_i p_i * A_i(z_s)          # [B, d]; note sum_i p_i = 1
z_hat = F.normalize(z_mix, dim=-1)
W_hat = F.normalize(W, dim=-1)              # W: [C_max, d]
logits = scale * (z_hat @ W_hat.t()) + bias  # scale: learnable, init 10.0
```

| | v1 | v2 |
| :-- | :-- | :-- |
| classifier params at C=100, d=768 | 20 x 51,300 = 1.03 M (in heads) | 76,800 |
| per-expert heads | yes | no |
| class growth | n/a | append rows to `W`/`bias` |

Growth policy: new classes append rows; **old rows are frozen by default**
(`--v2_cls_freeze_old true`). Protection for the old rows is `L_func` (section
6), not weight-space regularization. Ablation: unfrozen old rows + `L_func`.

`evaluation/heads.py` (`NCMHead`, `BiasCorrectionHead`) keeps working on top of
the logits; the bias-correction head needs stored exemplars, so under the
default zero-exemplar memory only `NCMHead` is unavailable and the plain
logits (or a class-prior correction computed from the coverage stats) are
used. This is an ablation item, not a blocker.

### 2.5 Forward pass (reference shapes, d=768, N=20, C=100, r=32, B=128)

```
1  x                [128, 3, 224, 224]
2  z_s = phi_0(x)   [128, 768]                 frozen, from cache
3  lift             [128, 768+256]
4  a = sigmoid(...) [128, N-1]                 per gate head, bias-shifted
5  p(e|z)           [128, N]                   nested residual product
6  idx, g = topk    [128, k], [128, k]         k=1 default
7  z_mix = z_s + sum_{i in idx} g_i A_i(z_s)   [128, 768]
8  logits           [128, C]                   cosine + scale + bias
9  loss             scalar
```

Steps 4-6 touch no expert parameters. Steps 7-8 touch only the selected
adapters (one adapter at k=1) and the classifier. Per-sample active
parameters: 49,152 (adapter) + 76,800 (classifier) ~= 126 K plus ~19 K of
gates, vs 682 K in v1.

### 2.6 Parameter ledger (reference geometry)

| | v1 (measured) | v2 |
| :-- | --: | --: |
| total params | 13.36 M | 1.08 M |
| trainable at any time | 682 K | ~55 K (one adapter + new rows + one gate) |
| active per sample | 682 K | ~126 K (adapter + classifier) + ~19 K (gates) |
| memory at P=536 | 14.23 MB | 1.76 MB |

These are targets to be measured and reported, not estimates to be claimed.

---

## 3. Memory: `DistributionMemory`

### 3.1 Format

Per prototype (the only stored per-sample state):

| field | shape | dtype | bytes | role |
| :-- | :-- | :-- | --: | :-- |
| `v_p` | `[d]` | fp32 | 3,072 | anchor point for `L_func`, negative for `L_gate` |
| `p_anchor` | `[C_max]` | fp16 | 200 | old function output `softmax(logits_t(z_p))`, non-zero only on classes seen at registration |
| `task_id` | scalar | int16 | 2 | reporting, per-task routing matrix |
| `label` | scalar | int16 | 2 | reporting, per-class coverage |
| `count` | scalar | int32 | 4 | EMA weight, coverage |
| **total** | | | **3,280** | |

Per expert (coverage statistics):

| field | shape | dtype | bytes | role |
| :-- | :-- | :-- | --: | :-- |
| `mu_i` | `[d]` | fp32 | 3,072 | expert centroid in `z_s` |
| `log_var_i` | `[d]` | fp32 | 3,072 | per-dimension variance, EMA |
| `n_i` | scalar | int64 | 8 | effective sample count |
| `q_i` | scalar | fp32 | 4 | running 90th percentile of `d_i` on own samples (scale reference) |

Total: 3.28 KB/prototype + 6.15 KB/expert. At P=536, N=20: **1.76 MB +
0.12 MB = 1.88 MB**, a 7.6x reduction against the measured v1 store.
`p_anchor` is fp16; zero entries contribute nothing to `L_func`
(cross-entropy against a probability target), so fp16 rounding is harmless.

### 3.2 What is deleted, and why it is safe

| v1 field | v2 | why safe |
| :-- | :-- | :-- |
| `o_p [N, C]` | `p_anchor [C]` | the global classifier replaces per-expert heads; the anchor is now the *model's* output, 30% of v1 bytes at the measured geometry |
| `x_p [k, d]` | dropped by default | fact 11: latent exemplars add 1.4 points on CIFAR-10 and are *negative* on CIFAR-100/ViT; fact 12: calibration (their other use) is redundant with distillation. Ablation `--v2_exemplars k` re-adds them at fp16 |
| `r_p [N]` | **derived** | gates are frozen after their expert's creation window and the stack only grows by appending, so `p(e\|z_p)` is recomputable exactly at any later time. Storing it would store a value the model can recompute |
| `owner_expert` | deleted | hard assignment is the thing that forced `task -> expert`; soft responsibility is `p(e\|z_p)` |
| `s_p [C]` | deleted | the generalist pathway is the identity adapter, i.e. `z_s` itself; `shared_expert` / `shared_gate` are deprecated |
| `raw_x` | deleted | frozen encoder: no representation refresh exists |
| exemplar `y_p` | deleted | no exemplar CE; `p_anchor` is the label |

`refresh_representations`, `refresh_anchors`, `get_routing_matrix`,
`get_expert_anchor_matrix`, `get_output_matrix`, `get_shared_anchor_matrix`
and the padding/rescaling logic all disappear with them.

### 3.3 Soft expert responsibility

Responsibility is not stored; it is a *query*:

```python
def responsibility(z_s, task_id=None) -> Tensor:   # [P, N] or [N]
    p = router.dense(z_s)                          # nested residual gates
    if task_id is not None: p = p[memory.task_id == task_id]
    return p.mean(0)
```

Used by: coverage updates (soft assignment weights), the merge criterion,
the reporting metric `task x expert` responsibility matrix, and the reuse
branch of the allocation controller. This directly replaces the
`owner_expert` hard target and the owner CE distillation of
`_distill_router_anchors`.

### 3.4 Byte accounting

`memory_bytes()` must count the actual dtypes (the v1 estimator assumes 4
bytes/float; v2 stores fp16 `p_anchor` and, in the ablation, fp16 `x_p`). The
equal-byte protocol in `BENCHMARK.md` depends on this being exact. A unit test
asserts `sum(dtype_size(field)) == memory_bytes()`.

---

## 4. Allocation controller

Input: a window `B_t` (task-aware: the first `W` samples of the new task;
task-free: the next fixed-length window). Output: `REUSE` or `EXPAND`.

### 4.1 Candidate filter (cheap, runs always)

Feature-space coverage is **not** label-space competence: a new class can sit
inside the existing feature support and still be badly separated by the
current classifier. So the cheap stage is a *filter that decides whether the
probe is worth running*, not a REUSE decision.

Two signals, both label-free and inference-only on cached features:

```
d_i(z) = sum_j (z_j - mu_ij)^2 / (var_ij + eps)      # diagonal Mahalanobis
nu(z)  = min_i d_i(z) / q_i                          # scale-free coverage deficit
nu_t   = median_{z in B_t} nu(z)

margin(z) = top1_logit(z) - top2_logit(z)            # decision-boundary distance
s_t       = mean_{z in B_t} sigmoid(-margin(z))      # functional surprise
```

`var_ij` is shrunk toward the global per-dimension variance:
`var = (1-rho) * var_i + rho * var_bar`, `rho = 0.1`. A diagonal covariance
avoids a `d x d` per-expert matrix (2.4 MB each at d=768); a shared low-rank
whitening (RanPAC-style PCA on the first task's features) is an optional
extension, not a default.

Decision table:

| `nu_t` | `s_t` | action |
| :-- | :-- | :-- |
| low | low | `REUSE` (probe skipped) |
| low | high | `PROBE` |
| high | any | `PROBE` |

Defaults `tau_cov = 2.0`, `tau_surprise = 0.5`. The margin-based surprise is
used instead of prediction confidence on purpose: a new class wedged between
two old classes is predicted *confidently wrong* by the frozen old rows, so
confidence would report low surprise exactly where the probe is needed.

This filter is a cost/recall trade-off, and it is reported as such: E2 must
log the **filter miss rate** = fraction of windows where the filter says
`REUSE` but the probe would have chosen `EXPAND`. For a task-free stream,
require the trigger on `K=2` consecutive windows before probing (hysteresis;
prevents single-batch expert explosion).

### 4.2 Counterfactual probe (runs only when the filter fires)

```
REUSE branch   : unfreeze the adapter of the top-1 expert by aggregate
                 responsibility on B_t. Old gates stay frozen. Train on B_t.
EXPAND branch  : add a fresh ResidualAdapter(r) and a gate head; fit and
                 calibrate the gate (2.3.1); train on B_t.
```

**Old gates are never unfrozen.** The entire routing-stability argument rests
on the frozen gate stack; re-opening a historical gate in the REUSE branch
would invalidate it and re-introduce the drift that `L_router_stab` existed to
patch in v1. Routing changes only through EXPAND, and only within the
calibrated bound of section 2.3.1.

**Compute parity is mandatory, including the gate fit.** The EXPAND branch
spends `F_gate` FLOPs on the logistic fit plus `K` adapter steps; REUSE spends
`K` adapter steps. The budget is therefore fixed in FLOPs:

```
F_budget = K * step_flops + F_gate
REUSE gets  K + ceil(F_gate / step_flops)  adapter steps
EXPAND gets K adapter steps + the gate fit
```

At the reference geometry the gate fit costs ~8 adapter steps
(512 sampled rows x 1024 lifted dims x 100 steps vs 128 x 768 x 32 x 2), so
parity costs ~4% of the probe budget. Without this, EXPAND receives free
optimization budget and a reviewer will flag it as a confound.

Also matched: trainable parameter count (one rank-`r` adapter + one gate in
EXPAND; one rank-`r` adapter in REUSE), batch order, and seed.

Utility:

```
J = acc_new - lambda_f * forget_anchor - lambda_m * dbytes/B - lambda_c * dflops/F

acc_new       : accuracy on a held-out slice of B_t (not used for training)
forget_anchor : mean KL( p_anchor[p] || softmax(logits(z_p)) ) over stored
                prototypes, minus its value before the probe
dbytes        : parameter bytes added by the branch
dflops        : extra forward FLOPs per sample
```

Decision: `EXPAND` iff `J_expand > J_reuse + eps` with `eps = 0.01`
(accuracy units). To reduce probe noise, run `S=2` data orders and compare the
means; `S` and `eps` are config knobs and the sensitivity is an ablation.

**Honest framing**: this is a greedy one-step lookahead under a stochastic
probe, not a causal counterfactual estimate. It can under-expand on a hard
task whose learnability only shows after more steps. That is exactly what the
ablation in section 9 E2 measures (always-reuse / always-expand / coverage-only
/ probe).

### 4.3 Why the probe is affordable

With the frozen feature cache, a probe step is an adapter forward/backward on
`[128, 768]`, no conv, no encoder. `K=200`, `S=2`, two branches = 800 adapter
steps per expansion decision. The v1 `ExpertBuilder` already spends more than
this on candidate training plus gate validation (3 epochs + validation).

### 4.4 Gate fit and calibration (the logistic probe)

`w_k, b_k` are fit by logistic regression on
`{lift(z) : z in B_t} -> 1` vs `{lift(z_p) : p in memory} -> 0`, with both
sides capped at 256 sampled rows (the full window would make the fit cost more
than the probe itself), class-weighted, `L2=1e-3`, 100 steps, full-batch. Then
the bias is calibrated per section 2.3.1:

```
t* = min over protected anchors of (m_p - delta) / (1 + m_p - delta)
t  = min(eps0, t*)
b_k = logit(t) - max over stored prototypes of (w_k . lift(z_p))
```

which may move the bias either way. This gives the gate a working direction on
the new region (escaping the `w, b` saddle that a zero-initialized
multiplicative gate would create) while preserving the historical top-1 on the
protected anchors. The fitted separability is also the discriminative coverage
readout logged for the allocation decision.

---

## 5. Lifecycle state machine

Replaces `ContinualTrainer.train_task` (494 lines, `adaptation/ttt.py:358`).

```
OBSERVE      window B_t from the stream (task-aware or task-free)
ENCODE       z_s = phi_0(x)                      (cache hit in the normal path)
COVERAGE     update/query expert stats, nu_t, s_t
ALLOCATE     candidate filter -> probe -> REUSE | EXPAND
ADAPT        train: classifier new rows, gate, selected adapter(s)
             losses: L_task + L_gate + L_func  (+ optional L_balance)
PROTECT      freeze the adapter and the gate; append p_anchor to memory
MEMORY       register prototypes (distance-thresholded, EMA), update coverage
OPTIMIZE     optional merge / compression under the byte budget
EVALUATE     stream metrics, routing retention, responsibility matrix
```

Module map:

| state | component | new file |
| :-- | :-- | :-- |
| OBSERVE | stream/window iterator | `pal_moe/data/` (reuse `split_folder`, `domain_shift`) |
| ENCODE | frozen encoder + cache | `pal_moe/models/encoder.py`, `pal_moe/data/feature_cache.py` (reuse) |
| COVERAGE | `ExpertCoverageStats` | `pal_moe/v2/coverage.py` |
| ALLOCATE | `AllocationController` | `pal_moe/v2/allocation.py` |
| ADAPT | `AdapterBank`, `ResidualGateRouter`, `GrowingCosineClassifier`, losses | `pal_moe/v2/{adapter,router,classifier,losses}.py` |
| PROTECT / MEMORY | `DistributionMemory` | `pal_moe/v2/memory.py` |
| OPTIMIZE | `function_space_merge` | `pal_moe/v2/merge.py` |
| EVALUATE | existing metrics + new ones | `pal_moe/evaluation/` (extend) |
| orchestration | `ContinualLearnerV2.run_window` | `pal_moe/v2/lifecycle.py` |

---

## 6. Losses

Core objective - three terms:

```
L = L_task + lambda_g * L_gate + lambda_f * L_func
```

Defaults: `lambda_g = 1.0`, `lambda_f = 1.0`.

`L_balance` is **not** in the core. It is an opt-in ablation
(`--v2_use_balance`, default off) and a diagnostic, for the reasons in 6.5.

### 6.1 `L_task`

`CE(logits(z_mix), y)` on the current window. Nothing else is needed for the
new classes; old rows of `W` are frozen.

### 6.2 `L_gate` (negative boundary, on the gate)

For the newest gate `a_new`:

```
L_gate = BCE(a_new(lift(z_p)), 0) over all stored prototypes      (negative)
       + BCE(a_new(lift(z)),   1) over the current window          (positive)
```

This is v1's OOD negative-boundary mechanism (facts 1, 2, 12) moved from the
expert's logits onto the routing variable, where it is directly interpretable:
"the new expert must not claim old regions". It replaces the entropy/energy
OOD losses, `ood_mode`, `ood_margin`, `lambda_ood` and the periodic
`--ood_every` machinery.

### 6.3 `L_func` (function anchor)

```
L_func = CE( p_anchor[p], softmax(logits(z_p)) )    over stored prototypes
```

`p_anchor` is a proper distribution over the classes seen at registration, so
mass leaking to later classes *increases* this term automatically: the
negative boundary on the *prediction* side comes for free. This single term
replaces: expert-output MSE stability (`lambda_e`), shared-expert anchoring,
LwF snapshot distillation, EMA representation distillation, and the
joint-calibration CE (fact 12: distillation is the dominant mechanism; the
others are redundant or negative).

### 6.4 Collapse diagnostic, not a balance loss

The recency funnel (fact 2) is already controlled by `L_gate`: the negative BCE
`a_new(z_p) -> 0` directly penalizes the new expert for claiming old regions.
Switch-style load balancing is a *different objective* - uniform token
distribution over a fixed pool - and it conflicts with the thesis of v2, which
is deliberate specialization. If a task is genuinely explained by one expert
(`p(E7|z) = 0.91`), that is the intended outcome, and `L_balance` would
penalize it. Worse, it would push the router to spread the new task's data
across old experts - the opposite of what the allocation controller just
decided - so the training objective would fight the allocation decision.

Instead of a loss, v2 reports a **collapse diagnostic** per task:

```
utilization entropy  H(mean_task p(e|z)) / log N
max_expert_share     max_e mean_task p(e|z)
newest_share         mean_task p(newest|z)
```

Flag a run when `max_expert_share > 0.8` across *all* tasks (a single expert
absorbing the whole stream) - that is the pathology worth catching. A single
task with `newest_share ~= 1` is normal specialization, not collapse.

`--v2_use_balance` re-adds the Switch term as an ablation for E2/E3. It is
expected to lose; the point of running it is to have the measurement.

### 6.5 Dropped from the v2 core

| v1 term | reason |
| :-- | :-- |
| `L_router_stab` = KL(g_old \|\| g_new) | routing is invariant by construction; there is no `g_old` to drift from |
| owner CE distillation (`--router_anchor_steps`) | no owner; `L_gate` supplies the cross-task routing signal |
| `L_ood` (entropy/energy) | subsumed by `L_gate` + `L_func` |
| `L_enc` (encoder stability) | frozen encoder |
| `L_ema` | no EMA encoder |
| `L_lwf` | subsumed by `L_func` |
| `L_replay`, `L_gen` | no exemplars, no generator in the core (ablation knobs only) |
| `L_sep` (expert separation margin) | redundant with `L_gate`; was considered and dropped |
| `L_proto` (prototype geometry) | the frozen representation + distance-thresholded registration already control this |

Any of these can be re-added as an ablation flag, but none is in the default
path.

---

## 7. Function-space merge

v1 merges by parameter averaging (`models/moe.py:383`) after picking the pair
by cosine similarity of `fc2.weight` (`builder/expert_builder.py:489`). Both
steps are wrong for nonlinear modules: weight-space proximity is not
function-space proximity, and `f((t1+t2)/2) != (f(t1)+f(t2))/2`.

### 7.1 Candidate selection (function space)

On a fixed anchor set `Z_a` (stored prototypes, subsampled to 256):

```
S_ij = mean_{z in Z_a, g_i+g_j>0} cos( A_i(z), A_j(z) )
D_ij = mean_{z in Z_a} KL( softmax(cls(z + A_i(z))) || softmax(cls(z + A_j(z))) )
```

Merge candidate iff `S_ij > tau_s` (default 0.8) and `D_ij < tau_d` (default
0.05). The v1 "most similar pair" heuristic is replaced by this test; the
byte budget decides *whether* to merge, the test decides *what* to merge.

### 7.2 Merge target

The correct target is the *responsibility-weighted* mixture contribution, not
the individual adapter. Experts `i` and `j` contribute `p(i)A_i + p(j)A_j`;
after merging into a slot `m` with `p(m) = p(i)+p(j)` the contribution must be
`p(m)A_m`, so

```
omega(z) = (1 - a_j(z)) a_i(z) / ( (1 - a_j(z)) a_i(z) + a_j(z) )
A_m*(z)  = omega(z) A_i(z) + (1 - omega(z)) A_j(z)
```

(the `prod_{k>j}(1-a_k)` factors cancel in the ratio; the remaining scale is
handled by the gate merge in 7.3). Fit a student adapter `A_m` (rank `<= r_i +
r_j`, default `r`) by `K=300` Adam steps minimizing
`|| A_m(z) - A_m*(z) ||^2` on `Z_a` plus a
`CE(logits(z + A_m(z)), y)` term on the anchors' labels.

### 7.3 Gate merge

For **adjacent** experts in creation order (`j = i+1`) the combined mass has a
closed form:

```
a_i'*(z) = (1 - a_j(z)) * a_i(z) + a_j(z)        and
1 - a_i'* = (1 - a_j(z)) * (1 - a_i(z))          (remaining mass unchanged)
```

This is the exact target for the merged gate, but it is not a sigmoid of a
linear function in general, so it is a regression target, not a copy:

- when `a_j ~= 0` wherever `a_i > 0` (the similar-expert case the candidate
  test selects for), `a_i'* ~= a_i`, so gate `i` is kept unchanged and the
  refit is a no-op;
- otherwise fit one gate head to `a_i'*` on the anchors (100 Adam steps, BCE
  against the soft target). If the fit fails the accept/reject test, fall
  back to *merge the adapters, keep both gates*: routing stays exact, the
  function is approximated, and one adapter (49 K params) is still saved.

For non-adjacent `i < j`, the combined mass depends on the intervening gates;
it is exact only when they are `~0` on the union region. Check
`max_z a_l(z) < 1e-3` for `i < l < j` on the anchors and reject otherwise.
This is a real constraint on merge order, and it is why the byte-budget
optimizer should prefer adjacent pairs.

### 7.4 Accept / reject

After the merge, re-evaluate on the anchors: commit iff
`dAcc > -epsilon` and `dForget < epsilon` (defaults `1e-3`), else roll back.
Parameter averaging is never used, and there is no "merge because the pool is
full" path: the byte budget triggers *candidate generation*, not a forced
merge.

---

## 8. Configuration surface

New flags (all with defaults, all added to `pal_moe/config.py::_RANGES`
where numeric - note that unknown keys are hard errors in that module):

```
--method palmoe_v2                 # new method id in run_benchmark.py
--v2_rank 32
--v2_lift_dim 256
--v2_lift_seed 0
--v2_tau_cov 2.0
--v2_tau_surprise 0.5
--v2_cov_windows 2
--v2_probe_steps 200
--v2_probe_repeats 2
--v2_probe_eps 0.01
--v2_probe_lambda_f 1.0
--v2_probe_lambda_m 1.0
--v2_probe_lambda_c 1.0
--v2_lambda_gate 1.0
--v2_lambda_func 1.0
--v2_use_balance false            # ablation only (6.4)
--v2_cls_scale 10.0
--v2_cls_freeze_old true
--v2_exemplars 0                   # ablation: >0 re-adds latent exemplars
--v2_memory_bytes -1               # budget B; -1 = unlimited
--v2_merge off                     # off | adjacent | any
--v2_merge_tau_s 0.8
--v2_merge_tau_d 0.05
```

Configs live in `configs/v2_*.json` and are validated by the existing strict
loader.

---

## 9. File map

### 9.1 New (additive, `pal_moe/v2/`)

| file | contents |
| :-- | :-- |
| `pal_moe/v2/__init__.py` | public surface: `PALMoEv2`, `ContinualLearnerV2` |
| `pal_moe/v2/adapter.py` | `ResidualAdapter`, `AdapterBank` (add/freeze/params/bytes) |
| `pal_moe/v2/router.py` | `ResidualGateRouter`: gate stack, `dense()`, `add_gate()`, `merge_adjacent()`, `retention()` |
| `pal_moe/v2/classifier.py` | `GrowingCosineClassifier`: `grow(n_new)`, freeze policy, bytes |
| `pal_moe/v2/coverage.py` | `ExpertCoverageStats`: update, `mahalanobis`, `deficit`, shrinkage |
| `pal_moe/v2/allocation.py` | `AllocationController`: candidate filter, logistic gate fit + top-1-safe calibration, counterfactual probe |
| `pal_moe/v2/memory.py` | `DistributionMemory`: prototype store, `p_anchor`, byte accounting |
| `pal_moe/v2/losses.py` | `l_task`, `l_gate`, `l_func`, `l_balance` |
| `pal_moe/v2/merge.py` | `similarity`, `merge_target`, `fit_student`, `merge_adjacent` |
| `pal_moe/v2/model.py` | `PALMoEv2` (encoder + router + bank + classifier, forward) |
| `pal_moe/v2/lifecycle.py` | `ContinualLearnerV2.run_window` state machine |
| `tests/test_v2.py` | invariance, parity, byte accounting, merge exactness (section 10 E6) |

### 9.2 Reused unchanged

| file | role |
| :-- | :-- |
| `pal_moe/models/encoder.py` | `SharedEncoder` (ResNet/ViT ImageNet weights), frozen |
| `pal_moe/data/feature_cache.py` | `CachedFeatureEncoder`, feature cache |
| `pal_moe/data/*` | stream splitters |
| `pal_moe/evaluation/*` | metrics, diagnostics, geometry, task-free evaluator |
| `pal_moe/baselines/*` | all baselines and the equal-byte protocol |
| `pal_moe/persistence.py` | extend with a v2 state-dict branch |
| `experiments/run_benchmark.py` | add `palmoe_v2` to `method_keys` and one method block |
| `experiments/run_benchmark_multi.py`, `experiments/recipes/*` | reuse as-is |

### 9.3 v1 files that v2 bypasses (not deleted)

| v1 file | v2 replacement | note |
| :-- | :-- | :-- |
| `models/expert.py::MLPExpert` | `v2/adapter.py` | keep for v1 and baselines |
| `models/router.py` (3 routers) | `v2/router.py` | keep for v1 |
| `models/moe.py::DynamicMoE` | `v2/model.py` | keep for v1 |
| `memory/prototype_memory.py` | `v2/memory.py` | keep for v1 |
| `trigger/expert_trigger.py` | `v2/allocation.py` | keep for v1 |
| `builder/expert_builder.py` | probe + accept in `v2/allocation.py` | keep for v1 |
| `adaptation/ttt.py::ContinualTrainer` | `v2/lifecycle.py` | keep for v1 |
| `merge.py` (soup/TIES/task-arithmetic) | `v2/merge.py` | keep as a serving toolbox |

Deletion of the v1 paths happens only after the paper's v1 tables are frozen.

---

## 10. Experiments

### E0 - prerequisite: representation ceiling and adapter headroom

Purpose: decide whether the *premise* holds - frozen pretrained
representation + small residual adapters - before any v2 package code is
written. E0 is a standalone script (`experiments/e0_representation_ceiling.py`)
on the cached features, not part of `pal_moe/v2/`. E0 and M1 must not be
developed in parallel: if the headroom is small, the rank and adapter
decisions change.

- **E0a - provenance.** The v1 CIFAR-100/ViT row was produced at commit
  `82b5684`; `git diff 82b5684..HEAD` on `pal_moe/` and `experiments/` is
  cosmetic for this recipe (black formatting, one print, one result key, one
  method-name string, ResNet identity head). Confirm with a 1-seed re-run at
  HEAD rather than a 3-seed repeat: PAL-MoE + iCaRL only, same config.
- **E0b - representation ceiling.** Joint linear/cosine probe on all 100
  classes of the frozen ViT features. This is the absolute upper bound of the
  frozen space and the number that decides whether adapters have room.
- **E0c - zero-adapter CL reference.** Class-incremental cosine classifier,
  frozen old rows, no adapters, prototype registration. This is the M4 red
  line measured standalone.
- **E0d - adapter headroom.** Per-task residual adapters, always-expand, old
  adapters frozen, `L_func` only. Rank sweep `r in {0, 8, 32, 64}`; `r=0` must
  equal E0c. The gap `E0d(r) - E0c` is the marginal value of the adapter
  architecture, i.e. whether v2 has anything to allocate at all.

Do not tune v2 on the conv encoder: 15.65% average accuracy on a 100-class
problem is not a signal.

### E1 - v2 end-to-end

CIFAR-100 20-task, frozen ViT-B/16, feature cache, 3 seeds, `palmoe_v2`
defaults. Report: accuracy, forgetting, experts/task, reuse rate, routing
retention, `memory_bytes`, `total_params`, `active_params`, `fit_seconds`,
`task x expert` responsibility matrix.

Success criterion: **Pareto-dominates iCaRL at its own budget** (>= 64.94%
accuracy with <= 10.66 MB `stored_bytes` and <= 0.67 M trainable params), or
dominates the v1 row (59.34% / 18.64% / 14.23 MB) at a strictly smaller
budget.

### E2 - allocation ablation (the killer experiment)

Four policies on the same stream: (a) always reuse, (b) always expand
(v1-like), (c) candidate filter only, (d) full counterfactual probe. Also
reported: the **filter miss rate** (windows where the filter says REUSE but
the probe would have chosen EXPAND) and the `--v2_use_balance` cell. Report
the full metric set per task. This isolates whether the counterfactual
decision adds anything over the filter and over forced expansion. If (d) does
not beat (c) and (b) on the Pareto front, the allocation controller is not the
contribution and should be reported as a negative result.

### E3 - budget sweep

`--v2_memory_bytes` in {1, 2, 4, 8, 16} MB x `--v2_rank` in {8, 16, 32, 64}.
Produce accuracy/MB and forgetting/MB curves, and accuracy vs trainable
params. This is the paper's headline figure.

### E4 - merge on/off

`--v2_merge {off, adjacent, any}` at a fixed byte budget. Verifies the
adjacency exactness claim (E6) end-to-end and measures whether merge buys
accuracy per byte or only shrinks the pool.

### E5 - task-free / online

Tiny-ImageNet folder stream (`pal_moe.data.split_folder`; the runner does not
expose it yet - `EXPERIMENT_PLAN.md` item M3), window hysteresis on. Report
online accuracy, recent accuracy, surprise, and expert count over time
(`evaluation/task_free.py`).

### E6 - invariance and exactness tests (unit, no training)

1. Add a gate with `b_k <= -30`: `router.dense(z)` is unchanged to
   `atol=1e-6` before and after, for a random `z` batch.
2. `sum_e p(e|z) == 1` for random gates.
3. `A_i == 0` at init: `z_mix == z_s` exactly.
4. Expansion safety: with a probe-fitted and calibrated gate, the **top-1
   routing is preserved on every protected anchor**:
   `argmax p_before(z_p) == argmax p_after(z_p)`, and
   `max_p a_new(z_p) <= eps0`. Anchors with `m_p <= delta` are excluded and
   reported separately.
5. Adjacent merge with `a_j = 0` on the anchors: `p_after == p_before`
   (`torch.equal`); with `a_j > 0`, `p_after` matches the closed-form target
   `a_i'*` to numerical tolerance.
6. `memory_bytes()` equals the sum of the stored dtypes.
7. Zero-exemplar parity: with `top_k=1` and one expert, `logits == frozen
   linear probe` (cosine classifier on `z_s`) to numerical tolerance.

### E7 - routing retention (the metric v1 could not have)

On a fixed probe set, record `p(e|z)` after every task. `RR_t` = mean total
variation between `p_t` and `p_{t-1}` restricted to experts alive at `t-1`.
v2 predicts `RR_t ~= 0` except on the new task's own region; v1 measured
routing capture (fact 2). Report per task.

---

## 11. Risks and open questions

1. **Frozen backbone ceiling.** v2's premise is that the representation is
   strong enough that adapters can close the gap. If the ViT features are
   already saturated, `L_task` has no headroom and adapters will not help.
   Mitigation: E0, plus the rank sweep (E3).
2. **Filter recall.** The candidate filter is a cost/recall trade-off and can
   say REUSE when the probe would have expanded. Mitigation: the margin-based
   surprise signal, `K=2` hysteresis, and the measured **filter miss rate** in
   E2. If the miss rate is high, the filter's thresholds are wrong, not the
   probe.
3. **Recency funnel in the nested stack.** The newest gate is outermost.
   Mitigation: `L_gate`, the candidate filter, and `L_func`; E7 measures it.
   The collapse diagnostic (6.4) is the tripwire, not a balance loss.
4. **Frozen old classifier rows.** If adapters move `z` for old classes, the
   frozen rows may miscalibrate. Mitigation: `L_func`; ablation with unfrozen
   old rows; the existing `BiasCorrectionHead` (needs exemplars, so it is
   tied to `--v2_exemplars`).
5. **Zero-exemplar read-out.** Without `x_p` there is no NCM head and no
   bias-correction prior. Ablation: `--v2_exemplars {0,2,5}`.
6. **Diagonal covariance may under-model coverage.** Shrinkage plus the
   optional PCA whitening; E3 measures the effect on the expert count.
7. **Merge order constraint.** Only adjacent merges are exact. The optimizer
   must respect it; `--v2_merge any` is the unsafe mode and must show a
   measured loss.
8. **Equal-byte honesty.** v2 must be compared at equal `stored_bytes`,
   including `p_anchor` and coverage stats. The v1 protocol already exists;
   do not invent a second one.
9. **v1 freeze.** No v1 file may change behaviour while the paper tables are
   being produced. Enforce by keeping v2 in its own package and by running the
   existing 105 tests plus a v1 smoke before/after each v2 commit.

---

## 12. Milestones

Each milestone has a guard test; none of them touches v1 behaviour.

| # | deliverable | guard |
| :-- | :-- | :-- |
| M1 | `v2/adapter.py`, `v2/router.py` | E6.1-E6.5 (invariance, normalization, expansion safety, exact merge) |
| M2 | `v2/classifier.py`, `v2/model.py` | E6.3, E6.7 (zero-adapter parity with a linear probe) |
| M3 | `v2/memory.py`, `v2/coverage.py` | E6.6 (byte accounting), coverage sanity on cached features |
| M4 | `v2/lifecycle.py` reuse-only path | **red line**: adapters disabled -> exactly the E0c frozen linear probe; `rank=0` -> identical logits; then `r in {8,16,32,64}` to measure the real adapter gain |
| M5 | expansion + candidate filter (no probe) | E2(b,c) |
| M6 | `v2/allocation.py` counterfactual probe | E2(d), E1 |
| M7 | `v2/merge.py` | E4, E6.5 |
| M8 | task-free windows + hysteresis | E5 |
| M9 | budget sweep, paper figures | E3, E7 |

M4 is the critical sanity gate: if the reuse-only path cannot match a linear
probe on frozen features, the classifier/adapter design is wrong and no
allocation policy will rescue it.

---

## Appendix A - notation

| symbol | meaning | shape |
| :-- | :-- | :-- |
| `z_s` | stable routing representation | `[B, d]` |
| `z_mix` | plastic representation entering the classifier | `[B, d]` |
| `A_i` | expert `i` residual adapter | `[B, d]` |
| `a_k` | scalar gate for expert `k >= 1` | `[B, 1]` |
| `p(e\|z)` | routing distribution (nested residual product) | `[B, N]` |
| `W`, `bias` | global classifier | `[C_max, d]`, `[C_max]` |
| `v_p` | prototype anchor feature | `[d]` |
| `p_anchor` | stored old-model output at `v_p` | `[C_max]` |
| `mu_i`, `var_i` | expert coverage statistics | `[d]`, `[d]` |
| `nu` | coverage deficit | `[B]` |
| `r` | adapter rank | scalar |

## Appendix B - what v1 code the spec retires (from the default path)

`QuantitativeTrigger`, `AlwaysTrigger` as the allocation policy,
`owner_expert`, `get_expert_anchor_matrix` (one-hot branch), the owner CE
distillation, `lock_historical_routing` + backward hooks + the zero-decay
router group, `_project_to_null_space` / `null_space_basis`, the shared
softmax expansion, `shared_expert` / `shared_gate` / `s_p`, `o_p`, `r_p`,
`x_p` (default), `raw_x`, `EMAEncoder`, `refresh_representations`,
`refresh_anchors`, `compute_stability_losses` (router/expert/encoder),
`energy_boundary_loss`, `enforce_capacity_control` (weight-similarity merge +
parameter averaging), `clone_function_preserving` (no parent to clone),
`widen`, `TestTimeAdapter` (re-add as an ablation on top of the v2 model if
needed).

Kept: the encoder zoo, the feature cache, the benchmark harness and baselines,
the equal-byte accounting, the result schema, the metrics, the checkpointing,
and the test suite.
