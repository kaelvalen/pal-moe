# P2-BOUND: why the expert bank adds < 1 pp over a ridge router, and whether confusion-aligned experts change it - pre-registration

Status: **proposed, not started.** Written during the v3 restructure (phase 5) and
committed before any of its runs. It follows `E_TID2_RESULTS.md`, whose P2 contrast
(+0.62 pp `coherent`, -0.03 pp `dispersed`) is the thing to be explained, and it is
the pre-registration that gates the experimental `by_confusion` consolidation policy
(`pal_moe/experts/policies.py`: `consolidate("by_confusion")` refuses to run unless
this document is named).

## 1. The questions

**Part A (mechanism).** Once routing is fixed by the continual ridge router, the bank
can only change an answer in two ways: *rescue* a sample whose ridge class is wrong but
whose routed task is right (the owner expert then decodes it), or *break* a sample
whose ridge class was right. Is P2 small because there is little rescuable mass, or
because the bank rescues poorly?

**Part B (intervention).** If the mass is the binding term, expert boundaries should
follow the router's confusion structure instead of the arrival order of the data. Does
a `by_confusion` consolidation of the same data beat `by_arrival`, and does it reach
what the known semantic grouping (`coherent`, one expert per superclass) reaches?

## 2. What is already known (and therefore not a test)

The rescuable mass is readable from stored aggregates, because a right class implies a
right task (the owner of the true class is the true task) and CIFAR-100 test tasks are
equal-sized:

```text
m = P(class wrong, task right) = ridge C@1 - ridge_alone accuracy
coherent    0.8616 - 0.7677 = 0.0939   (9.4 %)
dispersed   0.7745 - 0.7677 = 0.0068   (0.7 %)
```

These two numbers were computed by the author before this document from
`results/e_tid2/e_tid2_ridge_router.json`. Part A therefore does **not** test `m`; it
measures the terms `m` multiplies.

## 3. Part A design: per-sample 2x2 decomposition of E-TID2

Re-run the 12 E-TID2 cells (2 regimes x seeds 42, 1, 2, 3, 4, 5) with
`experiments/e_tid2_ridge_router.py`'s exact code path (float32 ridge, the stored
recipe), additionally recording per test sample: `r` = ridge class correct,
`tau` = ridge-routed task correct, `s` = `ridge_routed` system correct. No computation
changes; only outputs are added.

The decomposition is an identity (a guard, not a finding):

```text
P2 = P(s) - P(r)
   = m * rho  +  P(not r, not tau) * rho'  -  P(r) * beta

m    = P(not r, tau)          rescuable mass
rho  = P(s | not r, tau)      rescue rate of the owner expert
rho' = P(s | not r, not tau)  wrong-expert luck (expected ~0)
beta = P(not s | r)           break rate
```

**Endpoints (per cell, then paired over seeds within a regime):** `m`, `rho`, `rho'`,
`beta`, the rescue contribution `m * rho`, the break contribution `P(r) * beta`, and the
mass-bound ceiling `P2_max = m + P(not r, not tau) * rho'` (P2 if every rescuable sample
were rescued and none broken).

**Outcome table (fixed in advance):**

| result | reading |
| :-- | :-- |
| `rho >= 0.5` in both regimes and `P(r) * beta <= 0.5 * m * rho` | the bank rescues well; **P2 is mass-bound**. The lever is the mass: expert boundaries that put confused classes into the same expert. Part B is motivated |
| `rho < 0.5` in either regime | the owner expert rescues poorly: the bank's value path, not its boundaries, limits P2. Part B is run (it is cheap and pre-registered) but its motivation is weakened, and the next variable is the expert's value path |
| `P(r) * beta >= m * rho` in a regime | breakage cancels rescue in that regime: the bank is net-neutral **by interference**, not by redundancy; report it as such |

## 4. Part B design: consolidation policy on identical data and an identical router

All arms run through the v3 API (`pal_moe.api.PalMoE`, float64 medium path) on the
same written batches, so the ridge router statistics are bitwise identical across arms
(guarded). Only the grouping of data into experts changes.

```text
A0  by_arrival      one expert per written task (the E-TID2 system, float64 router)
A1  by_confusion    20 experts of 5 classes, spectral clustering of the router's
                    confusion matrix (balanced, pal_moe.experts.policies.by_confusion)
A2  by_superclass   20 experts = the 20 CIFAR-100 superclasses (the coherent
                    construction used as a consolidation policy on the same data)
```

Regimes: `dispersed` (primary: `m` is small, the arrival order is semantically
scattered) and `coherent` (secondary: arrival already equals superclass, so A0 == A2 up
to training order, and A1 tests whether confusion structure improves on semantics).

**No test data in the grouping.** The confusion matrix for A1 is computed from
**5-fold cross-fitted** closed-form ridge predictions on the training split only (the
S6b construction reuses the test split as `val`, so validation data would leak). The
fold assignment is fixed by `seed`.

Routing for every arm: the `ridge_class` router with class ownership = the arm's
grouping (task = expert that owns the argmax class). Expert recipe: the S11 L3 recipe
(10 epochs, lr 1e-3, batch 128, `lambda_func = 1.0`, rank 8), `set_seed(seed)` once per
consolidation.

**Primary family** (paired over six seeds, `dispersed`):

```text
P3  Acc(A1) - Acc(A0)     does confusion-aligned consolidation beat arrival order?
P4  Acc(A1) - Acc(A2)     does it reach the semantic grouping? (TOST at +/-1 pp)
```

**Read together:** `m` under each grouping (the mass the grouping creates),
`rho`, `beta`, each arm minus `ridge_alone` (the P2 analogue), expert-size balance,
and the adjusted Rand index between the A1 and A2 groupings.

**Outcome table (fixed in advance):**

| result | reading |
| :-- | :-- |
| `P3 >= +1 pp` (6/6 seeds) and `m(A1) > m(A0)` | expert boundaries that follow router confusion **create** rescuable mass and the bank earns its keep over the ridge router; `by_confusion` becomes the v3 default candidate (a separate confirmatory run on a second backbone before it is adopted) |
| `|P3| < 1 pp` and `m(A1) > m(A0)` | mass was created but not converted: the rescue rate is the limit (read with Part A's `rho`); the bank stays redundant given a ridge router |
| `|P3| < 1 pp` and `m(A1) <= m(A0)` | the clustering failed to create mass; the policy, not the hypothesis, is refuted at this scale |
| `P3 <= -1 pp` | confusion-aligned experts break more than they rescue; do not adopt |
| P4 equivalent within +/-1 pp | router confusion recovers what semantic supervision gives, without labels for it |

## 5. Vetoes

```text
decomposition identity   Part A terms sum to P2 per cell within 1e-12
E-TID2 reproduction      Part A re-run reproduces every stored cell within 1e-6 on all
                         five arms (the float32 runner, as in the v3 anchor check)
identical router         the medium-path statistics digest is bitwise equal across
                         A0 / A1 / A2 within a (regime, seed) cell
no test leakage          the A1 confusion matrix is computed from train-split
                         cross-fitted predictions only (asserted: no test index used)
balanced grouping        every A1 / A2 expert owns exactly 5 classes
router purity            router trainable parameters 0, asserted by the API guard
reversibility            every write and consolidation passes the API's trial
                         undo/redo guard bitwise
one consolidation        set_seed then exactly one consolidation per (arm, regime, seed)
feasibility              measured cell cost projects the grid under the 2 h ceiling
```

## 6. Statistics

Six paired differences per contrast and regime; per-seed values first, then mean / sd,
exact two-sided paired permutation (`pal_moe.eval.stats.paired_stats`), max-statistic
Westfall-Young over the primary family `{P3}` plus TOST for `P4`
(`pal_moe.eval.stats.westfall_young`, `tost`). The six-pair floor is 0.0312. Part A is
descriptive with per-seed values and 95 % intervals; its readings use the thresholds in
its table, not p-values.

## 7. Feasibility

```text
Part A    12 cells x ~40 s (one L3 training each)                 ~8 min
Part B    3 arms x 2 regimes x 6 seeds = 36 consolidations x ~40 s ~25 min
ceiling   2 h
```

## 8. Out of scope

```text
a learned router            v3 forbids it (router purity); not a variable here
more experts / more rank    capacity saturates at rank 8 (F3, F11); not the variable
other backbones / datasets  a confirmatory replication is a separate document
the LM path                 docs/V3_LLM_PREREG.md
```

## 9. What this licenses, and what it does not

- **Licenses:** a statement about *why* the bank adds < 1 pp over the ridge router at
  `T = 20` on this backbone (mass-bound vs rescue-bound vs interference), and whether
  confusion-aligned consolidation changes it on these two constructions.
- **Does not license:** adopting `by_confusion` as the default without a replication; any
  claim about the LM path; any claim beyond `T = 20`, CIFAR-100 on the ViT-B/16 cache,
  and six seeds.

## Amendment 1 (2026-09-26, before any run)

The v3 medium path changed after review: one float64 accumulator pair, `forget` is a
subtraction, and the API guards measure order invariance and reversibility against a
tolerance instead of forcing them by a canonical summation order. Two vetoes in
section 5 are restated accordingly; nothing else changes.

```text
identical router   within a (regime, seed) cell all three arms write the same batches
                   in the same arrival order, so the medium-path statistics digest is
                   still bitwise equal across A0 / A1 / A2 (it is a same-order
                   determinism check, not an order-invariance claim)
reversibility      every write and consolidation passes the API guard: undo max|dW|
                   <= 1e-10 and canary argmax identical; consolidations additionally
                   bitwise (the bank object is swapped back)
```

## Amendment 2 (2026-09-26, after review, before any run)

Review approved Part A as written and Part B conditional on this amendment. The
questions, the arms and the existing outcome rows are unchanged.

**What Part B measures.** A0 (`by_arrival`) can train each expert when its task arrives
and discard the task's features afterwards. A1 (`by_confusion`) and A2
(`by_superclass`) regroup at the end: they need **every task's features retained** and
A1 uses a confusion matrix computed **after all 100 classes were seen** (hindsight). P3
therefore measures a **hindsight, offline upper bound** of confusion-aligned
consolidation - not a continual policy.

**Added endpoint (per arm):**

```text
stored_feature_bytes   training features the policy must hold to consolidate:
                       A0 = the largest single task's train split (one task at a time),
                       A1 / A2 = every task's train split (all 20 at once);
                       float32 features + int64 labels, computed from the cached tensors
```

**Added outcome row (takes precedence over the P3 >= +1 pp row's "default candidate"
clause):**

| result | reading |
| :-- | :-- |
| any P3 | Part B is the hindsight-offline upper bound of confusion-aligned consolidation at the stored-bytes cost reported. A positive P3 does **not** make `by_confusion` a default candidate: a continual variant, in which the grouping is decided as classes arrive and without retaining past features, must be tested first under its own pre-registration |

**Added to section 9, "does not license":** any claim that confusion-aligned
consolidation works under the continual constraint (no retained features, no hindsight).

**Added veto:**

```text
A0 reproduction   in every (regime, seed) cell of Part B, A0 run through the v3 API
                  reproduces the stored E-TID2 `ridge_routed` cell within 1e-6, so that
                  A0 -> A1 / A2 differs only in the grouping
```

**Implementation facts fixed now (so the runner cannot choose them later):**

```text
A1 confusion     5-fold cross-fitted float64 ridge (lambda = 1.0) on the written train
                 batches only; fold assignment by torch.randperm with seed = cell seed;
                 rows = true class, columns = cross-fitted argmax
A1 clustering    pal_moe.experts.policies.by_confusion, 20 groups, balanced (capacity
                 5), KMeans random_state = cell seed
A2 groups        the 20 CIFAR-100 superclasses (pal_moe.core.constructions.superclass_of)
expert order     groups are trained in index order (A1: cluster index; A2: superclass
                 index), one set_seed(seed) per consolidation
accuracy         per-task mean over the construction's 20 test tasks, as in E-TID2
```

**Reviewer's expectation, written before the run (recorded, not tested).** In
`dispersed`, A2 should raise `m` sharply because the ridge router's errors lie within
superclasses; `rho` may fall because separating five similar classes of one superclass
is a harder expert task (the owner oracle may move from ~97.3 toward the `coherent`
~87.3). Then `m` rises, `rho` falls and the net is unclear - the "mass created but not
converted" row.
