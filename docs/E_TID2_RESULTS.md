# E-TID2: a continual class-level ridge router over the unchanged L3 bank - results

Pre-registration: **none committed.** The arms, the two primary contrasts, the guards
and the outcome table were fixed in the docstring of
`experiments/e_tid2_ridge_router.py` before it was run, but the script was not
committed before the run (it was written as `e_tid_ridge_router.py`, renamed to match
its output file, and first committed in the Phase 0 record commit of
`v3-restructure`, after the JSON existed). Read everything below as **exploratory**.

Data: `results/e_tid2/e_tid2_ridge_router.json` - 12 cells (2 regimes x seeds
42, 1, 2, 3, 4, 5), the S11 recipe (10 epochs, lr 1e-3, batch 128,
`lambda_func = 1.0`, rank 8, one prototype, top-1), ridge `lambda = 1.0`, device `cuda`.

Status: **run, all three guards passing. P1: the ridge router lifts the system by
+4.10 pp (`coherent`) / +6.08 pp (`dispersed`), 6/6 seeds each. P2: once routing is
fixed, the expert bank adds +0.62 pp (`coherent`, 6/6 positive) and -0.03 pp
(`dispersed`, 6/6 negative) over the ridge readout alone - both inside the 1 pp
SESOI.** Under the docstring's table that is "the bank is redundant given a ridge
router".

## 1. Design

Evaluation-only on the selection, like AC3: one trained `L3_per_task` cell per
(regime, seed), and only the expert id per sample changes between arms. The ridge
router is `RidgeReadout(768, 100)` fitted **one task at a time** (continual
accumulation of `A = ZᵀZ`, `B = ZᵀY`); the routed task is the owner of the argmax
class.

| arm | rule |
| :-- | :-- |
| `proto` | pinned E0 route (`PrototypeRouter` top-1) - anchor |
| `ridge_routed` | expert = owner task of argmax ridge class - **primary** |
| `ridge_masked` | `ridge_routed`, readout restricted to the routed task's classes (exploratory) |
| `ridge_alone` | the continual ridge's own class prediction (== `L1_ridge`) |
| `oracle` | owner expert given (L4, the ceiling) |

```text
P1  ridge_routed - proto        does the better router realise its coverage?
P2  ridge_routed - ridge_alone  does the bank add anything once routing is fixed?

P2 > 0 in every seed, both regimes   the bank earns value over the best readout
|P2| < 1 pp                          the bank is redundant given a ridge router
P2 < 0                               ridge alone wins; the experts dissolve into it
P1 <= 0 despite higher C@1           selections are not decodable (AC3's gap, again)
```

## 2. Results (means over six seeds)

| regime | `proto` | `ridge_routed` | `ridge_masked` | `ridge_alone` | `oracle` | proto `C@1` / `C@3` | ridge `C@1` / `C@3` |
| :-- | --: | --: | --: | --: | --: | :-- | :-- |
| `coherent` | 73.29 | **77.39** | 77.44 | 76.77 | 87.30 | 0.8195 / 0.9494 | **0.8616 / 0.9681** |
| `dispersed` | 70.66 | **76.74** | 76.75 | 76.77 | 97.30 | 0.7130 / 0.8915 | **0.7745 / 0.9302** |

Paired contrasts (`s11.paired_stats`, exact sign / permutation over six seeds):

| regime | contrast | mean | 95 % CI | +/- | perm p |
| :-- | :-- | --: | :-- | :-- | --: |
| `coherent` | P1 `ridge_routed - proto` | **+0.0410** | [+0.0405, +0.0414] | 6/0 | 0.031 |
| `coherent` | P2 `ridge_routed - ridge_alone` | **+0.0062** | [+0.0051, +0.0073] | 6/0 | 0.031 |
| `coherent` | X1 `ridge_masked - ridge_alone` | +0.0067 | [+0.0056, +0.0077] | 6/0 | 0.031 |
| `dispersed` | P1 `ridge_routed - proto` | **+0.0608** | [+0.0606, +0.0611] | 6/0 | 0.031 |
| `dispersed` | P2 `ridge_routed - ridge_alone` | **-0.0003** | [-0.0006, +0.0000] | 0/6 | 0.031 |
| `dispersed` | X1 `ridge_masked - ridge_alone` | -0.0002 | [-0.0006, +0.0001] | 1/5 | 0.063 |

Six seeds give a minimum attainable two-sided exact p of 0.031; no multiplicity
correction was declared.

## 3. Guards

```text
G1 evaluator equivalence   proto arm vs model.evaluate_task: max |delta| 6.9e-08
G2 oracle equivalence      oracle arm vs evaluate_task(oracle=True): max |delta| 7.3e-08
G3 ridge incrementality    continual vs one-shot ridge: 0 argmax mismatches on every
                           test sample of every cell; max |dW| 1.26e-03 (coherent)
                           / 1.09e-03 (dispersed) in float32
```

Cross-checks against the stored S11 study (same recipe, same seeds): `proto` equals
the S11 `L3_per_task` cells within 1.4e-07, `oracle` equals `L4_oracle` within
1.8e-07, `ridge_alone` equals `L1_ridge` within 7.7e-08.

G3 is the reason the v3 medium path accumulates in **float64**: the argmax is exact,
but a weight difference of 1e-3 is not what an "exact" closed-form edit should leave.

## 4. Reading

- **P1.** `C@1` rises by 4.2 / 6.2 pp and accuracy follows almost one-for-one
  (+4.1 / +6.1 pp). The last row of the table does not apply: the better router's
  selections are decodable.
- **P2.** Both regimes fall in the "|P2| < 1 pp" row: **the bank is redundant given a
  ridge router.** The table's rows are not mutually exclusive, and the signs are
  consistent across seeds - `coherent` is positive in 6/6, `dispersed` negative in
  6/6 - so the precise statement is: the bank adds a small, reliable amount where
  tasks are geometrically coherent, and a small, reliable nothing where they are
  dispersed. Neither reaches the SESOI.
- **Why (hypothesis, not tested here).** The bank can only change the answer where
  the router's class is wrong but its task is right (it then decodes with the owner
  expert) - or break it where the class was right. How much mass sits in "class wrong,
  task right" is set by task geometry, and dispersed tasks leave almost none. This is
  the mechanism `docs/P2_BOUND_PREREG.md` pre-registers.

## 5. What this does not say

- Not confirmatory: no committed pre-registration.
- It does not say experts are useless - it says that experts whose boundaries follow
  the **arrival order** of tasks add < 1 pp over the router's own readout.
- `ridge_masked` is a different decoder (S5's masking factor), exploratory only.
- No claim beyond `T = 20`, these two constructions, six seeds, one ViT-B/16 cache.

## 6. Records

```text
experiments/e_tid2_ridge_router.py        the script (docstring = the readings)
results/e_tid2/e_tid2_ridge_router.json   12 cells + report, untracked
this                                      the write-up
```
