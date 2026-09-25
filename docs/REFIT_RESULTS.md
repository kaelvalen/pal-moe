# Readout refit / replay: decode mismatch or representation loss? - results

Pre-registration: `docs/REFIT_PREREG.md` (`6ef995e`). Data:
`results/refit/refit_study.json` - 18 fresh training cells x three readout
treatments (`R0` frozen, `R1` refit-current, `R2` refit-replay).

Status: **run, all guards passed. Fixed reading: the owner-side accuracy loss is
predominantly consistent with a readout/decode mismatch, and the recovery depends
substantially on replay / old-task evidence.**

## 1. Guards

```text
anchor invariance   all 18 R0 evaluations bitwise equal to the stored cells of
                    coupling, intervention and owner_side on the five metrics,
                    and per-task accuracy bitwise equal to owner_side
C@3 invariance      coverage_at_3 bitwise identical across R0/R1/R2 in all 18
                    cells - only g changed
procedure control   Delta_replay for C0 is +0.055 pp (exact p = 0.875), not
                    significantly negative
reproducibility     re-running OWNER-ONLY / coherent / 42 reproduced R0, R1 and
                    R2 bitwise, every metric and per-task accuracy
```

## 2. Primary family (`Acc`): `R2 - R0`

| arm | six paired differences (pp) | mean | exact two-sided p |
| :-- | :-- | --: | --: |
| OWNER-ONLY | +49.72, +52.47, +52.17, +41.93, +41.27, +39.86 | **+46.24** | **0.0312** |
| C1 | +20.80, +20.29, +17.59, +20.21, +20.12, +21.40 | **+20.07** | **0.0312** |

Westfall-Young within the primary family: **0.0312**. Fixed reading: **recovery is
consistent with a readout/decode mismatch** - the class information is still
decodable from the drifted evidence once the readout is refitted on stored
training evidence.

## 3. Secondary family (`Acc`): current vs old

| arm | component | six paired differences (pp) | mean | exact p |
| :-- | :-- | :-- | --: | --: |
| OWNER-ONLY | `Delta_current = R1 - R0` | -4.00, -1.46, -1.56, -0.50, -2.11, -0.66 | **-1.71** | 0.0312 |
| C1 | `Delta_current` | -0.37, -0.90, -0.82, +0.16, -0.16, -0.02 | -0.35 | 0.1250 |
| OWNER-ONLY | `Delta_old = R2 - R1` | +53.72, +53.93, +53.73, +42.43, +43.38, +40.52 | **+47.95** | 0.0312 |
| C1 | `Delta_old` | +21.17, +21.19, +18.41, +20.05, +20.28, +21.42 | **+20.42** | 0.0312 |

Westfall-Young within the secondary family: **0.0312**. Fixed reading: `R1 ~ R0`
(no recovery; for OWNER-ONLY a small but significant loss) together with `R2 > R1`
(significant) means **the recovery depends substantially on replay / old-task
evidence**. Refitting on the current task's evidence alone does not recover
old-task accuracy.

## 4. Where the recovery lands

Means over seeds; the recovered share is of the arm-to-C0 gap:

| regime | arm | R0 | R1 | R2 | C0 | recovered share by R2 |
| :-- | :-- | --: | --: | --: | --: | --: |
| coherent | OWNER-ONLY | 17.97 | 15.63 | **69.42** | 74.16 | **91.6%** |
| coherent | C1 | 13.57 | 12.88 | **33.13** | 74.16 | 32.3% |
| dispersed | OWNER-ONLY | 23.73 | 22.64 | **64.75** | 70.14 | **88.4%** |
| dispersed | C1 | 16.87 | 16.86 | **37.44** | 70.14 | 38.6% |

The owner-side arm's refit reaches 88-92% of the no-coupling boundary, while C1
recovers only 32-39% - consistent with the non-owner path additionally damaging the
evidence itself (its routing-dominant component). A residual 8-12% of the
owner-side gap (4.7-5.4 pp against C0) is not recovered by refitting and is
reported as the remaining unexplained part.

## 5. What this licenses, and what it does not

- **Licenses:** the owner-side accuracy loss is consistent with a decode/readout
  mismatch; replay-based readout refitting recovers most of it (91.6 % / 88.4 % of
  the gap to C0); current-task-only refitting does not (and for OWNER-ONLY it
  slightly hurts). The flat C0 control is what makes the procedure credible.
- **Does not license:** that the representation is fully intact (`R2 > R0` does not
  test intactness, and a residual gap remains); equivalence from the `R1`
  non-significance (six pairs cannot license it - `R1 ~ R0` is read only as "no
  substantial readout-only recovery"); any claim about where inside the readout the
  mismatch lives; any memory/budget or deployment reading (stored features are not
  proposed as a method); anything beyond the pinned formulation.

## 6. Chain position and claim status

```text
COUPLING
   |
   v
INTERFERENCE
   |   descriptive mechanistic support
   v
INTERVENTION
   +-- non-owner paths -> the dominant causal component of routing damage
   v
OWNER-SIDE
   +-- owner-side update -> the dominant causal component of accuracy damage
   v
READOUT REFIT
   +-- current-task evidence -> does not recover old classes
   +-- replay / old-task evidence -> recovers most of the loss
          +-- consistent with a decode/readout mismatch
          +-- 8-12% residual gap -> unexplained
```

The claims, separated:

| claim | status |
| :-- | :-- |
| owner-side update has a causal effect on accuracy | **supported** |
| most of that effect is consistent with a readout/decode mismatch | **supported** |
| replay / old-task evidence is required for the recovery | **supported** |
| the representation is fully intact | not shown |
| the residual gap is representation degradation | not shown |
| the whole owner-side damage is readout mismatch | rejected / not supported |

The chain's tightest formulation:

> Non-owner path intervention produces the dominant causal component of routing
> damage; owner-side updating produces the dominant causal component of accuracy
> damage. Most of the owner-side accuracy loss is recoverable by post-hoc readout
> refitting with replayed old-task evidence, consistent with a decode/readout
> mismatch, while a small residual accuracy gap remains unexplained.

The large `R2 - R1` difference is the empirical counterpart of the
`pal_moe/arch/readouts.py` warning: re-opening rows on current-task data alone does
not serve old classes. The residual is left unexplained on purpose; separating
representation degradation, expert degradation or normalisation effects needs a new
intervention of its own.

## 7. Records

```text
6ef995e   the pre-registration
67d4623   the runner
this      the 18 cells and this document
```
