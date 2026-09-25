# Intervention: cutting the non-owner evidence path to old projections - results

Pre-registration: `docs/INTERVENTION_PREREG.md` (`e58d3cb`) with Amendment 1
(definitional, pre-run; committed with the implementation, `1b54c89`). Data:
`results/intervention/intervention_study.json` (18 cells; the six `owner_only`
cells are the new data, C1 and C0 are re-runs of the coupling cells).

Status: **run. All vetoes passed. The first row of the fixed outcome table: the
non-owner evidence path is a causal contributor to the collapse - and the two
endpoints separate cleanly.** Routing coverage recovers most of the way to the
no-coupling boundary; accuracy does not.

## 1. Vetoes

```text
cut audit (V3)          19 task checks, all pass: forward values torch.equal;
                        P, g, W_t, E_t gradients bitwise equal (max delta 0.0);
                        every old W_j gradient equals the owner-only reference
                        (1/P) * sum_{p: owner=j} CE_p exactly (max rel err 0.0,
                        stronger than the pinned 1e-6 tolerance)
anchor invariance (V2)  all 12 C1/C0 re-runs equal the coupling cells exactly on
                        all five metrics
one construction        set_seed then exactly one E2Model per cell; the C1
                        pre-check reproduced 0.1304 / 0.9041000545024872 before
                        the study ran (the interference study's root cause)
```

## 2. Primary: `OWNER-ONLY - C1`, six pairs

| metric | paired differences | mean | exact two-sided p |
| :-- | :-- | --: | --: |
| `Acc` (pp) | +6.30, +4.30, +2.59, +7.18, +5.74, +7.67 | **+5.63** | **0.0312** |
| `C@3` | +0.027, +0.024, +0.023, +0.042, +0.032, +0.038 | **+0.031** | **0.0312** |

Pairs are (coherent, 42/1/2) then (dispersed, 42/1/2). All six are positive on both
metrics; the Westfall-Young max-statistic over the `{Acc, C@3}` family gives the
same 0.0312, the six-pair floor of the exact test. Pre-registered reading: **row
1** - the non-owner evidence path is a causal contributor.

## 3. Boundary: which endpoint recovers

| regime | metric | C1 | OWNER-ONLY | C0 | recovered share |
| :-- | :-- | --: | --: | --: | --: |
| coherent | `Acc` | 13.57 | 17.97 | 74.16 | 7.3% |
| coherent | `C@3` | 0.9031 | 0.9279 | 0.9319 | 86.2% |
| dispersed | `Acc` | 16.87 | 23.73 | 70.14 | 12.9% |
| dispersed | `C@3` | 0.8078 | 0.8452 | 0.8548 | 79.6% |

Means over seeds; C0 is the no-coupling boundary re-run inside this study. The
dissociation is the result: **cutting the non-owner path repairs routing coverage
almost to C0 (80-86% of the gap) while accuracy stays near the collapse
(7-13%)**. The side metrics move with coverage: `conditional_oracle@3` +0.058,
`ceiling@3` +0.056, `oracle_accuracy` +0.151, all with the same six-pair pattern
at p = 0.0312.

Accuracy therefore has a damage component the non-owner cut does not remove. Since
C0 - no old-`W` gradient at all - restores accuracy and OWNER-ONLY - owner-term
gradient kept - does not, the remaining candidate is the owner-term re-alignment
itself: old projections still move (section 5) while the old-class readout rows are
frozen, so their evidence drifts out of the frozen decision boundaries. That is a
pointer for the next pre-registration, not a result of this study.

## 4. Count calibration: `asym_pp` against the `19.65x` headline

Loss-form per-prototype measurement at the last task boundary (`q = T-1`), means
over seeds; coherent / dispersed:

| arm | `asym_pp` | `asym_sum` |
| :-- | --: | --: |
| C1 | 1.0342 / 1.0340 | 19.650 / 19.645 |
| OWNER-ONLY | 1.0264 / 1.0263 | 19.501 / 19.500 |
| C0 | 1.0289 / 1.0272 | 19.549 / 19.517 |

The C1 sum-form values reproduce the interference ladder's `19.650 / 19.645` to the
printed precision. The count-adjusted counterpart is ~1.03 on every arm: at the trained
state an old projection's per-prototype cross-entropy on other tasks' prototypes
is only 2.6-3.4% above its own-prototype value. So the `19.65x` headline is
essentially the prototype count (~19 other prototypes per own prototype), exactly
as the interference pre-registration warned it could not be read as per-prototype
severity. The sum form is dominated by count and is nearly unchanged by the
intervention, while coverage moves by 2.5-4 points - it is not the endpoint.

## 5. Rewrite mass (raw norms, secondary)

`mean ||delta_W_j||` over old projections at the final task:

```text
coherent    C1 6.6690   OWNER-ONLY 4.1054 (-38%)   C0 0.0000
dispersed   C1 6.0844   OWNER-ONLY 3.1420 (-48%)   C0 0.0000
```

The cut reduces but does not zero the movement: old projections still re-align,
now only on their own task's prototypes.

## 6. What this licenses, and what it does not

- **Licenses:** with every value and every other gradient bitwise unchanged,
  removing the non-owner gradient path to old `W_j` raises `Acc` and `C@3`
  significantly (all six pairs, WY 0.0312) and restores routing coverage most of
  the way to the no-coupling boundary. The non-owner path is a cause of the
  routing damage.
- **Does not license:** a complete explanation - accuracy stays collapsed, and the
  section 3 pointer is a hypothesis, not a tested result; any per-prototype
  severity statement - section 4 shows the sum form is counting prototypes; any
  statement about `P` or `g`, which were unchanged in every arm; anything beyond
  the pinned formulation and the 20-expert operating point.
- The primary test was defined a priori and does not condition on
  `INTERFERENCE_RESULTS.md`; the count calibration in section 4 is a secondary
  passive measurement, reported as calibration.

## 7. Records

```text
e58d3cb   the pre-registration
1b54c89   the owner_only arm, the cut audit and the runner (+ Amendment 1)
this      the 18 cells and this document
```
