# Owner-side residual: the two-component decomposition of the E2 collapse - results

Pre-registration: `docs/OWNER_SIDE_PREREG.md` (`67f38f9`, with the pre-run wording
note `3c5ce6f`). Data: `results/owner_side/owner_side_study.json` - 18 cells, all
re-run fresh; the earlier records serve only as anchors.

Status: **run, all vetoes passed. Both fixed readings landed: the accuracy collapse
has an owner-side causal component, and the clean "routing <-> non-owner,
accuracy <-> owner-side" mapping is not supported.**

## 1. Vetoes

```text
cut audit            19/19 PASS (V3 of INTERVENTION_PREREG.md, Amendment 1)
anchors phase A      C1 and C0 bitwise equal to the coupling cells AND to the
                     intervention cells
anchors phase B      OWNER-ONLY bitwise equal to the intervention cells
every cell           equal to every applicable stored cell on all five metrics;
                     one float difference would have been a veto
passivity            the C1 pre-check with the drift probe active reproduced
                     0.1304 / 0.9041000545024872 exactly
one construction     set_seed then exactly one E2Model per cell
```

## 2. Primary family (`Acc`): the owner-side component is real

| component | six paired differences (pp) | mean | exact two-sided p |
| :-- | :-- | --: | --: |
| `Delta_nonowner = OWNER-ONLY - C1` | +6.30, +4.30, +2.59, +7.18, +5.74, +7.67 | **+5.63** | **0.0312** |
| `Delta_owner = C0 - OWNER-ONLY` | +54.99, +56.80, +56.77, +47.11, +46.44, +45.67 | **+51.30** | **0.0312** |

Pairs are (coherent 42/1/2, dispersed 42/1/2); all six are positive for both
components. Westfall-Young max-statistic within the `Acc` family: **0.0312**.

Fixed reading, row 1: **the residual accuracy degradation has an owner-side causal
component**. With the non-owner component replicating, the accuracy collapse
decomposes at the level of the arms into a non-owner part (+5.63 pp) and an
owner-side part (+51.30 pp), the latter about nine times larger. The decomposition
is exact per pair by construction: `(OWNER-ONLY - C1) + (C0 - OWNER-ONLY) =
C0 - C1`.

This does not license "the only cause is owner-side": the two components are what
was measured, the non-owner part was causally established on routing, and nothing
here identifies the origin of the owner-side part beyond the update-versus-no-update
contrast.

## 3. Secondary family (`C@3`): the clean mapping is not supported

| component | six paired differences | mean | exact two-sided p |
| :-- | :-- | --: | --: |
| `Delta_nonowner^C@3` | +0.0267, +0.0242, +0.0234, +0.0417, +0.0323, +0.0381 | **+0.0311** | **0.0312** |
| `Delta_owner^C@3` | +0.0020, +0.0039, +0.0060, +0.0101, +0.0086, +0.0100 | **+0.0068** | **0.0312** |

Westfall-Young within the `C@3` family: **0.0312**. The owner-side `C@3` component
is positive and significant, so by the fixed reading **the clean mapping
"routing <-> non-owner, accuracy <-> owner-side" is not supported**: both pathways
affect routing, with the non-owner component about 4.6x larger. The report records
that explicitly and does not re-map after the fact.

## 4. Passive mechanism secondaries (descriptive only)

Per arm, means over seeds; per-task accuracy is the mean over the 19 old tasks,
`drift_final` the mean over old tasks at the last boundary:

| regime | arm | `Acc` | `C@3` | per-task `Acc` | `drift_final` |
| :-- | :-- | --: | --: | --: | --: |
| coherent | C1 | 13.57 | 0.9031 | 9.83 | 0.8548 |
| coherent | OWNER-ONLY | 17.97 | 0.9279 | 15.66 | 0.7992 |
| coherent | C0 | 74.16 | 0.9319 | 73.99 | 0.0000 |
| dispersed | C1 | 16.87 | 0.8078 | 13.18 | 0.8010 |
| dispersed | OWNER-ONLY | 23.73 | 0.8452 | 21.77 | 0.7519 |
| dispersed | C0 | 70.14 | 0.8548 | 70.09 | 0.0000 |

Drift trajectory (coherent, seed 42; mean over old tasks at boundaries 4/9/14/19):
C1 0.864 / 0.881 / 0.883 / 0.853; OWNER-ONLY 0.769 / 0.811 / 0.824 / 0.806; C0
zero to floating-point precision at every boundary.

Within-arm Spearman between per-task accuracy and final drift across the 19 old
tasks: C1 -0.841 (cells -0.92, -0.77, -0.80, -0.73, -0.89, -0.94); OWNER-ONLY
-0.772 (-0.83, -0.42, -0.74, -0.84, -0.85, -0.95).

Reading: drift is zero where the old projections are frozen, near-total (0.75-0.85,
a cosine of roughly 0.15-0.25) wherever they update, and per-task accuracy tracks
it inversely both across arms and across tasks within an arm. This is consistent
with the candidate "old evidence drifts out of the frozen old-class readout rows".
It is **descriptive**: no causal claim, no test, and it cannot convert the
significant contrast of section 2 into a mechanism.

## 5. What this licenses, and what it does not

- **Licenses:** the accuracy collapse has an owner-side causal component
  (`C0 - OWNER-ONLY`, +51.30 pp, all six pairs, WY 0.0312); and the collapse
  decomposes into two causally established components at the level of the arms -
  non-owner (+5.63 pp `Acc`, +0.0311 `C@3`) and owner-side (+51.30 pp `Acc`,
  +0.0068 `C@3`) - with different endpoint weights: the non-owner pathway dominates
  routing, the owner-side pathway dominates accuracy, and, by the fixed reading,
  neither is endpoint-exclusive.
- **Does not license:** a single cause; any mechanism claim from the drift probe;
  statements about `P` or `g`, which were unchanged in every arm; anything beyond
  the pinned formulation and the 20-expert operating point; and C0 as a solution -
  it remains a reference, still below E0 on coverage (0.9319 / 0.8548 against
  0.9494 / 0.8915, coupling study).

## 6. Records

```text
67f38f9   the pre-registration
3c5ce6f   the conceptual-chain wording; per_task_accuracy added (additive)
1cfbad4   the runner and the passive drift probe
this      the 18 fresh cells and this document
```
