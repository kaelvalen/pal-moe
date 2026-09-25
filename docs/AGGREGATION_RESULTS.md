# Decision rule: winner-take-all vs uniform top-3 aggregation - results

Status: **run. The uniform top-3 mixture is significantly worse than
winner-take-all in both regimes, on all six seeds, with every identity guard
passing exactly.** Pre-registration: `docs/AGGREGATION_PREREG.md`.

## 1. The result

Primary, `Delta Acc = Acc_mixture - Acc_WTA`, paired by seed:

| contrast | per-seed (six) | mean | sd | exact permutation p | WY-adjusted p |
| :-- | :-- | --: | --: | --: | --: |
| `Delta Acc`, `coherent` | -0.0731 -0.0707 -0.0741 -0.0721 -0.0689 -0.0810 | **-0.0733** | 0.0042 | 0.0312 | **0.0312** |
| `Delta Acc`, `dispersed` | -0.0410 -0.0432 -0.0410 -0.0381 -0.0429 -0.0480 | **-0.0424** | 0.0033 | 0.0312 | **0.0312** |
| `Delta Acc_covered`, `coherent` | all negative | **-0.0772** | 0.0044 | 0.0312 | **0.0312** |
| `Delta Acc_covered`, `dispersed` | all negative | **-0.0475** | 0.0037 | 0.0312 | **0.0312** |
| `Delta C@3` (guard), both regimes | 0.0000 exactly | **0.0000** | 0.0000 | - | not tested (a guard) |

Per arm, six seeds:

| arm | accuracy | accuracy \| covered | C@3 |
| :-- | --: | --: | --: |
| WTA, `coherent` | 73.29 | 77.20 | 0.9494 |
| mixture, `coherent` | 65.96 | 69.48 | 0.9494 |
| WTA, `dispersed` | 70.66 | 79.26 | 0.8915 |
| mixture, `dispersed` | 66.42 | 74.51 | 0.8915 |

**This is the fourth row of the pre-registration's outcome table: the mixture
dilutes specialization.** Combining the three candidates uniformly loses 7.3
points (`coherent`) and 4.2 (`dispersed`) against using the top-ranked candidate,
and the loss is the same or larger on the samples whose owner expert is already
in `C_3` (-7.7 / -4.8): the decision rule does not fail to *use* the candidate
set, it actively destroys information the top-ranked candidate had. This is the
first result in the chain that is both family-wise significant and negative.

## 2. The guards, all passing exactly

```text
WTA anchor (S11 L3, per seed)     12/12 cells, max |delta| = 1.1e-07 (float noise)
Delta C@3                          0.0000 exactly in all 12 cells
candidate IDs, router scores       identical by construction
expert outputs (posterior hash)    identical across arms, 12/12
global-class posteriors            both arms received the same unmasked tensors
bank hash                          identical per (regime, seed)
covered fraction                    identical across arms
```

So the comparison is decision-rule-only: the ranking, the candidate set, the
expert outputs, the class space and the bank are the same, and coverage cannot
move by construction. The WTA arm reproducing S11's `L3` exactly is what licenses
that reading - and it is why the earlier masked variant was rejected before the
run (see the pre-registration: masking reintroduces the task/class masking
factor S5 separated out, and would have made the anchor fail by construction).

## 3. What this closes, and what it opens

The invariant every attempt has kept - one independent score per expert, a
competition between experts, a single winner - now has both of its components
tested:

```text
pointwise supervision          -> not the constraint   (RR, RRF, DR)
comparative supervision        -> not the constraint   (DR)
uniform top-3 aggregation      -> does not improve on winner-take-all under the
                                  fixed candidate set   (this study)
```

**The third line is deliberately narrow.** This study refutes the pre-registered
uniform top-3 aggregation alternative; it does not show that winner-take-all is
optimal, and it does not show that aggregation schemes in general are inferior.
Only that specific alternative, under that fixed candidate set, at that operating
point.

And the ranking side is already covered: capacity does not help (S8, S10),
resolution helps partially (S8, S11), the adapted space trades separability for
within-expert accuracy (RRF). With the decision rule now closed, **the remaining
untested variable is the expert formulation itself**: why experts cannot be
combined in one decision space, and what an expert should be such that its output
is comparable to its neighbours'.

## 4. What this does not say

- It does not say aggregation is wrong in general. It refutes the
  pre-registered **uniform top-3 full-class mixture**, nothing more: no
  temperature, no calibration and no learned weighting was tried, and a
  weighted or calibrated combination is a different hypothesis.
- It does not say WTA is optimal - only that it beats this fixed mixture at this
  operating point.
- It does not reopen anything: Stage 1, the Router Ranking Study, the factorial
  and the decision-routing study all stand as reported.
- The compute asymmetry is not part of the result: both arms evaluated the same
  three experts, so the mixture's loss is not a compute artefact. In deployment
  WTA could evaluate one expert, which would only widen its advantage.
