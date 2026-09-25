# Decision rule: winner-take-all vs uniform top-3 aggregation - pre-registration

Status: **proposed, awaiting approval, not started.** It follows
`DIAGNOSIS_SYNTHESIS.md` (frozen diagnosis) and `DECISION_ROUTING_RESULTS.md`
(the pointwise half of the invariant is closed: comparative supervision is not
better). Nothing here is evidence for those documents.

The chain the programme has measured so far:

```text
more expert capacity            -> does not solve it      (S8, S10)
more routing resolution         -> partial                (S8, S11)
better pointwise supervision    -> does not solve it      (RR, RRF)
comparative supervision         -> does not solve it      (DR)
```

and now the last untested piece of the invariant that all of them kept:
**the decision rule**.

## 1. The question

The synthesis asks whether routing failure is caused by the pointwise
winner-take-all formulation. This document tests the operational half of that,
stated so it can fail:

> **Given an identical ranked top-3 candidate set, does combining candidate
> experts outperform selecting the top-ranked expert?**

## 2. The comparison

**The ranking, the scores and the candidate set are identical by construction.**
Let `s_e(z)` be the router's scores and

```text
C_3(z) = Top_3 { s_e(z) }        the same three experts, in the same order
```

**WTA:** use the top-ranked candidate only.

```text
e*      = argmax_{e in C_3} s_e(z)
p_WTA   = p_{e*}(y | z)
```

**Mixture:** combine the three candidates uniformly.

```text
p_mix(y | z) = (1/3) * sum_{e in C_3} p_e(y | z)
```

Each expert's local posterior `p_e(y|z)` is the shared readout applied to that
expert's adapted feature, placed into the global class space with that expert's
own classes having mass and every other class having zero. There is **no learned
selector, no temperature and no weighting mechanism**: the uniform average is
chosen precisely so that the only difference is the decision rule.

**Both arms compute the same forward.** Both evaluate the same three experts on
the same input; WTA then discards two of them and the mixture combines all three.
The active-compute difference is therefore **not** a causal variable in this
study. It is recorded separately as a deployment note: a WTA deployment could
evaluate one expert, this implementation evaluates three, and the study does not
claim otherwise.

## 3. Endpoints

**Guards - all must be identical, and any failure voids the cell:**

```text
Delta C@3                exact 0        the ranking does not change
candidate IDs            exact same     C_3 is the same set in the same order
router scores            exact same
bank hash                identical
expert outputs           identical      if these move, it is not a decision-rule
                                        experiment
L4 / oracle path         identical
Ceiling@3                identical      it is built from the same candidate set
```

**Primary:**

```text
Delta Acc = Acc_mixture - Acc_WTA
```

Coverage is *not* the primary here: it cannot move. It is a guard.

**Secondary:**

```text
Delta Acc_covered = the same difference restricted to samples whose owner expert
                    is already in C_3
```

This is the quantity that shows whether the decision rule can use the candidate
set it already has.

## 4. Anchor

Because the WTA path is identical to the ladder's own path, the WTA arm must
reproduce S11's `L3` accuracy **per seed, exactly**. That is the anchor and it is
a veto: a failed anchor makes the mixture result unreadable rather than
downgraded.

## 5. Outcome reading, fixed in advance

| result | reading |
| :-- | :-- |
| `Delta Acc > 0` and `Delta Acc_covered > 0` | winner-take-all cannot use the candidates it already has; uniform aggregation is supported |
| overall `Delta Acc > 0`, `Delta Acc_covered` unchanged | the gain comes from something other than the candidate set; the mechanistic claim is weakened |
| `Delta Acc` ~ 0 | within the existing top-3 candidate set, WTA is not the bottleneck; the ranking and representation sides return to the front |
| `Delta Acc < 0` | the mixture dilutes specialization; WTA is the better rule in this setup |

A negative result refutes **the fixed uniform top-3 mixture hypothesis**, not
"aggregation" as an idea. If the third row holds, the invariant is nearly
exhausted experimentally and the next question is no longer "how should the
router score?" but **why experts cannot be combined in one decision space - the
expert formulation itself**.

## 6. Statistics

The S11 protocol. Families declared separately:

```text
primary      Delta Acc,               2 regimes
secondary    Delta Acc_covered,       2 regimes
```

Paired by seed, six seeds, exact sign and permutation tests, Westfall-Young
max-statistic correction within each family separately, TOST where equivalence
is claimed.

## 7. Cost

```text
2 arms x 2 regimes x 6 seeds = 24 cells
```

One expert bank per `(regime, seed)`, trained once and shared by both arms in one
process with one evaluation path, so the two arms differ only in the decision
rule. The banks are the ladder's `L3`, re-run here so the anchor can be checked
in the same process.

## 8. Out of scope

```text
learned selectors, temperature, calibration   no; the mixture is uniform and fixed
candidate-set size sweeps                     no; m = 3 is fixed, coverage is a guard
new expert types or representation changes    no
capacity or resolution sweeps                 no; measured
rehearsal or stored features                  no
```
