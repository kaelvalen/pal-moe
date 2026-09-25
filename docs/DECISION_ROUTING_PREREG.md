# Decision Routing: comparative supervision - pre-registration

Status: **proposed, awaiting approval, not started.** It follows
`DIAGNOSIS_SYNTHESIS.md`, which freezes what the three previous programmes
established. Nothing here is evidence for them.

---

## 1. The question

> Is routing failure caused by the pointwise winner-take-all formulation, rather
> than by the quality of the individual expert scores?

The synthesis names the invariant every previous attempt kept:

```text
z  ->  { s_1(z), ..., s_T(z) }  ->  argmax_e s_e(z)
```

and the measured constraint: cross-expert separability, which a pointwise score
has to produce as a *side effect* of each expert being individually good. The
evidence says it does not.

**This document tests the *pointwise* half only.** The winner-take-all decision
rule stays exactly as it is. The reason is the programme's own rule: the
winner-take-all half cannot be changed without also changing how experts combine
(a soft mixture is a different architecture), so it is a separate hypothesis for
a separate document.

## 2. The manipulation

Supervision changes; everything else does not.

```text
baseline   each expert's score is supervised pointwise: a prototype is registered
           for the sample's own class and compared against the sample
comparative  supervision is defined on *pairs*: for a given input z, the model is
             trained on comparisons  s(e_i, e_j | z)  or  P(e_i > e_j | z)
             between experts (or between the sample and another expert's
             evidence), so that the required separation between experts is an
             explicit training target rather than a side effect
```

The pinned form, so it cannot be chosen after seeing a result:

```text
targets     pairs are formed between (a) the sample's owner expert and (b) each
            other seen expert; the comparative objective is a margin/ranking loss
            over those pairs
inference   unchanged: task id unavailable, one score per expert, argmax
training    no rehearsal, no stored features beyond what the ladder already keeps
loss        L_total = L_task + lambda * L_comparative, lambda = 1.0
bank        untouched: the comparative term trains the routing mechanism only,
            never the experts or the readout
```

If a comparative objective improves coverage while the pointwise one does not,
the formulation was the limit. If it does not, the constraint is not the
supervision's shape either, and the winner-take-all half becomes the next
hypothesis with a stronger prior.

## 3. Control matrix

```text
fixed                                  changed
-----                                  -------
backbone (frozen ViT-B/16)             the supervision of the routing scores
partition (S6b): coherent, dispersed
expert bank, rank 8, one expert/task
prototypes per class (1)
candidate m = 3 primary
seeds 42, 1, 2, 3, 4, 5
bank budget, training budget, lr
decision rule: argmax over experts
```

The bank hash is recorded per cell and must be identical to the baseline's: the
comparative term never receives gradient through the experts.

## 4. Endpoints and guards

**Primary:** `Delta C@3` against the pointwise baseline, with the routing ground
truth being the **owner expert**, not `L4`'s prediction.

**Kept from the previous programmes, reported together:**

```text
conditional_oracle@3   a coverage gain with an oracle loss is a representation
                       cost, not a routing win (and the reverse is a routing win
                       even if accuracy does not follow)
ceiling@3 = coverage@3 x conditional_oracle@3
tax = L4 - L3
R_iso_ncm = (L3 - L0) / (L4 - L0)
headroom fraction recovered   descriptive, not a threshold
```

**Guards:**

```text
bank identity      hash-identical to the baseline, or the arm is void
L4 invariance      in the frozen-representation arms L4 must be unchanged exactly
ceiling identity   reported, and conditional_oracle reported rather than asserted
                     equal when coverage moves
one variable       no arm may differ from the baseline except in the supervision
```

## 5. Outcome reading, fixed in advance

| result | reading |
| :-- | :-- |
| comparative supervision improves `C@3` and lowers the tax | the pointwise formulation was the constraint; routing is a comparison problem |
| `C@3` improves with no tax change | coverage improved where the loss is not; the tax is not driven by the candidates missed pointwise |
| no `C@3` change | the supervision's shape is not the constraint; the next hypothesis is the winner-take-all decision itself |
| `C@3` falls | comparative supervision is harder to fit at this scale; the pointwise form was not the limit but the easier problem |

The third row is the one that most sharpens the programme: it would leave the
decision rule as the last untested piece of the invariant, with two independent
negative results (pointwise supervision, comparative supervision) behind it.

## 6. Statistics

The S11 protocol: paired by seed, six seeds (the floor at which an exact
two-sided test can reject), exact sign and permutation tests, Westfall-Young
max-statistic correction over the family (two regimes), TOST where the claim is
equivalence.

## 7. Cost

```text
2 arms (baseline, comparative) x 2 regimes x 6 seeds = 24 trainings
```

The baseline arm is S11's `L3` and is re-run in the same process so both arms
share one bank per seed and one evaluation path; reproducing it is the anchor.

## 8. Out of scope

```text
winner-take-all / soft mixture    a separate hypothesis, separate document
new expert types                  no; the bank is the existing residual adapter
representation changes            no; the RRF result moved the variable here
capacity sweeps                   no; S8/S11 measured it as not the constraint
rehearsal or stored features      no
```
