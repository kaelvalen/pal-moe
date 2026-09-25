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
comparative  supervision is defined on *pairs of experts*: for a given input z,
             the model is trained on comparisons between the owner expert and
             every other seen expert, so that the required separation between
             experts is an explicit training target rather than a side effect
```

The pinned form, so it cannot be chosen after seeing a result.

**Pairs.** For every training example `z`, with `e* = owner(z)` and `T_t` experts
seen so far, the pairs are exactly

```text
(e*, j)   for every   j in {1..T_t} \ {e*}
```

That is `T_t - 1` comparisons per example. **No hard-negative mining, no top-k
negatives, no random negative sampling, and no "nearest opponent" selection** -
each of those would leave sampling variance or a selection rule available to
explain a result after the fact. The only evidence in the comparison is the
sample and the measured evidence of both experts.

**Loss.** On the existing scalar scores `s_e(z)`:

```text
L_comp(z) = 1/(T_t - 1) * sum_{j != e*} max(0, gamma - [ s_{e*}(z) - s_j(z) ])
L         = L_task + lambda * L_comp          lambda = 1
gamma                                          = 0.2, fixed
```

`gamma = 0.2` is pinned because the prototype/cosine scores live on a bounded
scale near `[-1, 1]`, where a large margin stops being a constraint; the value is
fixed before the run and is not a free parameter.

**The mean over negatives is load-bearing.** `1/(T_t - 1)` is not decoration:
without it the comparative term's effective weight grows as tasks accumulate -
more negatives are summed per example - which would inject an implicit
curriculum on top of the loss shape and make any late-task difference
uninterpretable. Fixed here, the comparative term has the same weight at task 1
and at task 20.

**Protocol, and what the comparison holds fixed.** The evidence the loss is
computed on is the **stored class prototypes of the seen experts** - the same
evidence the prototype router uses and the same evidence the Representation x
Routing factorial's pointwise reference was trained on - not raw samples. A
sample-trained gate without row locking collapses in a first attempt (C@3 0.175
against the prototype router's 0.949): training on task `t`'s samples pushes the
older experts' rows down, and their own tasks' samples never check them again, so
both arms are degenerate and their difference is a difference between two broken
scorers. Pinning the evidence removes that failure mode from both arms equally.

Both arms train the *same* scorer architecture (a `z -> T` linear gate on the
L2-normalised feature) with the same optimizer, epochs, learning rate and seed,
on the same evidence, with the same inference (no task id, one score per expert, `argmax`). The rows are **not
locked**: R2's failure in the Router Ranking Study was one-vs-previous training
followed by a global argmax, and at every step the loss here is computed over
*all* experts seen so far, so no row is ever trained against a subset. The only
difference between the arms is the loss:

```text
pointwise     cross-entropy over the seen experts, target = the owner expert
comparative   the hinge above, over owner-vs-all pairs, where a training example
              is a stored prototype and e* is its owner expert
```

The expert bank, the partition, rank, prototype count, budget and seeds are
fixed, and the bank is not trained by either loss.

**A consequence of the pair rule, pinned before the run.** For the *first*
expert there are no negatives yet, so the comparative loss is empty and that row
receives no positive signal from its own task; the pointwise arm has no such
gap. The primary endpoint is therefore reported over all tasks, and a
pre-declared secondary recomputes `Delta C@3` over tasks `1..T-1`, where every
task has negatives. Both numbers are reported, and the aggregate is primary.

If a comparative objective improves coverage while the pointwise one does not,
the formulation was the limit. If it does not, the constraint is not the
supervision's shape either, and the winner-take-all half becomes the next
hypothesis with a stronger prior.

## 2b. The three arms, and what each isolates

| arm | training | role |
| :-- | :-- | :-- |
| prototype router | none | local non-parametric **absolute reference** |
| pointwise gate | cross-entropy on stored prototypes | learned global **pointwise reference** |
| comparative gate | the hinge on the same stored prototypes | **primary manipulation** |

```text
prototype -> pointwise      isolates "local rule -> global learned fit"
pointwise -> comparative    isolates the supervision's shape (one variable)
```

For that reason the primary endpoint is `comparative - pointwise`, and the
result is reported under a neutral name - the **comparative-minus-pointwise
coverage difference** - not as a "routing gain". The study measures the effect of
the supervision's form; it does not decide which form is right.

**Both anchors are kept, separately, because they are not the same quantity:**

```text
C@3 prototype router   0.9494 / 0.8915   (RRF frozen_off)
C@3 pointwise gate     0.9404 / 0.8465   (RRF frozen_on, prototypes + CE)
```

Reproduction is a **veto, not a tolerance**: if the pointwise gate does not
reproduce its anchor, the comparative result is not read at all; if the prototype
router does not reproduce its own, the run is invalid rather than partial.

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
| `C@3` improves against the pointwise arm but not against the prototype router | the loss shape helps, but neither learned scorer beats the local rule |
| the two arms differ on tasks `1..T-1` but not in the aggregate | the comparative signal works where it exists; the first task's structural gap masks it |
| `C@3` improves with no tax change | coverage improved where the loss is not; the tax is not driven by the candidates missed pointwise |
| no `C@3` change | the supervision's shape is not the constraint; the next hypothesis is the winner-take-all decision itself |
| `C@3` falls | comparative supervision is harder to fit at this scale; the pointwise form was not the limit but the easier problem |

The third row is the one that most sharpens the programme: it would leave the
decision rule as the last untested piece of the invariant, with two independent
negative results (pointwise supervision, comparative supervision) behind it.

## 6. Statistics

Families are declared separately so it is never ambiguous later what entered a
correction:

```text
primary family      comparative - pointwise,  2 regimes   (the loss-only contrast)
secondary family    comparative - prototype,  2 regimes   (against the local rule)
secondary diagnostic   the same contrasts over tasks 1..T-1, where negatives exist
```

Paired by seed, six seeds, exact sign and permutation tests, Westfall-Young
max-statistic correction **within each family separately**, TOST where the claim
is equivalence.

## 7. Cost

```text
2 trained arms + 1 training-free control
x 2 regimes x 6 seeds
= 36 evaluation cells
= 24 trainings + 12 training-free evaluations
```

One bank per (regime, seed) is trained and shared by all three arms, in one
process with one evaluation path, so the arms differ only in the scorer. The two
anchored arms (prototype router, pointwise gate) must reproduce their values or
the run is invalid.

## 8. Out of scope

```text
winner-take-all / soft mixture    a separate hypothesis, separate document
new expert types                  no; the bank is the existing residual adapter
representation changes            no; the RRF result moved the variable here
capacity sweeps                   no; S8/S11 measured it as not the constraint
rehearsal or stored features      no
```
