# Expert Formulation Study - pre-registration

Status: **proposed, awaiting approval, not started.** It follows
`DIAGNOSIS_SYNTHESIS.md` and the four negative/partial results of the chain.

## 1. The chain, and the question it leaves

```text
capacity                      -> not the constraint     (S8, S10)
resolution                    -> partial                (S8, S11)
pointwise learned ranking     -> negative               (RR, RRF)
comparative ranking           -> negative               (DR)
adapted representation        -> negative, trade-off    (RRF)
uniform top-3 aggregation     -> no improvement         (AGG)
```

The invariant all of them kept:

```text
z -> {E_1 ... E_T} -> {s_1 ... s_T} -> decision
```

and every attempt changed the *scores* or the *decision*, never the **object being
scored**. The question this study asks:

> **Can expert outputs be formulated in a common, comparable decision space
> rather than as independently task-local classifiers?**

The hypothesis behind it, and it is the one the measurements support: the scores
`{s_t(z)}` are **not intrinsically comparable across experts**, because each
expert is trained to say something about *its own* classes and nothing about how
its evidence relates to another expert's. The RRF dissociation - within-expert
discrimination up, cross-expert separability down - is exactly that.

## 2. Scope, and what this study may not touch

```text
new router                no        (RR, DR: supervision shape is closed)
new loss or objective     no        (DR: closed)
new training budget       no
new backbone              no
new memory mechanism      no
new aggregation rule      no        (AGG: closed)
                         ---
expert representation / output formulation   YES, this and only this
```

The expert bank, the partition, rank, prototype count, seeds, candidate `m` and
the decision rule (`argmax`, winner-take-all) are all held as they are.

## 3. The formulation variable, pinned

The ladder of possible formulations is, in the pre-registration's terms:

```text
E0   E_t : z -> logits_t             the current expert: a task-local classifier
E1   E_t : z -> h_t, evidence read in one shared space for every expert
E2   E_t : z -> evidence, and no expert-level classifier at all
```

**This study pins E1 and only E1.** E2 is deferred, and the reason is written
here so it cannot be chosen later: E2 necessarily changes the *training objective*
(the expert no longer classifies its own classes), and DR closed the objective
side of the invariant. Reopening it inside a formulation study would move two
variables at once.

**E1, operationalised.** The expert's output is no longer consumed as task-local
logits; every expert's adapted feature is projected into **one shared decision
space** and the class evidence comes from a shared mapping applied to that
projection:

```text
E0   s_t(z) and the class evidence are read from that expert's own local
     output, task by task
E1   the expert produces h_t = P(E_t(z)) in a single shared space, and the same
     evidence function g(h_t) is applied for every expert, so two experts'
     outputs are compared as two points in one space rather than as two
     task-local decisions
```

The manipulation is *where the expert's output lives*, not what it is trained
against: the expert's task loss, the readout family and the decision rule are
unchanged. The projection `P` is the only new parameter, it is shared by all
experts, and its cost is recorded.

**The projection's form is pinned here, so none of it can be chosen after the
run:**

```text
P            nn.Linear(dim, dim, bias=True)      no hidden layer
activation   none                                the projection is affine
normalisation none
initialisation  weight = identity, bias = 0      so E1 *is* E0 at step 0, and the
                                                 study measures what training does
                                                 to a shared decision space rather
                                                 than the effect of a random map
```

**Parameter sharing is a runtime fact, not a documentation claim.** A single `P`
module instance is used for every expert's output, and `g` is the ladder's single
shared readout; the run records one parameter hash for `P` and one for `g`, and
asserts per forward pass that no per-expert copy exists.

Both hooks apply the same transformation: the task loss in `_optimize` and the
function-preservation loss in `_l_func` read through `P`, so the two objectives
are computed in the same space. Without that, E1 would stop being a one-factor
experiment.

## 4. Endpoints and guards

**Primary:** `Delta Acc` against E0, paired by seed.

**Mechanistic secondary, all reported together** (a formulation that fixes routing
while breaking expert capacity must be visible as such):

```text
C@3                    does the candidate set improve?
conditional oracle@3   does the usable capacity survive?
ceiling@3 = coverage@3 x conditional_oracle@3
tax = L4 - L3
R_iso_ncm = (L3 - L0) / (L4 - L0)
```

The AGG lesson applies in reverse here: there, the decision mechanism changed and
the candidate set stayed identical; here coverage *may* move, so it is reported
rather than guarded, and the oracle/ceiling pair is what says whether the gain is
real.

**Guards, vetoes:**

```text
E0 anchor        the E0 arm must reproduce S11's L3 per seed exactly
bank hash        the expert bank's parameters are identical across arms
budget           same rank, prototypes, candidate m, seeds, epochs, lr
one variable     no arm may differ from E0 except in the expert's output space
```

**No success threshold.** There is no "＋X points counts as working" line. The
study reports the primary effect with the S11 statistics protocol (paired, six
seeds, exact tests, Westfall-Young within each declared family) and the four
outcome rows below.

## 5. Outcome reading, fixed in advance

| result | reading |
| :-- | :-- |
| `Delta Acc > 0` with the oracle and ceiling intact | a shared decision space is the missing ingredient; the object being scored was the constraint |
| `Delta Acc > 0` with a lower conditional oracle | the formulation trades expert capacity for comparability; a bounded, honest trade, not a fix |
| `Delta Acc ~ 0` | the formulation alone does not break the failure mode; the expert's *objective* (E2, deferred) is the next hypothesis |
| `Delta Acc < 0` | a shared projection is not enough and costs capacity; the failure is deeper than the output space |

**Veto order for the smoke, before any run:**

```text
1  E0 anchor      the E0 arm reproduces S11's L3 per seed exactly (its path is
                  byte-for-byte the unchanged ladder; `projection=None`).
                  The reference must be keyed by (construct, level, rank,
                  protos, num_tasks, seed): S11 stores two cells per such key,
                  part B (T = 20) and part E (the T = 5/25 endpoints), and a
                  key that omits num_tasks silently compares against the wrong
                  partition. The first check hit exactly that collision.
2  shared P/g     one parameter hash for P, one for g, and a per-forward assert
                  that no per-expert copy was created
3  E1 smoke       only then is the E1 arm measured
```

## 6. Statistics, cost, scope

```text
statistics   S11 protocol; primary family = Delta Acc over 2 regimes;
             secondary endpoints in their own family
cost         2 arms x 2 regimes x 6 seeds = 24 cells, one bank per (regime, seed)
             shared by both arms in one process with one evaluation path
out of scope anything in section 2, and any change to the decision rule
```
