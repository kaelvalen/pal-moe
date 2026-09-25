# Representation x Routing Objective - pre-registration

Status: **approved, run, reported in `REPRESENTATION_ROUTING_RESULTS.md`.** Outcome:
neither factor succeeds alone and the interaction is real but insufficient - the
fourth row of section 7. This document is kept as the pre-registration it was. Third programme document in
the sequence:

```text
STAGE1_RESULTS.md              Stage 1, frozen
ROUTER_RANKING_PREREG.md       the ranking hypothesis
ROUTER_RANKING_RESULTS.md        -> refuted in the opposite direction
REPRESENTATION_ROUTING_PREREG.md  this document
```

Nothing here is evidence for Stage 1 or for the Router Ranking Study. Stage 1
measured a failure mode; the Router Ranking Study tested one mechanism against
it and failed; this document tests *why* that mechanism failed, by separating the
two things it moved together.

---

## 1. The question

> Does routing success depend on the routing objective, on representation
> plasticity, or on the interaction?

The Router Ranking Study's R2 changed the routing objective while the
representation stayed frozen, and lost. But its own post-mortem found the reason
in the *protocol*, not in the idea: `lock_historical_routing` trains each expert's
scorer one-vs-previous, so the final argmax over all `T` rows is dominated by
later rows. It therefore **does not** show that a supervised routing objective
fails - only that a frozen-representation, post-hoc, one-vs-previous task scorer
does. This factorial is built to tell those apart.

## 2. The factorial

| representation | routing objective | what it isolates |
| :-- | :-- | :-- |
| frozen (raw `z`) | off: prototype ranking (R0) | baseline - Stage 1's ladder |
| frozen (raw `z`) | on: globally consistent supervised task-routing | **routing-objective** effect |
| trainable (candidate-specific adapted) | off: prototype ranking in the adapted space | **expert-adapted representation** effect |
| trainable (candidate-specific adapted) | on: both | **interaction** |

The three effects, all paired by seed:

```text
routing objective          C(frozen, on)      - C(frozen, off)
expert-adapted             C(trainable, off)  - C(frozen, off)
interaction                [C(trainable, on) - C(trainable, off)]
                         - [C(frozen, on)    - C(frozen, off)]
```

**Terminology, pinned.** The representation factor is *not* "representation
plasticity alone". Operationally the ranking happens in a **candidate-specific**
space:

```text
s_e(z) = max_c cos( E_e(z), p_{e,c} )
```

which changes two things at once: the space (trained rather than raw) and the
graph (each candidate is scored in its own adapted representation, not in one
shared space). So a positive result licenses

> "routing succeeds better when candidate ranking is performed in the
> expert-adapted representation"

and **not** "representation plasticity alone caused the gain". The design does
not separate those two, and the report must not claim it does. The frozen arms
score every candidate in the *same* raw space, which is exactly what makes the
baseline the comparison it is.

## 3. The two factors, pinned

### 3.1 Routing objective (the fix R2 did not have)

R2's failure was structural: each row was trained only against *earlier* rows, on
*its own* task, and then a global argmax was taken over all rows. The fix must be
globally consistent **without rehearsal**, because the ladder stores no past
features:

```text
training data    the stored class prototypes of every seen expert - the same
                 information the prototype router uses, so the contrast is
                 "the same evidence, used by a learned global fit instead of a
                 local non-parametric rule"
target           the observed task identity (the owner expert of each prototype)
loss             cross-entropy over ALL seen experts, every step
                 (no row is ever trained one-vs-previous)
inference        input z only; no task id; a ranking over T experts
```

The objective class and protocol are fixed here, before any result. If it
succeeds, the claim is "a globally consistent supervised task-routing objective
trained on stored prototypes improves ranking"; if it fails, the claim is that
this objective class is insufficient - not that the *idea* of a learned routing
objective is.

### 3.1b Prototype synchronisation, pinned

Three implementations of "the routing objective uses stored prototypes" are all
defensible and are not the same experiment:

```text
A  snapshot       prototypes fixed when the task is learned, never recomputed
B  recomputed     p_{e,c} = mean(E_e(z_c)) recomputed every step
C  stop-gradient  prototypes move, but no gradient flows through their construction
```

**This study pins A for the stored evidence, and C for the current expert's own
anchor**, because the pre-registration's central contrast is *the same stored
evidence used two ways* - and B would let the representation feed new evidence
back into the training loop, changing that contrast.

```text
old experts        stored task-local sufficient statistics (A), never recomputed
current expert     during task t's own training, a stop-gradient EMA of E_t(z)
                   serves as its positive anchor (C); it is not stored evidence
prototype timing   identical in every arm: the stored prototype for task t is
                   created from the representation state that exists right after
                   task t's training, which is also when the ladder registers its
                   own prototypes
gradients          within an arm, the routing objective never receives gradient
                   through prototype construction; expert gradients flow only
                   through the current encoded sample
```

Each arm records the prototype timing, a prototype hash and the representation
checkpoint it was taken from, so the chronology can be audited rather than
assumed. Without this, the `off` and `on` arms could silently differ in *when*
their prototypes were taken.

### 3.2 Representation plasticity

The backbone stays frozen. The trainable representation is the **existing
residual-adapter bank** - no new model family:

```text
frozen      the router scores the raw cached feature z, and prototypes are
            registered from z                        (Stage 1's ladder, unchanged)
trainable   the router scores the expert-adapted feature: for candidate expert e,
            s_e(z) = max_c cos( E_e(z), p_{e,c} )
            and prototypes are registered from the adapted features E_t(z)
```

So the representation axis is *which space the ranking happens in*, and the
`trainable` arms are trained representations by construction. The cost is `T`
adapter applications per sample (20 x 12k MACs at rank 8, about four times the
readout); it is identical in both `trainable` arms, so the comparison is fair.

When the routing objective is `on` in the `trainable` arm, the routing loss enters
the **expert's** training loop, so the expert learns features that make its own
prototypes distinguishable from the other experts'. That is the interaction term,
and it is the only arm where the routing objective can change the representation.

The loss is pinned: for each minibatch of the current task's features,

```text
scores_e(z) = max_c cos( E_e(z), p_{e,c} )   for e < t   (stored snapshots, A)
scores_t(z) = cos( E_t(z), anchor_t )                      (stop-gradient EMA, C)
L_route     = cross_entropy([scores_0 ... scores_t], target = t)
L_total     = L_task + lambda * L_route                    lambda = 1.0
```

`lambda = 1.0` matches the ladder's existing prototype-anchor weight, so the
interaction arm adds no new hyper-parameter family.

## 4. Operating point (identical to S8/S11/S12)

```text
dataset        CIFAR-100, frozen ViT-B/16, dim 768
partitions     dispersed and coherent, T = 20, the S8/S11 construction
expert         residual adapter, rank 8, one expert per task
prototypes     1 per class (the prototype count is not a factor here)
candidate m    3 (primary), 1/2/4/8/T reported
seeds          42, 1, 2, 3, 4, 5 (the S11 protocol)
bank budget    identical in all four arms
```

Expert capacity is deliberately **not** swept: S8 showed the tax is flat from
rank 2 to 128, so sweeping it again would add a confound without adding
information.

## 5. Endpoints

**Primary:** `Delta C@3` between arms, with the routing ground truth being the
**owner expert** (the task assignment), not `L4`'s prediction.

**Secondary, reported together because they can disagree:**

```text
conditional_oracle@3    coverage can rise while the experts' usable capacity
                        falls; that would mean the router ranks better in a
                        representation that carries less
ceiling@3 = coverage@3 x conditional_oracle@3
tax = L4 - L3
R_iso_ncm = (L3 - L0) / (L4 - L0)
headroom fraction recovered = (C_arm - C_baseline) / (Ceiling_baseline - C_baseline)
```

The headroom fraction is **descriptive**, not a success threshold.

## 6. Guards

```text
L4 invariance        in the two `frozen` arms L4 must equal the baseline exactly
                     (the oracle route does not use the router and the bank is
                     unchanged). In the `trainable` arms the bank changes, so the
                     guard becomes "the bank hash is recorded and identical across
                     arms that share it" rather than "L4 is unchanged".
ceiling identity     Ceiling@m = Coverage@m x Acc_conditional_oracle,m is an
                     identity; when coverage moves, conditional_oracle is
                     reported rather than asserted equal
coverage + oracle    a coverage gain with a conditional_oracle loss is reported
                     as a representation cost, not a routing win
one factor at a time no arm may differ from the baseline in more than the two
                     declared factors
```

## 7. Outcome reading, fixed in advance

| result | reading |
| :-- | :-- |
| routing objective + frozen representation succeeds | the ranking objective was the missing ingredient; representation was not the limit |
| expert-adapted representation + existing routing succeeds | the space the ranking happens in is the limiting factor; the objective was not |
| only joint training succeeds | the routing objective requires the candidate-specific adapted space to be trained with it - the interaction is the mechanism |
| neither succeeds | task-identity supervision over this expert formulation is insufficient; the next question is about the *formulation* (what an expert should be, or what supervision should say), not about routers or plasticity |

The fourth row matters most and is the one the Router Ranking Study could not
reach: it would rule out both single-factor explanations at once.

## 8. Statistics

The S11 protocol, unchanged: paired by seed, six seeds (the floor at which an
exact two-sided test can reject), exact sign and permutation tests, a
Westfall-Young max-statistic correction over the family, and TOST where the claim
is equivalence. The family is the three effects x two regimes = six tests.

## 9. Cost

```text
4 arms x 2 regimes x 6 seeds = 48 trainings, plus evaluations
```

The `frozen/off` arm is S11's `L3` cells and the `frozen/on` arm is the Router
Ranking Study's R2, both already measured at six seeds - they are re-run anyway
so that all four arms share one process, one bank per seed and one evaluation
path, and their reproduction is checked as the anchor.

## 10. Out of scope

```text
new PAL-MoE architecture     no; this is a causal follow-up, not a design
capacity sweep               no; S8/S11 measured it as not the constraint
new expert types             no; the bank is the existing residual adapter
rehearsal or stored features no; the routing objective uses stored prototypes only
corruptions or scale         no; S9/S10 measured those axes
```

## 11. Sequence

```text
Stage 1 (frozen)
  -> Router Ranking Study (R2 refuted)
  -> this factorial (representation vs routing objective vs interaction)
  -> a new PAL-MoE only if the factorial says the formulation is the limit
```
