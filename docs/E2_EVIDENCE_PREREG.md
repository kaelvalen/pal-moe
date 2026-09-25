# E2: evidence-producing experts - pre-registration

Status: **proposed, awaiting approval, not started.** It follows
`EXPERT_FORMULATION_RESULTS.md`, whose structural finding is what makes E2 both
necessary and different:

> the projection sat **between** the expert and the readout while the router
> scored the **raw feature**, so E1's candidate set could not change. `Delta C@3 =
> 0` was structural, not empirical.

So E2 is the first study in the chain in which the router reads **information the
expert itself produced**.

## 1. The question

> **Can an expert formulation that produces task-comparable evidence, rather than
> task-local logits, provide a routing signal that is usable across experts?**

The distinction the whole chain has been circling:

```text
task-local classification quality     !=   cross-expert evidence comparability
```

E1 measured the second without touching the first and lost a little of both. E2
makes the second an explicit target.

## 2. The formulation, pinned

### 2.1 Evidence

```text
h_t = normalize( W_e E_t(z) )        W_e : dim -> 128,  one per expert
```

`d_e = 128` fixed, L2-normalised, so every expert's evidence is a point on the
unit sphere of **one shared comparison space**. That is the whole point: two
experts' outputs are comparable because they are two directions in the same
space, not because a downstream rule rescales them.

### 2.2 The router's access point

```text
s_t = < P z , h_t >                  P : dim -> 128, shared, no activation
e* = argmax_t s_t                    winner-take-all, unchanged
```

The router scores the **evidence**, for **every** expert (`T` evidence
computations per sample, cost recorded); the query `P z` is shared, so the
comparison is a single shared bilinear form and comparability cannot come from a
per-expert scoring head. This is the deliberate and necessary difference from
E1, where the router read the raw feature.

### 2.3 The objective

```text
L = L_task + lambda * L_evidence        lambda = 1.0, fixed
L_task      the ladder's task loss through the expert, unchanged
L_evidence  cross-entropy over the SEEN experts on s_t, target = the owner expert
```

`L_evidence` keeps the *form* the baseline's supervised routing already used -
the decision-routing study closed supervision *shapes*, and this is not a new
shape - but it is now computed **through the experts' evidence**, so the experts'
own outputs are shaped to be comparable. Without it the study would repeat E1
with an extra projection. `L_task` alone is exactly what produced E1's trade-off.

### 2.4 What is held fixed

```text
partition, rank, prototypes, seeds, epochs, lr, budget   unchanged
decision rule                                            winner-take-all (AGG
                                                          closed aggregation)
readout for classification                               the ladder's shared one
```

## 3. Endpoints

**Primary:** `Delta Acc` against the E0 baseline, paired by seed.

**Read together, never singly** - this is the first study where the router can
move, so both sides can move:

```text
C@3                    cross-expert routing evidence
conditional_oracle@3   usable expert capacity
Acc                    end-to-end
Ceiling@3 = C@3 x oracle@3   which side moved
```

## 4. Outcome reading, fixed in advance

| result | reading |
| :-- | :-- |
| `C@3` up **and** `oracle` up | the strongest success form so far: routing evidence and usable capacity improve together |
| `C@3` up, `oracle` down | a trade: routing improves at the cost of expert usability |
| `C@3` down, `oracle` up | evidence quality improves but stays incomparable across experts |
| both down or both flat | comparable evidence is not reachable this way; the failure is not in the output formulation at all |

No success threshold: the effect and its family-wise corrected significance are
reported, and the row is named.

## 5. Guards

```text
E0 anchor            the baseline arm reproduces S11's L3 per seed exactly,
                     keyed by (construct, level, rank, protos, num_tasks, seed)
shared query         one P for every expert, by object identity, plus an assert
                     that no expert carries its own query
evidence dimension   d_e = 128 in every cell; recorded
O(T) cost            the evidence computation for all experts is recorded per cell
bank                 recorded; the classification readout is the ladder's own
scope                no aggregation change, no new router family, no capacity or
                     resolution sweep, no rehearsal
```

## 6. Statistics and cost

The S11 protocol: paired by seed, six seeds, exact sign and permutation tests,
Westfall-Young within each declared family (`Delta Acc` primary over two regimes;
the three mechanistic metrics secondary in their own family).

```text
2 arms (E0, E2) x 2 regimes x 6 seeds = 24 cells
```

## 7. Scope, stated to keep the family singular

This pre-registration tests **one** formulation: an evidence-producing expert with
a shared bilinear comparison and a routing loss through the evidence. It is not a
learned shared embedding study, not a prototype-learner study, not metric
learning, and not a contrastive-expert study; those are different families with
different failure modes, and folding them in would make a negative result name
nothing. If E2 fails, the honest sentence is "this evidence formulation does not
work", and the next pre-registration names the next one.
