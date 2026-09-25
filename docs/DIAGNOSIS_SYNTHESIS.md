# Diagnosis synthesis: Stage 1 -> Router Ranking -> Representation x Routing

Status: **frozen diagnosis, no new experiment.** This document exists so that the
next pre-registration cannot be shaped by what the evidence turned out to say.
It states what three programmes have jointly established, and nothing more.

## 1. The chain

**Stage 1** (`STAGE1_RESULTS.md`): expert capacity exists and saturates early;
scaling it does not close the routing tax (flat over rank 2 -> 128, and growing
with `T` while the oracle improves); routing is the bottleneck; the tax is a
property of the task geometry (13.98 `coherent` against 26.63 `dispersed`); router
resolution is the only lever measured to move it, and it does not finish the job
(11.24 / 23.93 left behind).

**Router Ranking Study** (`ROUTER_RANKING_PREREG.md`,
`ROUTER_RANKING_RESULTS.md`): replacing the prototype ranking with a `z -> T`
supervised task-compatibility gate on frozen features produced
`Delta C@3 = -0.079 / -0.212`. Its post-mortem located a protocol defect
(one-vs-previous rows, then a global argmax).

**Representation x Routing Objective** (`REPRESENTATION_ROUTING_PREREG.md`,
`REPRESENTATION_ROUTING_RESULTS.md`): the defect was fixed - every step uses
cross-entropy over *all* seen experts against their stored prototypes - and the
objective **still lost** (-0.0090 / -0.0450). The expert-adapted representation
also lost (-0.0502 / -0.0332). Both single-factor explanations are retired. The
interaction is positive and significant (+0.0063 / +0.0298) but an order of
magnitude too small to close either gap.

## 2. The three points that must not be lost

**First: "it was the one-vs-previous bug" is no longer available.** That bug was
real, and it was fixed. A globally consistent learned fit on *the same stored
evidence the prototype router uses* is still worse than the local non-parametric
rule. So the failure is not that the prototype metric lacks sophistication, and
not that the objective was trained badly.

**Second: the trade-off.** In the expert-adapted arms, coverage falls while the
conditional oracle rises:

```text
coherent     C@3 0.9494 -> 0.8992    conditional_oracle@3 0.8811 -> 0.8943
dispersed    C@3 0.8915 -> 0.8583    conditional_oracle@3 0.9847 -> 0.9913
```

A transformation that improves **within-expert discrimination** degrades
**cross-expert separability**. This is the strongest mechanistic finding the
programme has produced, and it locates the constraint precisely: not in the
quality of the experts, and not in the evidence, but in how experts relate to
each other in the ranking space.

**Third: the interaction reads as partial compensation, not as a solution.**
Representation and routing objective are not unrelated - training them together
is better than the sum of the two separate harms - but "joint training solves
it" does not follow. The correct statement is:

> **Joint adaptation partially offsets the harm introduced by either factor, but
> does not restore the baseline routing quality.**

## 3. The invariant that all attempts kept

Every routing mechanism tried so far, across three programmes, has the same
abstract shape:

```text
z  ->  { s_1(z), ..., s_T(z) }  ->  argmax_e s_e(z)
```

one independent score per expert, a competition between experts, and a single
winner. The prototype router does this, the learned gate does this, the
adapted-space router does this, and adding prototypes does this with sharper
scores.

The measured constraint is cross-expert separability (point two). Pointwise
scores with a winner-take-all decision is exactly the structure that has to
produce separability as a *side effect* of each expert's own score being good.
The evidence says it does not.

## 4. What follows, and what does not

The next variable is therefore the **decision structure**, not the
representation, not the router's capacity and not its features. The measured
question:

> Is routing failure caused by the pointwise winner-take-all formulation, rather
> than by the quality of the individual expert scores?

**Not claimed here:** that a different decision structure will work; that the
expert formulation is wrong; that representation plasticity is useless (the
interaction is positive); or anything that would reopen Stage 1, which stays
frozen.

**Also not claimed:** that the two factors of the RRF were fully separated. The
adapted arms changed both the space and the per-candidate graph, so the
dissociation in point two identifies *where* the constraint is, not which of
those two components caused it.

## 5. Where the next pre-registration starts

`DECISION_ROUTING_PREREG.md`, on the pointwise half of the question, because that
half can be manipulated while the decision rule is held fixed - one variable, as
the programme requires. The winner-take-all half is a separate hypothesis and
gets its own document if it is reached.
