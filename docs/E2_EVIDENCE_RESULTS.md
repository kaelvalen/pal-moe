# E2 (evidence-producing experts) - non-execution and postmortem

Status: **not executed.** The pre-registered 24-cell confirmatory run was **not
started**, because its first veto (a usable confirmatory operating point) failed
and because the pre-registered protocol is computationally infeasible at this
scale. The pre-registration is unchanged: `docs/E2_EVIDENCE_PREREG.md`.

## 1. What was built and what passed

The implementation exists (`experiments/e2_evidence.py`) and the full veto chain
passed on the seed-42 smoke:

```text
E0 anchor (S11, keyed with num_tasks)   max |delta| = 0.0            PASS
all-W gradient guard                    old_W_with_gradient == total  PASS
previous-adapter freeze guard           not in optimizer, frozen      PASS
d_e = 128, W bias = False                                              PASS
L_evidence contains no g                                               PASS
```

So the contract of section 2 of the pre-registration was executed as written:
every stored prototype, every seen expert, one shared query, averaged and never
summed, at every optimizer step.

## 2. The three observations, frozen

```text
single expert,  the W -> 128 -> g pathway        78.80 %   works
20 experts, lambda = 1   routing 0.9041, acc 13.04 %        collapse
20 experts, lambda = 0   routing 0.1459, acc  3.77 %        no rescue
```

**The `lambda = 0` result does not isolate the cause.** As the pre-registration's
own construction implies, removing `L_evidence` also removes the only training
signal for the shared query `P` and for the older experts' projections `W_e`;
`P` receives no gradient at all. The observed further collapse is therefore
almost the expected consequence of removing the objective, not evidence about
what the objective causes. The safe statement is:

> The `lambda = 0` diagnostic did not rescue the 20-expert system; removing the
> evidence objective caused both routing and classification to collapse further.
> Therefore, the collapse cannot be attributed to an over-strong evidence term
> alone. The diagnostic does not uniquely identify the remaining cause, because
> `lambda = 0` also removes the only training signal for the shared
> routing/evidence pathway.

What can be said safely, and only this:

> **The single-expert evidence pathway is valid, but this particular continual
> multi-expert training formulation does not scale to the 20-expert setting.**

It cannot yet be said that the shared readout is the cause.

## 3. Why the confirmatory run stops, in two separate senses

**(a) Scientific: the formulation is degenerate at this operating point.** 78.80 %
single-expert against 13.04 % at 20 experts means there is no usable confirmatory
operating point; running 24 cells would only repeat a collapsed measurement across
seeds.

**(b) Practical: the pre-registered protocol is infeasible.** One 20-expert cell
cost 3095 s, so the 24-cell grid is approximately **20.6 hours** under the pinned
contract (every optimizer step, full batch over all 100 prototypes times all seen
experts, with backward through every expert). This is a **protocol feasibility
failure, not a scientific falsification**, and the two are kept apart here.

Neither reason changes the pre-registration. What failed is the execution of the
registered plan, and the plan is not rewritten afterwards.

## 4. What this closes, and the shape of the next question

E2 was the last untested component of the invariant: it was the first study in
which the router reads information the expert itself produced, and the evidence
pathway itself works. What does not work is the **continual coupling across
accumulated experts**, and the next pre-registration must be neutral about which
component breaks it - `L_evidence`, the shared readout `g`, the shared query `P`,
expert accumulation, or the old/new expert interaction - rather than naming one of
them in advance:

> **How does continual coupling across accumulated experts prevent a locally valid
> evidence representation from remaining usable globally?**

That is a separate pre-registration. This document closes E2 as what it is: a
registered experiment that was not executable, with the diagnostics that
establish why, and without redefining the failed experiment's cause after the
fact.
