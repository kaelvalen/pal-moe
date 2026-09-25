# Representation x Routing Objective - results

Status: **run. Neither factor succeeds alone, and the interaction is real but far
too small to overcome either factor's cost** - the fourth row of the
pre-registration's outcome table. Pre-registration:
`docs/REPRESENTATION_ROUTING_PREREG.md`.

## 1. The four arms

`C@3` / `conditional_oracle@3` / `ceiling@3`, six seeds:

| arm | `coherent` | `dispersed` |
| :-- | :-- | :-- |
| `frozen_off` (baseline, Stage 1's ladder) | **0.9494** / 0.8811 / 0.8365 | **0.8915** / 0.9847 / 0.8778 |
| `frozen_on` (globally consistent routing objective) | 0.9404 / 0.8797 / 0.8273 | 0.8465 / 0.9833 / 0.8324 |
| `trainable_off` (candidate-specific adapted space) | 0.8992 / **0.8943** / 0.8042 | 0.8583 / **0.9913** / 0.8499 |
| `trainable_on` (interaction) | 0.8966 / 0.8835 / 0.7921 | 0.8322 / 0.9894 / 0.8104 |

The three pre-registered effects, paired over seeds, all six rejecting at the
Westfall-Young-adjusted p = 0.0312:

| effect | per-seed (six) | mean | sd |
| :-- | :-- | --: | --: |
| routing objective, `coherent` | -0.0093 -0.0089 -0.0095 -0.0083 -0.0090 -0.0088 | **-0.0090** | 0.0004 |
| routing objective, `dispersed` | -0.0452 -0.0449 -0.0449 -0.0451 -0.0448 -0.0451 | **-0.0450** | 0.0002 |
| expert-adapted space, `coherent` | -0.0478 -0.0523 -0.0501 -0.0531 -0.0436 -0.0542 | **-0.0502** | 0.0040 |
| expert-adapted space, `dispersed` | -0.0379 -0.0296 -0.0293 -0.0276 -0.0326 -0.0420 | **-0.0332** | 0.0056 |
| interaction, `coherent` | +0.0053 +0.0049 +0.0044 +0.0090 +0.0013 +0.0131 | **+0.0063** | 0.0041 |
| interaction, `dispersed` | +0.0374 +0.0300 +0.0267 +0.0286 +0.0281 +0.0278 | **+0.0298** | 0.0039 |

**Verdict: neither single factor improves coverage, and the interaction - though
real and significant - is an order of magnitude too small to close either gap.**
The pre-registration's fourth row: task-identity supervision over this expert
formulation is insufficient, and the next question is about the *formulation*,
not about routers or plasticity.

## 2. What this does to the Router Ranking Study's reading

The Router Ranking Study attributed R2's failure to its protocol: rows trained
one-vs-previous, then a global argmax. This study fixes exactly that - every step
uses cross-entropy over *all* seen experts against their stored prototypes, so no
row is ever trained one-vs-previous - and the objective **still loses**
(-0.0090 / -0.0450). So the protocol was a real defect but not the whole story:

> a globally consistent learned fit on the *same stored evidence* the prototype
> router uses is still worse than the local non-parametric rule.

That is a stronger statement than the Router Ranking Study could make, and it
retires the "the objective just needed to be trained consistently" explanation.

## 3. The dissociation that is the real result

In the adapted arms, **coverage falls while the conditional oracle rises**:

```text
coherent     C@3 0.9494 -> 0.8992   conditional_oracle@3 0.8811 -> 0.8943
dispersed    C@3 0.8915 -> 0.8583   conditional_oracle@3 0.9847 -> 0.9913
```

The expert-adapted representation makes each expert **more accurate within its own
task** and **less distinguishable from the other experts**. The same
transformation that improves within-expert discrimination degrades the
cross-expert separability the ranking needs. This is the pre-registration's
coverage/oracle guard firing in the reverse direction from the one it was written
for, and it is the sharpest mechanistic statement the programme has produced:

> the routing constraint is **cross-expert separability in the ranking space**,
> and it is not fixed by using the same evidence differently (a learned global
> fit) or by moving to the adapted space (which trades it away for within-expert
> accuracy).

It also explains why S8/S11's resolution axis helps but does not finish the job:
more prototypes sharpen local estimates, which is a within-expert (and slightly
cross-expert) improvement, not a change in the separability structure.

## 4. Guards

| guard | result |
| :-- | :-- |
| bank identity across `frozen_off` / `frozen_on` / `trainable_off` | identical hashes, 6 seeds each, both regimes |
| `trainable_on` bank | differs from the shared bank, as designed |
| `frozen_off` reproduces S11's `L3` | 12/12 cells, `max |delta| = 0.0` |
| prototype source recorded per arm | raw `z` for the frozen arms, `E_e(z)` snapshots for the trained ones |

So the only thing that varied between the three shared-bank arms was the ranking,
and the interaction arm's difference is the routing loss in the expert's training
loop - which is what the bank hash difference records.

## 5. What this does not say

- It does not say a learned routing objective is impossible. It says
  task-identity supervision, on stored prototypes, with this expert formulation,
  in either space, is not enough.
- It does not say plasticity is harmful in general: the interaction term is
  positive, so training the representation *with* the objective is better than
  not, just not better than the baseline.
- The adapted-space arms change two things (the space and the per-candidate
  graph); a positive result would have been reported as "ranking in the
  expert-adapted representation", and the same limitation applies in reverse to
  the negative result - the space and the graph are not separated here.
- Nothing here reopens Stage 1, which is frozen.

## 6. The next pre-registration

The measured question is now:

> what should an expert be, or what should its supervision say, so that the
> routing decision becomes separable across experts without giving up
> within-expert accuracy?

Everything the programme has tried so far keeps the pairwise structure of the
problem fixed: one expert per task, a class-mean or fitted score per expert, and
a win-by-maximum decision. The dissociation in section 3 says that structure is
what fails. So the next pre-registration should vary the *decision structure*
itself - for example whether the ranking is a competition between experts at all,
or whether supervision should be defined on comparisons (this pair vs that pair)
rather than on individual scores.

`docs/REPRESENTATION_ROUTING_PREREG.md` names the fourth row as "the formulation
is the limit". That is where the evidence now points, and the chain is frozen in
`DIAGNOSIS_SYNTHESIS.md` so the next pre-registration cannot be shaped by what
the evidence turned out to say. The next pre-registration is
`DECISION_ROUTING_PREREG.md`: it tests the *pointwise* half of the winner-take-all
formulation - comparative supervision, with the decision rule held fixed - because
the winner-take-all half changes how experts combine and is therefore a separate
hypothesis.
