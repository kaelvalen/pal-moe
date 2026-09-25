# Router Ranking Study - results

Status: **run, hypothesis refuted in the opposite direction.** The
pre-registration is `docs/ROUTER_RANKING_PREREG.md`; this document reports the
outcome and nothing in it may be read back into Stage 1, which is frozen at
`STAGE1_RESULTS.md`.

## 1. Outcome

The pre-registered primary endpoint was `Delta C@3 > 0`. It came out
significantly **negative**, in both regimes, on all six seeds:

| endpoint | per-seed (six seeds) | mean | sd | WY-adjusted p |
| :-- | :-- | --: | --: | --: |
| `Delta C@3` (`coherent`) | -0.0782 -0.0801 -0.0809 -0.0785 -0.0768 -0.0783 | **-0.0788** | 0.0015 | 0.0312 |
| `Delta C@3` (`dispersed`) | -0.2174 -0.2166 -0.2186 -0.2107 -0.2017 -0.2090 | **-0.2123** | 0.0065 | 0.0312 |
| `Delta tax` (`coherent`) | +0.1227 +0.1270 +0.1321 +0.1271 +0.1267 +0.1239 | **+0.1266** | 0.0033 | 0.0312 |
| `Delta tax` (`dispersed`) | +0.2730 +0.2715 +0.2749 +0.2619 +0.2587 +0.2655 | **+0.2676** | 0.0065 | 0.0312 |
| `Delta R_iso_ncm` (`coherent`) | -0.7282 -0.7431 -0.7771 -0.7490 -0.7475 -0.7344 | **-0.7465** | 0.0169 | 0.0312 |
| `Delta R_iso_ncm` (`dispersed`) | -1.0130 -1.0059 -1.0193 -0.9718 -0.9585 -0.9874 | **-0.9926** | 0.0242 | 0.0312 |

Point estimates, for orientation:

| regime | arm | accuracy | coverage@1 | coverage@3 |
| :-- | :-- | --: | --: | --: |
| `coherent` | R0 prototype | 73.21 - 73.46 | 0.820 | 0.949 |
| `coherent` | R2 learned gate | 60.15 - 60.94 | 0.693 | 0.869 |
| `dispersed` | R0 prototype | 70.62 - 70.66 | 0.713 | 0.891 |
| `dispersed` | R2 learned gate | 43.36 - 44.79 | 0.440 | 0.674 - 0.690 |

The descriptive headroom fraction recovered at `m = 3` is **-1.22**
(`coherent`) and **-1.57** (`dispersed`): R2 does not merely fail to recover the
selector headroom, it loses more accuracy than the headroom is worth, because its
routing is worse than the baseline's top-1.

This is the fourth row of the pre-registration's outcome table - "a learned gate
is worse than nearest-prototype ranking on frozen features" - and the third row's
sharper reading applies too: the prototype scorer was not the limitation, and
neither was the *linear gate* the missing piece.

## 2. The guards held, so the effect is ranking-only

| guard | result |
| :-- | :-- |
| R0 reproduces S11's `L3` | 12/12 cells, `max |delta| = 0.0` |
| the oracle route before vs after the router swap | 12/12 cells, `max |delta| = 0.0` |
| the bank/readout parameter hash after installing the gate | unchanged in 12/12 |

So the expert bank, the readout, the partition, the budget and the seeds were
identical in both arms, and `L4` is untouched. The difference is a pure ranking
effect - which is what makes the negative result interpretable rather than a
confound.

## 3. The sensitivity arm: no budget rescues it

The declared gate budget is the expert's own (10 epochs, batch 128, `lr 1e-3`).
The exploratory arm trains the same gate far longer:

| regime | 10 epochs, `lr 1e-3` (declared) | 50 epochs, `lr 1e-3` | 200 epochs, `lr 1e-2` |
| :-- | --: | --: | --: |
| `coherent` coverage@3 | 0.869 | 0.787 | **0.321** |
| `dispersed` coverage@3 | 0.683 | 0.602 | **0.216** |

More training makes it monotonically **worse**. That is the opposite of a tuning
artefact, and it is the first clue to the mechanism.

## 4. Why it fails: the frozen-rows protocol is one-vs-previous

`lock_historical_routing(t)` freezes the first `t` expert rows, so when task `t`
arrives only its own row is optimised - against the experts seen so far, with the
task identity as the target. Every row therefore learns to beat *earlier* rows on
*its own* task, and **no row is ever constrained against later tasks**.

The final routing decision is an argmax over all `T` rows, so a later row that
happens to score high on an earlier task's features wins it. Two measurements
show this directly:

```text
training loss          0.0000   (the row perfectly fits its own task)
final coverage@1       0.166    (and still loses the argmax at the end)
```

and the monotone degradation with more training in section 3: more fitting of
later rows means more domination of the final argmax.

The failure is also much larger in `dispersed` (-0.212 against -0.079), which is
where the task geometry overlaps and an unconstrained later row is most likely to
outscore an earlier task's row.

## 5. What this says about the baseline

The prototype router's advantage is **not** that cosine similarity is a better
metric than a learned linear score. It is that the ranking is **assembled from
local estimates** - class means registered incrementally, with no global fit -
so it stays consistent as the bank grows. A globally fitted classifier trained
one task at a time under a freeze policy does not, for the structural reason in
section 4.

That reading also explains a Stage 1 result the study was not designed to touch:
S8/S11's memory axis *helps* (16 prototypes per class move the tax by -2.7) while
this learned gate does not, and the difference is consistency-by-construction,
not capacity. More prototypes means more local estimates; a bigger gate means a
more over-confident global fit.

## 6. The next question, sharpened

A *globally consistent* learned ranking would need one of:

```text
(a) statistics or rehearsal of past tasks        the ladder is rehearsal-free;
                                                 this costs memory and is a
                                                 different protocol
(b) a ranking objective trained jointly with     this is the representation
    the experts                                  question - the next study
(c) a non-parametric local estimator             which is what the prototypes
                                                 already are
```

The pre-registration named (b) as the follow-up if `Delta C@3 = 0`. The measured
outcome is stronger than that: the failure is not that the prototype scorer was
already optimal, it is that **the routing signal must be made consistent, and
neither more resolution nor a supervised task scorer on frozen features makes it
so**. So the next pre-registration is the representation-vs-routing study, and
its question is now specific:

> Does routing need a different representation, or a routing objective trained
> jointly with the experts - and if jointly, does that change what the experts
> learn?

## 7. What this does not say

- It does not say learned routers cannot work. It says this protocol - frozen
  rows, one-vs-previous targets, no rehearsal, a linear gate on frozen features -
  is worse than nearest-prototype ranking, by a margin that grows with training.
- It does not say the prototype scorer is optimal. R1 (16 prototypes per class)
  still beats R0, so resolution is a real lever; the gate simply is not.
- It does not revisit Stage 1. The diagnosis Stage 1 produced is what motivated
  the hypothesis; refuting the hypothesis leaves the diagnosis where it was and
  narrows the next question.
- The `dispersed`-vs-`coherent` difference in the *size* of the failure is
  descriptive; the study was not powered to compare regimes on this endpoint.
