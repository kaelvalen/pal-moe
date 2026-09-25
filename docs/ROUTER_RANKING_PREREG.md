# Router Ranking Study - pre-registration

Status: **pre-registered, run, and refuted in the opposite direction.** The
outcome is reported separately in `ROUTER_RANKING_RESULTS.md`; this document is
kept as the pre-registration it was. This is a new programme, not a Stage 1
stage. Stage 1 is frozen at `STAGE1_RESULTS.md`; nothing here is evidence for
its conclusions, and nothing here may be reported as if it were.

The distinction matters because of how this programme was produced. Stage 1
measured a failure mode; this study proposes a mechanism aimed at that failure
mode. A mechanism that works would not retroactively validate Stage 1's
diagnosis, and a mechanism that fails would not falsify it - it would only show
that this particular fix does not address it. The pre-registration exists so
that both outcomes are informative and neither is post-hoc.

---

## 1. The question

> Can better ranking realize the existing expert capacity?

The mechanism the study bets on, stated so it can fail:

```text
existing expert bank
      -> better candidate ranking
      -> the correct expert enters the top-m candidates
      -> the existing selector can exploit it
```

## 2. What Stage 1 hands over (all measured, all cited)

| finding | evidence |
| :-- | :-- |
| capacity is not the constraint | S8: the tax is flat over a 64x rank sweep and if anything widens; S10: `L4` gains +3.8 to +6.5 points with `T` while `L3` is flat |
| the failure is in the **ranking** | S10: at a fixed candidate budget `m`, coverage falls with `T` while the selector headroom at fixed `m` is constant |
| resolution is the only lever that moves the tax | S8/S11: 16 prototypes per class move the tax by -2.76 / -2.71 points, leaving 11.24 / 23.93 |
| the failure mode is **diffuseness**, not confident error | S6, S9: assignment `max-share` and `recall@3` fall together |
| the selector is not the scaling bottleneck | S10: `headroom(3)` is 0.162 at `T = 5` and 0.170 at `T = 25` |

**One correction to the hand-over, and it matters for the design.** "Selector
headroom is constant" does *not* mean the selector is worthless: at `m = 3`
there are still ~16-17 points of conditional headroom in `dispersed`, which is a
usable secondary opportunity. The claim is about scaling, not about value.

## 3. The control matrix

```text
fixed                          changed
-----                          -------
backbone (frozen ViT-B/16)     router ranking mechanism
partition (S6b constructions)  and nothing else
expert bank (one per task)
expert parameters (rank 8)
prototypes per class (1)
candidate budget m (1, 2, 3, 4, 8, T)
seeds (six, the S11 protocol)
training budget (10 epochs, lr 1e-3)
```

Expert capacity does not move. This is the confound S10 could not remove (bank
size and candidate count are collinear when one expert owns one task) and the
one thing this study must not reintroduce.

**Expert training and router training are separate, and the bank is provably
untouched.** The ladder trains its expert and readout with the *task index* as
the routing target (`fit_task` passes `ids = task_index`, never the router's
output), so the bank's training trajectory does not depend on the router at all.
R2's gate is therefore fitted *post hoc* on the cached features, after the bank
is trained, and the bank is literally the same object in both arms. The gate also
receives no gradient from the experts: its loss is a function of `gate(z)` alone.
The checklist this buys:

```text
backbone           identical (the cache)
partition          identical (S6b's construction)
trained expert bank identical (same object; a parameter hash is recorded per cell)
prototype memory    identical
rank                identical
budget              identical (declared above)
seed                identical
--------------------------------------------------------------
router state/objective   R0: registered class means
                         R2: the learned gate
```

If R2's training changed any expert parameter, `Delta C@3` would not be a ranking
effect and the comparison would be void.

## 4. The router ladder (minimal, and one arm is already measured)

```text
R0  prototype ranking, one mean per class     the S8/S11 baseline, measured
R1  prototype ranking, 16 prototypes/class    the S8/S11 memory axis, measured
R2  learned compatibility ranking             the new hypothesis
```

R1 is a **control, not a candidate**: it is exactly the memory axis S8 and S11
already measured at six seeds, so its `Delta C@3` is known before this study runs
(+0.008 `coherent`, +0.021 `dispersed`). The new hypothesis is that R2 improves
coverage *beyond memory resolution*:

```text
S8/S11:  more router memory      -> coverage improves
this:    a different ranking     -> coverage improves beyond that
```

**R2 is not a new architecture, and its objective is pinned exactly.** It is the
v1 `DynamicRouter` gate (`pal_moe/models/router.py`), a single
`nn.Linear(dim, num_experts)`, and it is not a generic "compatibility" scorer: it
is a **`z -> T` supervised task-compatibility scorer trained against the observed
task identity**.

```text
training     input z (the task's features), target = the observed task identity
             cross-entropy over the experts seen so far
             `lock_historical_routing(t)` freezes the first t expert rows
inference    input z only. No task id. Output: T routing logits -> top-m
```

So the claim a positive result would test is specific: **observed task identity
during training can produce a better routing ranking for task inference.** R2
does not get its power from a better similarity function; it gets it from using
task supervision as the routing objective. It enters through the S1 router
registry, so the ladder code path is unchanged, and the oracle protocol remains
excluded because no task id is available at inference.

**The gate's budget is declared, not tuned.** Ten epochs at batch size 128 and
`lr = 1e-3` - the same budget the expert and the readout receive - so no budget
asymmetry can explain an R0/R2 difference. Because "the learned gate loses" could
still be a tuning artefact, the study *also* reports an exploratory sensitivity
arm at 50 epochs and 200 epochs with a higher learning rate. That arm is
secondary and cannot rescue the primary endpoint.

**Resource position at `T = 20`, `dim = 768`:**

| router | stored | note |
| :-- | --: | :-- |
| R0 | 307 KB | 100 class means |
| R1 | 4.9 MB | 1,600 prototypes, 16x R0 |
| R2 | **61.5 KB** | 20 x (768 + 1) floats, 5x cheaper than R0 |

So R2 is the cheapest router in the ladder. If it wins, it wins on mechanism and
not on budget.

## 5. Endpoints

**Primary (mechanism):**

```text
Delta C@3 = C_R2@3 - C_R0@3
```

Coverage is the router's own quantity, so this is the direct test of the bet. A
new router that raises accuracy without raising coverage has improved something
else and does not count.

**Coverage's ground truth is pinned:** `C@m = P(e* in Top_m(R(z)))` where `e*` is
the **expert bank's known owner expert** - the task assignment - and *not* `L4`'s
prediction.

```text
routing ground truth        = the owner expert / task assignment
classification ground truth = the class label
```

Keeping those apart is what makes `Delta C@3` a measure of ranking quality rather
than a second accuracy number.

**Secondary:**

```text
Delta tax       = (L4 - L3)_R2 - (L4 - L3)_R0
Delta R_iso_ncm = (L3 - L0)/(L4 - L0) | R2  -  the same | R0
```

**The guard that makes the primary endpoint interpretable**, stated as the
identity it is:

```text
Ceiling@m = Coverage@m x Acc_conditional_oracle,m
```

With the bank and the candidate-conditioned selector fixed, the guard is
`Delta L4 = 0` **exactly**: the oracle's overall accuracy is router-independent,
so any movement means the manipulation leaked into the bank. `conditional_oracle@m`
is reported rather than asserted equal, because it is conditioned on the covered
set and the covered set changes when coverage changes. The study checks the
invariance empirically by evaluating the oracle route before and after the router
swap (it must be bit-identical) and records a parameter hash of the bank and
readout in both arms.

**The candidate-set decomposition is carried over from S10**, with `m = 3`
primary and the same table shape:

| m | coverage | ceiling | headroom | selection gap |
| -: | -------: | ------: | -------: | ------------: |
| 1 | | | | |
| 3 | | | | |
| 8 | | | | |
| T | | | | |

## 6. Success criterion, fixed in advance

```text
success      Delta C@3 > 0   AND   Delta tax < 0
preferred    and Delta R_iso_ncm > 0
strong       and a meaningful share of the m = 3 headroom is closed
```

All three quantities are paired over the six seeds and tested with the S11
protocol: exact sign and permutation tests, and a Westfall-Young max-statistic
correction over this study's family. `Delta C@3` is the primary; the other two
are secondary and reported as such.

A "meaningful share" of the headroom is deliberately *not* given a number in
advance, because the headroom at `m = 3` is 16-17 points and any threshold would
be arbitrary. The study reports the descriptive quantity

```text
headroom fraction recovered = (C_R2 - C_R0) / (Ceiling_R0 - C_R0)
```

and the reader judges it. It is **not** a success threshold.

**The budget comparison is a confound control, not a claim.** R2 is the cheapest
router in the ladder (61.5 KB against R0's 307 KB and R1's 4.9 MB), but the
study's claim is a *ranking* advantage, not a resource advantage. The bytes are
reported so that a win cannot be attributed to a budget difference; they are not
part of the success criterion.

## 7. What each outcome means

| outcome | reading |
| :-- | :-- |
| `Delta C@3 > 0`, `Delta tax < 0` | the ranking was the binding constraint, and a learned compatibility score addresses it |
| `Delta C@3 > 0`, `Delta tax = 0` | coverage improved but not where the loss is: the tax is not driven by the candidates the ranking misses |
| `Delta C@3 = 0` | the prototype scorer was not the limitation. The Stage 1 diagnosis strengthens and moves outward: **expert assignment may require a representation or routing objective that frozen-space ranking cannot provide** |
| `Delta C@3 < 0` | a learned gate is worse than nearest-prototype ranking on frozen features; a negative result about learned routers at this scale |

The third row is the most valuable failure. It does not weaken Stage 1 - it
sharpens the next question from "which router?" to "does routing need a
different *representation*, or a routing objective trained jointly with the
experts?".

## 8. Cost and anchors

Because R0, R1, `L0`, `L1`, `L2b` and `L4` are all already measured at six seeds
in S11, the study's new cells are only **`L3` under R2: 2 regimes x 6 seeds = 12
trainings**, plus the coverage decomposition at `m in {1, 2, 3, 4, 8, T}`.

Anchors, checked exactly:

- `L4` under R2 must equal S11's `L4` to the digit (the oracle route does not
  use the router, so a difference means the manipulation leaked).
- `L0`, `L1`, `L2b` are reused unchanged and must match S11 exactly.
- The `R0` arm must reproduce S11's `L3` exactly before R2 is compared to it.

## 9. Out of scope

```text
representation changes        no; that is the *next* study, and only if this one fails
expert capacity               no; S8/S10 measured it as not the constraint
new losses, merge rules,
rejection heads               no
PAL-MoE changes               no; only if the representation-vs-routing study asks for them
```

## 10. Programme order

```text
Router Ranking Study          this document
  -> representation-vs-routing study   only if Delta C@3 = 0
  -> new PAL-MoE                       only if the representation study asks for it
```

Each arrow is a new pre-registration. No step inherits the previous step's
evidence as its own justification.
