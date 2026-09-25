# AC3: address-space ablation - fixed retrieval vs the learned evidence address - pre-registration

Status: **proposed, awaiting approval, not started.** It follows
`AC1_ADDRESS_FREEZE_RESULTS.md`, which retired "freeze more of the address path" and
moved the variable to the address **form**, and it is the `fixed-address control` the
v2 programme is gated on before AC2 (value plasticity) becomes meaningful.

## 1. The question

> Does the routing address have to be the learned bilinear evidence address
> `(P, W_j)`, or does a **fixed retrieval address** in the frozen feature space -
> class prototypes, cosine, winner-take-all - work as well **inside the E2 system**?

Three cells of the 2x2 `{address: learned, fixed} x {value path: L3 direct, E2
evidence}` already exist:

```text
L3  + fixed address     E0 / the aggregation study's WTA arm   73.29 / 70.66   0.9494 / 0.8915
E2  + learned address   C0                                    74.16 / 70.14   0.9319 / 0.8548
E2  + fixed address     this study
```

The missing cell matters because in E2 the address and the value are **the same
object**: `h_j = W_j E_j(z)` is both the routing key and the vector the readout
decodes. A fixed address breaks that identity - selection is made in the raw feature
space while the selected value is still `h_j`. Whether the readout can still decode
what a fixed-space ranking picks is exactly the unmeasured question.

## 2. Design: an evaluation-only re-scoring ablation

In the pinned E2 contract **routing is inference-only**. The training loss
(`L_task + lambda L_evidence`) contains no routing term (`experiments/e2_evidence.py`),
so the address can be swapped at evaluation time without retraining and without
changing a single trained weight.

```text
same trained cell (C0: w_alignment=current, p_alignment=plastic)
    address = bilinear     the pinned address, s_j = <normalize(Pz), normalize(W_j E_j z))   [anchor]
    address = fixed_proto  s_j = max_{c in task j} <normalize(z), normalize(p_c)>           [AC3]
```

`fixed_proto` is the E0 rule, literally: `pal_moe.arch.PrototypeRouter.expert_scores`,
registered per class with the owner task as the expert id. The prototypes are the
stored per-class train means E2 already keeps (`E2Model.prototypes`), which are
computed by the same code path the ladder uses
(`feats[mask].mean(dim=0)` on `task["splits"]["train"]`, `s2_ladder.register_task`).

Everything else is held fixed: the trained values, the readout, the candidate set
definition (owner task in top-k), the winner-take-all decision (closed by the
aggregation study), `T = 20`, the seeds.

Reference cells are re-run in the same study: E0 = `L3_per_task` per (regime, seed),
so the comparison to the published fixed-address reference is measured, not quoted.

## 3. Endpoints

**Guards (identities, not tests).**

```text
bilinear anchor       the anchor evaluation must reproduce the stored coupling /
                      intervention / owner-side C0 cells bitwise on all five metrics
address identity      fixed_proto coverage@1 and coverage@3 must be **bitwise equal**
                      to the in-study E0 cells' coverage@1 / coverage@3 - same rule,
                      same prototypes, same data. A deviation is a defect signal and
                      stops the study; it is not a scientific outcome.
evaluator equivalence the re-scoring loop with the bilinear score function must
                      reproduce `E2Model.evaluate` bitwise on every returned field
                      (including per_task_accuracy)
prototype fidelity    the router's registered means must be bitwise equal to
                      `E2Model.prototypes`
address is param-free the fixed address path contains no learned parameter
                      (`sum(p.numel() for p in router.parameters()) == 0`, and the
                      score function reads only `z` and registered buffers)
```

**Primary family** (paired over regime x seed, n = 6, one test per family member):

```text
Delta Acc        Acc_fixed_proto - Acc_bilinear
Delta ceiling@3  the accuracy upper bound the address allows
                 (coverage@3 x P(correct | covered))
```

**Read together:** `coverage@1`, `coverage@3`, `conditional_oracle@3`,
`conditional_learned@3`, `selection_gap@3`, `oracle_accuracy`, the routed accuracy
split by covered / uncovered, and the top-1 expert-share entropy.

**Secondary:** equivalence of `Acc_fixed_proto` and `Acc_bilinear` at a +/-1 pp SESOI
(`s11_confirmatory.tost`), so "no difference" is a stated result rather than an
absence of one.

## 4. Outcome reading, fixed in advance

| result | reading |
| :-- | :-- |
| `Delta Acc >= 0` (or equivalent within +/-1 pp) and the fixed address's `ceiling@3` is not lower | the learned address is **not necessary**: a fixed-space retrieval is a drop-in for the routing path, and the higher coverage (`0.9494` against `0.9319`) comes for free. `P` and `W` can leave the address path; the remaining variable is the value path - **AC2 becomes meaningful** |
| `Delta Acc < 0` beyond the SESOI while `coverage@3` is at least the bilinear arm's | the address ranks *better* on the coverage measure but its selections are **not decodable** by the readout: in E2 the address/value coupling is functional, not incidental. The next variable is the value/readout path with the address held fixed - still AC2, but with a value-side objective, and the fixed-address "drop-in" claim is refuted |
| `Delta ceiling@3 > 0` with `Delta Acc` near zero | the address leaves headroom the readout does not use: report the decomposition and treat the value/readout path as the binding constraint |

All three readings are decision-relevant and all three name the next variable. A null
is a result (`DECISION_ROUTING_RESULTS.md` is the precedent for the wording).

## 5. Vetoes

```text
bilinear anchor       6/6 cells bitwise against the stored C0 cells, all five metrics
address identity      bitwise coverage@1/@3 equality with the in-study E0 cells, 6/6
evaluator equivalence bitwise on all returned fields, 6/6 cells
prototype fidelity    bitwise, per registered class
param-free address    router parameter count 0, asserted at runtime
one construction      set_seed then exactly one E2Model and one reference model per
                      (regime, seed) cell
feasibility           measured cell cost projects the grid under the 1 h ceiling
```

## 6. Statistics

Six paired differences (2 regimes x 3 seeds) per contrast, the owner-side convention:
per-pair values first, then mean / sd, exact two-sided paired permutation and the
max-statistic Westfall-Young correction within the primary family
(`s11_confirmatory.paired_stats`, `westfall_young`). The six-pair floor is 0.0312.

## 7. Feasibility

```text
declared grid       6 C0 training cells + 6 E0 reference cells = 12 trainings
measured cell cost  ~50 s (GPU, CUDA, epochs=10, from AC1)
grid projection     ~10 min
hard ceiling        1 h
```

## 8. Out of scope

```text
retraining without L_evidence    a different training recipe, a separate prereg
P: tracking vs data              AC1's open alternative; only meaningful if a
                                 learned address stays necessary, which this study tests
value plasticity                 AC2, gated on this study's reading
the decision rule                WTA is held fixed (the aggregation study closed mixture)
readout / decode refit           withdrawn (2026-09-25)
domain / ability CL, LLM port    not this rung
```

## 9. What this licenses, and what it does not

- **Licenses (drop-in reading):** that the E2 architecture's routing does not require a
  learned address, and that a fixed, parameter-free retrieval address is at least as
  good on the coverage endpoint - i.e. "stable keys" can be literal frozen prototypes
  rather than an immutable learned map. It also licenses combining the two results
  into the v2 statement: keys that never move, a selection rule that needs no learned
  query, and a value path as the only plastic part.
- **Licenses (coupling reading):** that in E2 the address and the value representation
  are functionally coupled (the key doubles as the decoded value), which is a concrete
  architectural constraint on "stable address + plastic value" designs that keep the
  evidence projection.
- **Does not license:** any training-side claim (this is evaluation-only), any claim
  about `L_evidence`'s usefulness, any claim that the E2 value path is redundant, any
  accuracy claim beyond `T = 20` and these two regimes, or any statement that the
  prototype space is the right final address for a foundation model - it is the
  control, not the architecture.
