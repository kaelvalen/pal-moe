# AC1: address-freeze completion - the shared query in the routing path - pre-registration

Status: **proposed, awaiting approval, not started.** It opens the v2 programme
(`stable address / plastic value`) and it is the variable the coupling study itself
deferred (`COUPLING_PREREG.md`, section 7: "the shared query `P` - likewise [a
separate coupling variable, separate pre-registration]").

## 1. Why this is the first v2 test

The follow-up chain decomposed the E2 collapse at the level of the arms:

```text
non-owner coupling   OWNER-ONLY - C1     +5.63 pp Acc   +0.0311 C@3
owner-side update    C0 - OWNER-ONLY    +51.30 pp Acc   +0.0068 C@3
```

(`COUPLING_RESULTS.md`, `INTERVENTION_RESULTS.md`, `OWNER_SIDE_RESULTS.md`.) In
**every** arm of that chain the experts `E_j` are frozen - the runtime guard requires
`previous_experts_in_optimizer == 0` and `previous_experts_frozen` at every task
(`experiments/e2_evidence.py`). The measured collapse is therefore **address-side**:
the values never moved, and the routing evidence

```text
s_j(z) = < normalize(P(z)) , normalize(W_j E_j(z)) >
```

moved because `W_j` and `P` moved.

The coupling study froze one half of that address, `W_1..W_{t-1}`, and the result is
C0: accuracy restored to the untouched baseline (74.16 / 70.14 against E0's 73.2 /
70.7) but coverage still below it:

```text
coherent    C0 0.9319    E0 0.9494
dispersed   C0 0.8548    E0 0.8915
```

`W` does not move in C0, so that residual cannot be W drift. One address component is
still plastic in every arm of the chain: the **shared query `P`**, trained on every
task.

The v2 invariant this serves,

```text
stable address  +  plastic value  +  immutable consolidation  +  associative memory
```

requires the address path to be frozen **as a whole**: an address that is the pair
`(P, W_j)` cannot be stable while `P` is plastic. AC1 is the smallest test that can
falsify the completeness of C0, and it is the gate for AC2 (plastic values):
introducing plasticity into the values is only readable once the address path is
closed.

## 2. The two arms

```text
C0            w_alignment=current,  p_alignment=plastic        (the anchor)
C0+P_frozen   w_alignment=current,  p_alignment=consolidated
```

`consolidated` is defined operationally, as the exact analogue of C0's treatment of
`W`: `P` trains during **task 0** under the pinned contract (same optimizer, learning
rate, epochs, `L_task + lambda L_evidence`), and from the first task boundary onward
it is `requires_grad_(False)` and absent from the optimizer's parameter list, for
every `t >= 1`. A runtime guard records, per task, `P_frozen`, `P_in_optimizer` and
`P_grad_nonzero`.

Everything else is unchanged: `E_j` frozen exactly as in the chain, the current `W_t`
trainable, older `W` frozen, the readout `g` untouched, the `L_task` / `L_evidence`
form, the winner-take-all decision, the inference path, the seeds, the budget,
`T = 20`.

**`g` is deliberately not touched, and the values are deliberately left frozen.**
Moving two axes at once is what the earlier studies established is the way to learn
nothing; the readout/decode line was withdrawn by decision (2026-09-25), and values
are AC2.

## 3. Endpoints

**Primary mechanistic metric:** `Delta C@3` between `C0+P_frozen` and `C0`, paired
over (regime, seed) - the question is whether the residual coverage cost of C0 is
query drift.

**Secondary family:** `Acc` (routing-mediated; read as a co-movement, not as an
independent claim).

**Read together:** `conditional_oracle@3`, `ceiling@3`, `oracle_accuracy`. A change
in the oracle between arms is possible - a frozen query changes the gradient that
shaped each `W_j` during its own task - so the outcome table applies to `C@3`, and
the oracle columns say whether any coverage change is usable.

**Passive trajectories** (no autograd; deepcopy + `torch.no_grad()`, the owner-side
drift probe):

```text
key drift     1 - cos( h_j(z_c) at t , h_j(z_c) at the end of task j )    h = W_j E_j
query drift   1 - cos( P_t(z_c)     , P_j(z_c)     at the end of task j )
```

Key drift is expected to be zero to floating-point precision in both arms (`W_j` and
`E_j` are frozen after task j); query drift is expected to be zero in the new arm and
positive in the anchor. The trajectories are what makes a null result readable: they
state whether the manipulation did what it claims to do.

## 4. Outcome reading, fixed in advance

| result | reading |
| :-- | :-- |
| `Delta C@3 > 0`, gap to E0 closes (residual <= ~0.005) | the residual is query drift; "the whole address path must be immutable" is supported, and the v2 address is a frozen map |
| `Delta C@3 ~ 0` (inside the anchor band, or far smaller than the gap to E0) | the residual is not drift at all - it is the cost of the **form** of the evidence address (bilinear `P.W` on adapted features) relative to training-free prototypes; the next variable is the address form, not its plasticity |
| `Delta C@3 < 0` | the shared query must be re-fitted across tasks to keep accumulated keys mutually comparable; the address is not decomposable into independent frozen halves at this scale |

All three readings are decision-relevant, and each names the next variable. A null is
a result; it is not "no effect" by default (`DECISION_ROUTING_RESULTS.md` records the
precedent and its wording).

## 5. Vetoes

```text
anchor equivalence   the six C0 cells, run with the passive drift probe active,
                     must match the stored coupling / intervention / owner-side
                     cells within |delta| <= 1e-6 on accuracy, ceiling_at_3 and
                     oracle_accuracy, and within |delta| <= 5e-4 on
                     coverage_at_3 and conditional_oracle_at_3. The execution
                     substrate is not bitwise-identical to the stored runs, and
                     the two coverage metrics are discrete top-3 statistics whose
                     boundary flips under a different floating-point order (the
                     pilot below measures the size). 5e-4 stays 35x below the
                     smallest effect this study is designed to read (>= 0.01).
                     This is also the passivity demonstration (probe active
                     against the probe-free stored cells).
implementation       the pre-AC1 implementation must return the pilot's cells
equivalence          bitwise on the same substrate (see the pilot), so that the
                     p_alignment change is behaviour-preserving on the plastic
                     path. The runner records the harness revision with the study.
P freeze guard       new arm, every task t >= 1: P_frozen, P_in_optimizer == 0,
                     P_grad_nonzero == False; anchor: P plastic, in the optimizer,
                     gradient present. Recorded per task by the model's own guard.
manipulation check   key drift zero to floating-point precision (|d| <= 1e-5) in
                     both arms (the probe compares a class-mean vector against a
                     clone, and a float32 self-cosine is ~1e-7, not bitwise 1.0);
                     query drift <= 1e-5 in the new arm and >= 1e-3 in the anchor
                     at the final boundary (the pilot reads >= 0.007 per task,
                     mean 0.086).
one construction     set_seed then exactly one E2Model per cell (the interference
                     study's root cause).
feasibility          the measured cell cost projects the 12-cell grid under the 1 h
                     ceiling.
```

Substrate pilot (2026-09-25, CPU, `torch 2.14.0+cu130`, no CUDA device visible; the
stored cells were produced on another substrate). C0, seed 42, `epochs=10`, probe
active, against the stored cells:

```text
regime      metric                  stored                  pilot                 |delta|
coherent    accuracy                0.7433                  0.7433                 0
            coverage_at_3           0.9328000485897064      0.9328000009059906     4.8e-08
            conditional_oracle@3    0.7968481989708405      0.7968481989708405     0
            ceiling_at_3            0.74330003871862        0.743300000721937      3.8e-08
            oracle_accuracy         0.8797                  0.8797                 0
dispersed   accuracy                0.7025                  0.7025                 0
            coverage_at_3           0.8579000413417817      0.8581000059843064     2.0e-04
            conditional_oracle@3    0.8188600069938221      0.8186691527793963     1.9e-04
            ceiling_at_3            0.7025000338531315      0.702500004899167      2.9e-08
            oracle_accuracy         0.977                   0.977                  0
```

The pristine pre-AC1 implementation (`git show 4628b47:experiments/e2_evidence.py`)
returns exactly the pilot values on this substrate, so the `p_alignment` change is
behaviour-preserving on the plastic path. Measured cell cost: 41-42 s per cell
(probe active, CPU).

## 6. Statistics

Six paired differences (2 regimes x 3 seeds) per contrast, the owner-side convention:
per-pair values first, then mean / sd, exact two-sided paired permutation
(`s11_confirmatory.paired_stats`) and the max-statistic Westfall-Young correction
within each family. The six-pair floor is 0.0312.

## 7. Feasibility

```text
declared grid       2 arms x 2 regimes x 3 seeds = 12 cells
measured cell cost  41-42 s (CPU, C0, seed 42, both regimes, epochs=10, probe active)
grid projection     12 x ~45 s ~ 10 min
hard ceiling        1 h
```

The study is cheap enough that the ceiling is not a design constraint here; it is
recorded because it was one for E2, and the record stays honest about cost.

## 8. Out of scope

```text
value plasticity              AC2 (the experts stay frozen here, as in the chain)
the address form              prototype / retrieval addresses are AC3
the readout / decode path     untouched; the refit line was withdrawn (2026-09-25)
domain / ability CL           not this rung
the LLM / VLM port            gated on a positive AC1-AC3 chain
```

## 9. What this licenses if it reads, and what it does not

- **Licenses (positive):** that the residual routing cost of the frozen-`W` arm has a
  query-side component, i.e. that the address path had a second plastic component and
  freezing it recovers coverage. That is the first positive architectural statement of
  the v2 programme, and it makes AC2 (plastic values, disjoint address) the next
  variable.
- **Licenses (null):** that the residual is a property of the evidence *form*. That
  retires "freeze more of the address" and moves the programme to AC3 (retrieval over
  frozen memory), which is also where the last untested piece of the frozen diagnosis
  lives (the winner-take-all decision; `DIAGNOSIS_SYNTHESIS.md` section 4).
- **Does not license:** any claim about the values (frozen throughout), the readout,
  accuracy as an endpoint of its own, scaling beyond `T = 20`, or the training-free
  prototype baseline being "the" target rather than the reference it is.
