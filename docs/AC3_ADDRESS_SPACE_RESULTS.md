# AC3: address-space ablation - fixed retrieval vs the learned evidence address - results

Pre-registration: `docs/AC3_ADDRESS_SPACE_PREREG.md` (`e6879c2`), development smoke in
its section 5. Data: `results/ac3/ac3_address_space_study.json` - 6 re-scored cells
plus 6 reference cells, harness `0e43673`, device `cuda`.

Status: **run, every veto passing exactly. The pre-declared first reading landed: the
fixed retrieval address is a drop-in for the routing path.** `Delta Acc = +0.048` pp,
equivalent to zero within the +/-1 pp SESOI, the ceiling is not lower, and the owner
path is bitwise unchanged - while coverage rises to E0's value exactly.

## 1. The two addresses on the same weights (means over three seeds)

Routing is inference-only in the pinned contract, so both rows are the same trained
C0 cells; only the selection changes.

| regime | address | `Acc` | `C@1` | `C@3` | `cond_routed@3` | `cond_owner@3` | `gap@3` | `ceiling@3` | `oracle_acc` |
| :-- | :-- | --: | --: | --: | --: | --: | --: | --: | --: |
| `coherent` | bilinear (pinned) | 74.16 | 0.8200 | 0.9319 | 0.7958 | 0.8926 | 0.0968 | 0.7416 | 0.8786 |
| `coherent` | **fixed_proto** | 73.61 | 0.8195 | **0.9494** | 0.7754 | 0.8868 | 0.1114 | 0.7361 | 0.8786 |
| `coherent` | E0 reference | 73.34 | 0.8195 | 0.9494 | - | 0.8812 | - | - | - |
| `dispersed` | bilinear (pinned) | 70.14 | 0.7068 | 0.8548 | 0.8205 | 0.9890 | 0.1685 | 0.7014 | 0.9772 |
| `dispersed` | **fixed_proto** | 70.78 | 0.7130 | **0.8915** | 0.7939 | 0.9868 | 0.1929 | 0.7078 | 0.9772 |
| `dispersed` | E0 reference | 70.68 | 0.7130 | 0.8915 | - | 0.9849 | - | - | - |

Per cell (`Acc`, `C@3`):

```text
coherent/42    bilinear 74.33 / 0.9328    fixed 73.67 / 0.9494
coherent/1     bilinear 73.78 / 0.9299    fixed 73.46 / 0.9494
coherent/2     bilinear 74.36 / 0.9329    fixed 73.71 / 0.9494
dispersed/42   bilinear 70.25 / 0.8579    fixed 70.78 / 0.8915
dispersed/1    bilinear 69.99 / 0.8521    fixed 70.79 / 0.8915
dispersed/2    bilinear 70.17 / 0.8543    fixed 70.76 / 0.8915
```

## 2. Primary family: the difference is a wash, and that is the result

`Delta = fixed_proto - bilinear`, paired over regime x seed:

| contrast | per-pair (six) | mean | sd | exact p | WY-adjusted p |
| :-- | :-- | --: | --: | --: | --: |
| `Delta Acc` | -0.0032, -0.0065, -0.0066, +0.0053, +0.0059, +0.0080 | **+0.00048** | 0.0067 | 0.875 | 1.000 |
| `Delta ceiling@3` | -0.0032, -0.0065, -0.0066, +0.0053, +0.0059, +0.0080 | **+0.00048** | 0.0067 | 0.875 | 1.000 |

Equivalence (TOST at the pre-registered +/-1 pp SESOI): **equivalent**, mean
`+0.048` pp, 90 % CI `[-0.50, +0.60]`, `p_lower = 0.0060`, `p_upper = 0.0086`.

The ceiling does not move either, so the pre-registration's third reading ("the
address leaves headroom the readout does not use") does **not** apply: no headroom is
left unused at this scale.

## 3. What the switch does move: coverage up, top-1 decodability down, net zero

- **Coverage.** `C@3` rises to E0's value exactly (+1.75 pp `coherent`, +3.67 pp
  `dispersed`), and `C@1` is unchanged. The identity is by construction - the
  registered address-identity guard measures `|delta| = 0.0` on coverage@1/@3 against
  the in-study E0 cells - so the coverage column is an implementation guard, not the
  finding. The finding is what the *accuracy* does with it.
- **The owner path is untouched.** `oracle_accuracy` is bitwise identical between the
  two addresses in every cell (six of six), because the oracle path contains no
  address. The whole difference is selection, exactly as in AC1.
- **The selection gap widens.** `gap@3 = cond_owner@3 - cond_routed@3` grows by
  +1.5 pp `coherent` and +2.4 pp `dispersed`: on the same covered samples, the owner
  expert would deliver 88.7 / 98.7 %, while the routed top-1 delivers 77.5 / 79.4 %
  (against 79.6 / 82.1 % for the learned address). So the fixed address finds the
  owner's expert in the top-3 **more** often, but its top-1 pick is decodable
  **less** often, and the two effects cancel exactly (the ceiling is flat).
- **The address costs no parameters** - the router parameter count is 0, asserted at
  runtime - and it is derived from the prototype bank E2 already stores.

Descriptively, the fixed address on the **E2 value path** edges out E0's own value
path at identical coverage (73.61 against 73.34 `coherent`, 70.78 against 70.68
`dispersed`). That is not a tested contrast (different training recipes), but it says
the E2 evidence path is not what E0's configuration had wrong - the address was.

## 4. Vetoes, all passing exactly

```text
bilinear anchor       6/6 cells, |delta| = 0.0 on all five metrics, against all
                      three stored sources (coupling, intervention, owner-side)
evaluator equivalence |delta| = 0.0 on all six fields, both regimes, 6/6 cells
address identity      |delta| = 0.0 on coverage@1 and coverage@3 against the
                      in-study E0 cells
prototype fidelity    exact, 100 registered classes per cell
param-free address    router parameter count 0, all cells
E0 reference          max |delta| 1.09e-07 against the aggregation WTA cells
                      (band 1e-6), coverage@1 exactly 0.0
one construction      set_seed then exactly one E2Model and one reference model per
                      cell
feasibility           19.9 s mean cell, 0.07 h projected, 1 h ceiling
```

## 5. The reading, as pre-registered

The first row of the outcome table was:

> the learned address is **not necessary**: a fixed-space retrieval is a drop-in for
> the routing path, and the higher coverage (`0.9494` against `0.9319`) comes for
> free. `P` and `W` can leave the address path; the remaining variable is the value
> path - **AC2 becomes meaningful**

That is what the data says, with one precision the table did not spell out: "drop-in"
is an equivalence at the accuracy and ceiling endpoints, not an improvement. The
learned address buys top-1 decodability, the fixed address buys top-3 coverage, and
at `T = 20` on these two constructions the two exactly compensate. The second row's
signature (better coverage, less decodable selections) is present in the gap column;
it is just not the dominant term.

## 6. What this does not say

- It is an evaluation-only ablation. Nothing here licenses a claim about `L_evidence`,
  about the value path's training, or about removing `P`/`W` from the *training*
  contract - only from the routing path.
- It does not say the class-prototype space is the right final address for a
  foundation model. It is the control, and it is the fixed space this rung already had
  a reference for.
- It does not say the address/value coupling is absent - the gap widened - only that
  it costs nothing net at this scale.
- It does not say E0's value path is redundant, and it makes no claim beyond
  `T = 20`, these two constructions, and these three seeds.

## 7. Consequence for the v2 contract

The v2 statement survives the switch with one word changed:

```text
stable KEYS  +  plastic READER  +  immutable consolidation  +  associative memory
```

The address path can be a **parameter-free retrieval over frozen prototypes**, so the
failure modes AC1 and C1 measured - a learned shared query or a learned evidence
projection moving under continual training - are removed **by construction** rather
than by freezing, and the cost is zero at this scale. Two consequences:

1. **AC2 is now the open variable.** With the address fixed and parameter-free, the
   remaining question is what may stay plastic on the value side, and that is what
   AC2 was defined to test.
2. **The top-1 gap is the address's bill.** The fixed address's selections are less
   decodable; the compensation is coverage. A next-generation address would have to
   keep the fixed-space coverage *and* recover the top-1 decodability - but that is a
   value/readout-side question (what makes a selected expert's output decodable),
   which is again AC2's territory, not a reason to re-learn the address.

## 8. Records

```text
e6879c2    the pre-registration (with the smoke calibration in section 5)
0e43673    the run (harness revision recorded in the study JSON)
this       the 6 re-scored cells, the 6 reference cells and this document
```
