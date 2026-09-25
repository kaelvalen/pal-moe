# Coupling ablation: accumulated evidence re-alignment - results

Status: **run. The third pre-declared reading: continually re-aligning accumulated
experts' evidence projections produces destructive interference.** All three
vetoes passed. Pre-registration: `docs/COUPLING_PREREG.md`.

## 1. The two arms, twelve cells

```text
C1   all W_1..W_t trainable      (the pinned E2 contract)
C0   current W_t trainable, W_1..W_{t-1} frozen
```

| regime | arm | accuracy | C@3 | conditional oracle@3 |
| :-- | :-- | --: | --: | --: |
| `coherent` | C1 | 13.04 / 12.68 / 15.00 | 0.9041 / 0.9018 / 0.9035 | 0.144 / 0.140 / 0.166 |
| `coherent` | **C0** | **74.33 / 73.78 / 74.36** | **0.9328 / 0.9299 / 0.9329** | **0.797 / 0.793 / 0.797** |
| `dispersed` | C1 | 15.96 / 17.81 / 16.83 | 0.8061 / 0.8112 / 0.8062 | 0.197 / 0.218 / 0.208 |
| `dispersed` | **C0** | **70.25 / 69.99 / 70.17** | **0.8579 / 0.8521 / 0.8543** | **0.819 / 0.821 / 0.821** |

Means and paired differences (C1 - C0, all three seeds negative):

| metric | `coherent` | `dispersed` |
| :-- | --: | --: |
| `Delta C@3` | **-0.0288** | **-0.0472** |
| `Delta Acc` | **-60.6 points** | **-53.3 points** |
| `Delta Oracle@3` | **-0.646** | **-0.613** |

**This is the third row: `C1 < C0`.** Freezing the accumulated experts' evidence
projections turns a collapsed system (13.6 % / 16.9 %) into one within a few
points of the untouched baseline (74.2 % / 70.1 %, against E0's 73.2 / 70.7 and
its coverage 0.9494 / 0.8915). Re-aligning **all** projections at every step
destroys both routing and usability, and the damage is an order of magnitude larger
than anything the earlier studies measured.

## 2. Vetoes, all passing

```text
C1 anchor (seed 42)        accuracy delta 0.0, C@3 delta 5.5e-08   PASS
C0 gradient isolation      current W has gradient; old W frozen and
                           absent from the optimizer, every task       PASS
feasibility                18.4 s per cell, projected total 0.06 h
                           against the 12 h ceiling                     PASS
```

## 3. A correction to the E2 postmortem

`E2_EVIDENCE_RESULTS.md` reported a cell cost of 3095 s and concluded that the
registered protocol was infeasible. **That cost figure came from the
pre-vectorisation implementation** (a per-prototype Python loop in the evidence
loss). With the batched loss the same 20-expert cell takes **18.4 s**, so the
feasibility objection is **withdrawn**: E2's non-execution stands on the
scientific ground alone - no usable confirmatory operating point at 20 experts -
and not on cost. The postmortem records this correction.

## 4. What this says about E2's collapse

The two arms differ **only** in whether the older evidence projections receive
gradient. `P`, `g`, `E_t`, `L_task`, `L_evidence`, the decision rule, the
inference path, the seeds and the budget are identical. So E2's collapse is
located: **the continual re-alignment of accumulated evidence projections is the
destructive mechanism**, not the shared readout and not the shared query.

The plausible mechanism, to be tested rather than asserted: every stored prototype
contributes to the loss of **every** old projection, so an old expert's evidence
is shaped by prototypes of tasks it never saw, and the interference grows with the
number of accumulated experts. That is a hypothesis for the next pre-registration,
not a conclusion of this one.

## 5. What this does not say

- It does not say the shared readout `g` is innocent, only that it is not the
  variable measured here.
- It does not say freezing old projections is a solution: C0 remains below E0 on
  coverage (0.9319 against 0.9494 `coherent`, 0.8548 against 0.8915 `dispersed`),
  so the evidence formulation still costs something relative to the ladder.
- It is one formulation - one shared bilinear query, one evidence dimension, one
  scale - and the same ablation on other coupling variables is separate work.
