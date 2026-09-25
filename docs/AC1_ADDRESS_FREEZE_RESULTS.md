# AC1: address-freeze completion - the shared query in the routing path - results

Pre-registration: `docs/AC1_ADDRESS_FREEZE_PREREG.md` (`74fc783`), with Amendment 1
(`8eb84bd`, the anchor band) registered before execution. Data:
`results/ac1/ac1_address_freeze_study.json` - 12 fresh cells, harness revision
`8d6e991`, device `cuda`.

Status: **run, all vetoes passed. The pre-registered third reading landed: freezing
the shared query collapses routing (-0.1507 C@3, -28.7 pp accuracy, 6/6 pairs), so
the address is not decomposable into independent frozen halves at this scale.**

## 0. The execution substrate, recorded

The first execution (CPU) stopped at the anchor veto and produced Amendment 1. At
re-execution the GPU was available: this torch build is cu130 and the driver
libraries live outside the default loader path
(`LD_LIBRARY_PATH=/run/opengl-driver/lib`). With that, the stored runs turn out to be
from **this same GPU**: every anchor cell reproduces the stored coupling /
intervention / owner-side cells **bitwise** - 18/18 comparisons, `0.0` on all five
metrics. Amendment 1's 1.5e-3 band was therefore never binding, and the CPU deltas it
was written from were substrate noise. Both the amendment and the substrate change
are recorded here rather than edited back.

```
LD_LIBRARY_PATH=/run/opengl-driver/lib .venv/bin/python \
    experiments/ac1_address_freeze.py --device cuda --epochs 10 --seeds 42,1,2
```

## 1. The two arms, twelve cells (means over seeds)

```text
C0            w_alignment=current,  p_alignment=plastic        (the anchor)
C0+P_frozen   w_alignment=current,  p_alignment=consolidated   (P trains on task 0 only)
```

| regime | arm | `Acc` | `C@3` | `cond_oracle@3` | `ceiling@3` | `oracle_acc` |
| :-- | :-- | --: | --: | --: | --: | --: |
| `coherent` | C0 | 74.16 | 0.9319 | 0.7958 | 0.7416 | 0.8786 |
| `coherent` | **C0+P_frozen** | **49.15** | **0.8216** | 0.5982 | 0.4915 | 0.8788 |
| `dispersed` | C0 | 70.14 | 0.8548 | 0.8205 | 0.7014 | 0.9772 |
| `dispersed` | **C0+P_frozen** | **37.69** | **0.6636** | 0.5679 | 0.3769 | 0.9767 |

Per-pair differences (`P_frozen - C0`):

```text
                 dAcc (pp)    dC@3        dOracle (pp)
coherent/42        -25.06     -0.1152        +0.070
coherent/1         -25.69     -0.1070        -0.010
coherent/2         -24.28     -0.1087         0.000
dispersed/42       -33.26     -0.1991         0.000
dispersed/1        -30.97     -0.1838        -0.150
dispersed/2        -33.10     -0.1906         0.000
```

## 2. Primary family (`C@3`): the frozen query destroys coverage

| contrast | six paired differences | mean | sd | exact permutation p | sign p |
| :-- | :-- | --: | --: | --: | --: |
| `P_frozen - C0`, `C@3` | -0.1070, -0.1087, -0.1152, -0.1838, -0.1906, -0.1991 | **-0.1507** | 0.0446 | **0.0312** | 0.0312 |

One test in the family, so the pre-registered Westfall-Young step reduces to this p.
All six pairs are negative; the effect is 8.6x the coherent C0 -> E0 gap (0.0175).

## 3. Secondary family (`Acc`): the same, larger

| contrast | six paired differences (pp) | mean | sd | exact p |
| :-- | :-- | --: | --: | --: |
| `P_frozen - C0`, `Acc` | -25.06, -25.69, -24.28, -30.97, -33.10, -33.26 | **-28.73** | 4.17 | **0.0312** |

## 4. The owner path is (almost) untouched: the damage is in selection

`oracle_accuracy` moves by `-0.15` to `+0.07` pp between the arms (mean `-0.015` pp)
against a `-28.7` pp accuracy collapse - three of the six seeds at exactly `0.000`. The
oracle path uses no `P`; the residual movement is the frozen query changing the
gradient that shaped each `W_j` during its own task, which the pre-registration
anticipated. So the collapse is not in the values or the owner readout: it is in
**which expert is selected**, for nearly all of the effect.

This is the mirror image of the owner-side study. There, the values were constant in
every arm and the address moved; here the values are constant *and* the address is
partially frozen, and the damage is still in selection.

## 5. Vetoes, all passing

```text
anchor equivalence    18/18 comparisons against coupling / intervention /
                      owner-side, |delta| = 0.0 on all five metrics (bitwise;
                      the amended 1.5e-3 band was not binding)
implementation         implied by the above: the stored cells were produced by the
equivalence           pre-AC1 implementation on this GPU, and the anchor arm
                      reproduces them exactly
P freeze guard        12 cells x 19 tasks = 228 per-task rows, all pass:
                      new arm P_frozen and P_in_optimizer == 0 and no P gradient;
                      anchor arm P plastic, in the optimizer, gradient present
manipulation check    key drift <= 1.6e-08 in both arms (float32 floor);
                      query drift 0.080-0.093 in the anchor, 0.00000 in the new arm
one construction      the runner builds exactly one E2Model per cell after set_seed
feasibility           48.6 s mean cell, 0.16 h projected, 1 h ceiling
```

## 6. The reading, as pre-registered

The outcome table's third row was fixed in advance:

> the shared query must be re-fitted across tasks to keep accumulated keys mutually
> comparable; the address is not decomposable into independent frozen halves at this
> scale

Two further sentences are licensed by the guards, no more:

- The collapse is not "P was untrained": the manipulation check shows the anchor's
  query drift trajectory is real (0.080-0.093) and the frozen arm's is exactly zero,
  so the two arms differ in exactly the registered variable.
- The collapse is not in the expert values or their readouts: `oracle_accuracy` is
  essentially invariant.

**The plausible mechanism, to be tested rather than asserted.** With the older `W_j`
frozen, the shared query is the only parameter that touches every accumulated expert's
evidence row, and so the only place a *cross-expert* constraint can be enforced; each
`W_t` is fitted only against its own task's rows. Freezing `P` removes that enforcement
point, and the keys stop being mutually comparable. Two alternatives are not separated
by this study: the query may need to *track* the keys, or the query may simply need
*many more tasks of training data* than task 0 provides. A follow-up can separate them
(e.g. a query trained on accumulated evidence at every step but with the update
frozen between tasks, against the task-0-only query).

## 7. What this does not say

- It does not say the residual C0 -> E0 coverage gap (0.9319 against 0.9494
  `coherent`) has no query-side component. It says the component cannot be removed by
  freezing `P`: a query that is plastic in both arms cannot explain a difference
  between them. Whether a *less* plastic query (EMA, smaller step) would help is
  unmeasured.
- It does not license "the query is the separability mechanism" as a conclusion; that
  is the mechanism hypothesis above.
- It does not test the fourth cell of the 2x2, `W` plastic with `P` frozen. That is a
  separate pre-registration, and the chain already has both other mixed cells (C1,
  C0, and now `P_frozen`).
- It does not touch the values (frozen everywhere), the readout, `T > 20`, or the
  LLM/VLM port.

## 8. Consequence for the v2 contract

The registered non-positive reading retires "freeze more of the address path" and
moves the variable to the address **form**. Together with the chain, the v2 invariant
has to be restated:

```text
stable KEYS  +  plastic READER  +  immutable consolidation  +  associative memory
```

not "stable address". The two constraints that follow are design rules for AC3, not
findings:

1. **Immutable keys, plastic reader.** The identifying side of the address (`W_j`,
   `E_j`) must be consolidated and never re-aligned - that is the C0 result and the
   owner-side decomposition. The reading side (query, readout, retrieval weights) must
   stay free to adapt to the fixed keys - that is this result, and it is the same
   shape the readout/decode line pointed at before it was withdrawn.
2. **Separability by construction, not by a learned shared query.** A fixed space
   that does not need cross-task maintenance removes the failure mode entirely; the
   training-free prototype space is the existing reference (E0: 0.9494 / 0.8915
   coverage). That is AC3, and it is also where the last untested piece of the frozen
   diagnosis lives (the winner-take-all decision; `DIAGNOSIS_SYNTHESIS.md` section 4).

## 9. Records

```text
74fc783    the pre-registration
8eb84bd    Amendment 1: the anchor band, from the CPU substrate pilot
8d6e991    the run (harness revision recorded in the study JSON)
this       the 12 cells and this document
```
