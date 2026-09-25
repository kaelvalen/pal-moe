# Interference mechanism: the expert-count ladder - results

Pre-registration: `INTERFERENCE_PREREG.md`, with **Amendment 1** (committed before
the re-run as `34c1289`) and its post-run clarification. Data:
`results/interference/interference_study.json` (30 cells, seeds 42 / 1 / 2). The
first, vetoed ladder is preserved verbatim as `interference_study.void.json`.

Status: **executed on the pinned `C1` trajectory.** The owner/non-owner
measurement is the **loss form** of Amendment 1 - no `torch.autograd.grad`, no
`retain_graph`, no `.grad` access anywhere in the executed path. The pattern below
is **mechanistic support, not a causal claim** (section 2.2 of the
pre-registration).

## 1. Two vetos, both correct

**Veto 1 - the runner was not the pinned path.** The anchored cell
(`coherent / T=20 / seed=42`) read `12.23 / 0.9032` through `run()` while every
individual piece of instrumentation reproduced `13.04 / 0.9041`:

```text
no hooks                          13.04 / 0.9041
no-op hooks                       13.04 / 0.9041
snapshot only                     13.04 / 0.9041
deepcopy only                     13.04 / 0.9041
full probe through run()          12.23 / 0.9032   <- veto
```

The cause was one line in the runner: `E2Model` was constructed **twice** after
`set_seed(seed)`, so the second construction consumed the global RNG and training
started from different weights. After the fix (one construction) the same matrix
returns `13.04 / 0.9041` in all four cells, including the full probe and no hooks.
The probe was innocent throughout; so were the hooks, the snapshots and the
deepcopy.

**Veto 2 - the measurement form.** Amendment 1 replaced the originally specified
gradient decomposition with the loss form, because the gradient build did not
reproduce the anchor. That observation is now attributed to Veto 1's constructor
bug, and Amendment 1's "not passive" rationale is withdrawn in the clarification;
the loss form stays as the executed, conservative choice.

The first ladder - 30 cells whose anchored cell read `12.23 / 0.9032` - is **void
in full**. Its cells, and its `1.8 -> 6.4` non-owner/owner pattern, are withdrawn,
and nothing in this document derives from them.

## 2. Anchor

`coherent / T=20 / seed=42` reproduces the coupling study's `C1` cell (arm `all`)
**exactly at full float precision**:

```text
interference ladder        0.1304   coverage_at_3 0.9041000545024872
coupling C1                0.1304   coverage_at_3 0.9041000545024872
conditional_oracle@3 and ceiling@3 also identical
```

## 3. The ladder

Pinned contract (epochs 10, lr 1e-3, batch 128, rank 8, evidence lambda 1.0,
`w_alignment=all`, seeds 42 / 1 / 2, eval = the E2 inference path). Values are
means over the three seeds. `asym` is `mean non-owner mass / mean owner mass`
(Amendment 1); `||dW||` is the mean over old projections at the final task.

**coherent**

| T | owner mass | non-owner mass | asym | mean ||dW|| | Acc | C@3 | Oracle@3 | dC@3 |
| --: | --: | --: | --: | --: | --: | --: | --: | --: |
| 2 | 3.53 | 3.45 | 0.98 | 4.3631 | 65.53 | 1.0000 | 0.6553 | - |
| 4 | 2.52 | 11.62 | 4.61 | 5.9400 | 39.20 | 0.9880 | 0.3961 | -0.0120 |
| 8 | 3.91 | 30.38 | 7.77 | 7.5049 | 25.55 | 0.9518 | 0.2681 | -0.0362 |
| 12 | 5.03 | 59.71 | 11.88 | 6.8479 | 21.47 | 0.9375 | 0.2290 | -0.0142 |
| 20 | 6.74 | 132.37 | 19.65 | 6.6690 | 13.57 | 0.9031 | 0.1501 | -0.0344 |

**dispersed**

| T | owner mass | non-owner mass | asym | mean ||dW|| | Acc | C@3 | Oracle@3 | dC@3 |
| --: | --: | --: | --: | --: | --: | --: | --: | --: |
| 2 | 3.43 | 3.46 | 1.01 | 4.5980 | 81.33 | 1.0000 | 0.8133 | - |
| 4 | 2.48 | 11.88 | 4.79 | 6.9228 | 37.92 | 0.9457 | 0.4004 | -0.0543 |
| 8 | 3.93 | 31.79 | 8.10 | 5.3595 | 25.56 | 0.8676 | 0.2926 | -0.0781 |
| 12 | 5.03 | 59.93 | 11.91 | 5.4496 | 19.12 | 0.8438 | 0.2250 | -0.0237 |
| 20 | 6.72 | 132.07 | 19.65 | 6.0844 | 16.87 | 0.8078 | 0.2074 | -0.0360 |

Primary endpoint: `dC@3` between consecutive ladder points is negative at every
step in both regimes, with `Acc` and `conditional_oracle@3` falling alongside it.

## 4. Correlations (the weaker form, per the pre-registration)

Across the 30 cells, loss-based asymmetry against each metric (Pearson /
Spearman):

```text
Acc        -0.829 / -0.970
C@3        -0.789 / -0.878
Oracle@3   -0.826 / -0.952
within regime, asym vs C@3   coherent -0.982   dispersed -0.940
```

Both quantities are near-deterministic functions of `T` across these cells, so
this mostly re-expresses the ladder ordering; the pre-registration designates the
correlation as explicitly the weaker of the two forms of evidence, and it is
reported at that level.

## 5. Reading against the fixed outcome table

The pattern matches the first row of the pre-registration's outcome table:

> `nonowner >> owner` and both grow with `T`, metrics fall -> the interference is
> genuinely cross-task, and it accumulates

At `T=2` the two masses are equal (`asym` about 1.0); by `T=20` the non-owner mass
is about 20x the owner mass (132 vs 6.7), while `C@3`, `Acc` and `Oracle@3` all
fall. The other three rows are not triggered: the non-owner mass never tracks the
owner mass, the metrics do fall, and the asymmetry does grow with `T`.

Three limits are part of the reading, not footnotes:

**Mechanistic support only.** The measurement lives on the same trajectory as the
metrics; no intervention varies the non-owner mass independently. "The non-owner
evidence caused the collapse" is stronger than this design can license, exactly as
section 2.2 states.

**The asymmetry is a sum over a growing count.** For each old `W_j`, the non-owner
term sums one cross-entropy per stored prototype not owned by task `j` - a count
that grows with `T` - while the owner term sums one task's prototypes. Part of the
rise in the measured quantity is therefore arithmetic. The pre-registration
forbids adding a count-adjusted variant ("no metrics beyond this document"), so
none is computed and "accumulates" is stated at the level of the measured
quantity.

**The rewrite mass does not keep growing.** `mean ||dW||` rises to `T=8`
(7.50 coherent) and then is flat to slightly lower (6.85, 6.67): the per-task
movement of old projections peaks mid-ladder while the measured asymmetry keeps
rising. This is a raw-norm observation only, per the endpoint list.

What this supports: the destructive interference localised by the coupling
ablation is consistent with a cross-task evidence signal that grows with the
number of accumulated experts. What it does not do: identify that signal as the
cause, or quantify how much of the metric drop it explains.

## 6. Records

```text
cd42c54   the verified E2 refactor (train / train_task split, anchor checked)
34c1289   Amendment 1: loss-form decomposition (before the re-run)
7264f9b   the runner fix: one construction, hooks instrumentation, no copied body
ecf7dc5   prereg instrumentation lines aligned with Amendment 1
this      the 30 corrected cells and this document
```

Void record: `results/interference/interference_study.void.json` (the vetoed first
ladder). It is kept only as the record of the stop; nothing cites it.
