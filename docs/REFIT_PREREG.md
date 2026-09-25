# Readout refit / replay: decode mismatch or representation loss? - pre-registration

Status: **proposed, awaiting approval, not started.** It follows
`OWNER_SIDE_RESULTS.md`: the owner-side update is an independent causal component
of the accuracy collapse, and the passive drift probe left the mechanism open -
near-total evidence drift is *consistent* with a decode/readout mismatch, but no
mechanism claim was made. This study tests that candidate directly, without
re-interpreting any earlier result.

## 1. The question

> **Does post-hoc readout refitting recover the owner-side accuracy loss,
> consistent with a decode/readout mismatch, or does the loss persist despite
> refitting, consistent with representation-level degradation?**

## 2. Design: pinned trajectories, one post-hoc intervention on `g`

Training is untouched: the three arms re-run fresh under the pinned contract
(18 cells), with the usual bitwise anchors. After training, each trained model is
evaluated under three readout treatments that differ only in the readout `g`:

```text
R0  frozen         the trained readout (the pinned evaluation; the anchor)
R1  refit-current  g refitted on the last task's training evidence
R2  refit-replay   g refitted on all tasks' training evidence
```

The owner-update factor (C1, OWNER-ONLY, C0) and the readout treatment (R0, R1, R2)
form the 3x3 grid. C0 is the procedure control: its evidence is intact, so a
systematic loss there indicts the refit, not the arm.

### 2.1 The refit protocol, pinned identically for R1 and R2

```text
data         for every included task j and every training feature z:
                 h = normalize(W_j(E_j(z)))   (no_grad, frozen trained model)
             label = the class of z; R1 uses j = T-1 only, R2 uses every j
model        a deepcopy of the trained readout; all class rows are trainable
             (the readout's own trainable_hook over all classes)
budget       Adam lr 1e-3, 10 epochs over the treatment's data, batch 128,
             cross-entropy over all classes (no mask_unseen - every class is
             seen at the end), batch order from a seeded generator, fixed
             stopping (no early stop); identical for R1 and R2, only the data
             scope differs
no test data the refit never sees a test split; the evaluation stays honest
order        R0 is evaluated before any refit; R1 and R2 operate on separate
             deepcopies, so the stored model state is never modified
```

Recorded for transparency: `pal_moe/arch/readouts.py` warns that re-opening every
class row during *sequential* training collapses a readout (S2 measured 7.69 %).
This study re-opens the rows **after** training, on the final frozen
representation, with the full training data and a control arm; that is the
intended intervention, and section 5 pins the procedure check.

## 3. Endpoints

**Primary family (`Acc`):** `Delta_replay = R2 - R0` for OWNER-ONLY and C1, six
pairs each (regime x seed), exact two-sided permutation over the `2^6` sign
patterns, Westfall-Young max-statistic within the family of two. No directional
claim is pre-registered: the outcome table in section 4 maps positive, null and
negative results. Every contrast is paired inside a single trained model, so the
training-seed variance does not enter it.

**Secondary family (`Acc`), kept separate:**

```text
Delta_current = R1 - R0    for OWNER-ONLY and C1
Delta_old     = R2 - R1    for OWNER-ONLY and C1
```

same exact machinery, Westfall-Young within this family, reported separately and
never merged with the primary.

**Procedure control (`Acc`, not a test of the hypothesis):** `Delta_replay` for C0,
exact two-sided test. A significantly negative value is a procedure veto
(section 5).

**Structural invariance (veto, not an endpoint):** `coverage_at_3` is
readout-independent and must be bitwise identical across R0, R1 and R2 within
every cell.

**Descriptive:** `conditional_oracle@3`, `ceiling_at_3`, `oracle_accuracy` and
per-task accuracy under each treatment. The drift numbers of `OWNER_SIDE_RESULTS`
are not re-used here.

## 4. Outcome reading, fixed in advance

| result | reading |
| :-- | :-- |
| `R2 > R0`, primary-family significant | recovery is consistent with a readout/decode mismatch |
| `R2 ~ R0`, not significant | no substantial readout-only recovery; representation loss remains consistent with the data |
| `R1 ~ R0` (not significant) and `R2 > R1` (secondary-family significant) | the recovery depends substantially on replay / old-task evidence |
| `R1 > R0` (secondary-family significant) | current-task evidence alone is sufficient for substantial recovery |

Fixed limits on the reading:

- `R2 ~ R0` is non-significance at six pairs, **not** equivalence; it is read as
  "representation loss remains consistent", never as "representation loss is
  proven".
- `R2 > R0` supports a decode-mismatch reading; it does not prove the
  representation is fully intact, and it does not identify where in the readout
  the mismatch lives.
- With a procedure veto, the refit readings are not executed: the training cells
  stand, and the decode question stays open.

## 5. Guards

```text
one construction        set_seed then exactly one E2Model per cell
anchor invariance       bitwise, all five metrics, against the stored cells of
                        coupling, intervention and owner_side (C1, C0) and of
                        intervention and owner_side (OWNER-ONLY); the R0
                        per-task accuracy must equal owner_side's as well
procedure veto          Delta_replay for C0 must not be significantly negative;
                        if it is, the refit readings are not executed
C@3 invariance          coverage_at_3 bitwise equal across R0/R1/R2 per cell
passivity               R0 before the refits; refits on deepcopies; no test data,
                        no autograd.grad anywhere in the study path
reproducibility         re-running one cell reproduces its metrics and refit
                        results identically
no tuning               the pinned contract (epochs 10, lr 1e-3, batch 128,
                        rank 8, one prototype per class, T = 20, evidence
                        lambda 1.0, the same construction, cache and inference
                        path) plus the refit protocol of section 2.1
```

Artifacts: `experiments/readout_refit.py`, `results/refit/refit_study.json`,
`docs/REFIT_RESULTS.md`.

## 6. Feasibility

```text
cells       18 x ~20 s ~ 7 min
refits      R2 ~4k steps per cell, R1 ~0.2k; a few minutes in total at this scale
evaluation  three pinned passes per cell (R0/R1/R2); seconds
total       about 10-12 minutes, well under the 12 h ceiling
```

## 7. Out of scope

```text
memory / budget accounting   this is a mechanism test; the cached features are
                             not proposed as a memory-efficient method
deployment / CL method claims   the refit is post-hoc and uses stored features
routers, decisions, W, P     untouched; only g is intervened on
other datasets, backbones, formulations   the pinned contract
the accumulation curve       T in {4,8,20}; separate addendum
```
