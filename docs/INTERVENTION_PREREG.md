# Intervention: cutting the non-owner evidence path to old projections - pre-registration

Status: **proposed, awaiting approval, not started.** It follows
`COUPLING_RESULTS.md`, whose section 4 explicitly deferred the mechanism to the
next pre-registration:

> every stored prototype contributes to the loss of **every** old projection, so an
> old expert's evidence is shaped by prototypes of tasks it never saw, and the
> interference grows with the number of accumulated experts.

This study intervenes on exactly that path. Its test is defined a priori and does
not condition on any descriptive result - in particular not on `INTERFERENCE_RESULTS.md`,
which is a separate, non-causal study.

## 1. The question

> **Does cutting the gradient path that carries prototypes an old projection does
> not own - the non-owner evidence path - restore the collapsed system?**

## 2. Design: three arms, one manipulated path

All arms share the pinned E2/C1 contract (section 5). They differ only in which
evidence-loss contributions reach the old projections `W_j` (`j < t`):

```text
arm             gradient path to old W_j (j < t)
C1              every stored prototype's term        (pinned contract, re-run here)
OWNER-ONLY      only prototypes owned by task j      (the intervention)
C0              none - old W frozen                  (boundary reference, re-run here)
```

C0 is the no-coupling boundary already reported by the coupling ablation; it is a
reference, not a target.

### 2.1 What OWNER-ONLY changes, exactly

`_evidence_loss` scores every stored prototype against every seen expert and
averages the cross-entropies. In the intervention, at task `t`, for each old expert
`j < t`, the normalized projected feature is detached on every prototype row the
expert does not own:

```text
h_j(p) = normalize(W_j E_j(z_p))
for j < t and owner(p) != j   ->   h_j(p).detach()
owner rows (owner(p) == j)    ->   unchanged
```

**Forward consequence: none.** `detach` preserves the values, so the loss value and
every metric of the step are bitwise unchanged.

**Gradient consequence: exactly one path.** `W_j`'s gradient contains only the
prototypes of task `j` (the owner term of the decomposition), while every other
trainable parameter - the shared query `P`, the shared readout `g`, the current
projection `W_t` and the current adapter `E_t` - keeps the C1 gradient exactly.

The cut applies to old projections only. `W_t` keeps the full coupling, so the new
expert still aligns to the accumulated evidence; cutting it as well would change a
second variable (new-expert alignment) and is out of scope. `P` and `g` are
untouched.

### 2.2 Why this is the missing manipulation

The coupling ablation varied the old-`W` gradient between "all" and "none"; this
study inserts the middle level that separates the non-owner term from the owner
term inside "all". The primary causal contrast is therefore `OWNER-ONLY - C1`,
paired on the same seed and regime. A recovery under this manipulation is caused by
removing the non-owner path, at the level of the arms; C0 bounds it but does not
have to be reached.

### 2.3 Cells and prediction

```text
T = 20 (the collapsed operating point), 18 cells:
    arms    C1, OWNER-ONLY, C0
    regimes coherent, dispersed
    seeds   42, 1, 2
the six OWNER-ONLY cells are the new data; C1 and C0 are re-run inside the study
```

The C1 and C0 cells are re-run rather than copied so that every comparison is
paired within one process and the invariance veto has something to check against.
The pre-registered direction is `OWNER-ONLY - C1 > 0` on both `Acc` and `C@3`. The
full three-way ordering `C0 >= OWNER-ONLY >= C1` is **not** required for support.

## 3. Endpoints

**Primary (causal):** `OWNER-ONLY - C1`, paired by (regime, seed), on `Acc` and
`C@3` - six pairs. The family is `{Acc, C@3}`; the test is an exact two-sided
permutation over the `2^6` sign patterns of the paired differences (equivalently
the exact sign test at this size), with a Westfall-Young max-statistic adjustment
within the family. The p floor is `2 * (1/2)^6 = 0.0312`.

Support is read only if at least one metric is significant after the adjustment and
the other points in the same direction; a sign disagreement between the two is
reported as mixed, without support.

**Boundary (descriptive, no test):** the position of OWNER-ONLY relative to C0, and
the recovered share of the published C1-to-C0 gap, per regime. No threshold is
attached to either.

**Mechanistic secondaries (passive probes, same trajectories):**

```text
asym_pp                 mean_j(nonowner_mean_j) / mean_j(owner_mean_j), with
                        owner_mean_j    = mean{ CE_p : owner(p) = j }
                        nonowner_mean_j = mean{ CE_p : owner(p) != j }
                        the count-adjusted counterpart of the interference
                        study's 19.65x sum-form headline - calibration, not a test
||delta_W_j||           raw norm over old projections at the final task
owner / non-owner mass  the sum form, for continuity with the ladder table
side metrics            conditional_oracle@3, oracle_accuracy, ceiling_at_3
```

The headline boundary for the probes is the last task start (`q{T-1}`), mirroring
the interference ladder; every task boundary is stored. Probes are loss-form
records taken under `torch.no_grad()` on a deepcopy through the `E2Model.train`
hooks: no `autograd.grad`, no `.grad` access, no retained graphs.

## 4. Outcome reading, fixed in advance

| result | reading |
| :-- | :-- |
| `OWNER-ONLY - C1` significantly positive (after WY) | the non-owner evidence path is a causal contributor to the collapse; the recovered share of the C1-to-C0 gap and the position relative to C0 are reported as magnitudes, without thresholds |
| positive point estimate, not significant | direction consistent, not established; at six pairs this is the design floor and the result is reported as inconclusive - neither support nor refutation |
| non-positive point estimate, not significant | the non-owner path is not the destructive term under this manipulation; the remaining candidate is the owner-term re-alignment itself and the other coupling components |
| significantly negative | the non-owner terms are load-bearing; the hypothesis is refuted in the opposite direction - report, no tuning |

**What a positive result would and would not license.** The manipulation changes
exactly one gradient path - old `W_j` no longer receive non-owner evidence - so a
recovery licenses a causal statement about that path at the level of the arms. It
does not quantify what fraction of the collapse is explained (no thresholds), does
not identify per-prototype damage, does not generalise beyond the pinned
formulation, and says nothing new about the shared query or readout.

## 5. Guards

```text
one construction        set_seed(seed) then exactly one E2Model(...) per cell. A
                        second construction consumes the global RNG and breaks the
                        pinned trajectory; this was the interference study's root
                        cause, fixed in 3ddfdba, and is checked here by the anchor
anchor invariance       with the cut disabled, every C1 and C0 cell must equal the
                        coupling study's stored cell exactly on all metrics; one
                        float difference is a veto (not executed)
cut audit               one pre-run audited training, separate from the cells. At
                        each task t > 0, on a fixed batch:
                        (a) forward value: l_cut == l_full (torch.equal)
                        (b) for each old j, grad_cut(W_j) equals the gradient of
                            sum_{p: owner(p)=j} CE_p w.r.t. W_j, exactly
                        (c) every other trainable parameter (P, g, W_t, E_t):
                            grad_cut == grad_full, bitwise
                        any failure is a veto (not executed)
training guards         the pinned E2 guard per cell: every old W has a nonzero
                        gradient, previous experts frozen and absent from the
                        optimizer, d_e = 128, no W bias
reproducibility         re-running one cell reproduces its metrics identically
no tuning               epochs 10, lr 1e-3, batch 128, rank 8, one prototype per
                        class, T = 20, evidence lambda = 1.0, cosine readout, the
                        same construction, feature cache and inference path as the
                        coupling study
```

Implementation notes, pinned so the manipulation is unambiguous: the cut is a
property of the arm (`w_alignment = owner_only`, a new choice; `all` and `current`
are unchanged), implemented inside `_evidence_loss`; with the flag off the code
path is the pinned one. The audit may use `autograd.grad` because it is a separate
diagnostic run - never a study cell and never the training path.

Artifacts: `experiments/intervention.py`, `results/intervention/intervention_study.json`,
`docs/INTERVENTION_RESULTS.md`.

## 6. Feasibility

```text
audit        one 2-task audited training plus the comparisons, well under a minute
cells        18 x ~20 s ~ 6 min (the coupling study measured 18.4 s per cell)
total        well under the 12 h ceiling; the interference ladder's 30 cells ran in
             a single pass
```

## 7. Out of scope

```text
the accumulation curve    T in {4,8,20}; a separate pre-registration/addendum
the shared query P and the readout g    unchanged in every arm
other evidence dimensions or losses     the pinned contract
routers and decision rules              separate studies
other datasets or backbones             frozen backbone, source cache as pinned
the C1 vs C0 contrast      already reported in COUPLING_RESULTS
```
