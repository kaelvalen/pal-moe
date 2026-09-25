# Owner-side residual: the two-component decomposition of the E2 collapse - pre-registration

Status: **proposed, awaiting approval, not started.** It follows
`INTERVENTION_RESULTS.md`, whose narrow reading is the starting point: cutting the
non-owner evidence path to old projections restores routing coverage most of the
way to the no-coupling boundary while leaving most of the accuracy degradation
unresolved, and the residual is consistent with an owner-term or other residual
coupling mechanism. That is a hypothesis; this study pre-registers the test.

## 1. The question

> **Does old projections updating on their own task's prototypes carry the residual
> accuracy damage - measured as `C0 - OWNER-ONLY` on the same arms?**

The three arms form one chain, each step removing one gradient path into the old
projections `W_j` (`j < t`):

```text
C1            owner + non-owner evidence gradients   (the pinned contract)
OWNER-ONLY    the owner term only                    (non-owner paths cut)
C0            no gradient at all                     (old W frozen)
```

so both components are measured on the same trajectories:

```text
non-owner component    OWNER-ONLY - C1
owner-side component   C0 - OWNER-ONLY
```

## 2. Design

Eighteen cells, re-run fresh - nothing is taken from the earlier records:

```text
arms      C1, OWNER-ONLY, C0
regimes   coherent, dispersed
seeds     42, 1, 2
T         20 (the collapsed operating point)
```

The arms and the cut are exactly as pinned in `INTERVENTION_PREREG.md` section 2
(implementation `1b54c89`): identical values and identical gradients for `P`, `g`,
`W_t` and `E_t`; only the gradient paths into the old `W_j` differ. The re-run is
what makes the new contrast independent: every cell used in the analysis is
produced under this pre-registration, and the earlier records serve only as anchors
(section 5).

## 3. Endpoints

**Primary family (`Acc`), fixed in advance:**

```text
Delta_nonowner^Acc = OWNER-ONLY - C1     the replication
Delta_owner^Acc    = C0 - OWNER-ONLY     the new confirmatory test
```

Six pairs each (regime x seed), exact two-sided permutation over the `2^6` sign
patterns, Westfall-Young max-statistic **within this family of two**, p floor
`2 * (1/2)^6 = 0.0312`. The pre-registered direction is positive for both, and the
support reading is the family-wise adjusted p at the programme's usual 0.05.

**Secondary family (`C@3`), kept separate:** the same two components, computed and
tested with the same exact machinery, reported as their own family and never merged
with the primary. The pre-registered direction is positive for the non-owner
component; the owner-side `C@3` component is reported as measured - the earlier
records suggest it may not be zero (C0 was slightly above OWNER-ONLY on coverage),
and that is exactly what this family quantifies.

**Side metrics** (`conditional_oracle@3`, `ceiling_at_3`, `oracle_accuracy`) per
arm, descriptive.

**Passive mechanism secondaries - descriptive only, no causal claim, no
training-path modification.** Measured with `torch.no_grad()` and snapshots, never
inside a training graph:

```text
per-task accuracy   final-model accuracy on each old task's test split
evidence drift      for each old task j and class c:
                    h_{j,c} = mean over class-c test features of
                              normalize(W_j E_j(z))
                    snapshot at the end of task j (deepcopy, no_grad), recomputed
                    at every later task boundary; drift = 1 - cos(h_final,
                    h_task-end), headline at the final boundary, trajectory stored
```

The report reads drift against per-task accuracy descriptively and presents the
candidate mechanism ("old evidence drifts out of the frozen old-class readout
rows") as a hypothesis the probe characterises, not tests.

## 4. Outcome reading, fixed in advance

| result | reading |
| :-- | :-- |
| `Delta_owner^Acc` significantly positive (WY) | the residual accuracy damage has an owner-side component under this manipulation; with `Delta_nonowner` replicating, the collapse decomposes, at the level of the arms, into a non-owner part and an owner-side part |
| positive, not significant | direction consistent, not established at six pairs; reported as inconclusive |
| non-positive | the owner-side update does not carry the residual accuracy damage; the residual lies elsewhere (for example in `P`, `g`, or their interaction with the drifting evidence) - reported as a refutation of this candidate, no tuning |

Two readings are fixed for the secondary surfaces:

- A positive owner-side `C@3` component means freezing removes residual routing
  damage as well, so the clean mapping "routing <-> non-owner, accuracy <->
  owner-side" is not supported, and the report says so.
- The drift probe cannot turn a significant contrast into a mechanism: it is
  reported alongside the contrast, never as the cause of the `C0 - OWNER-ONLY`
  difference.

## 5. Guards

```text
one construction        set_seed then exactly one E2Model per cell
anchor invariance       bitwise, all five metrics, against BOTH studies:
                        new C1 = C1_coupling = C1_intervention
                        new C0 = C0_coupling = C0_intervention
                        new OWNER-ONLY = OWNER-ONLY_intervention
                        one float difference in any of them is a veto (not executed)
cut audit               the V3 audit of INTERVENTION_PREREG.md with Amendment 1,
                        re-run before the cells; failure is a veto
passive probes          no_grad + deepcopy for the evidence snapshots, no
                        autograd.grad anywhere in the study path, no retained graphs
reproducibility         re-running one cell reproduces its metrics identically
no tuning               epochs 10, lr 1e-3, batch 128, rank 8, one prototype per
                        class, T = 20, evidence lambda = 1.0, the pinned
                        construction, feature cache and inference path
```

Artifacts: `experiments/owner_side.py`, `results/owner_side/owner_side_study.json`,
`docs/OWNER_SIDE_RESULTS.md`.

## 6. Feasibility

```text
audit      one audited training run, about a minute
cells      18 x ~20 s ~ 7 min
probes     evidence snapshots per task end plus one final recompute; seconds
total      well under the 12 h ceiling
```

## 7. Out of scope

```text
readout refit / replay    a mitigation, separate study
dose or step-size arms    a sensitivity axis, separate study
P, g, the router, the inference path    unchanged in every arm
other formulations, datasets, backbones    the pinned contract
C0 as a solution          it remains a reference, still below E0 on coverage
the accumulation curve    T in {4,8,20}; separate addendum if wanted
```
