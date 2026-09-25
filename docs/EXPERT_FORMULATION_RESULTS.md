# Expert formulation E1 (shared decision space) - results

Status: **run. The shared projection leaves the candidate set untouched by
construction, mildly and non-significantly hurts expert usability in `coherent`,
and is neutral in `dispersed`.** Pre-registration:
`docs/EXPERT_FORMULATION_PREREG.md`.

## 1. The four metrics, seed-paired

`E1 - E0`, six seeds, paired:

| metric | regime | per-seed | mean | sd | permutation p | WY-adjusted p |
| :-- | :-- | :-- | --: | --: | --: | --: |
| `Delta Acc` | `coherent` | all negative | **-0.0023** | 0.0008 | 0.0312 | **0.0625** |
| `Delta Acc` | `dispersed` | mixed | **+0.0001** | 0.0004 | 0.6250 | 0.9688 |
| `Delta C@3` | `coherent` | all exactly 0 | **0.0000** | 0.0000 | - | - |
| `Delta C@3` | `dispersed` | all exactly 0 | **0.0000** | 0.0000 | - | - |
| `Delta Oracle@3` | `coherent` | all negative | **-0.0042** | 0.0014 | 0.0312 | **0.0625** |
| `Delta Oracle@3` | `dispersed` | mixed | **+0.0002** | 0.0009 | 0.6875 | 0.9688 |
| `Delta Ceiling@3` | `coherent` | all negative | **-0.0040** | 0.0014 | 0.0312 | **0.0625** |
| `Delta Ceiling@3` | `dispersed` | mixed | **+0.0002** | 0.0008 | 0.6875 | 0.9688 |

Per arm:

| arm | regime | accuracy | C@3 | conditional oracle@3 |
| :-- | :-- | --: | --: | --: |
| E0 | `coherent` | 73.29 | 0.9494 | 0.8811 |
| E1 | `coherent` | 73.06 | 0.9494 | 0.8769 |
| E0 | `dispersed` | 70.66 | 0.8915 | 0.9847 |
| E1 | `dispersed` | 70.66 | 0.8915 | 0.9849 |

## 2. `Delta C@3 = 0` is structural, and the pre-registration was wrong about it

Every one of the twelve cells has `Delta C@3` exactly zero, and that is not an
empirical finding: **the projection sits between the expert and the readout, while
the router scores the raw feature**, so the candidate set cannot change. The
pre-registration said coverage "may move" and would be reported rather than
guarded; that expectation was mistaken and is recorded here as a design
correction, not as a result.

The consequence for the reading is that E1 could only ever move the expert side,
and the four-case table collapses onto the oracle axis in this design.

## 3. Reading

Per the pre-registration's table, `coherent` is the fourth row - the formulation
costs capacity on both sides - and `dispersed` is the "~0" row:

```text
coherent   dC@3 = 0,   dOracle = -0.0042,  dAcc = -0.0023
dispersed  dC@3 = 0,   dOracle = +0.0002,  dAcc = +0.0001
```

with the honest qualification that **nothing survives the family-wise correction**
(every WY p is 0.0625 or higher; the six-seed floor is 0.0312). The direction in
`coherent` is consistent - all six seeds negative on accuracy, oracle and ceiling
- but small, about four tenths of a point of oracle, and not certified.

So: moving every expert's output through one shared, learned projection does not
break the failure mode. It does not fix the ranking (it provably cannot touch it),
and it does not improve expert usability; in the easy-routing regime it costs a
little, in the hard-routing regime it does nothing.

## 4. What this leaves

The chain's invariant now has every component tested, each with a negative or
partial result:

```text
capacity                  -> not the constraint
resolution                -> partial
pointwise learned ranking -> negative
comparative supervision   -> negative
adapted representation    -> trade-off
uniform top-3 aggregation -> no improvement
shared decision space     -> no improvement, structural zero on the ranking
```

**The formulation tested here still kept one expert per task, each an independent
adapter, combined by a decision rule.** What it did not test is the deferred E2:
an expert that produces *evidence* rather than a task-local classifier, which
necessarily changes the training objective as well. The measurements now make
that deferral the only remaining formulation hypothesis, and they also say what
it has to beat: the ranking side cannot be reached from the expert's output path
as long as the router reads the raw feature, so an E2-style expert must be
justified by the evidence side, not by coverage.

## 5. Guards

The E0 anchor (S11's `L3`, keyed with `num_tasks`) and the model-local sharing
check (one readout module and one projection module per model, by object
identity, with no expert carrying its own copy) both passed before the run on the
seed-42 smoke; the study file re-records them per cell for the audit trail. The
WTA-style veto discipline of AGG is not applicable here - there is no second
decision rule - so the anchors are the only vetoes.
