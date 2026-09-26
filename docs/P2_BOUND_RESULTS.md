# P2-BOUND: why the expert bank adds < 1 pp over a ridge router - results

Pre-registration: `docs/P2_BOUND_PREREG.md` (`2aea973`), amendment 1 (`4a30aad`),
amendment 2 (`af52036`, the review condition for Part B). Runner
`experiments/p2_bound.py` (`1b18147`), committed before the run. Data:
`results/p2_bound/p2_bound_study.json` (untracked) - 12 Part A cells and 12 Part B
cells (2 regimes x seeds 42, 1, 2, 3, 4, 5), device `cuda`, 1196 s.

Status: **run, every veto passing.**

- **Part A:** the owner expert rescues **30 % / 34 %** of the rescuable samples
  (`rho < 0.5` in both regimes). That is the second row of the pre-registered table:
  **the bank's value path, not its boundaries, limits P2.** In `dispersed` the third
  row also applies: breakage (0.27 pp) exceeds rescue (0.23 pp), so there the bank is
  net-neutral by interference.
- **Part B, `dispersed`:** confusion-aligned consolidation raises the rescuable mass
  **14-fold** (`m` 0.0068 -> 0.0975) and gains **+0.53 pp** over arrival order (6/6
  seeds, exact p = 0.031). That is below the 1 pp SESOI, which puts it in the "mass
  created but not converted" row. It is TOST-equivalent to the superclass grouping
  (-0.11 pp, +/-1 pp).
- **Part B is a hindsight-offline upper bound** (amendment 2). It needs **19.8x** the
  stored features of arrival order: 134.0 MB against 6.8 MB.

## 1. Vetoes

```text
decomposition identity     max |terms - P2| = 4.3e-18 (band 1e-12)
right class, wrong task    0 samples in 12 cells (the identity's premise holds)
E-TID2 reproduction        Part A, all five arms, 12/12 cells: |delta| = 0.0
A0 reproduction            Part B A0 via the API vs stored ridge_routed, 12/12: 0.0
identical router           medium-path statistics digest equal across A0/A1/A2, 12/12
no test leakage            only the written train batches reach consolidation
balanced grouping          every A1 / A2 expert owns exactly 5 classes
router purity              0 trainable parameters, every call
reversibility / order      every write, consolidation and forget passes the API guards
feasibility                1196 s (ceiling 2 h)
```

## 2. Part A: the decomposition of P2 (means over six seeds)

`P2 = m * rho + P(not r, not tau) * rho' - P(r) * beta`

| regime | `m` | `rho` | `rho'` | `beta` | rescue `m*rho` | break `P(r)*beta` | P2 | mass ceiling `P2_max` |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: |
| `coherent` | 0.0939 | **0.296** (0.291-0.302) | 0.0004 | 0.0282 | +0.0278 | -0.0217 | **+0.0062** | 0.0940 |
| `dispersed` | 0.0068 | **0.343** (0.250-0.397) | 0.0004 | 0.0035 | +0.0023 | -0.0027 | **-0.0003** | 0.0069 |

- `m` and `P(r)` are seed-independent: the ridge router is closed-form. `rho` and
  `beta` vary with the bank's training seed.
- **Reading, as pre-registered.**
  - Row 1 (`rho >= 0.5`, mass-bound) does not apply.
  - Row 2 (`rho < 0.5` in either regime) applies in both regimes: when the router
    already has the right task but the wrong class, the owner expert fixes only about
    a third of those samples.
  - Row 3 (`P(r) * beta >= m * rho`) applies in `dispersed` (0.0027 >= 0.0023). In
    `coherent` breakage is 78 % of rescue but below it.
- **What the numbers say beyond the rows.**
  - P2 is not small because the bank has nothing to work with. In `coherent` it could
    reach 9.4 pp, and it realises 0.6 pp.
  - Two terms of similar size cancel. The bank rescues 2.8 pp and breaks 2.2 pp of
    samples the ridge readout already had right.

## 3. Part B: consolidation policy on the same data and the same router

Means over six seeds. `A0` = by arrival, `A1` = by confusion (5-fold cross-fitted,
hindsight), `A2` = by superclass.

| regime | arm | accuracy | owner oracle | `m` | `rho` | `beta` | vs `ridge_alone` | stored features |
| :-- | :-- | --: | --: | --: | --: | --: | --: | --: |
| `dispersed` | A0 | 76.74 | 97.30 | 0.0068 | 0.343 | 0.0035 | -0.03 | 6.8 MB |
| `dispersed` | A1 | **77.28** | 87.09 | **0.0975** | 0.282 | 0.0295 | +0.50 | 134.0 MB |
| `dispersed` | A2 | 77.39 | 87.28 | 0.0939 | 0.295 | 0.0282 | +0.62 | 134.0 MB |
| `coherent` | A0 | 77.39 | 87.30 | 0.0939 | 0.296 | 0.0282 | +0.62 | 6.8 MB |
| `coherent` | A1 | 77.32 | 87.06 | 0.0977 | 0.286 | 0.0297 | +0.55 | 134.0 MB |
| `coherent` | A2 | 77.39 | 87.30 | 0.0939 | 0.296 | 0.0282 | +0.62 | 134.0 MB |

(`coherent` A2 == A0 exactly: the arrival order already is the superclass grouping,
ARI 1.0.)

Paired contrasts, six seeds:

| regime | contrast | per seed | mean | 95 % CI | +/- | exact p |
| :-- | :-- | :-- | --: | :-- | :-- | --: |
| `dispersed` | **P3** A1 - A0 | +0.53 +0.42 +0.68 +0.69 +0.35 +0.53 pp | **+0.53 pp** | [+0.39, +0.68] | 6/0 | 0.031 (WY 0.031) |
| `dispersed` | **P4** A1 - A2 | +0.06 -0.28 +0.01 +0.01 -0.27 -0.20 pp | -0.11 pp | [-0.27, +0.05] | 3/3 | 0.25 |
| `dispersed` | A2 - A0 (read together) | | +0.65 pp | [+0.55, +0.74] | 6/0 | 0.031 |
| `coherent` | P3 A1 - A0 | +0.23 -0.54 +0.15 +0.14 -0.25 -0.15 pp | -0.07 pp | [-0.38, +0.24] | 3/3 | 0.56 |

TOST on P4 at +/-1 pp: **equivalent** in both regimes (`dispersed` 90 % CI
[-0.24, +0.02] pp, p_lower 1.7e-5, p_upper 5.5e-6).

The A1 groupings agree with the superclasses only partly: ARI 0.44-0.49 in both
regimes.

**Reading, as pre-registered (with amendment 2).**

- `P3 = +0.53 pp` is below +1 pp, and `m(A1) = 0.0975` is 14x `m(A0) = 0.0068`. That
  is the second row: **mass was created but not converted; the rescue rate is the
  limit.** It agrees with Part A's `rho`.
- P4 is equivalent within +/-1 pp: in accuracy, router confusion recovers what the
  semantic grouping gives, without superclass labels. It does so with a different
  grouping (ARI ~0.46).
- Amendment 2's row takes precedence. Part B is the **hindsight-offline upper bound** of
  confusion-aligned consolidation, at 19.8x the stored feature bytes. A positive P3
  does not make `by_confusion` a default candidate. A continual variant needs its own
  pre-registration.

## 4. The reviewer's written expectation, against the data

Recorded in amendment 2 before the run:

> in `dispersed`, A2 should raise `m` sharply ... `rho` may fall ... the owner oracle
> may move from ~97.3 toward the `coherent` ~87.3 ... `m` rises, `rho` falls and the
> net is unclear - the "mass created but not converted" row.

Every part landed:

- `m`: 0.0068 -> 0.0939.
- `rho`: 0.343 -> 0.295.
- The owner oracle: 97.30 -> 87.28.
- The net: +0.65 pp, positive but below the SESOI - the named row.

## 5. What this says

Across every grouping tried - arrival, confusion, superclass - the whole system lands
between 76.7 and 77.4, within 0.7 pp of the ridge readout alone (76.77). Changing the
expert boundaries moves where the rescuable mass is. It does not change how much of it
the experts convert. The experts rescue about 30 % of the samples they own and break
about 3 % of the samples the readout already had right.

**The variable that remains is the expert's value path:** what an expert does with a
sample the router has placed correctly at the task level. This is the second time the
programme lands there (AC3's section 7 reached it from the address side).

## 6. What this does not say

- Nothing about a continual `by_confusion` (amendment 2): Part B retained all features
  and grouped in hindsight.
- Not that experts cannot help. It says these experts - rank-8 residual adapters with
  a cosine readout, the S11 L3 recipe - rescue about 30 %.
- No claim beyond `T = 20`, CIFAR-100 on the ViT-B/16 cache, six seeds.
- P4's equivalence is an accuracy statement. The groupings themselves differ
  (ARI ~0.46).

## 7. Records

```text
2aea973   pre-registration
4a30aad   amendment 1 (tolerance guards)
af52036   amendment 2 (hindsight-offline bound, stored bytes, A0 veto)
1b18147   the runner (committed before the run)
this      results/p2_bound/p2_bound_study.json summarised
```
