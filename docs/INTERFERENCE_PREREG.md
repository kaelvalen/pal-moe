# Interference mechanism: new prototypes against old projections - pre-registration

Status: **proposed, awaiting approval, not started.** It follows
`COUPLING_RESULTS.md`, which localised the break: with everything else identical,
training the older evidence projections (`C1`) collapses the system while freezing
them (`C0`) restores accuracy, so the continual re-alignment of accumulated
projections is the destructive mechanism.

What the ablation did **not** establish is *why*. The hypothesis is:

> every stored prototype contributes to the loss of every old projection, so an old
> expert's evidence is reshaped by tasks it never saw, and the interference
> accumulates with the number of experts.

## 1. The question

> **Does the destructive interference in the old `W_j` arise from new tasks'
> prototypes being applied to old experts, and how does it accumulate with the
> number of experts?**

## 2. Design: instrument the same trajectory, manipulate only the expert count

This study does **not** repeat `C1` vs `C0`. Repeat-measurement of that contrast
would not separate the mechanism from any other coupling effect. It measures the
mechanism directly on the `C1` trajectory, and adds one controlled manipulation.

**2.1 Direct measurements (no manipulation).** During `C1` training, per task `q`
and per old projection `W_j` (`j < q`):

```text
delta_W_j            the change in W_j over task q          (rewrite magnitude)
grad_norm(q -> j)    || grad_{W_j} L_q ||                   (which task damages which)
                     - see Amendment 1: restated in loss form because the
                       gradient probe was not passive
```

**2.2 The owner / non-owner decomposition** - the critical measurement. `L_q` is a
cross-entropy over all seen experts on all stored prototypes, so its gradient
w.r.t. an old `W_j` decomposes by prototype ownership:

```text
grad_{W_j} L_q  =  grad_{W_j} L_q^{owner}  +  grad_{W_j} L_q^{nonowner}
                   prototypes of task j        every other prototype
```

Only the second term is "someone else's evidence". The hypothesis predicts
`||nonowner|| >> ||owner||` and that both grow with the number of accumulated
experts.

**2.3 The expert-count ladder.** The `C1` protocol is run to `T in {2, 4, 8, 12,
20}` tasks and measured at each stopping point. The hypothesis predicts:

```text
T up  ->  non-owner gradient mass up  ->  interference up
      ->  C@3, Acc, Oracle@3 down
```

**What a positive pattern would and would not be.** Non-owner gradient mass
rising while the metrics fall, measured on the same trajectory, is **mechanistic
support** - not a causal claim. Without an intervention that varies the non-owner
mass independently of everything else, "the non-owner gradient caused the
collapse" is stronger than the design can license. The report is worded at the
support level, and the ladder's correlation is explicitly the weaker of the two
forms of evidence it collects.

**No metrics beyond this document.** In particular `Delta W_j` is reported as the
raw norm, as written above; a normalised variant would be a different contract and
is not added here, even if it would ease cross-projection comparison.

**Amendment 1 (pre-run, before any measurement exists under this wording).** The
gradient form of section 2.2 was built and found **not passive**: with no hooks,
no-op hooks, snapshot-only hooks and deepcopy-only hooks the anchored cell returns
`13.04 / 0.9041`, but the full probe returns `12.23 / 0.9032`. The study's own
invariance veto therefore fired, and no result has been produced under the original
wording. Rather than report a perturbing measurement, the decomposition is
restated in its loss form:

```text
owner mass_j    = sum over prototypes p with owner(p) = j     of CE_p
nonowner mass_j = sum over prototypes p with owner(p) != j    of CE_p
asymmetry       = mean nonowner mass / mean owner mass
```

measured on a deepcopy of the model with **no `torch.autograd.grad`, no
`retain_graph` and no `.grad` access at all**. The partition is unchanged (by
prototype ownership, never by an argmax), the question is unchanged, and only the
quantity's form changes - from a gradient norm to the loss the gradient would have
been taken of. `||delta_W_j||` is unchanged, since the snapshot-only probe was
verified passive.

This amendment is recorded before the re-run: the first ladder's cells remain
void, and no interference mechanism claim is made from them.

## 3. Endpoints

**Primary:** `Delta C@3` between consecutive ladder points (the routing damage
curve), with `Acc` and `conditional_oracle@3` reported alongside.

**Mechanistic secondaries**, related to the primary **on the same trajectory**:

```text
mean ||nonowner|| / mean ||owner||      the interference asymmetry
mean ||delta_W_j|| as T grows           the rewrite mass
correlation across ladder points between non-owner mass and each metric
```

## 4. Outcome reading, fixed in advance

| result | reading |
| :-- | :-- |
| `nonowner >> owner` and both grow with `T`, metrics fall | the interference is genuinely cross-task, and it accumulates |
| `nonowner ~ owner`, metrics still fall | the damage is not "someone else's prototype"; the mechanism is elsewhere in the coupling |
| non-owner mass grows but the metrics do not fall | the rewrite is not what breaks the system; the damage is larger than the projection movement |
| no accumulation with `T` | the effect is not cumulative; the 20-expert collapse has a threshold-like cause |

## 5. Guards

```text
instrumentation is passive     probes use torch.autograd.grad on the loss graph
                               BEFORE optimizer.step(), never touching .grad, so
                               the training trajectory is unchanged
exact anchor                   the `C1` run at T = 20 must reproduce
                               COUPLING_RESULTS' C1 numbers exactly
                               (13.04 / 0.9041 at seed 42, coherent)
owner definition               ownership is the stored prototype's task, never the
                               argmax of the current scores
no tuning                      loss, optimizer, epochs, lr, d_e = 128,
                               free-weights-no-bias, seeds and inference path are
                               the pinned E2 contract, unchanged
```

## 6. Feasibility

Measured under the current implementation: **18.4 s per 20-expert cell** (coupling
run), so the ladder is cheap:

```text
grid          T in {2,4,8,12,20} x 2 regimes x 3 seeds = 30 cells
projected     well under 1 h; the 12 h ceiling of the coupling study is not a risk
```

The shortened ladder points cost less than the 20-task cells, so the projection is
an upper bound.

## 7. Out of scope

```text
the shared readout g        a separate coupling variable
the shared query P          likewise
other evidence dimensions, losses or decision rules   the pinned E2 contract
the C1 vs C0 contrast       already localised; repeating it adds nothing
```
