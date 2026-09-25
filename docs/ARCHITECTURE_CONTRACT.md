# PAL-MoE Architecture Contract (S1)

> **Note (2026-09-26):** still in force; `V3_ARCHITECTURE.md` builds on these interfaces. The v1 modules named in section 3 now live in `pal_moe/legacy/` (old import paths are alias shims), and `pal_moe/evaluation` is `pal_moe/eval`.

Status: implemented, 2026-09-23. Companion to `MEASUREMENT_CONTRACT.md` (S0,
what a run must report) and `PALMOE_V2_SPEC.md` (the v2 design freeze, which
this supersedes on the abstraction question).

S1 changes **no behaviour**. It replaces the fused v1 abstraction with four
interfaces so that a capacity experiment, a readout experiment and a routing
experiment can be run separately. The guard is stated in section 5.

---

## 1. The composition

```
X --Backbone--> Z --Expert--> Z' --Readout--> Y
                       ^
                       |
                     Router
```

| interface | signature | implementations |
| :-- | :-- | :-- |
| `Backbone` | `encode(x) -> z`, `output_dim` | `mlp`, `conv`, `resnet18/34/50`, `vit_b_16/32/l_16`, `cached`, `random` |
| `RepresentationExpert` | `transform(z) -> z'` | `identity`, `residual_adapter`, `mlp` |
| `ClassificationExpert` | `classify(z) -> logits` | `legacy_mlp` (the v1 `MLPExpert`) |
| `Readout` | `fit(z, y)`, `predict(z) -> logits` | `ncm`, `cosine`, `linear`, `logistic`, `ridge`, `mlp` |
| `Router` | `route(z) -> [B, N]`, `top_k(z, k)` | `prototype`, `legacy_linear`, `legacy_distance`, `legacy_attention` |

Registries live in `pal_moe/arch/registry.py`; `describe()` returns the contents
and is the function a result record should call to state which axis was used.

---

## 2. Why two expert protocols, and why the router is separate

v1's `MLPExpert` is `Z -> Y`: it adapts *and* classifies in one module. That is
why the two effects the Stage 1 brief cares about could not be separated before
S1. The E0/L2 measurements make the cost of the conflation concrete:

| rung | configuration | accuracy | params |
| :-- | :-- | --: | --: |
| L2a | one shared expert, trained jointly | **76.62** | 89 K |
| L2b | one shared expert, sequential | 55.67 | 89 K |
| L3 | one expert per task | 70.56 | 323 K |
| L4 | L3 + oracle routing | **97.64** | 323 K |

A single expert has enough *capacity* (L2a beats the joint probe's 75.38), but
sharing it under sequential adaptation is what breaks (L2b), and the bank buys
*isolation* rather than capacity (L3). Then selection costs 27 points (L4).

None of those three sentences can be stated - let alone measured - if the expert
is a classifier head. Hence:

- `RepresentationExpert` (`Z -> Z`) is the MoE expert, and **must be the
  identity at initialization**, so adding one is function-preserving;
- `ClassificationExpert` (`Z -> Y`) exists only so v1 keeps working, and is
  registered under a different kind so the distinction is visible in a config;
- `Router` is its own interface because candidate recall (88.6% at k=3) and
  adaptation quality are different quantities with different failure modes.

---

## 3. Mapping of v1 modules (nothing is migrated, nothing breaks)

| v1 module | contract view | note |
| :-- | :-- | :-- |
| `SharedEncoder` | `SharedEncoderBackbone` | wrapper; `encode` delegates to forward |
| `CachedFeatureEncoder` | `CachedBackbone` | the feature-cache path, identity by definition |
| `MLPExpert` | `LegacyClassificationExpert` | wrapper, no parameters, no numerics changed |
| `DynamicRouter` / `DistanceRouter` / `AttentionRouter` | `LegacyRouter` | `top_k` derived from the same dense distribution |
| `DynamicMoE` | - | untouched; v1 keeps using it directly |

`pal_moe/factory.py` still constructs the v1 objects. The contract is additive:
`pal_moe/arch/` imports from `pal_moe/models/`, never the other way round.

---

## 4. What S1 deliberately does not do

No new loss, no new routing, no NCM optimisation, no dynamic allocation, no
encoder training, no change to any expert architecture, and no attempt to
improve any benchmark number. If a change would move a metric, it does not
belong in S1.

---

## 5. The guard (measured)

Acceptance criterion: `max |delta metric| = 0` on a fixed smoke, with only
timing and the additive contract field masked.

```
before: experiments/run_benchmark.py --dataset mnist --methods naive \
        --epochs 1 --device cpu --seed 42        (pre-S1, S0 commit)
after:  same command
result: 15 non-masked result keys compared, 0 differences (NaN-aware)
        3 apparent differences were `nan != nan`, not changes
```

Additional guards:

- the full test suite: **154 passed** (106 v1 + 18 S0 contract + 30 S1 arch);
- `ruff` and `black --check` clean;
- the registry reproduces a real experiment: `experiments/e0_representation_ceiling.py`
  was switched from its in-script adapter to `pal_moe.arch.ResidualAdapter` and
  re-run; the L2 numbers are identical (76.62 joint / 55.67 sequential,
  89,088 parameters).

---

## 6. What this unlocks for S2 onward

- The complexity ladder becomes a config: `backbone=ncm`-style readout swaps
  (`ncm -> ridge -> logistic -> linear -> mlp`) need no loop change.
- The backbone axis (rule 1) is a `--backbone` value, including `random`; the
  cached path keeps it cheap.
- The expert axis (rule 7) is a `--expert` value; `identity` is the zero-capacity
  control that every claim must beat (rule 8).
- The router axis is measurable in isolation: `top_k` gives candidate recall,
  which is the number the L4 gap is about.
- A result record can state `describe()` so a reader knows which axis moved.

---

## 7. Files

| file | contents |
| :-- | :-- |
| `pal_moe/arch/protocols.py` | the five `Protocol`s and the structural checks |
| `pal_moe/arch/registry.py` | `Registry` + the five registries + `describe()` |
| `pal_moe/arch/backbones.py` | `SharedEncoderBackbone`, `CachedBackbone`, `RandomProjectionBackbone`, `wrap_encoder` |
| `pal_moe/arch/experts.py` | `IdentityExpert`, `ResidualAdapter`, `ResidualMLPExpert`, `LegacyClassificationExpert` |
| `pal_moe/arch/readouts.py` | `NCMReadout`, `CosineReadout`, `LinearReadout`, `LogisticReadout`, `RidgeReadout`, `MLPReadout` |
| `pal_moe/arch/routers.py` | `PrototypeRouter`, `LegacyRouter` |
| `tests/test_arch.py` | 30 contract tests, including the identity-at-init rule |
