# PAL-MoE v3: one fixed address space, three time scales

Status: **implemented (phases 0-5 of the v3 restructure, 2026-09-26), no new
scientific claim.** Supersedes `PALMOE_V2_SPEC.md` as the architecture; extends the
S1 contract (`ARCHITECTURE_CONTRACT.md`) whose interfaces it reuses. Every design
decision below is traced to the result that motivates it (section 6), and every
stored result the restructure could touch was re-run and reproduced (section 7).

## 1. The constraints

1. The model must learn a single sentence or a new batch **at any time after
   deployment**. Learning is an API call, not a training run.
2. It is a Mixture-of-Experts or an equivalent sparse / modular system.
3. No existing result changes.

## 2. The architecture

```text
frozen base (immutable, content-hashed)
   └─► hidden state h at chosen layer(s) = FIXED ADDRESS (key)
          │  parameter-free retrieval (cosine, exact; ANN later behind the same calls)
   ┌──────┼──────────────────────────┬───────────────────────────────┐
   ▼                                 ▼                               ▼
 FAST: single item              MEDIUM: new batch               SLOW: consolidation
 append-only KV memory          closed-form linear edit          frozen representation experts
 key = h, value = label/text    float64 sufficient statistics    boundaries = a policy
 O(1) write, exact delete       one accumulator, forget=downdate (by_arrival | by_confusion*)
                                                                 immutable after the fit
```

`*` gated behind `P2_BOUND_PREREG.md`.

**Keys are never trained.** The backbone is frozen and hashed (`core/`), and whatever
it emits at the chosen layer is the address (`address/`). No component downstream
holds a learned address.

**Router.** Parameter-free, reads only frozen keys and registered buffers, asserted at
runtime. Two implementations behind one interface (`router/`):

| name | rule | origin |
| :-- | :-- | :-- |
| `prototype` | best cosine to a stored class mean, per expert | E0 / AC3 |
| `ridge_class` | continual class-level ridge on the medium path's statistics; task = owner of the argmax class | E-TID2 |

**Experts.** `RepresentationExpert` (`Z -> Z`, identity at init, function-preserving,
S1), trained only at consolidation and frozen afterwards. Which data trains which
expert is a pluggable policy (`experts/policies.py`): `by_arrival` (one expert per
written task, the Stage 1 behaviour, the control) and `by_confusion` (spectral
clustering of the router's confusion matrix, balanced; experimental, refuses to run
unless `P2_BOUND_PREREG.md` is named).

**Readout.** The S1 registry (`readout/`, same objects as `pal_moe.arch.readouts`).
The medium path's ridge is a separate float64 implementation (`edit/stats.py`); the
float32 `RidgeReadout` is kept for bitwise reproduction of stored results.

**Medium path mechanics.** For a ridge-fitted linear map, `A = sum KᵀK`, `B = sum KᵀV`
are additive over edits. They are held in **one float64 accumulator pair** whose size
does not depend on the number of edits; learning adds an edit's contribution,
`forget` **subtracts** it (a downdate). To be able to subtract, each edit keeps the
smaller of its two representations: its factor rows `(K, V)` when `n <= d'` (a fact, a
small batch) or its `(dA, dB)` when `n > d'` (a large batch, e.g. a 2500-sample vision
task at d' = 769). Order invariance and forget are therefore **fp-close, not bitwise**,
and the guards measure how close (section 4). Bitwise equality holds where it is real:
when the last medium edit is forgotten the accumulators are reset to the prior.

On the LM the same structure is the MEMIT closed form `Delta = (C0 + S)^-1 B`
(`edit/down_proj.py`): `S`, `B` are single accumulators (one `d_ff x d_ff` matrix, not
one per edit), each edit keeps its `(K, R)` rows (`n x d_ff`, ~150 KB per fact at
d_ff = 18944), residuals are computed against the base model so contributions stay
independent, and forget subtracts. `mode="woodbury"` solves the same statement over the
stacked rows without the `d_ff^2` accumulator for large `d_ff` and few edits. **`C0`
must come from a corpus**: `estimate_key_covariance(lm.collect_keys(corpus))` over every
token position; `DownProjEdit` refuses anything that is not such a `KeyPrior`, and a
prior estimated from fewer tokens than `d_ff` (rank-deficient). The removal of the
last edit removes the hook and restores the base model bitwise.

## 3. The API

```python
from pal_moe.api import PalMoE, Batch, Example

model = PalMoE(dim=768, num_classes=100, router="ridge_class", canary=canary_feats)
rec  = model.write(Batch(z, y, task=0))    # MEDIUM -> EditRecord
rec  = model.write(Example(z1, 7))         # FAST   -> EditRecord
model.forget(rec.id)                       # exact: memory delete / statistics removal
rep  = model.consolidate("by_arrival")     # SLOW   -> ConsolidationReport
pred = model.predict(z_test)               # labels, logits, routed expert ids + scores,
                                           # per-sample source (memory | experts | medium)
st   = model.state()                       # StateHash(base_hash, ordered edit log, digest)
```

| call | returns |
| :-- | :-- |
| `write(item)` | `EditRecord(id, kind, content_hash, order_hash, locality_report, reversibility_report, order_report, purity_report)` |
| `forget(edit_id)` | a report: the downdate's drift against a from-scratch re-sum, and (if the resulting state was visited before) canary argmax identity and max \|delta output\| against that visit |
| `consolidate(policy, prereg=None)` | `ConsolidationReport(record, policy, groups, experts_added, frozen_parameters)` |
| `predict(x)` | `Prediction(labels, logits, expert_ids, expert_scores, source, memory_hits, medium_labels)` |
| `state()` | `StateHash(base_hash, edits, digest)` |

The LM facade (`pal_moe.api.lm.PalMoELM`) has the same calls over
`pal_moe.core.hf_lm.HFCausalLM`: `write("a sentence")` is a FAST memory row
(retrieved into the context), `write(Batch(prompts, targets))` a MEDIUM down-projection
edit; its `consolidate` is declared and raises (the LM slow path is not built).

## 4. The guards (every `write` / `forget` / `consolidate`)

| # | guard | how it is checked | on failure |
| :-- | :-- | :-- | :-- |
| 1 | **Locality** | canary argmax flip rate <= epsilon (per path: fast default 0, medium / consolidation logged), plus max \|delta output\| | roll back, raise (or log) |
| 2 | **Reversibility** | inside every write a trial undo/redo: after the undo, max\|dW\| against the pre-write solution <= `tolerance` (1e-10), canary argmax identical and max\|delta output\| <= `output_tolerance`; `bitwise` reported separately (true for FAST writes, consolidations and a return to zero medium edits). On every forget: accumulator vs re-sum (downdate drift) <= `tolerance`, and the canary against the last visit of that state | roll back, raise |
| 3 | **Order invariance** | the accumulator vs the live contributions re-summed in a seeded random permutation: max\|dW\| <= `tolerance`, canary argmax identical. A corrupted accumulator fails it (tested) | roll back, raise |
| 4 | **Router purity** | trainable scalars reachable from the router == 0 (inspects the object, not its self-report) | raise |

Implementation: `pal_moe/api/guards.py` (`GuardedEditor`), shared by the vision and LM
facades.

## 5. Package layout

```text
pal_moe/
  core/        backbones (FrozenFeatureBackbone, FrozenModuleBackbone), hf_lm (HF causal LM
               with hidden-state / down-proj hooks), features + constructions (moved from
               the runners), hashing
  address/     ExactCosineIndex (parameter-free retrieval)
  router/      prototype, ridge_class, the purity guard
  memory/      FastMemory (fast path); prototype_memory / generative are v1 aliases
  edit/        LinearStats (float64 accumulator, downdate); DownProjEdit (LM, corpus prior)
  experts/     ladder (the Stage 1 bank, moved), policies (by_arrival, by_confusion)
  readout/     the S1 readout registry (same objects as pal_moe.arch.readouts)
  api/         PalMoE, PalMoELM, GuardedEditor, records
  eval/        metrics, schema (measurement contract), stats (moved from s11), editing
               (CounterFact / zsRE / MQuAKE / canary harness)
  legacy/      v1 modules, behaviour frozen bitwise
  arch/        the S1 contract (protocols, registries), unchanged
experiments/   thin runners; old names re-exported from the package
```

Moved code and its shims:

| moved from | to | shim |
| :-- | :-- | :-- |
| `experiments/s2_ladder.py` (LevelSpec, LADDER, LEVELS_BY_NAME, CLOSED_FORM_READOUTS, LadderModel, forward_transfer, `_entropy`) | `pal_moe/experts/ladder.py` | re-exported by `s2_ladder` |
| `experiments/s2_ladder.py` (load_tasks, iter_batches, set_seed) | `pal_moe/core/features.py` | re-exported by `s2_ladder` |
| `experiments/s11_confirmatory.py` (paired_stats, signed_rank_statistic, westfall_young, tost, holm) | `pal_moe/eval/stats.py` | re-exported by `s11_confirmatory` |
| `experiments/s11_confirmatory.py` (train_model, evaluate) | `pal_moe/experts/ladder.py` | re-exported by `s11_confirmatory` |
| `experiments/s6b_difficulty.py` (superclass_of, args_data_dir, build_construction, separability) | `pal_moe/core/constructions.py` | re-exported by `s6b_difficulty` |
| `pal_moe/evaluation/*` | `pal_moe/eval/*` | `sys.modules` alias package |
| `pal_moe/{models,adaptation,baselines,builder,trigger}` | `pal_moe/legacy/...` | `sys.modules` alias packages |
| `pal_moe/{factory,merge,persistence}.py`, `pal_moe/memory/{prototype_memory,generative}.py` | `pal_moe/legacy/...` | `sys.modules` alias modules |

An alias shim makes the old dotted name *be* the new module object, so a class is
never duplicated and a monkeypatch through either name is seen through both
(`pal_moe/legacy/_alias.py`; asserted in `tests/test_v3_anchors.py`).

## 6. Decision -> evidence

| decision | motivating result | file |
| :-- | :-- | :-- |
| keys / addresses are never trained | a learned shared query collapses routing when frozen (-28.7 pp, AC1); continually re-aligned evidence projections interfere destructively (C1) | `AC1_ADDRESS_FREEZE_RESULTS.md`, `COUPLING_RESULTS.md` |
| parameter-free retrieval is the routing path | fixed prototype retrieval is a drop-in for the learned address, TOST-equivalent at +/-1 pp (+0.048 pp), router params 0 | `AC3_ADDRESS_SPACE_RESULTS.md` |
| batch learning is closed-form, order-invariant, reversible | continual ridge == one-shot ridge (E-TID2 G3: 0 argmax mismatches); statistics are additive, so removal is exact | `E_TID2_RESULTS.md` |
| float64 statistics; order and forget measured against a tolerance | float32 continual vs one-shot max\|dW\| 1.26e-3 / 1.09e-3 (G3); float64 3.6e-13 | `E_TID2_RESULTS.md` |
| a corpus covariance prior for the LM edit | MEMIT's locality mechanism; on a tiny random model the corpus prior moves held-out keys about half as much per unit of edit gain as an edit-only prior (a geometry check, `tests/test_v3_lm.py`) | `edit/down_proj.py` |
| the router is the lever, not expert capacity | oracle routing +27 pp (F3); adapter capacity saturates at rank 8 (F2); capacity does not close the gap (F11); ranking degrades with bank size (F18, F19) | `STAGE1_RESULTS.md` §15 |
| a class-level router mapped to tasks, not a task classifier | offline `lin_class` +7.7 / +10.4 pp C@1 over prototypes; `lin_task` underfits its own training set (0.93 / 0.82) | `E_TID_RESULTS.md` |
| `ridge_class` is a first-class router | continual class-level ridge routing +4.10 / +6.08 pp over E0, 6/6 seeds (P1) | `E_TID2_RESULTS.md` |
| expert boundaries are a policy, `by_arrival` is only the control | the arrival-ordered bank adds +0.62 / -0.03 pp over ridge alone (P2); rescuable mass 9.4 % / 0.7 % | `E_TID2_RESULTS.md`, `P2_BOUND_PREREG.md` |
| `by_confusion` is gated | its motivation is a hypothesis (P2 mass-bound), not a result | `P2_BOUND_PREREG.md` |
| experts stay isolated and frozen | the bank buys isolation, not capacity (F6); sharing under sequential adaptation breaks (L2b) | `STAGE1_RESULTS.md` §15, `ARCHITECTURE_CONTRACT.md` |
| three separate time scales, not one mechanism | fast = one item (memory), medium = closed form (E-TID2), slow = trained experts (Stage 1 bank): three different cost / reversibility profiles | this document |

## 7. Anchors (measured on this machine, RTX 5060 Laptop GPU, cu130)

| anchor | cells | band | before the move | after the move | through the v3 API |
| :-- | --: | :-- | :-- | :-- | :-- |
| S11 E0 `L3_per_task`, T=20, rank 8 (73.29 / 70.66) | 12 | bitwise | max \|d\| 0.0 | max \|d\| 0.0 (accuracy and acc_matrix) | - |
| AC3 cells (`fixed_proto`, `bilinear`, anchor metrics) | 6 | 1e-6 | 0.0 | 0.0 | - |
| E-TID2 cells, all five arms + coverage + G1/G2 | 12 | 1e-6 | 0.0 | 0.0 | 0.0 on all five arms and both coverages, ridge arms included (float64 moved no argmax); G3 max\|dW\| 3.6e-13 (float32 run: 1.26e-3). Accumulator + downdate guards, measured: permutation max\|dW\| <= 4.2e-13, trial-undo max\|dW\| <= 4.2e-13 (tolerance 1e-10), canary argmax identical, consolidation bitwise. Storage: 5.3 MB accumulators + 5.3 MB per 2500-sample task (dA, the minimum to subtract when n > d') |
| v1 S1 smoke (mnist, naive + palmoe, 1 epoch, cpu, seed 42) | 1 | bitwise | - | 335 leaves, 0 differences | - |
| v1 checkpoints written by stage1-final (`tests/fixtures/`) | 2 | bitwise | - | whole-object pickle (old `pal_moe.models.*` paths) and `persistence` checkpoint load through the shims; forward output bitwise | - |

Runner: `experiments/v3_anchors.py` (writes `results/v3/anchors*.json`, untracked).

## 8. Built vs declared

| part | status |
| :-- | :-- |
| vision fast / medium / slow paths, both routers, the four guards | built, tested (`tests/test_v3.py`), E-TID2 reproduced through it |
| `by_confusion` | built, gated; no run exists |
| LM backend (4-bit HF, hooks, fast retrieval, closed-form down-proj edit, guards) | built, tested on a tiny random Llama only (`tests/test_v3_lm.py`); **no LM result exists** |
| LM slow path (frozen LoRA experts) | declared, raises |
| editing harness (CounterFact / zsRE / MQuAKE / canary) | built; loaders verified on the real public releases (21919 / 19086 / 3000 records, sha256 in `V3_LLM_PREREG.md` amendment 1) and metrics on synthetic records; the study is `V3_LLM_PREREG.md`, **not run** |
| ANN retrieval | declared; the exact index is the reference |
| incremental consolidation onto an existing bank | declared, raises (forget the previous consolidation and consolidate everything pending) |
