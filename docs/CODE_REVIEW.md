# Code Review - PAL-MoE codebase

Assessment of structure, modularity and comment style (2026-09-23), written
for the advisor meeting and as a refactor backlog. The run queue is active;
nothing here changes behaviour.

## Verdict

**Suitable for a research codebase: yes.** Clear layering, one concept per
module, consistent trainer/router interfaces, a single result schema, 105
tests and CI. The weak spots are three large files and a flat
`experiments/` directory (now indexed in `experiments/README.md`), not the
architecture.

## Layering (dependency direction is consistent)

```
data/ -> models/ -> memory/ -> adaptation/ + builder/ + trigger/ -> evaluation/
                                (training loop)                 (metrics)
experiments/  orchestrate everything; no method logic lives there.
pal_moe/factory.py  is the only place that constructs routers/models/memory.
```

- `models/`: `SharedEncoder` (MLP/conv/ResNet/ViT), `MLPExpert`,
  `DynamicRouter`/`DistanceRouter`/`AttentionRouter`, `DynamicMoE`.
- `memory/`: `PrototypeMemory` (`v_p/r_p/o_p/x_p/y_p/owner`) + generative
  variant; all persistent state is accounted for by `memory_bytes`.
- `baselines/`: nine trainers behind one interface (`train_task`,
  `update_buffer`, `memory_bytes`) - the equal-byte protocol depends on this
  consistency.
- `adaptation/ttt.py`: the trainer (task phase, calibration, router
  distillation, expansion/freeze, checkpoints).
- `builder/` + `trigger/`: candidate training + validation gate, separate from
  the trainer; the gate can be disabled without touching the loop.
- `evaluation/`: metrics, diagnostics, geometry, heads, calibration, task-free
  streaming.
- `config.py`, `persistence.py`, `merge.py`, `factory.py`: small utilities.

## Per-file notes

| File | Lines | Verdict |
| :-- | --: | :-- |
| `pal_moe/__init__.py` | 31 | clean public surface |
| `models/encoder.py` | 461 | cohesive; `pretrain_contrastive` (86) is the longest method, acceptable |
| `models/expert.py` | 123 | small and focused; function-preserving clone/widen well isolated |
| `models/router.py` | 576 | three routers sharing a method set; consistent, but could split per router later |
| `models/moe.py` | 461 | container + prototype routing hook; fine |
| `memory/prototype_memory.py` | 1061 | large but cohesive; one 137-line method (`_update_or_create_with_matrix`) is the only real smell |
| `memory/generative.py` | 170 | optional toolkit, isolated |
| `adaptation/ttt.py` | 942 | **god class**: `train_task` is 494 lines. Main refactor target |
| `adaptation/losses.py` | 45 | tiny, single purpose |
| `builder/expert_builder.py` | 519 | gate logic + candidate training; the 146-line validate method is long but linear |
| `trigger/*` | 278 | clean; three trigger types behind `evaluate` |
| `data/*` | 661 | simple per-dataset splitters with clear docstrings; no shared base class, but the duplication is shallow and stable |
| `evaluation/*` | 782 | metrics/diagnostics/geometry/heads/calibration/task-free, each small and single-purpose |
| `factory.py` | 174 | the anti-drift constructors; keep |
| `config.py` | 136 | validates configs strictly (unknown keys are errors); good |
| `baselines/*` | 1149 | nine trainers, uniform interface, 55-216 lines each; the most modular part |
| `experiments/run_benchmark.py` | 2173 | **monolith**: CLI + dataset setup + 14 method blocks + reporting. Largest structural weakness |
| `experiments/run_benchmark_multi.py` | 512 | thin driver; `--aggregate_only` supports repairs |
| `experiments/run_ablation.py` | 578 | separate controlled grid; some duplication with the runner, acceptable for now |
| `tests/test_pal_moe.py` | 3100 | one file for 105 tests; fine for a paper, split later if the suite grows |

## Comment style

Measured comment/docstring density: `ttt.py` 16%, `moe.py` 6%, `router.py` 5%,
`prototype_memory.py` 5%; data splitters ~0% comments but clear module and
function docstrings, and simple linear code.

- Comments are overwhelmingly **"why" comments tied to measured effects**
  (e.g. the exact routing-lock artifact, the BatchNorm drift fix, the
  zero-raw-replay caveat). For a research codebase this is the right kind of
  comment; they are the paper's provenance.
- A few block comments are long, but they document decisions that would
  otherwise be re-litigated (weight decay on locked rows, owner-aware
  prototype merging, cache invalidation). **Do not mass-trim.**
- No `TODO/FIXME/HACK` debt anywhere.
- What is missing is *orientation* documentation (how the layers fit), which
  `docs/RESEARCH_MAP.md`, `experiments/README.md` and this file now provide.

## Refactor backlog (after the paper run, in priority order)

1. **Split the benchmark runner** into
   `experiments/methods/{replay_family,palmoe}.py`, `dataset_setup.py` and
   `reporting.py`; keep the CLI thin. Guard: assert identical result JSONs on
   a 1-epoch smoke before/after.
2. **Split `ContinualTrainer.train_task`** into phases
   (`_maybe_expand`, `_train_newest`, `_calibrate`, `_distill_router`,
   `_finalize_lock`) with a unit test per phase.
3. **Extract `_PrototypeIndex`** (matrix caches, stability rows) out of
   `prototype_memory.py` so the memory module stays under ~700 lines.
4. **Move historical configs** (`cifar10_big*` variants cited by facts 11-14)
   into `configs/legacy/` once those facts are re-run with canonical configs.
5. **Split the test file** by module
   (`test_models.py`, `test_memory.py`, `test_adaptation.py`,
   `test_baselines.py`, `test_evaluation.py`); keep `-q` parity in CI.
6. Optional: group `experiments/` into `runners/` and `tools/`; deferred
   because recipes and docs reference the current paths.

None of these change behaviour; they are maintainability work for after the
paper deadline.
