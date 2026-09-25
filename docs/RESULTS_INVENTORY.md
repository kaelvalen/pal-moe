# Results Inventory

What every directory under `results/` contains, what it is evidence for, and
whether it is current. Regenerate the auto tables with
`python experiments/paper_report.py` (skips `results/archive/`).
Status legend: HEADLINE (quoted in README/SUNUM), EVIDENCE (design facts and
ablations), RUN-DAY (2026-09-22/23), PENDING (queued), ARCHIVE (exploratory,
not cited).

## Headline tables

| Directory / file | Contents | Status | Referenced by |
| :-- | :-- | :-- | :-- |
| `benchmark_multi.json` | Split-MNIST, 5 seeds, 12 methods (item budget) | HEADLINE | README §1 |
| `cifar10_conv_multiseed/` | Split-CIFAR-10 conv, 3 seeds | HEADLINE | README §2 |
| `cifar10_resnet18_multiseed/` | Split-CIFAR-10 ImageNet ResNet-18, 3 seeds | HEADLINE | README §3 |
| `cifar100_multiseed/` | Split-CIFAR-100 20-task conv, 3 seeds | HEADLINE | README §4 |
| `cifar100_resnet18/` | CIFAR-100 ResNet-18, single seed (fact 17) | HEADLINE | README §5 |
| `cifar100_resnet18_multiseed/` | CIFAR-100 ResNet-18, 3 seeds (run-day) | HEADLINE | SUNUM §5.2 |
| `cifar10_vit/` | CIFAR-10 ViT-B/16 single seed (pre-promotion) | EVIDENCE | BENCHMARK fact 18 |
| `cifar10_vit_multiseed/` | CIFAR-10 ViT-B/16, 3 seeds (+ latent-replay repair) | HEADLINE | BENCHMARK fact 18 |
| `cifar100_vit_multiseed/` | CIFAR-100 ViT-B/16, 3 seeds | HEADLINE | SUNUM §5.3 |

## Stage 1 (2026-09-25, measurement programme)

Status legend adds: STAGE1 (evidence for `docs/STAGE1_RESULTS.md`).

| Directory / file | Contents | Status | Referenced by |
| :-- | :-- | :-- | :-- |
| `e0/e0_all_vit_b_16_seed42.json` | representation ceiling, adapter ranks 0/8/32/64, oracle routing, router recall@K | STAGE1 | RESULTS F1-F5 |
| `e0/v1_repro_seed42/` | v1 + iCaRL re-run at HEAD (59.30 vs the published 59.34; iCaRL exact) | STAGE1 | RESULTS §2 |
| `e0/e0_adapter*, e0_ncm*, e0_shared*` | mixture / reranking / rejection negatives and the L2 rungs | STAGE1 | PLAN 3, RESULTS F5 |
| `s2/s2_ladder_study.json` | complexity ladder L0-L4, 3 seeds, CIFAR-100/ViT | STAGE1 | RESULTS §3 |
| `s3/s3_backbone_study.json` | six backbones, deltas, transfer, ViT consistency gate | STAGE1 | RESULTS §4 |
| `s3/<backbone>/s2_ladder_study_*.json` | per-backbone ladder, 3 seeds | STAGE1 | RESULTS §4 |
| `s3/s3_caches.json` | the projected-cache index (latent_dim 768, rep_seed) | STAGE1 | PLAN 5 |
| `s7/s7_transfer_cifar10.json` | per-checkpoint transfer curves, few-shot suite, evaluator validation | STAGE1 | RESULTS §5 |
| `s4/s4_dataset_study.json` | four datasets x six levels x three seeds, transfer per cell, S0 contracts, recipe stamp | STAGE1 | RESULTS §6 |
| `logs/` | per-stage logs from `experiments/run_all.py` | STAGE1 | - |

Feature caches (`results/s3/cache_*`, `results/s4/cache_*`, `results/feature_cache/*`)
are gitignored (`*.pt`); `experiments/s3_run.py` and `experiments/s4_datasets.py`
rebuild them from scratch.

## Follow-up chain (2026-09-25, Stage 1 -> owner-side)

The Stage-1 diagnosis and the separately pre-registered follow-ups. Each study has
its own pre-registration and results document; none is retro-fitted.
Status legend adds: CHAIN (evidence for the follow-up chain documents).

| Directory / file | Contents | Status | Referenced by |
| :-- | :-- | :-- | :-- |
| `rr/rr_ranking_study.json` | router ranking, 12 cells; the R2 gate refuted | CHAIN | ROUTER_RANKING_RESULTS.md |
| `rrf/rr_factorial_study.json` | representation x routing, 48 cells | CHAIN | REPRESENTATION_ROUTING_RESULTS.md |
| `dr/decision_routing_study.json` | decision routing, 36 cells | CHAIN | DECISION_ROUTING_RESULTS.md |
| `agg/aggregation_study.json` | aggregation, 24 cells | CHAIN | AGGREGATION_RESULTS.md |
| `ef/expert_formulation_study.json` | expert formulation E1, 24 cells | CHAIN | EXPERT_FORMULATION_RESULTS.md |
| - | E2 evidence, non-executed at 20 experts; postmortem only | CHAIN | E2_EVIDENCE_RESULTS.md |
| `coupling/coupling_study.json` | C1 vs C0, 12 cells | CHAIN | COUPLING_RESULTS.md |
| `interference/interference_study.json` | expert-count ladder, 30 cells (+ `*.void.json`, the vetoed first ladder) | CHAIN | INTERFERENCE_RESULTS.md |
| `intervention/intervention_study.json` | owner-only cut, 18 cells | CHAIN | INTERVENTION_RESULTS.md |
| `owner_side/owner_side_study.json` | two-component decomposition, 18 cells | CHAIN | OWNER_SIDE_RESULTS.md |
| `refit/refit_study.json` | readout refit / replay, 18 cells x 3 readout treatments | CHAIN | REFIT_RESULTS.md |

Chain state: the collapse is decomposed into two causally established components -
a non-owner pathway dominating routing damage and an owner-side update dominating
the accuracy damage - and neither pathway is endpoint-exclusive
(`OWNER_SIDE_RESULTS.md`, section 6). A post-hoc readout refit recovers 88-92% of
the owner-only gap to C0 while a current-task-only refit recovers none, so the
owner-side loss is consistent with a decode/readout mismatch, with a small residual
unexplained (`REFIT_RESULTS.md`).

## Protocol / correction evidence

| Directory | Contents | Status | Referenced by |
| :-- | :-- | :-- | :-- |
| `cifar10_final_full/` | Corrected CIFAR-10 table (BatchNorm artifact fix, fact 14) | EVIDENCE | BENCHMARK fact 14 |
| `cifar10_lockfix/` | Exact routing-lock reproduction (fact 15) | EVIDENCE | BENCHMARK fact 15 |
| `cifar10_big_frozen/`, `cifar10_big_frozen_full/` | Big CIFAR-10 runs behind facts 11-13 | EVIDENCE | BENCHMARK |
| `cifar100_big_frozen/` | Scaled 20-task CIFAR-100, single seed (fact 15/16) | EVIDENCE | BENCHMARK |
| `cifar100_20task/` | 20-task protocol run (fact 15 provenance) | EVIDENCE | BENCHMARK |
| `cifar100_relgate/`, `cifar100_gate_relative/`, `cifar100_gate_absolute/` | Validation-gate ablation (fact 16) | EVIDENCE | README gate note |
| `ablation/`, `ablation_results.json`, `ablation_comparison.png` | `run_ablation.py` controlled grid (fact 12) | EVIDENCE | BENCHMARK Ablations |
| `mnist_domainshift/` | Class-shared domain-shift pilot (rotate) | EVIDENCE | README §6 |
| `routing_asymmetry_debug.json` | Routing-funnel debug dump; referenced in `ttt.py` comments | EVIDENCE | code comments |
| `feature_cache/` | Persisted frozen features (CIFAR-10/100 ViT; Tiny-ImageNet) | data | recipes |
| `paper_report.md` | Auto-generated tables/figures (report script) | generated | - |

## Run-day experiments (2026-09-22, paper plan E1-E9)

| Directory | Contents | Status | Referenced by |
| :-- | :-- | :-- | :-- |
| `equalbyte/` | E4 feature-cache Pareto, CIFAR-10/100 (real seeds 42 1 2 at 1/4 MiB) | RUN-DAY | EXPERIMENT_PLAN findings |
| `equalbyte_raw/` | E4 raw-pipeline Pareto (raw ER 12,296 B vs PAL 2,184 B per item), plus reservoir cells; C10 3 seeds, C100 seed 42 | RUN-DAY | gncl, BENCHMARK fact 20 |
| `hybrid_raw/` | True hybrid vs pure in the raw pipeline, 3 seeds | RUN-DAY | SUNUM §5.5, BENCHMARK fact 21 |
| `ablation_final/` | E5 component ladder (7 variants × 3 seeds, CIFAR-10) | RUN-DAY | SUNUM §5.5 |
| `growth/` | E7 gated vs forced expansion, routing retention/matrices | RUN-DAY | SUNUM §5.8 |
| `drift/` | E8 anchor-refresh + inference-anchoring cells | RUN-DAY | SUNUM §5.7 |
| `capacity/` | E9 capacity sweep + parameter-matched baselines, 3 seeds | RUN-DAY | SUNUM §5.6 |
| `mir/` | E12 MIR baseline, 3 seeds | RUN-DAY | BENCHMARK fact 21 |
| `tinyimagenet_multiseed/` | E10 Tiny-ImageNet 20×10 class-IL, 3 seeds | RUN-DAY | docs/gncl.md, BENCHMARK fact 23 |
| `final_mnist_multiseed/`, `final_c10r18_multiseed/` | E3 regenerations with byte accounting (5/3 seeds) | RUN-DAY | SUNUM §5.1 |
| `final_c10conv_multiseed/`, `final_c100conv_multiseed/` | E3 conv regenerations with byte accounting, 3 seeds | RUN-DAY | SUNUM §5.1, BENCHMARK fact 22 |
| `mnist_domainshift_multiseed/` | E11 domain-shift, 5 seeds | RUN-DAY | SUNUM §5.8 |
| `latency/` | M4 latency JSONs, batch 1/128 (ResNet-18, ViT) | RUN-DAY | BENCHMARK fact 22 |
| `appendix/` | AO10 reservoir sampling (3 seeds) + online EWC (seed 42) | RUN-DAY | BENCHMARK fact 22 |

## Archive (exploratory, untracked, gitignored)

`results/archive/` holds 20 superseded/exploratory run directories (~1.6 GB)
that are not referenced by any document, code path or published number:
early anchor experiments (`mnist_anchor_*`, `cifar10_big_frozen_anchor`,
`cifar10_big_frozen_v2`), exploratory seed sweeps (`cifar10_seeds_a/b`,
`cifar10_final_a/b`, `cifar10_final_seed1/2/4`, `cifar10_multiseed`), the
mechanism sweeps (`sweep/`, `topk/`, `regpol/`, `regpol2/`, `recipe15/`) and
the older MNIST full suite (`mnist_final/`). They can be deleted without
affecting any published table; they are kept in case a design decision needs
to be traced.

## Conventions

- Every result JSON carries `memory_bytes` (stored data), `state_bytes`
  (Fisher/snapshots) and `stored_bytes`; PAL runs also carry
  `prototype_elements`, `stores_raw`, `experts_per_task` and, with
  `--track_routing`, `routing_retention`.
- `benchmark_meta_seed*.json` records the seed, git hash, args and duration -
  the provenance anchor for every number.
- A directory is only quoted in `README.md`/`docs/SUNUM.md` after it contains
  its full seed set; otherwise it is labelled single-seed or pending here.
