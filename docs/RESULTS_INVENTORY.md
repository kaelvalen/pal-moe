# Results Inventory

> What every directory under `results/` contains, what it is evidence for, and
> whether it is current. Regenerate the auto tables with
> `python experiments/paper_report.py` (skips `results/archive/`).
> Status legend: **HEADLINE** (quoted in README/SUNUM), **EVIDENCE**
> (design facts / ablations), **RUN-DAY** (2026-09-22), **PENDING** (queued
> tonight), **ARCHIVE** (exploratory, not cited).

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

## Protocol / correction evidence

| Directory | Contents | Status | Referenced by |
| :-- | :-- | :-- | :-- |
| `cifar10_final_full/` | Corrected CIFAR-10 table (BatchNorm artifact fix, fact 14) | EVIDENCE | BENCHMARK fact 14 |
| `cifar10_lockfix/` | Exact routing-lock reproduction (fact 15) | EVIDENCE | BENCHMARK fact 15 |
| `cifar10_big_frozen/`, `cifar10_big_frozen_full/` | Big CIFAR-10 runs behind facts 11–13 | EVIDENCE | BENCHMARK |
| `cifar100_big_frozen/` | Scaled 20-task CIFAR-100, single seed (fact 15/16) | EVIDENCE | BENCHMARK |
| `cifar100_20task/` | 20-task protocol run (fact 15 provenance) | EVIDENCE | BENCHMARK |
| `cifar100_relgate/`, `cifar100_gate_relative/`, `cifar100_gate_absolute/` | Validation-gate ablation (fact 16) | EVIDENCE | README gate note |
| `ablation/`, `ablation_results.json`, `ablation_comparison.png` | `run_ablation.py` controlled grid (fact 12) | EVIDENCE | BENCHMARK Ablations |
| `mnist_domainshift/` | Class-shared domain-shift pilot (rotate) | EVIDENCE | README §6 |
| `routing_asymmetry_debug.json` | Routing-funnel debug dump; referenced in `ttt.py` comments | EVIDENCE | code comments |
| `feature_cache/` | Persisted frozen features (CIFAR-10/100 ViT; Tiny-ImageNet) | data | recipes |
| `paper_report.md` | Auto-generated tables/figures (report script) | generated | — |

## Run-day experiments (2026-09-22, paper plan E1–E9)

| Directory | Contents | Status | Referenced by |
| :-- | :-- | :-- | :-- |
| `equalbyte/` | E4 feature-cache Pareto, CIFAR-10/100 (real seeds 42 1 2 at 1/4 MiB) | RUN-DAY | EXPERIMENT_PLAN findings |
| `equalbyte_raw/` | E4 raw-pipeline Pareto (raw ER 12,296 B vs PAL 2,184 B per item) | PENDING (wave 2) | — |
| `hybrid_raw/` | True hybrid vs pure in the raw pipeline | PENDING (wave 2) | — |
| `ablation_final/` | E5 component ladder (7 variants × 3 seeds, CIFAR-10) | RUN-DAY | SUNUM §5.5 |
| `growth/` | E7 gated vs forced expansion, routing retention/matrices | RUN-DAY | SUNUM §5.8 |
| `drift/` | E8 anchor-refresh + inference-anchoring cells | RUN-DAY | SUNUM §5.7 |
| `capacity/` | E9 capacity sweep + parameter-matched baselines | RUN-DAY (seeds 1/2 pending) | SUNUM §5.6 |
| `mir/` | E12 MIR baseline | PENDING (wave 2) | — |
| `tinyimagenet_multiseed/` | E10 Tiny-ImageNet 20×10 class-IL | PENDING (wave 2) | — |
| `final_mnist_multiseed/`, `final_c10r18_multiseed/` | E3 regenerations with byte accounting | PENDING (wave 1d) | — |
| `final_c10conv_multiseed/`, `final_c100conv_multiseed/` | E3 slow regenerations | PENDING (wave 2) | — |
| `mnist_domainshift_multiseed/` | E11 5-seed domain-shift pilot | PENDING (wave 2) | — |
| `latency/` | M4 latency JSONs (ResNet-18, ViT) | PENDING (wave 2) | — |
| `appendix/` | AO10 reservoir sampling + online EWC | PENDING (wave 2) | — |

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
- `benchmark_meta_seed*.json` records the seed, git hash, args and duration —
  the provenance anchor for every number.
- A directory is only quoted in `README.md`/`docs/SUNUM.md` after it contains
  its full seed set; otherwise it is labelled single-seed or pending here.
