# Experiments index

The runner scripts orchestrate `pal_moe/`; they do not contain method logic.
All of them share the factories in `pal_moe/factory.py` and the result schema
from `_record_baseline_result`, so method variants cannot drift apart.

## Runners

| Script | Purpose | Typical use |
| :-- | :-- | :-- |
| `run_benchmark.py` | The benchmark: 14+ method ids, all knobs, single seed | `--config configs/...` |
| `run_benchmark_multi.py` | Multi-seed driver (mean ± std), `--aggregate_only` to rebuild an aggregate | `--seeds "42 1 2" --config ...` |
| `run_ablation.py` | Controlled grid: loss components, init, gate, top-k, encoder | `--configs "OOD"` |
| `run_pure_explore.py` | Fast pure/hybrid mechanism sweeps during development | - |

## Tools

| Script | Purpose |
| :-- | :-- |
| `paper_report.py` | Scan `results/` → `results/paper_report.md` + Pareto/growth/latency figures (skips `results/archive/`) |
| `measure_latency.py` | Per-sample forward latency (batch 1/128) for the runner geometries |
| `prepare_tiny_imagenet.py` | Flatten the official Tiny-ImageNet train layout into an ImageFolder tree (symlinks) |
| `repair_missing_rows.py` | Merge a single-method re-run into existing per-seed JSONs (dry run by default) |
| `diagnose_checkpoint.py` | Per-task expert-accuracy / routing-share diagnosis (router vs expert bottleneck) |
| `debug_routing_asymmetry.py` | Early routing-funnel debug dump (`results/routing_asymmetry_debug.json`) |
| `plot_results.py` | Standalone figure generation for single/multi-seed JSONs |
| `merge_experts.py` | Merge trained experts into one serving head (soup/TIES/task arithmetic) |

## Recipes (`recipes/`)

| Script | Purpose |
| :-- | :-- |
| `paper_all.sh`, `paper_all_resume.sh` | Full paper queue (wave 1 → wave 2), one detached command |
| `paper_wave1b.sh`, `paper_wave1c.sh`, `paper_wave1d.sh` | Consolidation waves (equal-byte, growth, ablation, drift, capacity) |
| `paper_wave2.sh` | Raw equal-byte, MIR, Tiny-ImageNet, slow regenerations, latency, domain-shift |
| `paper_status.sh` | Progress/FAILED/summary check for the running queue |
| `multiseed_cifar.sh`, `cifar100_resnet18_multiseed.sh`, `vit_cifar_multiseed.sh` | 3-seed error bars per benchmark |
| `cifar100_full.sh`, `cifar100_resnet18_frozen.sh`, `cifar10_resnet18_frozen.sh`, `vit_cifar_quick.sh` | Single-benchmark runs |
| `cifar100_gate_ablation.sh`, `readout_ablation.sh` | Focused ablations |
| `memory_pareto.sh`, `mnist_domainshift_shared.sh` | Memory-budget sweep and domain-shift pilot |

`legacy/` keeps superseded one-off scripts (see its README).

## Running a benchmark

```bash
source .venv/bin/activate
export PYTHONPATH=.

# single seed, canonical MNIST recipe
python experiments/run_benchmark.py --config configs/mnist_default.json --device cuda

# 3-seed CIFAR-100 ResNet-18
python experiments/run_benchmark_multi.py --seeds "42 1 2" --device cuda \
  --config configs/cifar100_resnet18_frozen.json \
  --output_dir results/cifar100_resnet18_multiseed
```

The runner writes `benchmark_results_seed<s>.json` (metrics + byte accounting +
routing diagnostics) and `benchmark_meta_seed<s>.json` (seed, git hash, args,
duration) into the output directory.
