#!/usr/bin/env bash
# Relative-vs-absolute validation gate ablation on 20-task CIFAR-100.
# Single-seed evidence (results/cifar100_relgate vs results/cifar100_big_frozen)
# was mixed: pure accuracy up (+0.74) but forgetting up (+9.6); hybrid accuracy
# down (-0.7) but forgetting slightly better. This runs both modes across 3
# seeds with the shared pretrain cache for a verdict.
#
# Expected runtime: ~1 h (two pretrainings are cached per seed/config; seed 42
# is already cached from earlier runs).
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/run/opengl-driver/lib"
export SSL_CERT_FILE="${SSL_CERT_FILE:-/etc/ssl/certs/ca-bundle.crt}"

echo "=== relative gate (config default), seeds 42 1 2 ==="
.venv/bin/python experiments/run_benchmark_multi.py \
  --seeds "42 1 2" --device cuda --pretrain_cache \
  --config configs/cifar100_big_frozen.json \
  --methods palmoe,hybrid \
  --output_dir results/cifar100_gate_relative

echo "=== absolute gate (CLI override), seeds 42 1 2 ==="
.venv/bin/python experiments/run_benchmark_multi.py \
  --seeds "42 1 2" --device cuda --pretrain_cache \
  --config configs/cifar100_big_frozen.json \
  --gate_mode absolute \
  --methods palmoe,hybrid \
  --output_dir results/cifar100_gate_absolute

echo "GATE_ABLATION_DONE"
