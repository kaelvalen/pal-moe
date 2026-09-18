#!/usr/bin/env bash
# 3-seed error bars for the ViT-B/16 strong-backbone stack (PAL-MoE v2).
# Seeds 1/2 reuse the seed-invariant feature cache written by seed 42
# (identity projection: feature_dim == ViT width == 768).
#
# Expected total runtime: ~30-45 min after the first feature extraction.
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/run/opengl-driver/lib"
export SSL_CERT_FILE="${SSL_CERT_FILE:-/etc/ssl/certs/ca-bundle.crt}"

echo "=== 1/2: CIFAR-10 ViT-B/16, seeds 42 1 2 ==="
.venv/bin/python experiments/run_benchmark_multi.py \
  --seeds "42 1 2" --device cuda \
  --config configs/cifar10_vit.json \
  --output_dir results/cifar10_vit_multiseed

echo "=== 2/2: CIFAR-100 ViT-B/16, seeds 42 1 2 ==="
.venv/bin/python experiments/run_benchmark_multi.py \
  --seeds "42 1 2" --device cuda \
  --config configs/cifar100_vit.json \
  --output_dir results/cifar100_vit_multiseed

echo "VIT_MULTISEED_DONE"
