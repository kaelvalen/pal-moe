#!/usr/bin/env bash
# PAL-MoE v2 quick validation: CIFAR-10 + frozen ImageNet ViT-B/16, 15 epochs/task,
# one expert per task (5), refresh anchors after a 10-epoch calibration.
#
# The first run extracts ViT features (~60k images at 224) and saves them under
# results/feature_cache/cifar10_vit_b16/; later runs (including other seeds)
# reuse the cache. Expected: ~5-10 min feature extraction + ~10 min benchmark
# on the 8 GB laptop GPU.
set -u
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/run/opengl-driver/lib"
export SSL_CERT_FILE="${SSL_CERT_FILE:-/etc/ssl/certs/ca-bundle.crt}"

.venv/bin/python experiments/run_benchmark.py \
  --config configs/cifar10_vit.json \
  --device cuda

echo "VIT_CIFAR10_QUICK_DONE"
