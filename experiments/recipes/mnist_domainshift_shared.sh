#!/usr/bin/env bash
# Class-shared domain-shift stress test: the same 10 classes under rotating
# input phases; compares the pure recipe with the stabilized generalist expert.
#
# Expected runtime: ~10-15 min on GPU.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/run/opengl-driver/lib"
export SSL_CERT_FILE="${SSL_CERT_FILE:-/etc/ssl/certs/ca-bundle.crt}"

exec .venv/bin/python experiments/run_benchmark.py \
  --config configs/mnist_default.json \
  --dataset mnist \
  --methods palmoe \
  --domain_shift rotate \
  --shared_expert \
  --freeze_shared_after 2 \
  --task_free_eval \
  --device cuda
