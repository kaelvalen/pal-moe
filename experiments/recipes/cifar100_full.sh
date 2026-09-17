#!/usr/bin/env bash
# Full 20-task Split-CIFAR-100 recipe: frozen encoder + feature cache, 5
# epochs/task, router-owner distillation, all comparison methods.
#
# Expected runtime (RTX 5060 Laptop, 8 GB): ~1.5-2.5 h, dominated by the
# 50-epoch SimCLR pretraining (cached afterwards under ./data/pretrain_cache,
# so re-runs skip it).
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/run/opengl-driver/lib"
export SSL_CERT_FILE="${SSL_CERT_FILE:-/etc/ssl/certs/ca-bundle.crt}"

exec .venv/bin/python experiments/run_benchmark.py \
  --config configs/cifar100_big_frozen.json \
  --device cuda \
  --pretrain_cache
