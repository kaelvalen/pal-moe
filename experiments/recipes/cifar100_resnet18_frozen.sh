#!/usr/bin/env bash
# Strong-backbone long-horizon recipe: frozen ImageNet ResNet-18 + feature
# cache on 20-task Split-CIFAR-100 (relative validation gate).
#
# Expected runtime: ~15-25 min (no pretraining, one feature-cache build).
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/run/opengl-driver/lib"
export SSL_CERT_FILE="${SSL_CERT_FILE:-/etc/ssl/certs/ca-bundle.crt}"

exec .venv/bin/python experiments/run_benchmark.py \
  --config configs/cifar100_resnet18_frozen.json \
  --device cuda
