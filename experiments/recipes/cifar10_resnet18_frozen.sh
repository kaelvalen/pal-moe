#!/usr/bin/env bash
# Strong-representation recipe: ImageNet ResNet-18 backbone (frozen) + feature
# cache on Split-CIFAR-10, no pretraining needed.
#
# Expected runtime: minutes (only the feature cache build).
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/run/opengl-driver/lib"
export SSL_CERT_FILE="${SSL_CERT_FILE:-/etc/ssl/certs/ca-bundle.crt}"

exec .venv/bin/python experiments/run_benchmark.py \
  --config configs/cifar10_resnet18_frozen.json \
  --device cuda
